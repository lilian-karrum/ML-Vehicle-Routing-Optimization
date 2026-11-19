import os
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# configuration 
INSTANCES_DIR = Path(r"C:\Users\lilia\Downloads\CVRPTW\CVRPTW")
OUTPUT_FILE   = INSTANCES_DIR.parent / "cvrptw_results.xlsx"
ITERATIONS    = 300
CANDIDATES    = 8
ELITE_CHECKS  = 2
SEED          = 42

random.seed(SEED)
np.random.seed(SEED)

FLEET_PENALTY      = 1000
INFEASIBLE_PENALTY = 10**7


# load instances
def load_cvrptw(path):
    with open(path) as f:
        data = json.load(f)
    customers = {}
    for k, v in data.items():
        if k.startswith("customer_"):
            cid = int(k.split("_")[1])
            customers[cid] = {
                "x": float(v["coordinates"]["x"]),
                "y": float(v["coordinates"]["y"]),
                "demand": float(v["demand"]),
                "ready": float(v["ready_time"]),
                "due": float(v["due_time"]),
                "service": float(v["service_time"]),
            }
    depot = data["depart"]
    return {
        "name": path.stem,
        "capacity": float(data["vehicle_capacity"]),
        "max_vehicles": int(data.get("max_vehicle_number", 10**9)),
        "dist": np.array(data["distance_matrix"], dtype=float),
        "depot": (float(depot["coordinates"]["x"]), float(depot["coordinates"]["y"])),
        "customers": customers,
        "ids": sorted(customers.keys()),
    }


# check if a route is feasible 
def route_feasible(route, inst):
    dist, cap = inst["dist"], inst["capacity"]
    load = t = 0.0
    prev = 0
    for cid in route:
        c = inst["customers"][cid]
        load += c["demand"]
        if load > cap:
            return False
        t = max(t + dist[prev][cid], c["ready"])
        if t > c["due"]:
            return False
        t += c["service"]
        prev = cid
    return True


# calculate distance for one rout 
def route_distance(route, dist):
    if not route:
        return 0.0
    total = dist[0][route[0]]
    for i in range(len(route) - 1):
        total += dist[route[i]][route[i + 1]]
    return total + dist[route[-1]][0]

# calculate distance for all routes
def total_distance(routes, dist):
    return sum(route_distance(r, dist) for r in routes)

# calculate objective score
def score(routes, feasible, inst):
    val = FLEET_PENALTY * len(routes) + total_distance(routes, inst["dist"])
    if not feasible:
        val += INFEASIBLE_PENALTY
    return val


# greedy decoder to convert sequence of customers to routes
def build_routes(sequence, inst):
    dist = inst["dist"]
    routes = []
    for cid in sequence:
        best_ri = best_pos = None
        best_extra = float("inf")
        for ri, route in enumerate(routes):
            for pos in range(len(route) + 1):
                if route_feasible(route[:pos] + [cid] + route[pos:], inst):
                    before = 0 if pos == 0 else route[pos - 1]
                    after  = 0 if pos == len(route) else route[pos]
                    extra  = dist[before][cid] + dist[cid][after] - dist[before][after]
                    if extra < best_extra:
                        best_extra, best_ri, best_pos = extra, ri, pos
        if best_ri is not None:
            routes[best_ri].insert(best_pos, cid)
        else:
            nr = [cid]
            if not route_feasible(nr, inst):
                return routes + [nr], False
            routes.append(nr)
    if len(routes) > inst["max_vehicles"]:
        return routes, False
    return routes, True


# initial sequences 
#earliest deadlines first
def due_sequence(inst):
    return sorted(inst["ids"], key=lambda c: (inst["customers"][c]["due"], inst["customers"][c]["ready"]))

# earliest start time first
def ready_sequence(inst):
    return sorted(inst["ids"], key=lambda c: (inst["customers"][c]["ready"], inst["customers"][c]["due"]))

# largest demand first
def demand_sequence(inst):
    return sorted(inst["ids"], key=lambda c: (-inst["customers"][c]["demand"], inst["customers"][c]["due"]))

# always go to the nearest neighbour 
def nearest_neighbor_sequence(inst):
    dist = inst["dist"]
    remaining = set(inst["ids"])
    seq, current = [], 0
    while remaining:
        nxt = min(remaining, key=lambda c: dist[current][c])
        seq.append(nxt); remaining.remove(nxt); current = nxt
    return seq

# sort customers by angle around depot: creates circular routes
def angle_sequence(inst):
    dx, dy = inst["depot"]
    return sorted(inst["ids"], key=lambda c: np.arctan2(
        inst["customers"][c]["y"] - dy, inst["customers"][c]["x"] - dx))

def initial_sequences(inst):
    return [due_sequence(inst), ready_sequence(inst), demand_sequence(inst),
            nearest_neighbor_sequence(inst), angle_sequence(inst)]


# generates new candidate solutions
def mutate(seq):
    s = seq.copy(); n = len(s)
    if n < 2: return s
    op = random.choice(["swap", "reverse", "insert"])
    if op == "swap":
        i, j = random.sample(range(n), 2); s[i], s[j] = s[j], s[i]
    elif op == "reverse":
        i, j = sorted(random.sample(range(n), 2)); s[i:j+1] = s[i:j+1][::-1]
    else:
        i, j = random.sample(range(n), 2); node = s.pop(i); s.insert(j, node)
    return s


# improve route, and checks feasibility after swap
def two_opt_route(route, inst, max_passes=2):
    if len(route) < 4: return route[:]
    dist = inst["dist"]; best = route[:]
    for _ in range(max_passes):
        improved = False; n = len(best)
        for i in range(n - 1):
            for j in range(i + 1, n):
                a = 0 if i == 0 else best[i-1]; b = best[i]
                c = best[j];                     d = 0 if j == n-1 else best[j+1]
                if (dist[a][c] + dist[b][d]) - (dist[a][b] + dist[c][d]) < -1e-9:
                    cand = best[:i] + best[i:j+1][::-1] + best[j+1:]
                    if route_feasible(cand, inst):
                        best = cand; improved = True
        if not improved: break
    return best

# apply two opt to all routes 
def apply_2opt(routes, inst):
    return [two_opt_route(r, inst) for r in routes]

# move customers between routes
def relocate_between_routes(routes, inst, max_improvements=20):
    best = [r[:] for r in routes]; best_dist = total_distance(best, inst["dist"])
    improvements = 0; improved = True
    while improved and improvements < max_improvements:
        improved = False
        for i in range(len(best)):
            if improved: break
            for j in range(len(best)):
                if improved: break
                if i == j or not best[i]: continue
                for pos_from in range(len(best[i])):
                    if improved: break
                    cid = best[i][pos_from]
                    new_from = best[i][:pos_from] + best[i][pos_from+1:]
                    if new_from and not route_feasible(new_from, inst): continue
                    for pos_to in range(len(best[j]) + 1):
                        new_to = best[j][:pos_to] + [cid] + best[j][pos_to:]
                        if not route_feasible(new_to, inst): continue
                        new_routes = []
                        for k, r in enumerate(best):
                            if k == i:
                                if new_from: new_routes.append(new_from)
                            elif k == j: new_routes.append(new_to)
                            else:        new_routes.append(r[:])
                        nd = total_distance(new_routes, inst["dist"])
                        if len(new_routes) < len(best) or nd + 1e-9 < best_dist:
                            best = new_routes; best_dist = nd
                            improved = True; improvements += 1; break
    return best

# used for quick checks
def fast_eval(seq, inst):
    routes, feasible = build_routes(seq, inst)
    return score(routes, feasible, inst), routes, feasible

# final high-quality solution
def full_eval(seq, inst):
    routes, feasible = build_routes(seq, inst)
    if feasible:
        routes = apply_2opt(routes, inst)
        routes = relocate_between_routes(routes, inst)
        routes = apply_2opt(routes, inst)
    return score(routes, feasible, inst), routes, feasible


# convert solutions into numbers
def extract_features(routes, feasible, inst):
    if not routes: return np.zeros(12)
    dist = inst["dist"]
    rl = [len(r) for r in routes]
    rd = [sum(inst["customers"][c]["demand"] for c in r) for r in routes]
    rdist = [route_distance(r, dist) for r in routes]
    tw = [inst["customers"][c]["due"] - inst["customers"][c]["ready"] for r in routes for c in r]
    return np.array([
        len(routes), total_distance(routes, dist),
        float(np.mean(rl)), float(np.max(rl)), float(np.std(rl)),
        float(np.mean(rd)), float(np.max(rd)), float(np.std(rd)),
        float(np.mean(rdist)), float(np.std(rdist)),
        float(np.mean(tw)) if tw else 0.0,
        0.0 if feasible else 1.0,
    ])


# advanced search operator: remove customers, reinsert them optimally
def destroy_and_repair(sequence, inst, remove_frac=0.12):
    seq = sequence[:]
    n_remove = max(2, int(len(seq) * remove_frac))
    removed = random.sample(seq, n_remove)
    partial = [c for c in seq if c not in removed]
    routes, feasible = build_routes(partial, inst)
    if not feasible and not routes: return mutate(sequence)
    for cid in removed:
        best_ri = best_pos = None; best_extra = float("inf")
        for ri, route in enumerate(routes):
            for pos in range(len(route) + 1):
                cand = route[:pos] + [cid] + route[pos:]
                if route_feasible(cand, inst):
                    before = 0 if pos == 0 else route[pos-1]
                    after  = 0 if pos == len(route) else route[pos]
                    extra  = inst["dist"][before][cid] + inst["dist"][cid][after] - inst["dist"][before][after]
                    if extra < best_extra:
                        best_extra, best_ri, best_pos = extra, ri, pos
        if best_ri is not None:
            routes[best_ri].insert(best_pos, cid)
        else:
            nr = [cid]
            if route_feasible(nr, inst): routes.append(nr)
            else: return mutate(sequence)
    return [cid for route in routes for cid in route]


# no ML, just heuristics (greedy)
def solve_baseline(inst):
    seeds = initial_sequences(inst)
    best_score = float("inf"); best_routes = None; best_feasible = False
    for seq in seeds:
        s, routes, feasible = full_eval(seq, inst)
        if feasible and (not best_feasible or s < best_score):
            best_score, best_routes, best_feasible = s, routes, feasible
    if best_routes is None:
        for seq in seeds:
            s, routes, feasible = full_eval(seq, inst)
            if s < best_score:
                best_score, best_routes, best_feasible = s, routes, feasible
    return best_routes, best_score, best_feasible


# hybrid with ML
def solve_hybrid(inst, iterations=300, candidates_per_iter=8, elite_exact_checks=2):
    seeds = initial_sequences(inst)
    best_seq = None; best_score = float("inf"); best_routes = None; best_feasible = False

    for seq in seeds:
        s, routes, feasible = full_eval(seq, inst)
        if feasible and (not best_feasible or s < best_score):
            best_seq, best_score, best_routes, best_feasible = seq[:], s, routes, feasible
    if best_seq is None:
        for seq in seeds:
            s, routes, feasible = full_eval(seq, inst)
            if s < best_score:
                best_seq, best_score, best_routes, best_feasible = seq[:], s, routes, feasible

    X, y = [], []
    for base in seeds:
        seqs = [base[:]]
        for _ in range(20):
            s = base[:]
            if random.random() < 0.35: s = destroy_and_repair(s, inst)
            else:
                for _ in range(random.randint(1, 3)): s = mutate(s)
            seqs.append(s)
        for seq in seqs:
            routes, feasible = build_routes(seq, inst)
            sc = score(routes, feasible, inst)
            X.append(extract_features(routes, feasible, inst)); y.append(sc)
    X, y = np.array(X), np.array(y)
    scaler = StandardScaler().fit(X)
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=SEED, n_jobs=-1)
    model.fit(scaler.transform(X), y)

    exact_calls = 0
    for it in range(iterations):
        cand_seqs, cand_feats = [], []
        for _ in range(candidates_per_iter):
            seq = best_seq[:]
            r = random.random()
            if r < 0.30:   seq = destroy_and_repair(seq, inst)
            elif r < 0.80:
                for _ in range(random.randint(1, 3)): seq = mutate(seq)
            else:           seq = mutate(destroy_and_repair(seq, inst))
            routes_pred, feasible_pred = build_routes(seq, inst)
            cand_seqs.append(seq)
            cand_feats.append(extract_features(routes_pred, feasible_pred, inst))

        preds = model.predict(scaler.transform(np.array(cand_feats)))
        for idx in np.argsort(preds)[:elite_exact_checks]:
            exact_calls += 1
            s, routes, feasible = fast_eval(cand_seqs[idx], inst)
            if feasible and (not best_feasible or s < best_score):
                best_seq, best_score, best_routes, best_feasible = cand_seqs[idx][:], s, routes, True
            elif not best_feasible and s < best_score:
                best_seq, best_score, best_routes, best_feasible = cand_seqs[idx][:], s, routes, feasible

        if (it + 1) % 100 == 0:
            routes, feasible = build_routes(best_seq, inst)
            X = np.vstack([X, extract_features(routes, feasible, inst)])
            y = np.append(y, score(routes, feasible, inst))
            scaler = StandardScaler().fit(X)
            model.fit(scaler.transform(X), y)

    s, best_routes, best_feasible = full_eval(best_seq, inst)
    return best_routes, s, best_feasible, exact_calls


# save results 
def save_excel(results, output_path):
    wb = Workbook()
    ws = wb.active
    ws.title = "Results"

    DARK_BLUE   = "1F3864"
    MED_BLUE    = "2E75B6"
    LIGHT_GREEN = "E2EFDA"
    LIGHT_RED   = "FCE4D6"
    LIGHT_GRAY  = "F2F2F2"
    WHITE       = "FFFFFF"
    thin = Side(style='thin', color='BFBFBF')
    tb   = Border(left=thin, right=thin, top=thin, bottom=thin)

    section_defs = [
        ("Instance Info",  4, "1F3864"),
        ("Baseline",       5, "2E75B6"),
        ("Hybrid",         6, "1A5276"),
        ("Comparison",     3, "512E5F"),
    ]
    col = 1
    for label, span, color in section_defs:
        start = get_column_letter(col); end = get_column_letter(col + span - 1)
        c = ws.cell(row=1, column=col, value=label)
        c.font = Font(name='Arial', bold=True, color=WHITE, size=10)
        c.fill = PatternFill('solid', start_color=color)
        c.alignment = Alignment(horizontal='center', vertical='center')
        ws.merge_cells(f'{start}1:{end}1')
        col += span
    ws.row_dimensions[1].height = 18

    headers = [
        'Instance','# Customers','Capacity','Max Vehicles',
        'Baseline Obj','Baseline Vehicles','Baseline Distance','Baseline Time (s)','Baseline Feasible',
        'Hybrid Obj','Hybrid Vehicles','Hybrid Distance','Hybrid Time (s)','Hybrid Feasible','Exact Calls',
        'Δ Objective','Δ Obj (%)','Winner'
    ]
    bg_by_col = [DARK_BLUE]*4 + ["2E75B6"]*5 + ["1A5276"]*6 + ["6C3483"]*3

    ws.row_dimensions[2].height = 30
    for ci, (h, bg) in enumerate(zip(headers, bg_by_col), 1):
        c = ws.cell(row=2, column=ci, value=h)
        c.font = Font(name='Arial', bold=True, color=WHITE, size=9)
        c.fill = PatternFill('solid', start_color=bg)
        c.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        c.border = tb

    for ri, row in enumerate(results, 3):
        fill_color = WHITE if ri % 2 == 0 else LIGHT_GRAY
        ws.row_dimensions[ri].height = 16
        values = [
            row['instance'], row['num_customers'], row['capacity'], row['max_vehicles'],
            row['baseline_obj'], row['baseline_K'], row['baseline_distance'], row['baseline_time_s'], row['baseline_feasible'],
            row['hybrid_obj'], row['hybrid_K'], row['hybrid_distance'], row['hybrid_time_s'], row['hybrid_feasible'], row['exact_calls'],
            row['delta_obj'], row['delta_obj_pct'], row['winner']
        ]
        fmts = [None,None,None,None,'#,##0.00',None,'#,##0.00','0.00',None,'#,##0.00',None,'#,##0.00','0.00',None,None,'#,##0.0','0.000',None]
        for ci, (val, fmt) in enumerate(zip(values, fmts), 1):
            c = ws.cell(row=ri, column=ci, value=val)
            c.font = Font(name='Arial', size=9)
            c.border = tb
            c.alignment = Alignment(horizontal='center', vertical='center')
            if fmt: c.number_format = fmt
            col_name = headers[ci-1]
            if col_name == 'Winner':
                if val == 'hybrid':
                    c.fill = PatternFill('solid', start_color=LIGHT_GREEN)
                    c.font = Font(name='Arial', size=9, bold=True, color="1E8449")
                elif val == 'baseline':
                    c.fill = PatternFill('solid', start_color=LIGHT_RED)
                    c.font = Font(name='Arial', size=9, bold=True, color="C0392B")
                else:
                    c.fill = PatternFill('solid', start_color="FFF9C4")
                    c.font = Font(name='Arial', size=9, bold=True, color="7D6608")
            elif col_name == 'Δ Obj (%)':
                if isinstance(val, float) and val < 0:
                    c.fill = PatternFill('solid', start_color=LIGHT_GREEN)
                    c.font = Font(name='Arial', size=9, color="1E8449")
                elif isinstance(val, float) and val > 0:
                    c.fill = PatternFill('solid', start_color=LIGHT_RED)
                    c.font = Font(name='Arial', size=9, color="C0392B")
                else:
                    c.fill = PatternFill('solid', start_color=fill_color)
            elif col_name in ('Baseline Feasible', 'Hybrid Feasible'):
                c.fill = PatternFill('solid', start_color=LIGHT_GREEN if val else LIGHT_RED)
            else:
                c.fill = PatternFill('solid', start_color=fill_color)

    col_widths = [12,11,9,11,12,14,14,13,14,12,12,14,12,13,10,11,10,9]
    for i, w in enumerate(col_widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = 'A3'

    # Summary sheet
    ws2 = wb.create_sheet("Summary")
    df = pd.DataFrame(results)

    def hdr(r, col, val, bg=DARK_BLUE):
        c = ws2.cell(row=r, column=col, value=val)
        c.font = Font(name='Arial', bold=True, color=WHITE, size=10)
        c.fill = PatternFill('solid', start_color=bg)
        c.alignment = Alignment(horizontal='center', vertical='center')
        c.border = tb

    ws2.merge_cells('A1:F1')
    c = ws2['A1']
    c.value = "CVRPTW Hybrid Solver — Performance Summary"
    c.font = Font(name='Arial', bold=True, size=14, color=WHITE)
    c.fill = PatternFill('solid', start_color=DARK_BLUE)
    c.alignment = Alignment(horizontal='center', vertical='center')
    ws2.row_dimensions[1].height = 30

    ws2.merge_cells('A3:F3')
    hdr(3, 1, "Overall Results", bg=MED_BLUE); ws2.merge_cells('A3:F3')
    for ci, h in enumerate(['Metric','All','n=100','n=200','n=400'], 1):
        hdr(4, ci, h, bg=MED_BLUE)

    sizes = [100, 200, 400]
    sub = {s: df[df['num_customers']==s] for s in sizes}
    def pct(d): return f"{(d['winner']=='hybrid').mean()*100:.1f}%" if len(d) else 'N/A'

    stats = [
        ('Total Instances', len(df), *[len(sub[s]) for s in sizes]),
        ('Hybrid Wins', (df['winner']=='hybrid').sum(), *[(sub[s]['winner']=='hybrid').sum() for s in sizes]),
        ('Baseline Wins', (df['winner']=='baseline').sum(), *[(sub[s]['winner']=='baseline').sum() for s in sizes]),
        ('Ties', (df['winner']=='tie').sum(), *[(sub[s]['winner']=='tie').sum() for s in sizes]),
        ('Hybrid Win Rate', pct(df), *[pct(sub[s]) for s in sizes]),
        ('Avg Δ Obj (%)', f"{df['delta_obj_pct'].mean():.3f}", *[f"{sub[s]['delta_obj_pct'].mean():.3f}" if len(sub[s]) else 'N/A' for s in sizes]),
        ('Avg Hybrid Time (s)', f"{df['hybrid_time_s'].mean():.2f}", *[f"{sub[s]['hybrid_time_s'].mean():.2f}" if len(sub[s]) else 'N/A' for s in sizes]),
    ]
    for ri, row_data in enumerate(stats, 5):
        fill = WHITE if ri % 2 == 0 else LIGHT_GRAY
        for ci, v in enumerate(row_data, 1):
            c = ws2.cell(row=ri, column=ci, value=v)
            c.font = Font(name='Arial', size=10, bold=(ci==1))
            c.fill = PatternFill('solid', start_color=fill)
            c.alignment = Alignment(horizontal='center', vertical='center')
            c.border = tb

    for col in ['A','B','C','D','E','F']:
        ws2.column_dimensions[col].width = 22

    wb.save(output_path)
    print(f"\nSaved: {output_path}")


# plotting functions 
def plot_instance_routes(inst, routes, title, save_path):
    plt.figure(figsize=(10, 8))
    depot_x, depot_y = inst["depot"]
    
    # Plot customers
    cx = [c["x"] for c in inst["customers"].values()]
    cy = [c["y"] for c in inst["customers"].values()]
    plt.scatter(cx, cy, c='gray', s=15, alpha=0.6, label='Customers')
    
    # Plot depot
    plt.scatter(depot_x, depot_y, c='red', marker='s', s=120, label='Depot', zorder=5)
    
    # Plot routes
    colors = plt.cm.rainbow(np.linspace(0, 1, len(routes)))
    for r_idx, route in enumerate(routes):
        x_coords = [depot_x] + [inst["customers"][cid]["x"] for cid in route] + [depot_x]
        y_coords = [depot_y] + [inst["customers"][cid]["y"] for cid in route] + [depot_y]
        plt.plot(x_coords, y_coords, color=colors[r_idx], linewidth=1.5, alpha=0.7)

    plt.title(f"Routing Map: {title}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_performance(b_score, h_score, name, save_path):
    plt.figure(figsize=(6, 5))
    labels = ['Baseline', 'Hybrid']
    scores = [b_score, h_score]
    colors = ['#2E75B6', '#1A5276']
    
    plt.bar(labels, scores, color=colors, width=0.5)
    plt.title(f"Objective Score: {name}")
    plt.ylabel("Score (Lower is Better)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(scores):
        plt.text(i, v + (max(scores)*0.01), f"{v:.1f}", ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# main function
def main():
    instance_files = sorted(INSTANCES_DIR.glob("*.json"))
    if not instance_files:
        print(f"No JSON files found in {INSTANCES_DIR}")
        return

    # Setup plot directory
    plot_dir = INSTANCES_DIR.parent / "plots"
    plot_dir.mkdir(exist_ok=True)

    print(f"Found {len(instance_files)} instances in {INSTANCES_DIR}\n")
    results = []

    for i, path in enumerate(instance_files, 1):
        name = path.stem
        print(f"[{i}/{len(instance_files)}] {name}")
        try:
            inst = load_cvrptw(path)
            n = len(inst["ids"])

            # Baseline
            t0 = time.time()
            b_routes, b_score, b_feasible = solve_baseline(inst)
            b_time = time.time() - t0
            b_K    = len(b_routes) if b_routes else 0
            b_dist = total_distance(b_routes, inst["dist"]) if b_routes else 0.0

            # Hybrid
            t0 = time.time()
            h_routes, h_score, h_feasible, exact_calls = solve_hybrid(
                inst, iterations=ITERATIONS,
                candidates_per_iter=CANDIDATES,
                elite_exact_checks=ELITE_CHECKS)
            h_time = time.time() - t0
            h_K    = len(h_routes) if h_routes else 0
            h_dist = total_distance(h_routes, inst["dist"]) if h_routes else 0.0

            delta     = round(h_score - b_score, 4)
            delta_pct = round((delta / b_score) * 100, 4) if b_score else 0.0
            if delta < -1e-6:   winner = "hybrid"
            elif delta > 1e-6:  winner = "baseline"
            else:               winner = "tie"

            print(f"  Baseline: K={b_K}  dist={b_dist:.2f}  obj={b_score:.1f}  ({b_time:.2f}s)")
            print(f"  Hybrid:   K={h_K}  dist={h_dist:.2f}  obj={h_score:.1f}  ({h_time:.2f}s)  winner={winner}\n")

            # --- GENERATE PLOTS ---
            best_routes = h_routes if winner == "hybrid" else b_routes
            if best_routes:
                plot_instance_routes(inst, best_routes, f"{name} ({winner.capitalize()})", plot_dir / f"{name}_map.png")
            plot_performance(b_score, h_score, name, plot_dir / f"{name}_comparison.png")
            # ----------------------

            results.append({
                'instance': name, 'num_customers': n,
                'capacity': inst['capacity'], 'max_vehicles': inst['max_vehicles'],
                'baseline_obj': round(b_score, 2), 'baseline_K': b_K,
                'baseline_distance': round(b_dist, 2), 'baseline_time_s': round(b_time, 2),
                'baseline_feasible': b_feasible,
                'hybrid_obj': round(h_score, 2), 'hybrid_K': h_K,
                'hybrid_distance': round(h_dist, 2), 'hybrid_time_s': round(h_time, 2),
                'hybrid_feasible': h_feasible, 'exact_calls': exact_calls,
                'delta_obj': delta, 'delta_obj_pct': delta_pct, 'winner': winner,
            })

        except Exception as e:
            print(f"  ERROR on {name}: {e}\n")

    if results:
        save_excel(results, OUTPUT_FILE)
        # Also save CSV as backup
        pd.DataFrame(results).to_csv(OUTPUT_FILE.with_suffix('.csv'), index=False)
        print(f"CSV backup: {OUTPUT_FILE.with_suffix('.csv')}")
    else:
        print("No results to save.")

if __name__ == "__main__":
    main()