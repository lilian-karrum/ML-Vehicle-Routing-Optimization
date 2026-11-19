import os
import sys
import json
import random
import time
import subprocess
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

REPO = Path(__file__).resolve().parent
INSTANCES_DIR = REPO / "Instances" / "cvrp" / "json"
SOLUTIONS_DIR = REPO / "Solutions" / "cvrp"
BASELINE_DIR = SOLUTIONS_DIR / "baseline"
HYBRID_DIR = SOLUTIONS_DIR / "hybrid"
RESULTS_CSV = REPO / "results_cvrp.csv"
FLEET_PENALTY = 1000  # matches evaluator.py: fitness = 1000*K + distance


# load instances 
def load_cvrp(instance_name):
    path = INSTANCES_DIR / f"{instance_name}.json"
    with open(path) as f:
        data = json.load(f)
    customers = {}
    for k, v in data.items():
        if k.startswith("customer_"):
            cid = int(k.split("_")[1])
            customers[cid] = {
                "x": v["coordinates"]["x"],
                "y": v["coordinates"]["y"],
                "demand": v["demand"],
            }
    return {
        "name": instance_name,
        "capacity": data["vehicle_capacity"],
        "max_vehicles": data["max_vehicle_number"],
        "dist": np.array(data["distance_matrix"]),
        "depot": (data["depart"]["coordinates"]["x"], data["depart"]["coordinates"]["y"]),
        "customers": customers,
        "ids": sorted(customers.keys()),
    }


# function to turn sequence of customers to list of routes, greedy split by capacity
def build_routes(sequence, inst):
    routes, current, load = [], [], 0.0
    cap = inst["capacity"]
    for cid in sequence:
        d = inst["customers"][cid]["demand"]
        if load + d > cap and current:
            routes.append(current)
            current, load = [], 0.0
        current.append(cid)
        load += d
    if current:
        routes.append(current)
    return routes

# compute the distance of one route
def route_distance(route, dist):
    if not route:
        return 0.0
    total = dist[0][route[0]]
    for i in range(len(route) - 1):
        total += dist[route[i]][route[i + 1]]
    total += dist[route[-1]][0]
    return total

# compute distance of all routes
def total_distance(routes, dist):
    return sum(route_distance(r, dist) for r in routes)

# calculate objective score
def score(routes, dist):
    return FLEET_PENALTY * len(routes) + total_distance(routes, dist)


# 2-opt local search: to improve routes
# Try reversing segments
# Keep change if it reduces distance
def two_opt_route(route, dist):
    if len(route) < 4:
        return route
    best = list(route)
    n = len(best)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                a = 0 if i == 0 else best[i - 1]
                b = best[i]
                c = best[j]
                d = 0 if j == n - 1 else best[j + 1]
                if a == c or b == d:
                    continue
                delta = (dist[a][c] + dist[b][d]) - (dist[a][b] + dist[c][d])
                if delta < -1e-9:
                    best[i:j + 1] = best[i:j + 1][::-1]
                    improved = True
    return best

# apply 2-opt to all routes
def apply_2opt(routes, dist):
    return [two_opt_route(r, dist) for r in routes]

# full evaluation pipeline
def exact_eval(sequence, inst, use_2opt=True):
    routes = build_routes(sequence, inst)
    if use_2opt:
        routes = apply_2opt(routes, inst["dist"])
    return score(routes, inst["dist"]), routes


# Nearest Neighbor: always go to closest next customer
def nn_sequence(inst):
    dist = inst["dist"]
    remaining = set(inst["ids"])
    seq, current = [], 0
    while remaining:
        nxt = min(remaining, key=lambda c: dist[current][c])
        seq.append(nxt)
        remaining.remove(nxt)
        current = nxt
    return seq

# sort customers by angle around depot: creates circular routes
def angle_sequence(inst):
    dx, dy = inst["depot"]
    def angle(cid):
        c = inst["customers"][cid]
        return np.arctan2(c["y"] - dy, c["x"] - dx)
    return sorted(inst["ids"], key=angle)

# sort by highest demand first
def demand_sequence(inst):
    return sorted(inst["ids"], key=lambda c: -inst["customers"][c]["demand"])

# closest to depot first
def dist_from_depot_sequence(inst):
    d = inst["dist"]
    return sorted(inst["ids"], key=lambda c: d[0][c])

# used to seed the search
def initial_sequences(inst):
    return [nn_sequence(inst), angle_sequence(inst), demand_sequence(inst), dist_from_depot_sequence(inst)]


# modify a solution randomly
def mutate(seq):
    s = seq.copy()
    n = len(s)
    if n < 2:
        return s
    op = random.choice(["swap", "reverse", "insert"])
    if op == "swap":
        i, j = random.sample(range(n), 2)
        s[i], s[j] = s[j], s[i]
    elif op == "reverse":
        i, j = sorted(random.sample(range(n), 2))
        # cap reverse length to keep mutations local on big instances
        if j - i > 25:
            j = i + random.randint(2, 25)
        s[i:j + 1] = list(reversed(s[i:j + 1]))
    else:  # insert
        i, j = random.sample(range(n), 2)
        node = s.pop(i)
        s.insert(j, node)
    return s


# convert solution to numeric features
def extract_features(routes, inst):
    dist = inst["dist"]
    if not routes:
        return np.zeros(10)
    route_lengths = [len(r) for r in routes]
    route_demands = [sum(inst["customers"][c]["demand"] for c in r) for r in routes]
    route_dists = [route_distance(r, dist) for r in routes]
    cap = inst["capacity"]
    utils = [d / cap for d in route_demands]
    return np.array([
        len(routes),
        sum(route_dists),
        float(np.mean(route_lengths)),
        float(np.max(route_lengths)),
        float(np.std(route_lengths)),
        float(np.mean(route_dists)),
        float(np.std(route_dists)),
        float(np.mean(utils)),
        float(np.std(utils)),
        float(np.max(utils)),
    ])

# create dataset for ML model
def generate_training_data(inst, base_seqs, n_per_base=40):
    X, y = [], []
    for base in base_seqs:
        s, routes = exact_eval(base, inst, use_2opt=False)
        X.append(extract_features(routes, inst))
        y.append(s)
        for _ in range(n_per_base):
            seq = base.copy()
            for _ in range(random.randint(1, 3)):
                seq = mutate(seq)
            s, routes = exact_eval(seq, inst, use_2opt=False)
            X.append(extract_features(routes, inst))
            y.append(s)
    return np.array(X), np.array(y)


# find best initial solution
def _seed_best(inst, use_2opt):
    best_seq, best_score, best_routes = None, float("inf"), None
    for seq in initial_sequences(inst):
        s, routes = exact_eval(seq, inst, use_2opt=use_2opt)
        if s < best_score:
            best_seq, best_score, best_routes = seq, s, routes
    return best_seq, best_score, best_routes

# simple hill-climbing algorith
def solve_baseline(inst, iterations=500, use_2opt=True, verbose=True):
    t0 = time.time()
    best_seq, best_score, best_routes = _seed_best(inst, use_2opt)
    pbar = tqdm(range(iterations), desc=f"[baseline] {inst['name']:<14}", disable=not verbose,
                ncols=100, unit="it")
    pbar.set_postfix(obj=f"{best_score:.0f}", K=len(best_routes))
    for _ in pbar:
        seq = mutate(best_seq)
        s, routes = exact_eval(seq, inst, use_2opt=use_2opt)
        if s < best_score:
            best_seq, best_score, best_routes = seq, s, routes
            pbar.set_postfix(obj=f"{best_score:.0f}", K=len(best_routes))
    pbar.close()
    return {"routes": best_routes, "score": best_score, "time": time.time() - t0}

# slove using ML
def solve_hybrid(inst, iterations=500, candidates_per_iter=6, use_2opt=True, verbose=True):
    t0 = time.time()
    best_seq, best_score, best_routes = _seed_best(inst, use_2opt)

    if verbose:
        print(f"[hybrid]   {inst['name']}: training surrogate…")
    X, y = generate_training_data(inst, initial_sequences(inst), n_per_base=40)
    scaler = StandardScaler().fit(X)
    model = RandomForestRegressor(n_estimators=60, max_depth=10, n_jobs=-1, random_state=SEED)
    model.fit(scaler.transform(X), y)

    pbar = tqdm(range(iterations), desc=f"[hybrid]   {inst['name']:<14}", disable=not verbose,
                ncols=100, unit="it")
    pbar.set_postfix(obj=f"{best_score:.0f}", K=len(best_routes))
    exact_calls = 0
    for _ in pbar:
        cand_seqs, cand_feats = [], []
        for _ in range(candidates_per_iter):
            seq = mutate(best_seq)
            routes_pred = build_routes(seq, inst)
            cand_seqs.append(seq)
            cand_feats.append(extract_features(routes_pred, inst))
        preds = model.predict(scaler.transform(np.array(cand_feats)))
        idx = int(np.argmin(preds))
        s, routes = exact_eval(cand_seqs[idx], inst, use_2opt=use_2opt)
        exact_calls += 1
        if s < best_score:
            best_seq, best_score, best_routes = cand_seqs[idx], s, routes
            pbar.set_postfix(obj=f"{best_score:.0f}", K=len(best_routes))
    pbar.close()
    return {"routes": best_routes, "score": best_score,
            "time": time.time() - t0, "exact_calls": exact_calls}


# saves solution in required format
def write_solution(routes, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i, r in enumerate(routes, 1):
            f.write(f"Route #{i}: " + " ".join(str(c) for c in r) + "\n")

# checks feasibility and objective
def validate_with_evaluator(instance_name, solution_path):
    res = subprocess.run(
        [sys.executable, "evaluator.py", "cvrp", instance_name, str(solution_path)],
        cwd=str(REPO), capture_output=True, text=True,
    )
    return (res.stdout + res.stderr).strip()


# main function 
CSV_COLUMNS = [
    "instance", "num_customers", "capacity", "max_vehicles",
    "baseline_obj", "baseline_K", "baseline_distance", "baseline_time_s", "baseline_feasible",
    "hybrid_obj", "hybrid_K", "hybrid_distance", "hybrid_time_s", "hybrid_feasible",
    "hybrid_exact_calls",
    "delta_obj", "delta_obj_pct", "winner",
]

# extracts:feasibility, objective, distance, number of vehicles
def _parse_eval_output(output):
    """Return dict with feasible/objective/K/distance from evaluator.py stdout."""
    out = {"feasible": False, "objective": None, "K": None, "distance": None, "raw": output}
    if "infeasible" in output.lower():
        return out
    # Expected line: "Number of vehicles:  28 , Total distance:  30788.0 Objective value:  58788.0"
    for line in output.splitlines():
        if "Number of vehicles" in line and "Objective value" in line:
            try:
                parts = line.replace(",", " ").split()
                out["K"] = int(float(parts[parts.index("vehicles:") + 1]))
                out["distance"] = float(parts[parts.index("distance:") + 1])
                out["objective"] = float(parts[parts.index("value:") + 1])
                out["feasible"] = True
            except (ValueError, IndexError):
                pass
    return out


def run_instance(name, iterations=300):
    print("=" * 68)
    print(f"INSTANCE: {name}")
    print("=" * 68)
    inst = load_cvrp(name)
    print(f"  customers={len(inst['ids'])}  capacity={inst['capacity']}  "
          f"max_vehicles={inst['max_vehicles']}")

    baseline = solve_baseline(inst, iterations=iterations)
    print(f"  [baseline] obj={baseline['score']:.1f}  "
          f"K={len(baseline['routes'])}  time={baseline['time']:.1f}s")

    hybrid = solve_hybrid(inst, iterations=iterations)
    print(f"  [hybrid]   obj={hybrid['score']:.1f}  "
          f"K={len(hybrid['routes'])}  time={hybrid['time']:.1f}s")

    # Save BOTH solutions in their own subfolders
    baseline_path = BASELINE_DIR / f"{name}.txt"
    hybrid_path = HYBRID_DIR / f"{name}.txt"
    write_solution(baseline["routes"], baseline_path)
    write_solution(hybrid["routes"], hybrid_path)
    print(f"  [saved]   {baseline_path.relative_to(REPO)}")
    print(f"  [saved]   {hybrid_path.relative_to(REPO)}")

    # Verify both through the official evaluator
    baseline_check = _parse_eval_output(
        validate_with_evaluator(name, baseline_path.relative_to(REPO)))
    hybrid_check = _parse_eval_output(
        validate_with_evaluator(name, hybrid_path.relative_to(REPO)))
    print(f"  [verify]  baseline: {'OK' if baseline_check['feasible'] else 'INFEASIBLE'}  "
          f"obj={baseline_check['objective']}")
    print(f"  [verify]  hybrid:   {'OK' if hybrid_check['feasible'] else 'INFEASIBLE'}  "
          f"obj={hybrid_check['objective']}")

    delta = hybrid["score"] - baseline["score"]
    delta_pct = 100.0 * delta / baseline["score"] if baseline["score"] else 0.0
    winner = "hybrid" if hybrid["score"] < baseline["score"] else (
             "baseline" if baseline["score"] < hybrid["score"] else "tie")

    return {
        "instance": name,
        "num_customers": len(inst["ids"]),
        "capacity": inst["capacity"],
        "max_vehicles": inst["max_vehicles"],
        "baseline_obj": baseline["score"],
        "baseline_K": len(baseline["routes"]),
        "baseline_distance": round(total_distance(baseline["routes"], inst["dist"]), 1),
        "baseline_time_s": round(baseline["time"], 2),
        "baseline_feasible": baseline_check["feasible"],
        "hybrid_obj": hybrid["score"],
        "hybrid_K": len(hybrid["routes"]),
        "hybrid_distance": round(total_distance(hybrid["routes"], inst["dist"]), 1),
        "hybrid_time_s": round(hybrid["time"], 2),
        "hybrid_feasible": hybrid_check["feasible"],
        "hybrid_exact_calls": hybrid.get("exact_calls", 0),
        "delta_obj": round(delta, 1),
        "delta_obj_pct": round(delta_pct, 3),
        "winner": winner,
    }

# store results into CSV file
def write_csv(rows, path):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

# lists all dataset files
def all_instances():
    return sorted(p.stem for p in INSTANCES_DIR.glob("*.json"))


if __name__ == "__main__":
    # usage:
    #   python cvrp.py                 -> runs the 2 demo instances
    #   python cvrp.py X-n101-k25 ...  -> runs the named instances
    #   python cvrp.py --all           -> runs all 59 instances
    args = sys.argv[1:]
    if args and args[0] == "--all":
        names = all_instances()
    elif args:
        names = args
    else:
        names = ["X-n101-k25", "X-n106-k14"]

    results = []
    for n in names:
        results.append(run_instance(n))
        write_csv(results, RESULTS_CSV)  # incremental save — safe if run is interrupted

    print()
    print("=" * 80)
    print(f"{'instance':<18}{'baseline':>12}{'hybrid':>12}{'delta':>10}{'delta%':>10}{'winner':>10}")
    print("-" * 80)
    for r in results:
        print(f"{r['instance']:<18}{r['baseline_obj']:>12.1f}{r['hybrid_obj']:>12.1f}"
              f"{r['delta_obj']:>+10.1f}{r['delta_obj_pct']:>+10.2f}{r['winner']:>10}")
    print("-" * 80)
    print(f"CSV written to: {RESULTS_CSV.relative_to(REPO)}")
