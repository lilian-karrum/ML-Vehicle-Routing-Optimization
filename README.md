# Machine Learning for Vehicle Routing Problems (CVRP & CVRPTW)

A university Computer Science project exploring how **machine learning can guide evolutionary optimization** to produce efficient vehicle routes. The project investigates two variants of the Vehicle Routing Problem (VRP):

- **CVRP — Capacitated Vehicle Routing Problem:** Find efficient delivery routes while respecting vehicle capacity limits.
- **CVRPTW — Capacitated Vehicle Routing Problem with Time Windows:** Additionally ensure that customers are served within their permitted time windows.

The approach combines heuristic route construction, evolutionary search, local improvement methods, and a **Random Forest surrogate model** that ranks promising candidate solutions before exact evaluation.

## Approaches

### CVRP
- Heuristic initialization and capacity-feasible route construction
- Mutation-based evolutionary search and intra-route 2-opt improvement
- Random Forest-assisted candidate selection compared against a baseline evolutionary solver

### CVRPTW — My primary contribution
- Time-window- and capacity-aware route construction using feasible insertion
- Customer-order mutations (swap, reverse, insert) and destroy-and-repair diversification
- Random Forest regression to predict candidate objective scores and guide search
- Feasibility-preserving 2-opt and inter-route relocate improvements
- Route visualizations and baseline-versus-hybrid comparisons

## Technologies

Python · NumPy · pandas · scikit-learn · Matplotlib · openpyxl

## Reported results

Results below are from the academic project evaluation; they are benchmark results, not a guarantee of performance on new instances.

| Metric | CVRP | CVRPTW |
| --- | ---: | ---: |
| Benchmark instances solved | 59 / 59 | 33 / 33 |
| Instances where hybrid beat baseline | 48 / 59 | 23 / 33 |
| Mean objective improvement | 0.76% | 5.14% |
| Best single-instance improvement | 7.58% | 23.21% |

The hybrid methods improved solution quality on many benchmark instances, **at the cost of additional runtime**. For CVRPTW, the reported hybrid runtime was approximately 140 times that of the baseline across the tested instances.

## Repository files

- `cvrptw.py` — CVRPTW baseline and Random Forest-guided hybrid implementation, evaluation, and plotting
- `cvrp.py` — CVRP implementation *(include only if you have permission to publish your teammate's code)*

## Running the CVRPTW implementation

1. Install Python and the required packages:

   ```bash
   pip install numpy pandas scikit-learn matplotlib openpyxl
   ```

2. Obtain the compatible CVRPTW benchmark instance files in JSON format. The script expects a directory containing `*.json` files with the structure used in the project (including customer records, depot information, vehicle capacity, and distance matrix).
3. Open `cvrptw.py` and **change `INSTANCES_DIR`** near the top of the file to the folder containing those JSON files. The current path is a local Windows path and will not work on another computer.
4. Run:

   ```bash
   python cvrptw.py
   ```

The script compares baseline and hybrid results, writes an Excel results file and CSV backup, and generates route maps and comparison plots in a `plots` folder beside the instance directory.

**Note:** Benchmark JSON datasets are not included in this repository unless uploaded separately. The script requires them to run.

## Project contributions

This was a collaborative university project. **My primary contribution was the CVRPTW implementation**, including the capacity- and time-window-aware optimization and Random Forest-guided search. The CVRP implementation was developed as part of the wider team project; contributors should be credited according to their actual work.

## Context

Academic project completed at the **University of Sharjah**, Department of Computer Science.
