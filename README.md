# searching for local polarization


# Usage

## run query on graphs

currently, two query modes are supported:

- query a single seed: run a batch of single-seed queries on graph: `query_single_seed_in_batch.py`
  - save commands that query single seed (for parallel computing): `python3 print_query_single_seed_commands.py {graph} > cmds/{graph}.txt`
- query a seed pair (one node from each polarized side): run a batch of seed-pair queries on graph: `query_seed_pair_in_batch.py
  - save commands that query seed pairs: `python3 print_query_seed_pair_commands.py {graph} > cmds/{graph}_pairs.txt`

### notes on Aalto Triton computation infrastructure

- edit `sbatch_query_single_seed_in_batch.sh`  and make sure to update the following:
  - `--array=1-{n}`, where `n` is the number of commands to run (use `wc -l {cmds_path.txt}`) to get that number
  - `graph="{graph_name}"`: set the graph name accordingly
  - in addition, number of cpus, memory requirement, max running time can be set
- submit the job by `sbatch sbatch_run_queries_in_batch.sh
- the same applies to the other query mode, corresponding to file `sbatch_query_pairs_in_batch.sh`

## experiment on synthetic graphs

- effect of noise parameter: `run_experiment_effect_of_eta.py`
- effect of correlation parameter: `run_experiment_effect_of_kappa.py`
- effect of number of outlier nods: `run_experiment_effect_of_outlier_size.py`
- effect of number of seed: `run_experiment_effect_of_seed_size.py`

## jupyter notebooks along the process

the following notebooks highlights the thought process and how the project has involved:

- `signed-laplacian-eigen-value.ipynb`: demo on what the bottom-most eigen vector looks like on a toy graph (to understand better signed spectral theory in general)
- `proof-of-concept.ipynb`: the very early one that demos how this method works for small toy graphs and some investigation on the effect of kappa
- `experiment_on_synthetic_graphs.ipynb`: effect of different parameters on synetheic graphs
- `tuning-kappa.ipynb`: trying to understand better the effect of `kappa` on synthetic graphs
- `scalable-local-polarization.ipynb`: scalabel way to solve the linear equation (using conjugate gradient)
- `fast_sweeping.ipynb`: efficient way to sweep on `x` (reduces time cost by orders of magnitudes)
- `case-study-on-word-graph.ipynb`: manual checking the result on word graph + some visualization
- `why-constraint-not-tight.ipynb`: for some nodes typically with small degrees, `alpha` tends very close to `lambda_1`, making the constraint not tight
- `explore-query-result-on-real-graphs.ipynb`: checking query result on real graphs (more  statistics and some viz)
- `explore-fog-result-on-real-graphs.ipynb`: checking query result by [FOCG, KDD 2018](https://dl.acm.org/citation.cfm?id=2939672.2939855) on real graphs
- `FOCG-vs-Local.ipynb`: comparing [FOCG, KDD 2018](https://dl.acm.org/citation.cfm?id=2939672.2939855) with our method
- `dig-out-more-communities-on-word-graph.ipynb`: find out more polarized communities on "word" graph
- `overlapping-community-demo`: demo on overlapping communities in "word" graph