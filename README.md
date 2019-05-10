# searching for local polarization


# Usage

## run query on graphs

- run a batch of queries on graph: `run_queries_in_batch.py`
- print commands that each runs a batch of queries (for parallel computing): `print_batch_query_commands.py`

### notes on Aalto Triton

- run `python3 print_batch_query_commands.py {graph_name} > {cmds_path.txt}` to save the command list
- edit `sbatch_run_queries_in_batch.sh`: make sure to update the following:
  - `--array=1-{n}`, where `n` is the number of commands to run (use `wc -l {cmds_path.txt}`) to get that number
  - `graph="{graph_name}"`: set the graph name accordingly
  - in addition, number of cpus, memory requirement, max running time can be set
- submit the job by `sbatch sbatch_run_queries_in_batch.sh`

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
- `FOCG-vs-Local.ipynb`: comparing [FOCG, KDD 2018](https://dl.acm.org/citation.cfm?id=2939672.2939855) with our method
- `why-constraint-not-tight.ipynb`: for some nodes typically with small degrees, `alpha` tends very close to `lambda_1`, making the constraint not tight
- `explore-query-result-on-real-graphs.ipynb`: checking query result from real graphs (more  statistics and some viz)