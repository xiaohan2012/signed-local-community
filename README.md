# Searching for polarization in signed graphs: a local spectral approach, WebConf 2020

See the paper ([Arxiv version](https://arxiv.org/pdf/2001.09410.pdf))  


# Software dependency
## install python packages

Make sure you have [conda](https://docs.conda.io/en/latest/) installed.

Then run `conda env create -f environment.yml` to create the virtual environment.

Activate it by `conda activate polar`


## Install database (optional)

We use [postgres](https://www.postgresql.org/) to store results of experiments that are 1) repeated many times and 2) relatively time-consuming to run.
For example, results from seeding on real-world graphs are stored in database. 

You don't need database if you only call the API in Python (`core.py`).
However, if you want to reproduce the results in the paper, you need this.

## testing

run `pytest test*.py` (you might see very few errors e.g., usually 1  from `test_core.py` due to numerical instability)

# API usage 

a typical example of calling functions in Python is:

```python
import networkx as nx
from core import query_graph_using_sparse_linear_solver, sweep_on_x_fast
# read the graph
g = nx.read_gpickle('{path_of_graph}')

# compute the optimal x vector
x, obj_val = query_graph_using_sparse_linear_solver(g, [seeds1, seeds2], kappa=0.9, verbose=0, ub=g.graph['lambda1'])

# sweep on x to find C1 and C2
C1, C2, C, best_t, best_sbr, ts, sbr_list = sweep_on_x_fast(g, x, top_k=100)

print('community 1', C1)
print('community 2', C2)
```

# Command line usage

## run queries on graphs

currently, two query modes are supported:

- query a single seed: run a batch of single-seed queries on graph: `query_single_seed_in_batch.py`
  - save commands that query single seed (for parallel computing): `python3 print_query_single_seed_commands.py {graph} > cmds/{graph}.txt`
- query a seed pair (one node from each polarized side): run a batch of seed-pair queries on graph: `query_seed_pair_in_batch.py
  - save commands that query seed pairs: `python3 print_query_seed_pair_commands.py {graph} > cmds/{graph}_pairs.txt`

Note that the above two commands requires postgres being installed.

## exporting results from database 

use `export_single_seed_result_from_db.py|export_pair_result_from_db.py`

The output is in `pandas.DataFrame` format. 

## useful scripts

- `augment_result.py`: augment our result by various graph statistics

### pre/post-processing scripts for [FOCG, KDD 2016](https://www.kdd.org/kdd2016/papers/files/rpp0799-chuA.pdf)

- `prepare_data_for_matlab.py`: convert graph to Matlab format for FOCG to use
- `augment_focg_result.py`: augment FOCG result by various graph statistics


## scalability evaluation

- run `scalability_evaluation.py`

# Reproducing the figures/tables in the submission

## data pre-processing

- run `preprocess_graph.py` (remember to change the variable `graph`)
- or you can use the processed ones under `graphs/{graph}.pkl`

## Figure 1: the motivation plot

run `intro-plot.ipynb`

## Table 2: graph statistics

run `graph_stat_table.ipynb`

## Figure 4: synthetic graph experiment 

run the following (it takes ~1.5 hours on a 8-core machine in total):

- effect of noise parameter: `run_experiment_effect_of_eta.py`
- effect of number of outlier nods: `run_experiment_effect_of_outlier_size.py`
- effect of number of seed: `run_experiment_effect_of_seed_size.py`

then, make the plot using `experiment_on_synthetic_graphs.ipynb`

## Figure 5: real graph experiment

You need to have postgres installed in order to save the results. 

**run PolarSeeds**

Do the following for all graphs (word, bitcoin, epinions, etc):

- run `python3 print_query_seed_pair_commands.py {graph_name} > {cmd_list}.txt` to get the list of execution commands
  - `{cmd_list}.txt` will contain the list of commands to run to get the result
- run `python3 export_pair_result_from_db.py` to export the data (remember to change the `graph` variable in the script)
- run `python3 augment_pair_result.py {graph_name}` to add evaluation metric values

**run FOCG**

- before runnin FOCG, preprocess the graphs so they're Matlab-compatible: run `prepare_data_for_matlab.py` (remember to update te `graph` variable)
- check [this repository](https://github.com/xiaohan2012/KOCG.SIGKDD2016) and run the file `DemoRun.m` in Matlab
  - make sure the input graphs from previou step are in the right paths
  - copy the output `.mat` file under `outputs/focg-{graph_name}.mat`
- run `python3 augment_focg_result.py {graph_name}` to add evaluation metric values

**make the plot**

run `FOCG-vs-PolarSeeds.ipynb` to make the plot

# Figure 6: case studies

- (a) and (b): run `case-study-overlapping-community.ipynb`
- (c): run `case-study-distrust-radiation.ipynb`

# Jupyter notebooks along the way

the following notebooks are records of the thought process and how the project has involved:

- `signed-laplacian-eigen-value.ipynb`: demo on what the bottom-most eigen vector looks like on a toy graph (to understand better signed spectral theory in general)
- `proof-of-concept.ipynb`: the very early one that demos how this method works for small toy graphs and some investigation on the effect of kappa
- `experiment_on_synthetic_graphs.ipynb`: effect of different parameters on synetheic graphs
- `tuning-kappa.ipynb`: trying to understand better the effect of `kappa` on synthetic graphs
- `binary-search-on-alpha.ipynb`: binary search on alpha plus conjugate gradient method to solve the program
- `fast_sweeping.ipynb`: efficient way to sweep on `x` (reduces time cost by orders of magnitudes)
- `case-study-on-word-graph.ipynb`: manual checking the result on word graph + some visualization
- `why-constraint-not-tight.ipynb`: for some nodes typically with small degrees, `alpha` tends very close to `lambda_1`, making the constraint not tight
- `explore-seed-pair-query-result.ipynb`: checking query result on real graphs (some  statistics and viz)
- `explore-fog-result-on-real-graphs.ipynb`: checking query result by [FOCG, KDD 2018](https://dl.acm.org/citation.cfm?id=2939672.2939855) on real graphs
- `FOCG-vs-PolarSeeds.ipynb`: comparing [FOCG, KDD 2018](https://dl.acm.org/citation.cfm?id=2939672.2939855) with our method
- `dig-out-more-communities-on-word-graph.ipynb`: find out more polarized communities on "word" graph
- `case-study-overlapping-community`: demo on overlapping communities in "word" graph
- `case-study-distrust-radiation.ipynb`: case study of distrust radiation
- `intro-plot.ipynb`: plots of the motivational example

# Misc

### notes on sbatch ([Aalto Triton](https://scicomp.aalto.fi/triton/) ulitity script)

- edit `sbatch_query_single_seed_in_batch.sh`  and make sure to update the following:
  - `--array=1-{n}`, where `n` is the number of commands to run (use `wc -l {cmds_path.txt}`) to get that number
  - `graph="{graph_name}"`: set the graph name accordingly
  - in addition, number of cpus, memory requirement, max running time can be set
- submit the job by `sbatch sbatch_run_queries_in_batch.sh
- the same applies to the other query mode, corresponding to file `sbatch_query_pairs_in_batch.sh`

# To cite this paper

```bibtex
@inproceedings{xiao2020searching,
  title={Searching for polarization in signed graphs: a local spectral approach},
  author={Xiao, Han and Ordozgoiti, Bruno and Gionis, Aristides},
  booktitle={Proceedings of The Web Conference 2020},
  pages={362--372},
  year={2020}
}
```
	  
