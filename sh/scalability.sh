#! /bin/zsh

srun -c 8 --mem 10GB --time 480  --constraint=ivb python3 scalability_evaluation.py wikiconflict 1000000 100
