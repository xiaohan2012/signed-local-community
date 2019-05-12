#!/bin/bash
#SBATCH -c 4
#SBATCH -t 02:00:00
#SBATCH --mem-per-cpu=2500
#SBATCH --array=1-20
#SBATCH --constraint=[ivb]

graph="word"
n=$SLURM_ARRAY_TASK_ID                  # define n
line=`sed "${n}q;d" cmds/${graph}_pairs.txt`    # get n:th line (1-indexed) of the file

echo $line
srun $line

