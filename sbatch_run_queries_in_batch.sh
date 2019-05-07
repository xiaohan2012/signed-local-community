#!/bin/bash
#SBATCH -c 1 
#SBATCH -t 01:00:00
#SBATCH --mem-per-cpu=2500
#SBATCH --array=1-160

n=$SLURM_ARRAY_TASK_ID                  # define n
line=`sed "${n}q;d" cmds/slashdot1.txt`    # get n:th line (1-indexed) of the file

echo $line
srun $line

