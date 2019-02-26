"""
run an experiment suite specified by the name
"""
import os
import argparse
import math
import tempfile
from itertools import islice, takewhile

from helpers import random_str
from const import TMP_DIR
from sql import init_db


def gen_sbatch_string(
        job_name,
        script_name,
        hours_per_job,
        n_jobs,
        logfile_name,
        params_file_name,
        n_jobs_at_a_time=10,
        chunk_size=50,
        mem=5
):
    return """#!/bin/zsh

#SBATCH --job-name={job_name}
#SBATCH --output={logfile_name}  # %a does not work
#SBATCH --cpus-per-task 1 
#SBATCH --time {hours_per_job}:00:00
#SBATCH --mem={mem}G
#SBATCH --array=1-{n_array_jobs}%{n_jobs_at_a_time}

params_file={params_file_name}

arrayID=$SLURM_ARRAY_TASK_ID  # starting from 0

CHUNKSIZE={chunk_size}
(( lower = ($arrayID - 1) * $CHUNKSIZE + 1 ))
(( upper = $arrayID  * $CHUNKSIZE ))

for idx in $(seq $lower $upper); do
    params=`sed "${{idx}}q;d" ${{params_file}}`
    echo "srun python3 {script_name} ${{params}}"
    if [ ! -z "${{params}}" ]; then
        eval "srun python3 {script_name} ${{params}}"
    else
        echo "idx=${{idx}} has empty params"
    fi
done
""".format(
    job_name=job_name,
    hours_per_job=str(hours_per_job).zfill(2),
    n_jobs_at_a_time=n_jobs_at_a_time,
    n_array_jobs=math.ceil(n_jobs / chunk_size),
    logfile_name=logfile_name,
    params_file_name=params_file_name,
    mem=mem,
    script_name=script_name,
    chunk_size=chunk_size
)

def gen_tmp_file(prefix="", suffix=""):
    path = os.path.join(TMP_DIR, prefix + random_str() + suffix)
    return open(path, 'a')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n',
        '--name',
        required=True,
        help='the experiment name'
    )

    parser.add_argument(
        '-s',
        '--script_name',
        required=True,
        help='the script name'
    )

    parser.add_argument(
        '-d',
        '--debug',
        action='store_true',
        help="debug mode or not"
    )

    
    parser.add_argument(
        '-p',
        '--n_parallel',
        type=int,
        default=128,
        help="number of parallel jobs"
    )
    

    parser.add_argument(
        '-c',
        '--chunk_size',
        type=int,
        default=50,
        help="number of tasks inside one job"
    )
    
    args = parser.parse_args()
    iter_configs = getattr(__import__('experiment_configs'), args.name)
    
    configs = iter_configs(show_progress=(not args.debug))

    conn, cursor = init_db()
    
    if args.debug:
        print('debug')
        for config in configs:
            if not config.is_computed(cursor):
                config.print_commands(prefix="python3 {} ".format(args.script_name))
                print('\n')
                break
    else:
        sbatch_file = gen_tmp_file(prefix=args.name + '_sbatch_')
        params_file = gen_tmp_file(prefix=args.name + '_params_')
        log_file = gen_tmp_file(prefix=args.name + '_log_')
        
        print('writing params to {}'.format(params_file.name))

        n_jobs = 0
        for config in configs:
            # do some filtering
            if not config.is_computed(cursor):
                config.print_commands(fileobj=params_file)
                n_jobs += 1
        params_file.close()
        
        sbatch_string = gen_sbatch_string(
            args.name,
            args.script_name,
            config.hours_per_job,
            n_jobs,
            log_file.name,
            params_file.name,
            n_jobs_at_a_time=args.n_parallel,
            chunk_size=args.chunk_size
        )
        print("sbatch commands as follows\n{}\n".format('='*10))
        print(sbatch_string)
        print("{}\n".format('='*10))

        print('saved to {}'.format(sbatch_file.name))
        sbatch_file.write(sbatch_string)

        sbatch_file.close()
        log_file.close()
    conn.close()
