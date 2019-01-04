"""
run an experiment suite specified by the name
"""
import os
import argparse
import tempfile
from helpers import random_str
from const import TMP_DIR
from sql import init_db

def gen_sbatch_string(
        job_name,
        script_name,
        hours_per_job,
        minutes_per_job,
        n_jobs,
        logfile_name,
        params_file_name,
        n_jobs_at_a_time=32,
        mem=5
):
    return """#!/bin/zsh

#SBATCH --job-name={job_name}
#SBATCH --output={logfile_name}  # %a does not work
#SBATCH --cpus-per-task 1 
#SBATCH --time {hours_per_job}:{minutes_per_job}:00
#SBATCH --mem={mem}G
#SBATCH --array=1-{n_jobs}%{n_jobs_at_a_time}

params_file={params_file_name}

n=$SLURM_ARRAY_TASK_ID

params=`sed "${{n}}q;d" ${{params_file}}`

eval "srun python3 {script_name} ${{params}}"
""".format(
    job_name=job_name,
    hours_per_job=str(hours_per_job).zfill(2),
    minutes_per_job=str(minutes_per_job).zfill(2),
    n_jobs_at_a_time=n_jobs_at_a_time,
    n_jobs=n_jobs,
    logfile_name=logfile_name,
    params_file_name=params_file_name,
    mem=mem,
    script_name=script_name
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
        # required=True,
        default='experiment_on_community_graph.py',
        help='the script name'
    )
    
    parser.add_argument(
        '-d',
        '--debug',
        action='store_true',
        help="debug mode or not"
    )

    args = parser.parse_args()
    iter_configs = getattr(__import__('experiment_configs'), args.name)
    
    configs = list(iter_configs())

    if args.debug:
        for config in configs:
            config.n_rounds = 1
            config.arg_suffix = '--verbose 0 --debug'
            config.print_params(prefix="python3 {}.py ".format(args.script_name))
            print('\n')
    else:
        sbatch_file = gen_tmp_file(prefix=args.name + '_sbatch_')
        params_file = gen_tmp_file(prefix=args.name + '_params_')
        log_file = gen_tmp_file(prefix=args.name + '_log_')
        
        print('writing params to {}'.format(params_file.name))

        conn, cursor = init_db()

        for config in configs:
            # do some filtering
            if not config.is_computed(cursor):
                config.print_params(fileobj=params_file)
        params_file.close()
        
        n_jobs = sum(c.n_jobs for c in configs)
        sbatch_string = gen_sbatch_string(
            args.name,
            args.script_name,
            config.hours_per_job,
            config.minutes_per_job,
            n_jobs,
            log_file.name,
            params_file.name,
        )
        print("sbatch commands as follows\n{}\n".format('='*10))
        print(sbatch_string)
        print("{}\n".format('='*10))

        print('saved to {}'.format(sbatch_file.name))
        sbatch_file.write(sbatch_string)

        sbatch_file.close()
        log_file.close()
        conn.close()
