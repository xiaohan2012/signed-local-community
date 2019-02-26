#! /bin/zsh

# change the output path if needed
output_path="eval_results/congress.csv"


echo "id method f1 avg_cc diameter" > $output_path
psql -U xiaoh1 postgres -h 10.10.254.21 -U xiaoh1 postgres -h 10.10.254.21 -t -A -F" " -f sh/sql/extract_eval_result.sql >> $output_path
