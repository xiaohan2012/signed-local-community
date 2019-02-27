#! /bin/zsh

# change the output path if needed
output_path="eval_results/bitcoin.csv"

echo "id method query n m inter_neg intra_pos f1 avg_cc diameter" > $output_path
psql -U xiaoh1 postgres -h 10.10.254.21 -U xiaoh1 postgres -h 10.10.254.21 -t -A -F" " -f sh/sql/extract_eval_result.sql >> $output_path
