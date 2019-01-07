#! /bin/zsh

directory=$(dirname $0:A)
psql -U xiaoh1 postgres -h 10.10.254.21  -f ${directory}/sql/cg_neg_ratio.sql
