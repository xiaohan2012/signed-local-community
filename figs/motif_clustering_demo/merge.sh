#! /bin/zsh



for d in $(ls -d *); do
    if [ -d $d ]; then
	echo "processing" $d
	pdftk $d/graph.pdf \
	      $d/motif-graph.pdf \
	      $d/sweep-profile-plot.pdf \
	      $d/subgraph-selection.pdf \
	      $d/subgraph-selected.pdf \
	      $d/group-conductance-sweep.pdf \
	      $d/partitioning.pdf \
	      cat output \
	      $d/all.pdf
    fi
done

