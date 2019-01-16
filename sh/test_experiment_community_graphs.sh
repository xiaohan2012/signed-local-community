#! /bin/zsh

python3 experiment_on_community_graph.py  \
	-i 0 \
	-k 4 \
	-s 16 \
	-d 0.8 \
	-n 0.2 \
	-p 0.2 \
	-e 0.9 \
	-g ./graphs \
	-q 0 \
	--teleport_alpha 0.6 \
	-m hesitating_random_walk

