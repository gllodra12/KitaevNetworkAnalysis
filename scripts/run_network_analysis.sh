#!/bin/bash

for correlation in concurrence #mutual coherence 
do
    for measure in density # clustering
    do
        #python ./kitaev_network_analysis/network_analysis.py $measure $correlation 14 -w 1 -d 0.5 -t 0 -pT -s 0.01 
        python ./kitaev_network_analysis/network_analysis.py $measure $correlation 8 -w 1 -d 0.5 -t 0 -pF -s 0.01 
    done
done
