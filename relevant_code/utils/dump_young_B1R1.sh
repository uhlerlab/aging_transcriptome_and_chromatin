#!/bin/bash
resol=250000
MAPQ=30
num_chromosomes=22
for i in $(seq 1 22)
do
    for j in $(seq ${i} 22)
    do
        java -jar /home/braunger/juicer/juicer_tools_dump.jar dump observed NONE /home/braunger/rejuvenation_project/data/hic_raw/Young_B1R1_dedup_filter.dedup.hic ${i} ${j} BP ${resol} /home/braunger/rejuvenation_project/data/processed_hic_data/Young_B1R1/normalized_BP${resol}/chr${i}_chr${j}_NONE.txt       
        java -jar /home/braunger/juicer/juicer_tools_dump.jar dump observed SCALE /home/braunger/rejuvenation_project/data/hic_raw/Young_B1R1_dedup_filter.dedup.hic ${i} ${j} BP ${resol} /home/braunger/rejuvenation_project/data/processed_hic_data/Young_B1R1/normalized_BP${resol}/chr${i}_chr${j}_KR.txt
        java -jar /home/braunger/juicer/juicer_tools_dump.jar dump observed INTER_SCALE /home/braunger/rejuvenation_project/data/hic_raw/Young_B1R1_dedup_filter.dedup.hic ${i} ${j} BP ${resol} /home/braunger/rejuvenation_project/data/processed_hic_data/Young_B1R1/normalized_BP${resol}/chr${i}_chr${j}_INTERKR.txt 
        echo "Normalize Hi-C for chromosome pair ${i}, ${j} (NONE, KR, INTER_KR)"
    done
done
