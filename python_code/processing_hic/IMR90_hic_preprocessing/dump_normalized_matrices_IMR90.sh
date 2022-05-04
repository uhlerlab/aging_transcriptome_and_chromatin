#!/bin/bash
resol=250000
MAPQ=30
num_chromosomes=22
for i in $(seq 1 22)
do
    for j in $(seq ${i} 22)
    do
        java -jar /home/braunger/juicer/juicer_tools.jar dump observed NONE /home/braunger/masterthesis/data/hic_data/IMR90/GSE63525_IMR90_combined_30_excised.hic ${i} ${j} BP ${resol} /home/braunger/masterthesis/save/processed_hic_data/processed_hic_data_IMR90/normalized_BP${resol}/chr${i}_chr${j}_NONE.txt       
        java -jar /home/braunger/juicer/juicer_tools.jar dump observed SCALE /home/braunger/masterthesis/data/hic_data/IMR90/GSE63525_IMR90_combined_30_excised.hic ${i} ${j} BP ${resol} /home/braunger/masterthesis/save/processed_hic_data/processed_hic_data_IMR90/normalized_BP${resol}/chr${i}_chr${j}_KR.txt
        java -jar /home/braunger/juicer/juicer_tools.jar dump observed INTER_SCALE /home/braunger/masterthesis/data/hic_data/IMR90/GSE63525_IMR90_combined_30_excised.hic ${i} ${j} BP ${resol} /home/braunger/masterthesis/save/processed_hic_data/processed_hic_data_IMR90/normalized_BP${resol}/chr${i}_chr${j}_INTERKR.txt
        java -jar /home/braunger/juicer/juicer_tools.jar dump observed GW_SCALE /home/braunger/masterthesis/data/hic_data/IMR90/GSE63525_IMR90_combined_30_excised.hic ${i} ${j} BP ${resol} /home/braunger/masterthesis/save/processed_hic_data/processed_hic_data_IMR90/normalized_BP${resol}/chr${i}_chr${j}_GWKR.txt    
        echo "Normalize Hi-C for chromosome pair ${i}, ${j} (NONE, KR, INTER_KR, GW_KR)"
    done
done