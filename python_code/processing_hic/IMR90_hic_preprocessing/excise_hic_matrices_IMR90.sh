#!/bin/bash
resol=250000
MAPQ=30

java -Xmx5g -jar /home/braunger/juicer/hic_emt.jar excise -r ${resol} /home/braunger/masterthesis/data/hic_data/IMR90/GSE63525_IMR90_combined_30.hic /home/braunger/masterthesis/data/hic_data/IMR90
