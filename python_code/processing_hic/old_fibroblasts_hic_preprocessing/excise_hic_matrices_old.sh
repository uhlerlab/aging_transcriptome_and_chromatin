#!/bin/bash
resol=250000
MAPQ=30

java -Xmx5g -jar /home/braunger/juicer/hic_emt.jar excise -r ${resol} /home/braunger/masterthesis/data/hic_data/old_fibroblasts/ENCFF768UBD_hg19.hic /home/braunger/masterthesis/data/hic_data/old_fibroblasts
