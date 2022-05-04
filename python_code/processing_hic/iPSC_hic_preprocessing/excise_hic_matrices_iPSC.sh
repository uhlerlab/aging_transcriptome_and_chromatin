#!/bin/bash
resol=250000
MAPQ=30

java -Xmx5g -jar /home/braunger/juicer/hic_emt.jar excise -r ${resol} /home/braunger/masterthesis/data/hic_data/iPSC/GSM3576801_iPSC.hic /home/braunger/masterthesis/data/hic_data/iPSC
