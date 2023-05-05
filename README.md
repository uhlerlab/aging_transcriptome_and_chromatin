# Analysis of the regulatory network and chromatin organization in aging human fibroblasts
by Jana M. Braunger, Louis V. Cammarata, G.V. Shivashankar and Caroline Uhler

## Abstract
Since the human life expectancy is constantly increasing and aging has become a major risk factor for many diseases, a better understanding of the regulatory processes during aging is needed. Therefore, we used human skin fibroblasts as a model to study the regulation of aging by transcription factors, as well as by chromatin organization changes. Firstly, we identified transcription factors that drive the aging process by using the Prize-Collecting Steiner Tree algorithm to build protein-protein interactomes that capture regulatory changes at each life stage transition. Secondly, we mapped these regulatory findings to changes in chromatin organization using chromosomal conformation capture (Hi-C) data for young and old skin fibroblasts. The data was clustered using the Large Average Submatrices Algorithm, revealing important changes in interchromosomal intermingling. Co-regulated genes tended to colocalize in the young and old cell states, which supports the hypothesis of a tight coupling of transcriptional changes and chromatin reorganization during aging. 

## Structure of this repository
This github repository consists of the code we used for the analyses described in our paper:
1. R: 
    - RNAseq_processing.Rmd: This file contains the RNAseq data processing, the age group definition, as well as the differential gene expression analysis (Fig. 1).
    - DE_gene_robustness.R: This R script was used for obtaining DE genes of subsets of individuals to identify robust DE genes (Supplementary Fig. 2A).
    - Expression_pattern_DE_genes.Rmd: This markdown contains the heatmaps to show the expression over the five age groups of selected genes (DE genes between two consecutive age groups (Supplementary Fig. 3) and signature DE genes (Fig. 4A1/B1/C1/D1)). 
    - TF_expression_trajectories.Rmd: This markdown was used to visualize the expression of selected TFs over the five age groups (Supplementary Fig. 10).
    - TF_clustering.Rmd: This markdown was used to cluster the bridge TFs based on a variety of features (Fig. 5C and Supplementary Fig. 13) and create the boxplots for Fig. 5A and Fig. 5B.
2. Python: 
    - Steiner_networks: This folder contains the python script to run the prize-collecting Steiner tree (pcst_utils.py), as well as the jupyter notebook used for the analyses in Fig. 2 (Steiner_networks.ipynb) and Supplementary Fig. 7 (Steiner_networks_robustness.ipynb). 
    - Process_HiC_data: Inter- and intrachromosomal Hi-C data was processed separately using process_hic_contacts_intra.py and process_hic_contacts_inter.py to filter out repetive chromosome regions such as the centromeres and log-transform and z-score the data.
    - LAS: The LAS algorithm was run for each chromosome pair (LAS_intra.py, LAS_inter.py) and then the python script combine_all_submatrices.py was used to check whether submatrices found in one sample also have a high score in the others and only keeping submatrices whose score was above the threshold in both replicates.
    - Intermingling_difference_maps: The jupyter notebook intermingling_difference_maps.ipynb was used to create the binarized intermingling maps for young and old and create intermingling difference maps for selected genes (Fig. 3 and Fig. 4). The python script create_intermingling_networks.py was used to create interactive networks with dash cytoscape (Fig. 5D, Supplementary Fig. 14 and 15).
3. utils: This folder contains the juicer tools jar file used for the 
    - the pre command to convert the data from the .pair to the .hic format
    - the dump command for normalization
    
## Dependencies needed to run the code
1. R (version 4.2.1) with the following packages: 
    - tidyverse (version 1.3.2)
    - tximeta (version 1.14.1)
    - DESeq2 (version 1.36.0)
    - biomaRt (version 2.52.0)
    - UpSetR (version 1.4.0)
    - pheatmap (version 1.0.12)
2. Python (version 3.7.13) with the following packages:
    - networkx (version 2.4)
    - OmicsIntegrator 2 (version 2.3.10)
    - gseapy (version 0.10.8)
    - upsetplot (version 0.6.1)
