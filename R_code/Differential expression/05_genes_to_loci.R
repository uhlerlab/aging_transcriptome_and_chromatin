# Get loci for the targets of a TF
selected_TF = "HIF1A"

# Libraries
library(tidyverse)
library(biomaRt)

# Settings
dir = "/Users/janabraunger/Documents/Studium/Master/Boston/Masterarbeit/"
options(scipen = 999) 

# Load TF targets
data <- read.delim(paste0(dir, "Data/tf-target-information.txt")) %>%
  subset(select = c("TF", "target")) %>%
  distinct() %>%
  filter(TF == selected_TF) %>%
  dplyr::rename(gene = target) %>%
  subset(select = c(gene))

# Query gene information from biomart
mart <- useEnsembl(biomart = "ensembl", 
                   dataset = "hsapiens_gene_ensembl", 
                   mirror = "uswest")
gene_annotations <- getBM(filters="external_gene_name", 
                          attributes=c("external_gene_name", 
                                       "chromosome_name", "start_position", 
                                       "end_position", "description"), 
                          values=c(unique(data$gene), selected_TF), mart=mart) %>%
  data.frame() %>%
  arrange(chromosome_name) %>%
  group_by(external_gene_name) %>%
  summarize(chr = dplyr::first(chromosome_name), start = dplyr::first(start_position), 
            end = dplyr::first(end_position), description = dplyr::first(description)) %>%
  dplyr::rename(gene = external_gene_name)

# Add loci to gene dataframe
data_loci <- gene_annotations %>%
  mutate(locus = paste0("chr_", chr, "_loc_", 
                        plyr::round_any(start, 250000, floor))) %>%
  filter(chr %in% as.character(seq(1,22,1))) %>%
  arrange(as.numeric(chr), start) %>%
  subset(select = c(gene, locus))

# Save results
write.csv(data_loci, paste0(dir, "Data/", selected_TF, "_targets_loci.csv"), row.names = FALSE)
