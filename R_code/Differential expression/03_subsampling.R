# R script to get DE genes for subsampled patients

# load libraries
library(tidyverse)
library(magrittr)
library(tximeta)
library(DESeq2)

# data directory
dir <- "/Users/janabraunger/Documents/Studium/Master/Boston/Masterarbeit/"

# define age groups
age_groups <- data.frame(Age = as.character(seq(1, 96, 1)), 
                         age_group = c(rep('1-15', 15), rep('16-26', 11), 
                                       rep('27-60', 34), rep('61-85', 25),
                                       rep('86-96', 11)))


# get info about the runs
runs_info <- read.delim(paste0(dir, "Data/SraRunTable.txt"), sep = ",") %>%
  mutate(Age = gsub("y[rs]\\d+mos", "", Age)) %>%
  mutate(Age = gsub("yr", "", Age)) %>%
  left_join(age_groups) %>%
  mutate(age_group = factor(age_group, levels = unique(age_groups$age_group))) %>%
  filter(disease == "Normal") %>% 
  #filter(sex == "male") %>%
  mutate(Age = as.numeric(Age))
runs_info$names <- runs_info$Run
runs_info$files <- paste0(dir, "Data/rna_counts/", runs_info$names, "_quant/quant.sf")

# create summarized experiment with tximeta
se <- tximeta(runs_info)

# convert from ensembl transcript id to gene id
gse <- summarizeToGene(se)
gse <- gse[rowData(gse)$gene_biotype == "protein_coding", ]

# filter for genes that have counts of at least 10 in at least 5 samples
keep <- rowSums(assay(gse) >= 10) >= 5
gse <- gse[keep,]

# get DE genes for 100 sub-samples 
de_results <- c()
for (sim in seq(100)) {
  print(paste0("simulation ", sim))
  # Subsampling: selecting 80% of individuals per group
  set.seed(paste0(2022, sim))
  subsample <- data.frame(colData(gse)) %>% 
    group_by(age_group) %>% 
    slice_sample(prop = 0.8)
  gse_sub <- gse[, subsample$Run]
  
  dds <- DESeqDataSet(gse_sub, design = ~ age_group)
  
  # differential expression analysis
  dds <- DESeq(dds)
  foldchanges <- lapply(seq(1, length(unique(age_groups$age_group))-1, 1), function(ix) {
    t0 = unique(age_groups$age_group)[ix]
    t1 <- unique(age_groups$age_group)[ix+1]
    # calculate log2 foldchange between two age intervals 
    res <- results(dds, contrast=c("age_group", t0, t1))
    res$gene <- rowData(dds)$symbol
    res$transition <- paste0('fc_', t0, '_', t1)
    res <- subset(res, select = c(gene, transition, log2FoldChange, padj)) %>%
      as.data.frame() %>%
      rownames_to_column(var = "ensembl_id") %>%
      filter(gene != "")
    return(res)}) %>%
    bind_rows() %>%
    drop_na(padj) %>%
    filter(padj < 0.1) %>%
    subset(select = c(gene, transition)) %>%
    mutate(simulation = sim) %>%
    distinct()
  
  de_results <- bind_rows(de_results, foldchanges)
}
  
# Saving the results
write.csv(de_results, paste0(dir, "Data/subsampling_DE_0.1.csv"), row.names = FALSE)

