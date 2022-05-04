import sys, getopt
import json
import os, os.path
import numpy as np 
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import seaborn
import pandas as pd
import pickle
import pybedtools
from joblib import Parallel, delayed
from tqdm import tqdm
import time

''' This script counts the number of peaks per bin of Hi-C (250kb in the paper) for each genomic feature 
and outputs a matrix of feature x region for each chromosome separately.'''


def get_chrom_sizes(genome_dir,resol):
    '''
    Constructs dataframe of chromosome sizes (in bp and in loci counts at the chosen resolution)
    Args:
        genome_dir: (string) directory of genome data
        resol: (int) hic resolution
    Returns:
        A pandas datafrawith columns chr (chromosome), size, size_loci, size_roundup
    '''
    sizes_filename = genome_dir+'hg19.chrom.sizes'
    df_sizes = pd.read_csv(sizes_filename, sep = '\t', header = None, names=['chr','size'])
    df_sizes['chr'] = df_sizes['chr'].str.split('chr',expand=True)[1]
    df_sizes['size_loci'] = np.ceil(df_sizes['size']/resol).astype(int)
    df_sizes['size_roundup'] = np.ceil(df_sizes['size']/resol).astype(int)*resol
    df_sizes = df_sizes[~df_sizes['chr'].str.contains('hap|alt|Un|rand')]
    return(df_sizes)


def make_chrom_bed(chrom, chrom_size, resol):
    '''
    Create dataframe dividing the chromosome chrom into segments of resol length
    Args:
        chrom: (int) chromosome of interest
        chrom_size: (int) size of chrom
        resol: (int) Hi-C resolution
    Returns:
        A BED file containing the dataframe
    '''
    # Divide the chromosome into segments of HIC_RESOLN length
    stop_pos = np.arange(resol, chrom_size + resol, resol, dtype = 'int')
    df_chrom = pd.DataFrame()
    df_chrom['chrom'] = ['chr' + str(chrom)]*len(stop_pos)
    df_chrom['start'] = stop_pos - resol
    df_chrom['stop'] = stop_pos

    # Convert to bed file
    bed_chrom = pybedtools.BedTool.from_dataframe(df_chrom)
    return(bed_chrom)


def get_feature_matrix_chrom(bed_chrom, df, epigenome_dir, transcript_annotations, rna_seq_counts, chrom, resol):
    '''
    Create dataframe with features as indices and loci as columns for loci in bed_chrom
    Args:
        bed_chrom: (BED file) loci of interest (usually all loci of a given chromosome)
        df: (pandas DataFrame) epigenomic features metadata
        epigenome_dir: (string) directory of raw epigenomic data
        rna_seq_counts: (pandas DataFrame) TPM counts for transcript ids
        chrom: (int) number of the chromosome
        resol: (int) resolution of the HiC-Data
    Returns:
        A BED file containing the dataframe and a 
        Pandas DataFrame with the corresponding locus for each gene
    '''    
    bed_chrom_df = bed_chrom.to_dataframe()
    df = pd.DataFrame({'feature': df['name'], 
                       'filename': df['filename']})
    feature_matrix = pd.DataFrame(index = df['feature'].values, columns = bed_chrom_df['start'].values)
    
    for i in range(len(df)):
        f = df.loc[i,'filename']
        feature = df.loc[i,'feature']
        # Get bed file of the feature
        bed = pybedtools.BedTool(epigenome_dir + f).sort()
        # Get counts for this feature and this chromosome
        out = pybedtools.bedtool.BedTool.map(bed_chrom, bed, c = 4, o = 'count_distinct')
        counts = out.to_dataframe()['name'].values
        # Store results into matrix
        feature_matrix.loc[feature, :] = counts
        
    # Get RNA-seq counts per locus
    RNA_per_locus, gene_locus_chrom = get_rna_per_locus(bed_chrom, transcript_annotations, rna_seq_counts, chrom, resol)
    RNA_per_locus.columns = feature_matrix.columns
    
    # Add RNA-seq counts to the feature matrix
    feature_matrix = pd.concat([feature_matrix, RNA_per_locus])
    return(feature_matrix, gene_locus_chrom)


def get_rna_per_locus(bed_chrom, transcript_annotations, rna_seq_counts, chrom, resol):
    '''
    Create dataframe with RNA-seq count for each locus in bed_chrom 
    Args:
        bed_chrom: (BED file) loci of interest (usually all loci of a given chromosome)
        transcript_annotations: (pandas DataFrame) positions (chrom, start, end) of transcript ids
        rna_seq_counts: (pandas DataFrame) TPM counts for transcript ids
        chrom: (int) number of the chromosome
        resol: (int) resolution of the HiC-Data
    Returns:
        Pandas DataFrame with the summed RNA-seq count per locus and a
        Pandas DataFrame with the corresponding locus for each gene
    '''    
    # Get bed file of the transcript annotations
    bed = pybedtools.BedTool.from_dataframe(transcript_annotations).sort()
    # Get transcript ids that overlap with each locus
    out = pybedtools.bedtool.BedTool.map(bed_chrom, bed, c = 4, o = 'collapse', F = 0.5)
    counts = out.to_dataframe()['name'].values
    
    # Calculate sum of transcript counts per locus
    loci_counts = []
    gene_locus_chrom = pd.DataFrame({'locus': [], 'gene': []})
    loc = 0
    
    for loc_genes in tqdm(counts):
        time.sleep(.001)
        genes = loc_genes.split(r",")
        # save df with genes and locus
        locus_list = ["chr_"+str(chrom)+"_loc_"+str(loc)] * len(genes)
        gene_locus_chrom = pd.concat([gene_locus_chrom, pd.DataFrame({'locus': locus_list, 'gene': genes})])
        # create df with genes in loci and corresponding TPM counts
        genes_df = pd.DataFrame({'name': genes})
        gene_counts = pd.merge(genes_df, rna_seq_counts, on = "name", how = "left")
        # sum TPMs of all genes per locus
        loci_count = sum(gene_counts['count'])
        loci_counts.append(loci_count)
        loc = loc + resol

    loci_counts = pd.DataFrame({'RNA-seq':loci_counts}).fillna(0)
    return(loci_counts.T, gene_locus_chrom)


def get_filtered_chipseq(chrom, blacklist, processed_epigenome_data_dir):
    '''
    Removes blacklisted loci from epigenomic data for chromosome chrom
    Args:
        chrom: (int) chromosome of interest
        blacklist: (dict) dictionary of blacklisted loci for every chromosome
        processed_epigenome_data_dir: (string) directory of processes epigenomic data
    Returns:
        Epigenomic data for chromosome chrom without blacklisted loci
    '''
    df_chipseq = pd.read_csv(processed_epigenome_data_dir+'features_matrix_chr'+str(chrom)+'.csv', index_col = 0)
    # get all blacklisted loccations
    blacklist_chr = set(blacklist[chrom])
    # get a list of columns to keep
    allcols = set(map(int,df_chipseq.columns))
    cols2keep = allcols - blacklist_chr
    df_chipseq_filt = df_chipseq[list(map(str,cols2keep))]
    return df_chipseq_filt    
    

def get_mean_std(chr_list, blacklist, processed_epigenome_data_dir):
    '''
    Get mean and standard deviation for each epigenomic feature across all the genome (without considering
    blacklisted loci)
    Args:
        chr_list: (list) list of chromosomes
        blacklist: (dict) dictionary of blacklisted loci for every chromosome
        processed_epigenome_data_dir: (string) directory of processes epigenomic data
    Returns:
        An array containing the mean and standard deviation for each epigenomic feature
    '''
    # collect chipseq data across all chromosomes into one dataframe
    df_all = pd.DataFrame()
    for chrom in chr_list:
        df_chipseq_filt = get_filtered_chipseq(chrom, blacklist, processed_epigenome_data_dir)
        df_all = pd.concat([df_all, df_chipseq_filt], axis=1)
    # transform
    df_all = np.log(df_all + 1)
    # find mean and standard dev
    mean_features = np.mean(df_all, axis=1)
    std_features = np.std(df_all, axis=1)
    return mean_features, std_features


def normalize_chipseq(chr_list, mean_features, std_features, processed_epigenome_data_dir):
    '''
    For each chromosome, center each epigenomic feature using mean_features and scale it using std_features
    Args:
        chr_list: (list) list of chromosomes
        mean_features: (Numpy array) array of means for centering each epigenomic feature
        std_features: (Numpy array) array of standard deviations for scaling each epigenomic feature
        processed_epigenome_data_dir: (string) directory of processes epigenomic data
    Returns:
    '''
    for chrom in chr_list:
        # get chipseq data
        df_chipseq = pd.read_csv(processed_epigenome_data_dir+'features_matrix_chr'+str(chrom)+'.csv', index_col = 0)
        # transform
        df_chipseq = np.log(df_chipseq + 1)
        # normalize
        df_norm = (df_chipseq.T - mean_features)/std_features
        # transpose back
        df_norm = df_norm.T
        # save
        df_norm.to_csv(processed_epigenome_data_dir+'features_matrix_chr'+str(chrom)+'_norm.csv')

        
def parse_config(config_filename):
    '''
    Reads config file
    Args:
        config_filename: (string) configuration file name
    Returns:
        A dictionary specifying the main directories, cell type, resolution, quality, chromosomes
    '''
    config = json.load(open(config_filename))
    return(config)


def get_transcript_annotations(genome_dir):
    '''
    Constructs dataframe of transcript annotations 
    Args:
        genome_dir: (string) directory of genome data
    Returns:
        A pandas dataframe with columns chr (chromosome), start, end, name (ensembl id of transcripts), score (strand + or -)
    '''
    transcript_annotations = pd.read_csv(genome_dir+'gene_annotation.tsv', sep = '\t', header=0)
    selected_cols = ["chrom", "txStart", "txEnd", "name", "strand"]
    transcript_annotations = transcript_annotations.loc[:, selected_cols]
    transcript_annotations.columns = ["chrom", "start", "end", "name", "score"]
    return(transcript_annotations)


def get_rna_seq_counts(epigenome_dir, rna_seq_filename):
    '''
    Constructs dataframe of transcript annotations 
    Args:
        epigenome_dir: (string) directory of epigenome data
        rna_seq_filename: (string) filename of the transcriptomics data
    Returns:
        A pandas dataframe with columns name (ensembl transcript id), count (TPM)
    '''
    rna_seq_counts = pd.read_csv(epigenome_dir + rna_seq_filename, sep = '\t', header = 0)
    #select columns
    selected_cols = ["transcript_id", "TPM"]
    rna_seq_counts = rna_seq_counts.loc[:, selected_cols]
    #remove version number of transcript_id
    rna_seq_counts['transcript_id'] = rna_seq_counts['transcript_id'].str.split(r'.').str.get(0)
    #rename columns
    rna_seq_counts.columns = ["name", "count"]
    return(rna_seq_counts)
    
    
def main():
    
    # Parse command line arguments
    argv = sys.argv[1:]
    try:
        options, args = getopt.getopt(argv, "c:", ["config="])
    except:
        print('Incorrect arguments!')
        
    for name, value in options:
        if name in ('-c', '--config'):
            config_filename = value
    
    # Parse config file
    print('Options successfully parsed, read arguments...')
    config = parse_config(config_filename)
    genome_dir = config['GENOME_DIR']
    processed_hic_data_dir = config['PROCESSED_HIC_DATA_DIR']
    epigenome_dir = config['EPIGENOME_DIR']
    processed_epigenome_data_dir = config['PROCESSED_EPIGENOME_DIR']
    rna_seq_filename = config['RNA-SEQ_FILENAME']
    hic_celltype = config['HIC_CELLTYPE']
    resol_str = config['HIC_RESOLUTION_STR']
    resol = config['HIC_RESOLUTION']
    chr_list = config['chrs']
    
    # Get chromosome sizes
    df_sizes = get_chrom_sizes(genome_dir, resol)
    # Get dataframe listing available epigenomic data
    df = pd.read_csv(epigenome_dir + hic_celltype + '_regulatory_data.csv', sep=';', header=0)
    
    # RNA-Seq_Processing
    # Get transcript annotations (chrom, start, end, name, score)
    transcript_annotations = get_transcript_annotations(genome_dir)
    # Get RNA-Seq counts
    rna_seq_counts = get_rna_seq_counts(epigenome_dir, rna_seq_filename)
    
    # Get feature matrix for every chromosome in chr_list
    print('Get feature matrix for all chromosomes')
    genes_loci = pd.DataFrame({'locus': [], 'gene': []})
    for chrom in chr_list:
        print('Get feature matrix for chr'+str(chrom))
        # Get chromosome size
        chrom_size = int(df_sizes.loc[df_sizes['chr']==str(chrom)]['size'])
        # Divide the chromosome into segments of HIC_RESOLN length
        bed_chrom = make_chrom_bed(chrom, chrom_size, resol)
        # Process epigenomic and transcriptomic data for chrom
        feature_matrix, gene_locus_chrom = get_feature_matrix_chrom(bed_chrom, df, epigenome_dir, 
                                                                    transcript_annotations, rna_seq_counts, chrom, resol)
        # write feature matrix to file
        feature_matrix.to_csv(processed_epigenome_data_dir + 'features_matrix_chr' + str(chrom) + '.csv')
        
        # add gene_loci df per chromosome to the complete one
        genes_loci = pd.concat([genes_loci, gene_locus_chrom])
  
    # Normalize feature matrix for all chromosomes
    print('Normalize epigenomic data for all chromosomes')
    blacklist = pickle.load(open(processed_hic_data_dir+'blacklist_' + hic_celltype + '_INTERKR.pickle', 'rb'))
    mean_features, std_features = get_mean_std(chr_list, blacklist, processed_epigenome_data_dir)
    normalize_chipseq(chr_list, mean_features, std_features, processed_epigenome_data_dir)
    
    # Save genes_loci df
    genes_loci.to_csv(genome_dir+'genes_loci.csv', index = False)

if __name__ == "__main__":
    main()
    




