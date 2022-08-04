from importlib import reload
import sys, getopt
import json
import os, os.path
import pickle
import numpy as np 
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
plt.ioff()
import seaborn as sns
import pandas as pd
import itertools
from tqdm import tqdm
import time
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, inconsistent

''' This script clusters the inter and intra hic maps.'''

def get_interchromosomal_hic(cell_type, chr_list, hic_dir):
    '''
    Constructs dataframe in long format of all interchromosomal hic contacts
    Args:
        cell_type: (sting) 'IMR90' or 'old_fibroblasts'
        chr_list: (list) list of all chromosomes
        hic_dir: (string) directory of hic data
    Returns:
        A pandas datafrawith columns chr1, chr2, value
    '''
    print("Get interchromosomal hic data:")
    chr_pairs = list(itertools.combinations(chr_list, 2))

    hic_all_inter = pd.DataFrame({'chr1': [], 'chr2': [], 'value': []})

    for pair in tqdm(chr_pairs):
        time.sleep(.01)
        chr1, chr2 = pair

        # read pickle
        hic = pd.read_pickle(hic_dir + 'processed_hic_data_' + cell_type + '/final_BP250000_intraKR_interINTERKR/hic_chr' + str(chr1) + '_chr' + str(chr2) + '_norm1_filter3.pkl')
        # change colnames
        hic = hic.add_prefix('chr_'+str(chr2)+'_loc_')
        # change rownames and put rownames as a column
        hic = hic.T.add_prefix('chr_'+str(chr1)+'_loc_').T
        hic.reset_index(inplace=True)
        # convert df to long format
        hic = hic.melt(id_vars = ['index'])
        # change colnames 
        hic.columns = ['chr1', 'chr2', 'value']

        hic_all_inter = pd.concat([hic_all_inter, hic])
    return(hic_all_inter)


def get_intrachromosomal_hic(cell_type, chr_list, hic_dir):
    '''
    Constructs dataframe in long format of all intrachromosomal hic contacts
    Args:
        cell_type: (sting) 'IMR90' or 'old_fibroblasts'
        chr_list: (list) list of all chromosomes
        hic_dir: (string) directory of hic data
    Returns:
        A pandas dataframe with columns chr1, chr2, value
    '''
    print("Get intrachromosomal hic data:")
    hic_all_intra = pd.DataFrame({'chr1': [], 'chr2': [], 'value': []})
    
    for chrom in tqdm(chr_list):
        time.sleep(.01)
        
        # read pickle
        hic = pd.read_pickle(hic_dir + 'processed_hic_data_' + cell_type + '/final_BP250000_intraKR_interINTERKR/hic_chr' + str(chrom) + '_chr' + str(chrom) + '_norm1_filter3.pkl')
        # change colnames
        hic = hic.add_prefix('chr_'+str(chrom)+'_loc_')
        # change rownames and put rownames as a column
        hic = hic.T.add_prefix('chr_'+str(chrom)+'_loc_').T
        hic.reset_index(inplace=True)
        # convert df to long format
        hic = hic.melt(id_vars = ['index'])
        # change colnames 
        hic.columns = ['chr1', 'chr2', 'value']

        hic_all_intra = pd.concat([hic_all_intra, hic])
    return(hic_all_intra)


def normalize_hic(hic_long, quantile):
    '''
    Normalizes the hic values
    Args:
        hic_long: (pd DataFrame) data frame containing hic values in long format with columns chr1, chr2 and value
        quantile: (float) quantile to use for the normalization
    Returns:
        A pandas dataframe with columns chr1, chr2, value, norm_value
    '''
    # get value of the corresponding quantile
    q_hic = np.quantile(hic_long['value'], quantile)
    # divide by this quantile value
    hic_long['value_norm'] = hic_long['value'].div(q_hic)
    # set values greater than 1 to 1 (so that all values are between 0 and 1)
    hic_long['value_norm'][hic_long['value_norm'] > 1] = 1
    return(hic_long)


def subset_MEFISTO(hic_df, MEFISTO_dir, percentage):
    '''
    Normalizes the hic values
    Args:
        hic_df: (pd DataFrame) long df with all hic contacts
        MEFISTO_dir: (str) directory containing the file with the genes with most extreme weights in the first MEFISTO factor
        percentage: (float) percentage of how many genes should be selected
    Returns:
        A pandas dataframe with columns chr1, chr2, value, norm_value
    '''
    # get df with the top weighted genes from MEFISTO
    pos_weights = pd.read_csv(MEFISTO_dir+'top_genes_pos_'+str(percentage)+'_50.csv')
    pos_weights['sign'] = 'positive'
    neg_weights = pd.read_csv(MEFISTO_dir+'top_genes_neg_'+str(percentage)+'_50.csv')
    neg_weights['sign'] = 'negative'
    selected_genes = pd.concat([pos_weights, neg_weights])

    # filter Hi-C data to the MEFISTO loci
    hic_df_subset = hic_df[hic_df['chr1'].isin(selected_genes['locus']) & 
                           hic_df['chr2'].isin(selected_genes['locus'])]
    return(hic_df_subset)


def long_to_wide(df):
    '''
    Convert df in long format to a symmetric wide df
    Args:
        df: (pd DataFrame) long df with all hic contacts
    Returns:
        A pandas dataframe in wide format (loci by loci)
    '''
    # add values for switched chr1 and chr2 to expand to wide format
    inverted_df = pd.DataFrame({'chr1': df['chr2'], 'chr2': df['chr1'],
                                'value': df['value'], 'value_norm': df['value_norm']})
    df = pd.concat([df, inverted_df])

    # from long to wide format
    df_wide = df.pivot_table(index='chr1', columns='chr2', values='value_norm', sort = False).fillna(0)
    # sort columns in the order of the rows
    df_wide = df_wide[df_wide.index.tolist()]
    
    return(df_wide)


def plot_clusters_chrom(df, title):
    '''
    Plots clustered heatmap with color annotation according to chromosomes
    Args:
        df: (pd DataFrame) normalized hic data
    Returns:
        Clustered heatmap
    '''
    # if all zero column exists, remove column and row corresponding to that locus
    df = df.loc[:, (df != 0).any(axis=0)]
    df = df.T.loc[:, (df.T != 0).any(axis=0)].T
    
    # add color labels according to the chromosome number
    color_dict = {}
    palette = sns.color_palette("husl", 22)
    for loc in df.columns:
        chrom = loc.split("_")[1]
        color_dict[loc] = palette[int(chrom)-1]
    color_rows = pd.Series(color_dict)
    
    # plot clustered heatmap
    plt.figure()
    sns.clustermap(df,
                   method='average',
                   metric='cosine',
                   row_cluster=True, col_cluster=True,
                   figsize=(6,5),
                   xticklabels=False, yticklabels=False,
                   cmap='Reds', cbar_pos=(1, 0.5, 0.01, .4),
                   vmin=0, vmax=1,
                   dendrogram_ratio=(.1, .1),
                   row_colors=[color_rows], col_colors=[color_rows]).fig.suptitle(title)
    
    
def plot_clusters_sign(df, title, selected_genes):
    '''
    Plots clustered heatmap with color annotation according to sign of the MEFISTO weight
    Args:
        df: (pd DataFrame) normalized hic data
        title: (string) title of the plot
        selected_genes: (pd DataFrame) contains the signs of the selected genes
    Returns:
        Clustered heatmap
    '''
    
    # add color labels according to the sign of the gene in the first factor
    color_dict = {}
    palette = sns.color_palette()
    for loc in df.columns:
        if loc in selected_genes[selected_genes['sign'] == 'positive']['locus'].tolist():
            color_dict[loc] = palette[0]
        else:
            color_dict[loc] = palette[1]
    color_rows = pd.Series(color_dict)
    
    # plot clustered heatmap
    plt.figure()
    p = sns.clustermap(df,
                   method='average',
                   metric='cosine',
                   row_cluster=True, col_cluster=True,
                   figsize=(6,5),
                   xticklabels=False, yticklabels=False,
                   cmap='Reds', cbar_pos=(1, 0.5, 0.01, .4),
                   vmin=0, vmax=1,
                   dendrogram_ratio=(.1, .1),
                   row_colors=[color_rows], col_colors=[color_rows])
    
    order = p.dendrogram_row.reordered_ind
    p.fig.suptitle(title)
    
    # add legend
    handles = [Patch(facecolor=palette[name]) for name in [0, 1]]
    plt.legend(handles, ['positive', 'negative'], title='Loading',
               bbox_transform=plt.gcf().transFigure, bbox_to_anchor=(1., 0.3, 1., .102), loc='lower left')
    

def plot_clusters_pairs(df1, title1, df2, title2, selected_genes):
    '''
    Plots clustered heatmap with color annotation according to sign of the MEFISTO weight and second one in the same order
    Args:
        df1: (pd DataFrame) normalized hic data that should be clustered
        title1: (string) title of the left subplot
        df2: (pd DataFrame) normalized hic data that should be shown according to the clustering of df1
        title2: (string) title of the right subplot
        selected_genes: (pd DataFrame) contains the signs of the selected genes
    Returns:
        Clustered heatmaps for IMR90 and old fibroblasts in the same order
    '''
    
    # add color labels according to the sign of the gene in the first factor
    color_dict = {}
    palette = sns.color_palette()
    for loc in df1.columns:
        if loc in selected_genes[selected_genes['sign'] == 'positive']['locus'].tolist():
            color_dict[loc] = palette[0]
        else:
            color_dict[loc] = palette[1]
    color_rows = pd.Series(color_dict)
    
    # plot clustered heatmap
    plt.figure()
    p = sns.clustermap(df1,
                   method='average',
                   metric='cosine',
                   row_cluster=True, col_cluster=True,
                   figsize=(6,5),
                   xticklabels=False, yticklabels=False,
                   cmap='Reds', cbar_pos=(1, 0.5, 0.01, .4),
                   vmin=0, vmax=1,
                   dendrogram_ratio=(.1, .1),
                   row_colors=[color_rows], col_colors=[color_rows])
    
    order = p.dendrogram_row.reordered_ind
    p.fig.suptitle(title1)
    
    # add legend
    handles = [Patch(facecolor=palette[name]) for name in [0, 1]]
    plt.legend(handles, ['positive', 'negative'], title='Loading',
               bbox_transform=plt.gcf().transFigure, bbox_to_anchor=(1., 0.3, 1., .102), loc='lower left')
        
    # change ordering of df2 to the clustering of df1
    df2 = df2.iloc[order, order]
    
    # add color labels according to the sign of the gene in the first factor for df2
    color_dict = {}
    palette = sns.color_palette()
    for loc in df2.columns:
        if loc in selected_genes[selected_genes['sign'] == 'positive']['locus'].tolist():
            color_dict[loc] = palette[0]
        else:
            color_dict[loc] = palette[1]
    color_rows = pd.Series(color_dict)
    
    # plot 2
    p2 = sns.clustermap(df2,
                   method='average',
                   metric='cosine',
                   row_cluster=False, col_cluster=False,
                   figsize=(6,5),
                   xticklabels=False, yticklabels=False,
                   cmap='Reds', cbar_pos=(1, 0.5, 0.01, .4),
                   vmin=0, vmax=1,
                   dendrogram_ratio=(.1, .1),
                   row_colors=[color_rows], col_colors=[color_rows])
    p2.fig.suptitle(title2)
    # add legend
    handles = [Patch(facecolor=palette[name]) for name in [0, 1]]
    plt.legend(handles, ['positive', 'negative'], title='Loading',
               bbox_transform=plt.gcf().transFigure, bbox_to_anchor=(1., 0.3, 1., .102), loc='lower left')
    
    
def plot_hist(df1, df2, binwidth, celltype):
    '''
    Plots histogram to compare the distribution of loci with positive and negative weight
    Args:
        df1: (pd DataFrame) df in long format for positive loci
        df1: (pd DataFrame) df in long format for negative loci
    Returns:
        Histogram
    '''
    plt.hist(df1['value'], 
             bins=np.arange(min(df1['value']), max(df1['value']) + binwidth, binwidth), 
             alpha=0.5, label="positive weights")
    plt.hist(df2['value'], 
             bins=np.arange(min(df2['value']), max(df2['value']) + binwidth, binwidth), 
             alpha=0.5, label="negative weights")
    plt.xlabel("Normalized Hi-C values")
    plt.ylabel("Counts")
    plt.legend(loc='upper right')
    plt.title(celltype)
    

def get_4chrom_clustering(dend, genes, gene2chrom, inc=0.01, max_chrom=4, criterion='distance'):
    '''
    Function to determine the threshold to use to obtain a clustering with max 4 chromosomes per cluster
    (and the maximum number of genes)
    Args:
        dend: (Numpy array) the dendrogram
        genes: (Numpy array) genes (in the correct ordering with respect to dend)
        gene2chrom: (dict) dictionary mapping genes to the corresponding chromosome
        inc: (float) the incremental value for testing thresholds
        max_chrom: (int) maximum number of chromosome in a cluster
    Returns:
        A threshold to cut the dendrogram so that the resulting clustering has at most 4 chromosomes per cluster
    '''
    
    if criterion=='distance':
        # Determine height of dendrogram
        h = dend[-1,2]
        thresholds = np.arange(0,h+inc,inc)
        # Find threshold corresponding to max 4 chromosomes per cluster
        for t in thresholds:
            clustering = fcluster(dend, t, criterion=criterion)
            cluster_df = pd.DataFrame({'gene': genes,
                                       'cluster': clustering,
                                       'chrom': [gene2chrom[g] for g in genes]})
            max_chrom_current = cluster_df.groupby('cluster')['chrom'].nunique().max()
            if max_chrom_current >= max_chrom+1:
                break
        return(t-inc)
    
    elif criterion=='inconsistent':
        # Determine maximum inconsistency
        h = np.max(inconsistent(dend)[:,3])
        thresholds = np.arange(0,h+inc,inc)
        # Find threshold corresponding to max 4 chromosomes per cluster
        for t in thresholds:
            clustering = fcluster(dend, t, criterion=criterion)
            cluster_df = pd.DataFrame({'gene': genes,
                                       'cluster': clustering,
                                       'chrom': [gene2chrom[g] for g in genes]})
            max_chrom_current = cluster_df.groupby('cluster')['chrom'].nunique().max()
            if max_chrom_current >= max_chrom+1:
                break
        return(t-inc)


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
    hic_dir = config['HIC_DIR']
    mefisto_dir = config['MEFISTO_DIR']
    resol_str = config['HIC_RESOLUTION_STR']
    resol = config['HIC_RESOLUTION']
    chr_list = config['chrs']
    cell_type = config['CELL_TYPE']
    
    # Load interchromosomal hic data and normalize df
    inter_hic = get_interchromosomal_hic(cell_type, chr_list, hic_dir)
    inter_hic_norm = normalize_hic(inter_hic, 1)
    
    # Load intrachromosomal hic data and normalize df
    intra_hic = get_intrachromosomal_hic(cell_type, chr_list, hic_dir)
    intra_hic_norm = normalize_hic(intra_hic, 1)
    
    # Combine inter and intra
    hic = pd.concat([inter_hic_norm, intra_hic_norm])
    
    # Subset to MEFISTO genes (10% with highest positive and 10% with highest negative weight in factor 1)
    hic_subset = subset_MEFISTO(hic, mefisto_dir, 0.1)
    
    # Convert from long to wide format
    hic_wide = long_to_wide(hic_subset)    
    
    # Save results
    hic_wide.to_csv(hic_dir + 'processed_hic_data_' + cell_type + '/final_BP250000_intraKR_interINTERKR/MEFISTO_50.csv')
    

if __name__ == "__main__":
    main()
    





