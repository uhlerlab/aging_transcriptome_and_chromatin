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

''' This script creates a binarized map for selected loci based on whether a loci is part of a large average submatrix'''

def bin_intermingling_regions_hic_resoln(df_intermingling, chr1, chr2, resol):
    # break up large intermingling regions into 250kb bins (or resolution of HiC data)
    combos_list = []
    for row in df_intermingling.iterrows():
        row = row[1]
        start_row = range(int(row['start row']), int(row['stop row']) + resol, resol)
        start_col = range(int(row['start col']), int(row['stop col']) + resol, resol)
        combos = list(itertools.product(start_row, start_col))
        combos_list.append(combos)

    combos_list = np.asarray(list(itertools.chain.from_iterable(combos_list)))

    df_intermingling_binned = pd.DataFrame(combos_list, columns=['start row', 'start col'])
    
    df_intermingling_binned['chr1'] = "chr_" + str(chr1) + "_loc_" + df_intermingling_binned['start row'].astype(str)
    df_intermingling_binned['chr2'] = "chr_" + str(chr2) + "_loc_" + df_intermingling_binned['start col'].astype(str)
    df_intermingling_binned['value'] = 1

    return df_intermingling_binned[['chr1', 'chr2', 'value']].drop_duplicates()


def get_binarized_hic(selected_loci, chr_list, hic_dir, cell_type, resol, threshold):
    '''
    Constructs dataframe in long format of binarized map for selected loci
    Args:
        selected_loci: (list) list of loci with entries like 'chr_1_loc_155250000'
        chr_list: (list) list of all chromosomes
        hic_dir: (string) directory of processed hic data
        cell_type: (string) 'IMR90' or 'old_fibroblasts'
        resol: (string) resolution of the HiC maps
        threshold: (int) LAS threshold
    Returns:
        A pandas datafrawith columns chr1, chr2, value
    '''
    chr_pairs = list(itertools.combinations(chr_list, 2))

    hic_selection = pd.DataFrame({'chr1': [], 'chr2': [], 'value': []})
    
    # create df with all combinations of selected loci
    combos = list(itertools.product(selected_loci, selected_loci))
    hic_binarized = pd.DataFrame(combos, columns=['chr1', 'chr2'])

    for pair in tqdm(chr_pairs):
        time.sleep(.01)
        chr1, chr2 = pair
        
        # add value based on whether the loci combination is part of a large average submatrix
        fname = hic_dir + 'processed_hic_data_' + cell_type + '/LAS-'+ str(threshold) +'/intermingling_regions.chr' + str(chr1) + '_chr' + str(chr2) + '.avg_filt.csv'
        if (os.path.isfile(fname) == True):
            df_intermingling = pd.read_csv(fname, index_col = 0)
            # check if dataframe is empty
            if (len(df_intermingling) > 0):
                df_intermingling_binned = bin_intermingling_regions_hic_resoln(df_intermingling, chr1, chr2, resol)
                hic_selection = pd.concat([hic_selection, df_intermingling_binned])
                
    hic_binarized = pd.merge(hic_binarized, hic_selection, how = 'left', on = ['chr1', 'chr2'])
    hic_binarized['value'] = hic_binarized['value'].fillna(0)
    return(hic_binarized)


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
                                'value': df['value']})
    df = pd.concat([df, inverted_df])

    # from long to wide format
    df_wide = df.pivot_table(index='chr1', columns='chr2', values='value', aggfunc = "sum")
    
    return(df_wide)


def plot_intermingling_regions(chr_list, hic_dir, cell_type, count, threshold, ax):
    '''
    Plots a heatmap of the number of intermingling regions per chromosome pair
    Args:
        chr_list: (list) list of all chromosomes
        hic_dir: (string) directory of processed hic data
        cell_type: (string) 'IMR90' or 'old_fibroblasts'
        count: (string) 'LAS' or 'intermingling_regions' (depending on whether to visualize the number of submatrices or the number of pixels in the submatrices)
        threshold: (int) LAS threshold
        ax: axes to plot the subplot
    Returns:
        Heatmap
    '''
    # get numbers of intermingling regions
    hic_intermingling = LAS_statistics(chr_list, hic_dir, cell_type, threshold)
    hic_intermingling = hic_intermingling[['chr1', 'chr2', count]]
    hic_intermingling.columns = ['chr1', 'chr2', 'value']
    
    # conversion to long format
    hic_intermingling = long_to_wide(hic_intermingling).fillna(0)
    hic_intermingling.columns = hic_intermingling.columns.astype(int)
    hic_intermingling.index = hic_intermingling.index.astype(int)
    
    # plot heatmap
    if count == "intermingling_regions":
        sns.heatmap(hic_intermingling, cmap = "Reds", ax = ax, vmin = 0, vmax = 4000)
        ax.set_title(cell_type + ": Number of intermingling regions")
    elif count == "LAS":
        sns.heatmap(hic_intermingling, cmap = "Reds", ax = ax, vmin = 0, vmax = 15)
        ax.set_title(cell_type + ": Number of submatrices")
    ax.set_xlabel("")
    ax.set_ylabel("") 
    
    
def LAS_statistics(chr_list, hic_dir, cell_type, threshold):
    '''
    Calculates the number of intermingling regions and submatrices per chromosome pair
    Args:
        chr_list: (list) list of all chromosomes
        hic_dir: (string) directory of processed hic data
        cell_type: (string) 'IMR90' or 'old_fibroblasts'
        threshold: (int) LAS threshold
    Returns:
        pd DataFrame with number of intermingling regions and LAS submatrices per chr pair
    '''
    chr_pairs = list(itertools.combinations(chr_list, 2))
    hic_intermingling = pd.DataFrame({'chr1': [], 'chr2': [], 'intermingling_regions': [], 'LAS': []})

    for pair in chr_pairs:
        chr1, chr2 = pair

        fname = hic_dir + 'processed_hic_data_' + cell_type + '/LAS-' + str(threshold) + '/intermingling_regions.chr' + str(chr1) + '_chr' + str(chr2) + '.avg_filt.csv'
        if (os.path.isfile(fname) == True):
            df_intermingling = pd.read_csv(fname, index_col = 0)
            # check if dataframe is empty
            if (len(df_intermingling) > 0):
                intermingling_regions = bin_intermingling_regions_hic_resoln(df_intermingling, chr1, chr2, 250000)
                hic_intermingling = hic_intermingling.append({'chr1': chr1, 'chr2': chr2, 
                                                              'intermingling_regions': intermingling_regions.shape[0], 
                                                              'LAS': len(df_intermingling)}, ignore_index = True)
    return hic_intermingling
    
    
def plot_binarized_maps(young, old):
    '''
    Plots the binarized map for young and old fibroblasts next to each other
    Args:
        young: (pd DataFrame) binarized matrix with IMR90 data
        old: (pd DataFrame) binarized matrix with old_fibroblast data
    Returns:
        Two heatmaps
    '''
    fig, axs = plt.subplots(1, 2, figsize = (12, 4))
    sns.heatmap(young, cmap = "Reds", ax = axs[0], xticklabels=False, yticklabels=False)
    axs[0].set_ylabel('') 
    axs[0].set_title('Young fibroblasts (16 weeks)')

    sns.heatmap(old, cmap = "Reds", ax = axs[1], xticklabels=False, yticklabels=False)
    axs[1].set_ylabel('') 
    axs[1].set_title('Old fibroblasts (53 years)')

    
def plot_clustered_anno(df, selected_loci):
    '''
    Plots the binarized map with hierarchical clustering and color annotation for DE genes 
    Args:
        df: (pd DataFrame) binarized HiC map
        selected_loci: (pd DataFrame) annotation for loci, genes, DE
    Returns:
        Clustered heatmap
    '''
    # color annotation bar
    color_dict = {}
    palette = sns.color_palette()
    for loc in df.columns:
        if loc in selected_loci[selected_loci['DE'] == 'True']['locus'].tolist():
            color_dict[loc] = 'blue'
        else:
            color_dict[loc] = 'whitesmoke'
    color_rows = pd.Series(color_dict)

    # Plot clustered map
    plt.figure()
    p = sns.clustermap(df,
                   method='ward',
                   metric='euclidean',
                   row_cluster=True, col_cluster=True,
                   figsize=(4,4),
                   xticklabels=False, yticklabels=False,
                   cmap='Reds', cbar_pos=(1, 0.5, 0.01, .4),
                   dendrogram_ratio=(.1, .1),
                   row_colors=[color_rows], col_colors=[color_rows])
    
    # add legend
    handles = [Patch(facecolor='blue'), Patch(facecolor='whitesmoke')]
    plt.legend(handles, ['True', 'False'], title='DE gene',
               bbox_transform=plt.gcf().transFigure, bbox_to_anchor=(1., 0.3, 1., .102), loc='lower left')
        
    
    #return(p)
    

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
    tf_dir = config['TF_DIR']
    genome_dir = config['GENOME_DIR']
    resol_str = config['HIC_RESOLUTION_STR']
    resol = config['HIC_RESOLUTION']
    chr_list = config['chrs']
    cell_type = config['CELL_TYPE']
    subset = config['SUBSET']
    threshold = config['LAS_THRESHOLD']
    
    print("Cell type: ", cell_type)
    
    # Create binarized map for selected loci
    print("Binarize hic data for selected loci:")
    selected_loci = pd.read_csv(tf_dir + subset + '_targets_loci.csv')['locus'].unique()
    hic_binarized = get_binarized_hic(selected_loci, chr_list, hic_dir, cell_type, resol, threshold)
    
    # Convert from long to wide format and save results
    hic_wide = long_to_wide(hic_binarized)   
    hic_wide = hic_wide.loc[selected_loci, selected_loci]
    hic_wide.to_csv(hic_dir + 'processed_hic_data_' + cell_type + '/binarized_maps/' + 
                    subset + '_subset.csv')
    
    # Create binarized map for random loci (same amount per chromosome) as a comparison
    print("Binarize hic data for random loci:")
    all_gene_loci = pd.DataFrame({'locus': pd.read_csv(genome_dir+'all_gene_loci.csv')['locus'].unique()})
    all_gene_loci[['chr','chr_number', 'loc', 'loc_number']]=all_gene_loci.locus.str.split('_',expand=True)
    
    # get list of chromosome for the loci set of interest
    chromosome_list = [locus.split("_")[1] for locus in selected_loci]
    random_loci = []
    for chrom in chr_list:
        n = chromosome_list.count(str(chrom))
        loci = all_gene_loci[all_gene_loci['chr_number'] == str(chrom)].sample(n, random_state=2022).sort_index()['locus']
        random_loci.append(loci)
    # flatten list 
    random_loci = list(itertools.chain(*random_loci))

    hic_binarized = get_binarized_hic(random_loci, chr_list, hic_dir, cell_type, resol, threshold)
    hic_wide = long_to_wide(hic_binarized)   
    hic_wide = hic_wide.loc[random_loci, random_loci]
    hic_wide.to_csv(hic_dir + 'processed_hic_data_' + cell_type + '/binarized_maps/' + 
                    subset + '_random.csv')
    

if __name__ == "__main__":
    main()
    
