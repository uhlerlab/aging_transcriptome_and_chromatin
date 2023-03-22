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
import random
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, inconsistent

''' This script runs 100 simulations with TFs not included in the Steiner trees and outputs the percentage of intermingling regions in the targets'''

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

    for pair in chr_pairs:
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


def plot_intermingling_regions(chr_list, hic_dir, cell_type, ax):
    '''
    Plots a heatmap of the number of intermingling regions per chromosome pair
    Args:
        chr_list: (list) list of all chromosomes
        hic_dir: (string) directory of processed hic data
        cell_type: (string) 'IMR90' or 'old_fibroblasts'
        ax: axes to plot the subplot
    Returns:
        Heatmap
    '''
    chr_pairs = list(itertools.combinations(chr_list, 2))
    hic_intermingling = pd.DataFrame({'chr1': [], 'chr2': [], 'value': []})

    for pair in chr_pairs:
        chr1, chr2 = pair

        fname = hic_dir + 'processed_hic_data_' + cell_type + '/LAS/intermingling_regions.chr' + str(chr1) + '_chr' + str(chr2) + '.avg_filt.csv'
        if (os.path.isfile(fname) == True):
            df_intermingling = pd.read_csv(fname, index_col = 0)
            # check if dataframe is empty
            if (len(df_intermingling) > 0):
                intermingling_regions = bin_intermingling_regions_hic_resoln(df_intermingling, chr1, chr2, 250000)
                hic_intermingling = hic_intermingling.append({'chr1': chr1, 'chr2': chr2, 
                                                              'value': intermingling_regions.shape[0]}, ignore_index = True)

    hic_intermingling = long_to_wide(hic_intermingling).fillna(0)
    hic_intermingling.columns = hic_intermingling.columns.astype(int)
    hic_intermingling.index = hic_intermingling.index.astype(int)
    
    sns.heatmap(hic_intermingling, cmap = "Reds", ax = ax)
    ax.set_title(cell_type + ": Number of intermingling regions")
    ax.set_xlabel("")
    ax.set_ylabel("")    
    

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
    tf_dir = config['TF_DIR']
    pcst_dir = config['PCST_DIR']
    genome_dir = config['GENOME_DIR']
    resol_str = config['HIC_RESOLUTION_STR']
    resol = config['HIC_RESOLUTION']
    chr_list = config['chrs']
    subset = config['SUBSET']
    
    threshold_young = 15
    threshold_old = 30
    random.seed(202208)
    
    # Load TFs that are included in the Steiner trees
    incl_TFs = set(pd.read_csv(pcst_dir + 'incl_TFs_design2.csv')['TF'])
    
    # Load TF target interactions with loci annotation of the targets
    tf_targets = pd.read_csv(tf_dir + 'TF_targets_anno.csv', sep = ',')
    # Filter for TFs that have more than 20 targets
    number_of_targets = tf_targets['TF'].value_counts().rename_axis('TF').reset_index(name='n_targets')
    number_of_targets = number_of_targets[number_of_targets['n_targets'] > 100]
    tf_targets = tf_targets[tf_targets['TF'].isin(number_of_targets['TF'])]
    
    # Get non-included TFs
    non_incl_TFs = sorted(list(set(tf_targets['TF']).difference(incl_TFs)))
    print('Number of TFs that are not included in one of the Steiner trees and ',  
          'have more than 100 targets: ', len(non_incl_TFs))
    
    # Randomly select 100 non-included TFs
    non_incl_TFs = random.sample(non_incl_TFs, 100)
    
    # DataFrame to save the results
    intermingling = pd.DataFrame({'young': [], 'old': []})
    differences = pd.DataFrame({'no_intermingling': [], 'young_specific': [], 'old_specific': [], 'shared': []})
    
    # 100 simulations for non-included TFs
    print('Start 100 simulations of randomly selected loci')
    for n_sim in tqdm(range(100)):
        time.sleep(0.01)
        
        # Select loci based on targets of a non-included TF
        print('TF: ', non_incl_TFs[n_sim])
        random_loci = tf_targets.loc[tf_targets['TF'] == non_incl_TFs[n_sim], 'locus'].tolist()
        print('number of target loci: ', len(random_loci))
        
        # Binarize map for young and old fibroblasts
        hic_IMR90 = get_binarized_hic(random_loci, chr_list, hic_dir, 'IMR90', resol, threshold_young)
        hic_old = get_binarized_hic(random_loci, chr_list, hic_dir, 'old_fibroblasts', resol, threshold_old)
        
        # Calculate percentage of intermingling regions 
        total = hic_IMR90.shape[0]
        row = pd.DataFrame({'young': [hic_IMR90['value'].sum() / total], 
                            'old': [hic_old['value'].sum() / total]}, index = [non_incl_TFs[n_sim]])
        intermingling = pd.concat([intermingling, row])
        
        # Get percentages of differences between young and old
        diff = hic_IMR90['value'] + 2 * hic_old['value']
        row = pd.DataFrame({'no_intermingling': [diff[diff == 0].shape[0] / total],
                            'young_specific': [diff[diff == 1].shape[0] / total],
                            'old_specific': [diff[diff == 2].shape[0] / total],
                            'shared': [diff[diff == 3].shape[0] / total]}, index = [non_incl_TFs[n_sim]])
        differences = pd.concat([differences, row])
        
    
    intermingling.to_csv(hic_dir + 'simulation_intermingling_non_incl_TFs.csv')
    differences.to_csv(hic_dir + 'simulation_diff_non_incl_TFs.csv')
    

if __name__ == "__main__":
    main()
    
