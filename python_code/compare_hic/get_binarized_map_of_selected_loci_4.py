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
from matplotlib.colors import ListedColormap
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


def get_binarized_hic(selected_loci, chr_list, hic_dir, cell_type, resol, inter_threshold, intra_threshold):
    '''
    Constructs dataframe in long format of binarized map for selected loci
    Args:
        selected_loci: (list) list of loci with entries like 'chr_1_loc_155250000'
        chr_list: (list) list of all chromosomes
        hic_dir: (string) directory of processed hic data
        cell_type: (string) 'IMR90' or 'old_fibroblasts'
        resol: (string) resolution of the HiC maps
        inter_threshold: (int) LAS threshold for interchromosomal intermingling
        intra_threshold: (int) LAS threshold for intrachromosomal intermingling
    Returns:
        A pandas dataframe with columns chr1, chr2, value
    '''
    
    # create df with all combinations of selected loci
    combos = list(itertools.combinations_with_replacement(selected_loci, 2))
    hic_binarized = pd.DataFrame(combos, columns=['chr1', 'chr2'])
    print(hic_binarized.head())

    # Interchromosomal interactions (different chromosomes)
    chr_pairs = list(itertools.combinations(chr_list, 2))
    hic_selection = pd.DataFrame({'chr1': [], 'chr2': [], 'value': []})
    print('Load interchromosomal interactions')
    for pair in tqdm(chr_pairs):
        time.sleep(.01)
        chr1, chr2 = pair
        
        # add value based on whether the loci combination is part of a large average submatrix
        fname = hic_dir + cell_type + '/LAS-'+ str(inter_threshold) +'/intermingling_regions.chr' + str(chr1) + '_chr' + str(chr2) + '.avg_filt.csv'
        if (os.path.isfile(fname) == True):
            df_intermingling = pd.read_csv(fname, index_col = 0)
            # check if dataframe is empty
            if (len(df_intermingling) > 0):
                df_intermingling_binned = bin_intermingling_regions_hic_resoln(df_intermingling, chr1, chr2, resol)
                hic_selection = pd.concat([hic_selection, df_intermingling_binned])
                
    # Intrachromosomal interactions
    print('Load intrachromosomal interactions')
    for chr0 in tqdm(chr_list):
        time.sleep(.01)
        
        # add value based on whether the loci combination is part of a large average submatrix
        fname = hic_dir + cell_type + '/LAS_intra-'+ str(intra_threshold) +'/intermingling_regions.chr' + str(chr0) + '_chr' + str(chr0) + '.avg_filt.csv'
        if (os.path.isfile(fname) == True):
            df_intermingling = pd.read_csv(fname, index_col = 0)
            # check if dataframe is empty
            if (len(df_intermingling) > 0):
                df_intermingling_binned = bin_intermingling_regions_hic_resoln(df_intermingling, chr0, chr0, resol)
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
    df = pd.concat([df, inverted_df]).drop_duplicates()

    # from long to wide format
    df_wide = df.pivot_table(index='chr1', columns='chr2', values='value', aggfunc = "sum")
    
    return(df_wide)


#def plot_intermingling_regions(chr_list, hic_dir, cell_type, count, threshold, ax):
#    '''
#    Plots a heatmap of the number of intermingling regions per chromosome pair
#    Args:
#        chr_list: (list) list of all chromosomes
#        hic_dir: (string) directory of processed hic data
#        cell_type: (string) specifies the sample name
#        count: (string) 'LAS' or 'intermingling_regions' (depending on whether to visualize the number of submatrices or the number of pixels in the submatrices)
#        threshold: (int) LAS threshold
#        ax: axes to plot the subplot
#    Returns:
#        Heatmap
#    '''
#    # get numbers of intermingling regions
#    hic_intermingling = LAS_statistics(chr_list, hic_dir, cell_type, threshold)
#    hic_intermingling = hic_intermingling[['chr1', 'chr2', count]]
#    hic_intermingling.columns = ['chr1', 'chr2', 'value']
#    
#    # conversion to wide format
#    hic_intermingling = long_to_wide(hic_intermingling).fillna(0)
#    hic_intermingling.columns = hic_intermingling.columns.astype(int)
#    hic_intermingling.index = hic_intermingling.index.astype(int)
#    
#    # plot heatmap
#    if count == "intermingling_regions":
#        sns.heatmap(hic_intermingling, cmap = "Reds", ax = ax, vmin = 0, vmax = 30000)
#        ax.set_title(cell_type + ": Number of intermingling regions")
#    elif count == "LAS":
#        sns.heatmap(hic_intermingling, cmap = "Reds", ax = ax, vmin = 0, vmax = 80)
#        ax.set_title(cell_type + ": Number of submatrices")
#    ax.set_xlabel("")
#    ax.set_ylabel("") 
    
    
def plot_specific_intermingling(chr_list, diff_map, cell_type, ax):
    '''
    Plots a heatmap of the number of cell-type-specific intermingling regions per chromosome pair
    Args:
        chr_list: (list) list of all chromosomes
        diff_map: (pd DataFrame) difference map of the intermingling regions
        cell_type: (string) specifies the sample name (young, old or RJ)
        ax: axes to plot the subplot
    Returns:
        Heatmap
    '''
    # get counts of intermingling per chromosome pair for specified cell type
    counts = get_intermingling_counts(diff_map, cell_type, chr_list)

    # from long to wide format
    counts_wide = counts.pivot_table(index='chr1', columns='chr2', values='count').fillna(0)
    counts_wide.columns = counts_wide.columns.astype(int)
    counts_wide.index = counts_wide.index.astype(int)
    counts_wide = counts_wide.loc[chr_list, chr_list]
    
    # plot heatmap
    sns.heatmap(counts_wide, cmap = "Reds", ax = ax, vmin = 0, vmax = 1000)
    ax.set_title(cell_type + ": Number of intermingling regions")
    ax.set_xlabel("Chromosomes")
    ax.set_ylabel("Chromosomes") 

    
def plot_intermingling_diff(chr_list, diff_map, sample1, sample2, ax, vmin, vmax):
    '''
    Plots a heatmap of the difference in the number of intermingling regions per chromosome pair in two samples
    Args:
        chr_list: (list) list of all chromosomes
        diff_map: (pd DataFrame) difference map of the intermingling regions
        sample1: (string) first sample name
        sample2: (string) second sample name
        ax: axes to plot the subplot
        vmin: (int) min value for heatmap color
        vmax: (int) max value for heatmap color
    Returns:
        Heatmap
    '''
    # get numbers of intermingling for the two cell types
    intermingling_s1 = get_intermingling_counts(diff_map, sample1, chr_list)
    intermingling_s1.columns = [sample1, 'chr1', 'chr2']
    intermingling_s2 = get_intermingling_counts(diff_map, sample2, chr_list)
    intermingling_s2.columns = [sample2, 'chr1', 'chr2']
    
    # calculate differences
    diff_intermingling = intermingling_s1.merge(intermingling_s2, on = ['chr1', 'chr2'])
    diff_intermingling['diff'] = diff_intermingling[sample1] - diff_intermingling[sample2]
    diff_intermingling = diff_intermingling[['chr1', 'chr2', 'diff']]
    
    # from long to wide format
    diff_wide = diff_intermingling.pivot_table(index='chr1', columns='chr2', values='diff').fillna(0)
    diff_wide.columns = diff_wide.columns.astype(int)
    diff_wide.index = diff_wide.index.astype(int)
    diff_wide = diff_wide.loc[chr_list, chr_list]
    
    # plot heatmap
    color_df = pd.DataFrame({'Young': [240], 'Old': [28], 'RJ': [120]})
    cmap = sns.diverging_palette(color_df[sample2], color_df[sample1], s = 90, l = 60, as_cmap=True)
    #cmap = "RdBu"
    sns.heatmap(diff_wide, cmap = cmap, ax = ax, vmin = vmin, vmax = vmax)
    ax.set_title("Differences for " + sample1 + " - " + sample2)
    ax.set_xlabel("Chromosomes")
    ax.set_ylabel("Chromosomes") 

    
#def plot_intermingling_diff(chr_list, hic_dir, sample1, sample2, count, threshold, ax, vmin, vmax):
#    '''
#    Plots a heatmap of the difference in the number of intermingling regions per chromosome pair in two samples
#    Args:
#       chr_list: (list) list of all chromosomes
#        hic_dir: (string) directory of processed hic data
#        sample1: (string) first sample name
#        sample2: (string) second sample name
#        count: (string) 'LAS' or 'intermingling_regions' (depending on whether to visualize the number of submatrices or the number of pixels in the submatrices)
#        threshold: (int) LAS threshold
#        ax: axes to plot the subplot
#        vmin: (int) min value for heatmap color
#        vmax: (int) max value for heatmap color
#    Returns:
#        Heatmap
#    '''
#    # get numbers of intermingling regions for young and old
#    intermingling_s1 = LAS_statistics(chr_list, hic_dir, sample1, 20)
#    intermingling_s1 = intermingling_s1[['chr1', 'chr2', count]]
#    intermingling_s1.columns = ['chr1', 'chr2', sample1]
#
#    intermingling_s2 = LAS_statistics(chr_list, hic_dir, sample2, 20)
#    intermingling_s2 = intermingling_s2[['chr1', 'chr2', count]]
#    intermingling_s2.columns = ['chr1', 'chr2', sample2]
#
#    # calculate differences
#    diff_intermingling = intermingling_s1.merge(intermingling_s2, on = ['chr1', 'chr2'])
#    diff_intermingling['diff'] = diff_intermingling[sample1] - diff_intermingling[sample2]
#    diff_intermingling = diff_intermingling[['chr1', 'chr2', 'diff']]
#    diff_intermingling.columns = ['chr1', 'chr2', 'value']
#
#    # conversion to wide format
#    diff_intermingling = long_to_wide(diff_intermingling).fillna(0)
#    diff_intermingling.columns = diff_intermingling.columns.astype(int)
#    diff_intermingling.index = diff_intermingling.index.astype(int)
#    
#    sample1 = sample1.split("_")[0]
#    sample2 = sample2.split("_")[0]
#    
#    sns.heatmap(diff_intermingling, cmap = "RdBu", vmin = vmin, vmax = vmax, ax = ax)
#    ax.set_title("Differences for " + sample1 + " - " + sample2) 
#    ax.set_xlabel("")
#    ax.set_ylabel("")

    
def get_intermingling_counts(diff_map, cell_type, chr_list):
    '''
    Returns dataframe with number of cell-type specific intermingling regions per chromosome
    Args:
        diff_map: (pd DataFrame) difference map of the intermingling regions
        cell_type: (string) specifies the sample name (young, old)
        chr_list: (list) list of all chromosomes
    Returns:
        pd DataFrame
    '''
    # difference matrix to long format
    diff_map['loc1'] = diff_map.index
    specific_interactions_long = diff_map.melt(id_vars = 'loc1', var_name = 'loc2')
    
    # filter for cell-type specific intermingling
    if cell_type == "Young":
        specific_interactions_long = specific_interactions_long[specific_interactions_long['value'] == 1]
    elif cell_type == "Old":
        specific_interactions_long = specific_interactions_long[specific_interactions_long['value'] == 2]
    elif cell_type == "Shared":
        specific_interactions_long = specific_interactions_long[specific_interactions_long['value'] == 3]
    else:
        print('Error: Cell type has to be Young, Old or Shared')
    
    # add chromosomes as columns
    specific_interactions_long['chr1'] = [loc.split("_")[1] for loc in specific_interactions_long['loc1']]
    specific_interactions_long['chr2'] = [loc.split("_")[1] for loc in specific_interactions_long['loc2']]
    specific_interactions_long['chr-pair'] = specific_interactions_long['chr1'] + "_" + specific_interactions_long['chr2']

    # group by chr-pair and get number of rows per pair
    counts = pd.DataFrame({'count': specific_interactions_long.groupby(['chr-pair'])['chr-pair'].count()})
    counts['chr1'] = [loc.split("_")[0] for loc in counts.index]
    counts['chr2'] = [loc.split("_")[1] for loc in counts.index]
    
    # add zero-counts for missing chr-pairs
    chr_pairs = list(itertools.permutations(chr_list, 2))
    chr_pairs = [str(chr1) + "_" + str(chr2) for chr1, chr2 in chr_pairs]
    missing_chr_pairs = set(chr_pairs).difference(set(counts.index))
    for chr_pair in missing_chr_pairs:
        counts.loc[chr_pair] = [0, chr_pair.split("_")[0], chr_pair.split("_")[1]]
    
    return counts
    
    
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

        fname = hic_dir  + cell_type + '/LAS-' + str(threshold) + '/intermingling_regions.chr' + str(chr1) + '_chr' + str(chr2) + '.avg_filt.csv'
        
        if (os.path.isfile(fname) == True):
            df_intermingling = pd.read_csv(fname, index_col = 0)
            # check if dataframe is empty
            if (len(df_intermingling) > 0):
                intermingling_regions = bin_intermingling_regions_hic_resoln(df_intermingling, chr1, chr2, 250000)
                
                hic_intermingling = hic_intermingling.append({'chr1': chr1, 'chr2': chr2, 
                                                              'intermingling_regions': intermingling_regions.shape[0], 
                                                              'LAS': len(df_intermingling)}, ignore_index = True)
    return hic_intermingling
    
    
def plot_binarized_maps(young, old, diff_map, filtering = False):
    '''
    Plots the binarized map for young and old fibroblasts next to each other
    Args:
        young: (pd DataFrame) binarized matrix with young data
        old: (pd DataFrame) binarized matrix with old data
        diff_map: (pd DataFrame) diff_map = young + 2 * old
        filtering: (boolean) indication of whether to plot only bins that intermingle with at least one other bin
    Returns:
        Three heatmaps
    '''
    if filtering: 
        young  = young.loc[(young!=0).any(axis=1), (young!=0).any(axis=1)]
        old  = old.loc[(old!=0).any(axis=1), (old!=0).any(axis=1)]
        diff_map  = diff_map.loc[(diff_map!=0).any(axis=1), (diff_map!=0).any(axis=1)]
    
    # get indices for chromosome 17 and 19 loci
    chr17_loci = [locus for locus in young.columns if locus.split("_")[1] == '17']
    chr17_start = young.columns.get_loc(chr17_loci[0])
    chr17_end = young.columns.get_loc(chr17_loci[-1])
    chr19_loci = [locus for locus in young.columns if locus.split("_")[1] == '19']
    chr19_start = young.columns.get_loc(chr19_loci[0])
    chr19_end = young.columns.get_loc(chr19_loci[-1])
    
    fig, axs = plt.subplots(1, 3, figsize = (16, 4))
    sns.heatmap(young, cmap = "Reds", ax = axs[0], xticklabels=False, yticklabels=False)
    axs[0].set_ylabel('') 
    axs[0].set_title('Young')
    # add lines defining the chr 17 and 19 region
    axs[0].plot([chr19_start, chr19_start], [chr17_start, chr17_end], 'k-', lw = 2)
    axs[0].plot([chr19_end, chr19_end], [chr17_start, chr17_end], 'k-', lw = 2)
    axs[0].plot([chr19_start, chr19_end], [chr17_start, chr17_start], 'k-', lw = 2)
    axs[0].plot([chr19_start, chr19_end], [chr17_end, chr17_end], 'k-', lw = 2)

    sns.heatmap(old, cmap = "Reds", ax = axs[1], xticklabels=False, yticklabels=False)
    axs[1].set_ylabel('') 
    axs[1].set_title('Old')
    # add lines defining the chr 17 and 19 region
    axs[1].plot([chr19_start, chr19_start], [chr17_start, chr17_end], 'k-', lw = 2)
    axs[1].plot([chr19_end, chr19_end], [chr17_start, chr17_end], 'k-', lw = 2)
    axs[1].plot([chr19_start, chr19_end], [chr17_start, chr17_start], 'k-', lw = 2)
    axs[1].plot([chr19_start, chr19_end], [chr17_end, chr17_end], 'k-', lw = 2)
    
    # define colors
    palette = sns.color_palette("tab10")
    colors = ['whitesmoke', 'blue', 'magenta', 'lightgrey']
    cmap = ListedColormap(colors)

    sns.heatmap(diff_map, cmap = cmap, ax = axs[2], xticklabels=False, yticklabels=False)
    axs[2].set_ylabel('') 
    axs[2].set_title('Difference map')
    # add lines defining the chr 17 and 19 region
    axs[2].plot([chr19_start, chr19_start], [chr17_start, chr17_end], 'k-', lw = 2)
    axs[2].plot([chr19_end, chr19_end], [chr17_start, chr17_end], 'k-', lw = 2)
    axs[2].plot([chr19_start, chr19_end], [chr17_start, chr17_start], 'k-', lw = 2)
    axs[2].plot([chr19_start, chr19_end], [chr17_end, chr17_end], 'k-', lw = 2)
    
    
    
def plot_binarized_maps_chr_pair(young, old, diff_map, chr_pair):
    '''
    Plots the binarized map for young and old fibroblasts for a selected chromosome pair
    Args:
        young: (pd DataFrame) binarized matrix with young data
        old: (pd DataFrame) binarized matrix with old data
        diff_map: (pd DataFrame) young + 2 * old
        chr_pair: (tuple) tuple containing number of two chromosomes
    Returns:
        Three heatmaps
    '''
    chr1, chr2 = chr_pair
    # diff map for one chr pair
    chr1_loci = [locus for locus in diff_map.columns if locus.split('_')[1] == str(chr1)]
    chr2_loci = [locus for locus in diff_map.columns if locus.split('_')[1] == str(chr2)]
    
    fig, axs = plt.subplots(1, 3, figsize = (16, 4))
    sns.heatmap(young.loc[chr1_loci, chr2_loci], cmap = "Reds", ax = axs[0], xticklabels=False, yticklabels=False)
    axs[0].set_ylabel('') 
    axs[0].set_title('Young chr ' + str(chr1) + ' and ' + str(chr2))

    sns.heatmap(old.loc[chr1_loci, chr2_loci], cmap = "Reds", ax = axs[1], xticklabels=False, yticklabels=False)
    axs[1].set_ylabel('') 
    axs[1].set_title('Old chr ' + str(chr1) + ' and ' + str(chr2))
    
    # define colors
    palette = sns.color_palette("tab10")
    colors = ['whitesmoke', 'blue', 'magenta', 'lightgrey']
    cmap = ListedColormap(colors)

    sns.heatmap(diff_map.loc[chr1_loci, chr2_loci], cmap = cmap, ax = axs[2], xticklabels=False, yticklabels=False)
    axs[2].set_ylabel('') 
    axs[2].set_title('Difference map chr ' + str(chr1) + ' and ' + str(chr2))

    
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
    
    
def plot_clustered_diff_map(diff_map, TF, save_dir):
    '''
    Hierarchical clustering of the difference map for the targets of a TF
    Args:
        diff_map: (pd DataFrame) difference map for the targets of a TF
        TF: (string) TF of interest
        save_dir: (string) save directory
    Returns:
        clustered diff_map
    '''
    # define difference colors
    palette = sns.color_palette("tab10")
    colors = ['whitesmoke', 'blue', 'magenta', palette[2]]
    cmap = ListedColormap(colors)

    plt.figure()
    p = sns.clustermap(diff_map,
                   method='complete',
                   metric=color_metric,
                   row_cluster=True, col_cluster=True,
                   figsize=(5,5),
                   xticklabels=False, yticklabels=False,
                   cmap=cmap, cbar_pos=(1, 0.5, 0.01, .4),
                   vmin=0, vmax=3,
                   dendrogram_ratio=(.1, .1))


def plot_replicate_scores(sample, threshold, hic_dir, ax):
    '''
    Plots the score of a submatrix in one replicate vs its score in the other 
    Args:
        sample: (string) name of the cell type sample
        threshold: (int) LAS threshold
        hic_dir: (string) directory of processed hic data
        ax: axes to plot the subplot
    Returns:
        Scatterplot 
    '''
    fname = hic_dir + sample + '/LAS-'+ str(threshold) + '/intermingling_regions_all_chr_unfiltered.csv'
    all_submatrices = pd.read_csv(fname)

    ax.scatter(all_submatrices['score_r1'], all_submatrices['score_r2'])
    ax.axhline(y = threshold, color='r', linestyle='-')
    ax.axvline(x = threshold, color='r', linestyle='-')
    ax.set_xlim([0, 80])
    ax.set_ylim([0, 80])
    ax.set_title(sample)
    ax.set_xlabel("Submatrix scores in replicate 1")
    ax.set_ylabel("Submatrix scores in replicate 2")


def get_TF_intermingling(TFs, tf_dir, hic_dir, threshold_young, threshold_old):
    '''
    Generates dfs with the percentages of intermingling per cell type as well as the percentages of interaction types 
    (no, young specific, old specific, shared intermingling) between the targets of selected TFs
    Args:
        TFs: (list) list of TFs
        tf_dir: (string) directory of the TF_target file
        hic_dir: (string) hic directory 
        threshold_young: LAS threshold for young HiC map
        threshold_old: LAS threshold for old HiC map
    Returns:
        two pd DataFrames with the percentages of intermingling and interaction types between TF targets
    '''
    chr_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    resol = 250000
    
    # Load TF target interactions with loci annotation of the targets
    tf_targets = pd.read_csv(tf_dir + 'TF_targets_anno.csv', sep = ',')
    
    # DataFrame to save the results
    intermingling = pd.DataFrame({'young': [], 'old': []})
    differences = pd.DataFrame({'no_intermingling': [], 'young_specific': [], 'old_specific': [], 'shared': []})
    
    for TF in TFs: 
        selected_loci = tf_targets.loc[tf_targets['TF'] == TF, 'locus'].tolist()
        
        # Binarize map for young and old fibroblasts
        hic_IMR90 = get_binarized_hic(selected_loci, chr_list, hic_dir, 'IMR90', resol, threshold_young)
        hic_old = get_binarized_hic(selected_loci, chr_list, hic_dir, 'old_fibroblasts', resol, threshold_old)
        
        # Calculate percentage of intermingling regions 
        total = hic_IMR90.shape[0]
        row = pd.DataFrame({'young': [hic_IMR90['value'].sum() / total], 
                            'old': [hic_old['value'].sum() / total]}, index = [TF])
        intermingling = pd.concat([intermingling, row])
        
        # Get percentages of differences between young and old
        diff = hic_IMR90['value'] + 2 * hic_old['value']
        row = pd.DataFrame({'no_intermingling': [diff[diff == 0].shape[0] / total],
                            'young_specific': [diff[diff == 1].shape[0] / total],
                            'old_specific': [diff[diff == 2].shape[0] / total],
                            'shared': [diff[diff == 3].shape[0] / total]}, index = [TF])
        differences = pd.concat([differences, row])
    return intermingling, differences


def color_dist(l1, l2): 
    '''
    Calculates the distance between two interaction types (0: no intermingling, 1: only intermingles in first matrix, 
    2: only intermingles in second one, 3: intermingling in both) for a difference matrix comparing two HiC maps
    Args:
        l1: (float) type of interaction for a loci pair
        l2: (float) type of interaction for a loci pair
    Returns:
        integer that is the difference between the interactions
    '''
    if l1 == l2:
        diff = 0 
    elif (l1 == 0) & (l2 == 3):
        diff = 2
    elif (l1 == 3) & (l2 == 0):
        diff = 2
    elif (l1 == 1) & (l2 == 2):
        diff = 2
    elif (l1 == 2) & (l2 == 1):
        diff = 2
    else: 
        diff = 1
    return diff


def color_metric(r1, r2): 
    '''
    Calculates the distance between two row vectors for a difference matrix comparing two HiC maps
    Args:
        r1: (pd Series) row vector 1
        r2: (pd Series) row vector 2
    Returns:
        integer that is the mean difference between the two row vectors
    '''
    dist = [color_dist(r1[ix], r2[ix]) for ix in range(len(r1))]
    return np.mean(dist)


def get_blacklist_all_samples(hic_dir): 
    '''
    Creates a list of all blacklisted loci (centromeres, repeats, outlier in one of the samples)
    Args:
        hic_dir: (str) hic directory
    Returns:
        list of all blacklisted loci
    '''
    blacklist1 = hic_dir + 'Young_B1R1/final_BP250000_intraKR_interINTERKR/blacklist_Young_B1R1_INTERKR.pickle'
    blacklist2 = hic_dir + 'Young_B1R2/final_BP250000_intraKR_interINTERKR/blacklist_Young_B1R2_INTERKR.pickle'
    blacklist3 = hic_dir + 'Old_B1R1/final_BP250000_intraKR_interINTERKR/blacklist_Old_B1R1_INTERKR.pickle'
    blacklist4 = hic_dir + 'Old_B2R2/final_BP250000_intraKR_interINTERKR/blacklist_Old_B2R2_INTERKR.pickle'
    
    blacklist = [pd.read_pickle(blacklist1).tolist(), pd.read_pickle(blacklist2).tolist(),
                pd.read_pickle(blacklist3).tolist(), pd.read_pickle(blacklist4).tolist()]
    blacklist = set.union(*map(set,blacklist))
    return blacklist


def get_GO_intermingling_percentages(diff_map, GO_term, pathways, norm):
    '''
    Calculates the percentages of cell-state specific intermingling in a GO subset of the difference matrix
    Args:
        diff_map: (pd DataFrame) difference matrix with 1 = young-specific intermingling, 2 = old-specific intermingling 
                    and 3 = RJ specific intermingling
        GO_term: (string) GO pathway of interest
        pathways: (pd DataFrame) df that includes all genes and loci belonging to a GO pathway
        norm: (string) "all": normalization by dividing through total pixel number or
                       "intermingling": normalization by dividing through total number of intermingling pixels
    Returns:
        pd DataFrame with percentages of cell-state specific intermingling
    '''
    # Subset difference map to loci belonging to the pathway of interest
    if GO_term == "All loci":
        subset_diff_map = diff_map
    else:
        loci = set(pathways.loc[pathways['pathway'] == GO_term, 'locus'])
        subset_diff_map = diff_map.loc[diff_map.index.isin(loci), diff_map.index.isin(loci)]
    
    print(GO_term, ": ", subset_diff_map.shape)
    
    # Calculate number of intermingling regions 
    young = (subset_diff_map == 1).sum(axis=1).sum()
    old = (subset_diff_map == 2).sum(axis=1).sum()
    RJ = (subset_diff_map == 3).sum(axis=1).sum()
    if norm == 'intermingling':
        total = young + old + RJ
    elif norm == 'all':
        total = subset_diff_map.shape[0] * subset_diff_map.shape[0]
    else:
        print("norm parameter must be either intermingling or all")
    
    results = pd.DataFrame({'pathway': GO_term, 'cell_state': ['young', 'old', 'RJ'],
                            'percentages': [young/total, old/total, RJ/total]})
    return results


def diff_map_updown(diff_map, gene_order, transition, de_dir):
    """ Creates diff map sorting first all up- and then downregulated genes
    
    Args:
        diff_map: (pd DataFrame) difference matrix with 1 = young-specific intermingling, 2 = old-specific intermingling 
                    and 3 = RJ specific intermingling
        gene_order: (list) list of genes ordered according to their chromosome positions
        transition: (string) name of the transition
        de_dir: (string) path to the directory where the DE genes are saved
    
    Returns:
        Sorted diff map
    """
    # Load differentially expressed genes
    DE_genes = pd.read_csv(de_dir+'DE_updown.csv')
    DE_genes = DE_genes[DE_genes['transition'] == transition]
    DE_genes = DE_genes[DE_genes['updown'] == "up"]

    # Sort DE genes: first up, then downregulated ones
    order = [gene for gene in gene_order if gene in list(DE_genes['gene'])]
    up_len = len(order)
    order = order + [gene for gene in gene_order if gene not in order]
    
    # Colors
    palette = sns.color_palette("tab10")
    colors = ['whitesmoke', 'blue', 'magenta', 'lightgrey']
    cmap = ListedColormap(colors)

    g = sns.heatmap(diff_map.loc[order, order], cmap = cmap, xticklabels=False, yticklabels=False)
    # add lines defining the upregulated region
    plt.plot([up_len, up_len], [0, len(order)], 'k-', lw = 0.8)
    plt.plot([0, len(order)], [up_len, up_len], 'k-', lw = 0.8)
    plt.ylabel('') 
    return(g)


def get_diff_map_DE(diff_map, group, updown, data_dir, save_dir):
    """ Creates diff map with all up or downregulated DE genes of one group
    
    Args:
        diff_map: (pd DataFrame) diff_map containing all loci
        group: (string) name of the group ('Group1' or 'Group4_5')
        updown: (string) select 'up' or 'down' regulated genes
        data_dir: (string) path of the data directory
        save_dir: (string) path of the save directory
    
    Returns:
        Sorted diff map with DE genes
    """
    # load DE genes
    DE_genes = pd.read_csv(data_dir+'de_data/DE_' + group + '.csv')
    # filter for up- or downregulated genes genes
    #DE_genes = DE_genes[DE_genes['updown'] == updown]
    print(DE_genes.shape)

    # load loci of all DE targets
    all_gene_loci = pd.read_csv(data_dir+'genome_data/all_gene_loci.csv')
    DE_genes = DE_genes.merge(all_gene_loci, on = "gene")
    DE_genes = DE_genes[['gene', 'locus']]
    
    # save order of the genes according the corresponding loci
    locus_order = [loc for loc in diff_map.columns if loc in DE_genes['locus'].unique()]
    DE_genes = DE_genes.set_index('locus', drop = False)
    DE_genes = DE_genes.loc[locus_order, :]
    gene_order = DE_genes['gene']
    
    # create diff_map for DE targets
    diff_map_long = diff_map.copy()
    diff_map_long['loc1'] = diff_map_long.index
    diff_map_long = pd.melt(diff_map_long, id_vars = 'loc1', var_name = 'loc2')
    diff_map_long = diff_map_long[diff_map_long['loc1'].isin(DE_genes['locus']) & diff_map_long['loc2'].isin(DE_genes['locus'])]
    diff_map_long = DE_genes.rename(columns={"gene": "gene1", "locus": "loc1"}).merge(diff_map_long, on = 'loc1')
    diff_map_long = DE_genes.rename(columns={"gene": "gene2", "locus": "loc2"}).merge(diff_map_long, on = 'loc2')
    diff_map_long = diff_map_long[['gene1', 'gene2', 'value']]
    DE_diff_map = diff_map_long.pivot_table(index='gene1', columns='gene2', values='value')

    # sorting according to chromosomes
    DE_diff_map = DE_diff_map.loc[gene_order, gene_order]
    return(DE_diff_map)


def diff_map_DE(diff_map, group, updown, data_dir, save_dir):
    """ Plots diff map with all up- or downregulated DE genes of one group
    
    Args:
        diff_map: (pd DataFrame) diff_map containing all loci
        group: (string) name of the group ('Group1' or 'Group4_5')
        updown: (string) select 'up' or 'down' regulated genes
        data_dir: (string) path of the data directory
        save_dir: (string) path of the save directory
    
    Returns:
        Sorted diff map with DE genes
    """
    DE_diff_map = get_diff_map_DE(diff_map, group, updown, data_dir, save_dir)
    gene_order = DE_diff_map.columns
    
    # ordering: first genes with cell-state specific intermingling, then the ones without
    spec_im_genes = DE_diff_map.index[DE_diff_map.isin([1,2]).any(axis=1)].tolist()
    order = [gene for gene in gene_order if gene in spec_im_genes]
    im_len = len(order)
    order = order + [gene for gene in gene_order if gene not in order]
    DE_diff_map = DE_diff_map.loc[order, order]
    print(DE_diff_map.shape)

    palette = sns.color_palette("tab10")
    colors = ['whitesmoke', 'blue', 'magenta', 'lightgrey']
    cmap = ListedColormap(colors)
    
    plt.figure(figsize = (6,5))
    g = sns.heatmap(DE_diff_map, cmap = cmap, xticklabels=False, yticklabels=False)
    plt.ylabel('') 
    plt.xlabel('')
    # add lines defining the specific intermingling region
    plt.plot([im_len, im_len], [0, len(order)], 'k-', lw = 0.8)
    plt.plot([0, len(order)], [im_len, im_len], 'k-', lw = 0.8)
    plt.title('Intermingling of ' + updown + ' DE genes in ' + group)

    
def get_diff_map_non_DE(diff_map, group, updown, data_dir, save_dir, seed):
    """ Creates diff map with non DE genes with same numbers of genes per chromosome as in a group
    
    Args:
        diff_map: (pd DataFrame) diff_map containing all loci
        group: (string) name of the group ('Group1' or 'Group4_5')
        updown: (string) select 'up' or 'down' regulated genes
        data_dir: (string) path of the data directory
        save_dir: (string) path of the save directory
        seed: (int) seed used for random sampling
    
    Returns:
        Sorted diff map with non DE genes (and number of genes with cell state specific intermingling)
    """
        
    # load all gene loci and remove loci of DE genes
    all_gene_loci = pd.read_csv(data_dir+'genome_data/all_gene_loci.csv')
    all_gene_loci = all_gene_loci[all_gene_loci['locus'].isin(diff_map.columns)]
    DE_genes = pd.read_csv(data_dir+'de_data/DE_' + group + '.csv')
    DE_genes = DE_genes[DE_genes['updown'] == updown]
    DE_genes = DE_genes.merge(all_gene_loci, on = "gene")
    all_gene_loci = all_gene_loci[~all_gene_loci['locus'].isin(DE_genes['locus'])]
    all_gene_loci[['chr','chr_number', 'loc', 'loc_number']]=all_gene_loci.locus.str.split('_',expand=True)
  
    # get number of DE genes per chromosome 
    DE_genes = DE_genes[DE_genes['locus'].isin(diff_map.columns)]
    chromosome_list = [locus.split("_")[1] for locus in DE_genes['locus']] 
    
    # Select random non DE genes (same amount per chromosome as in the transition of interest)
    random_genes = pd.DataFrame({'gene': [], 'locus': []})
    for chrom in range(1, 23):
        n = chromosome_list.count(str(chrom))
        genes = all_gene_loci[all_gene_loci['chr_number'] == str(chrom)].sample(n, random_state=seed).sort_index()[['gene', 'locus']]
        random_genes = pd.concat([random_genes, genes])
    gene_order = random_genes['gene']
    
    # create diff_map for non DE genes
    diff_map_long = diff_map.copy()
    diff_map_long['loc1'] = diff_map_long.index
    diff_map_long = pd.melt(diff_map_long, id_vars = 'loc1', var_name = 'loc2')
    diff_map_long = diff_map_long[diff_map_long['loc1'].isin(random_genes['locus']) & diff_map_long['loc2'].isin(random_genes['locus'])]
    diff_map_long = random_genes.rename(columns={"gene": "gene1", "locus": "loc1"}).merge(diff_map_long, on = 'loc1')
    diff_map_long = random_genes.rename(columns={"gene": "gene2", "locus": "loc2"}).merge(diff_map_long, on = 'loc2')
    diff_map_long = diff_map_long[['gene1', 'gene2', 'value']]
    diff_map_non_DE = diff_map_long.pivot_table(index='gene1', columns='gene2', values='value')
    # sorting according to chromosomes
    diff_map_non_DE = diff_map_non_DE.loc[gene_order, gene_order]
 
    # ordering: first genes with cell-state specific intermingling, then the ones without
    spec_im_genes = diff_map_non_DE.index[diff_map_non_DE.isin([1,2]).any(axis=1)].tolist()
    order = [gene for gene in gene_order if gene in spec_im_genes]
    im_len = len(order)
    order = order + [gene for gene in gene_order if gene not in order]
    diff_map_non_DE = diff_map_non_DE.loc[order, order]
    return(diff_map_non_DE, im_len)


def quantify_im_changes(diff_map, group, updown, data_dir, save_dir, young_specific_im, old_specific_im, shared_im):
    """ Creates three histograms with the null distribution of the intermingling types in random non DE genes
    
    Args:
        diff_map: (pd DataFrame) diff_map containing all loci
        group: (string) name of the group ('Group1' or 'Group4_5')
        updown: (string) select 'up' or 'down' regulated genes
        data_dir: (string) path of the data directory
        save_dir: (string) path of the save directory
    
    Returns:
        Histograms with null distribution and actual value of specific intermingling and p-value
    """
    null_dist = pd.DataFrame({'young_im': [], 'old_im': [], 'shared_im': []})
    for sim in tqdm(range(1000)):
        time.sleep(0.01)
        diff_non_DE, im_length = get_diff_map_non_DE(diff_map, group, updown, data_dir, save_dir, 202302+sim)
        y_im = np.count_nonzero(diff_non_DE.to_numpy() == 1)
        o_im = np.count_nonzero(diff_non_DE.to_numpy() == 2)
        s_im = np.count_nonzero(diff_non_DE.to_numpy() == 3)
        null_dist = pd.concat([null_dist, pd.DataFrame({'young_im': [y_im], 
                                                       'old_im': [o_im],
                                                       'shared_im': [s_im]})])
    # Plot histograms
    fig, axs = plt.subplots(1, 3, figsize = (11, 5))
    axs[0].hist(null_dist['young_im'], bins = 20)
    axs[0].axvline(young_specific_im, color = 'red')
    axs[0].set_title('Young-specific intermingling') 
    axs[0].set_xlabel('Number of young-specific \n intermingling pixels')
    axs[0].set_ylabel('Number of simulations')
    p_young = len([y_im for y_im in null_dist['young_im'] if young_specific_im < y_im]) / null_dist.shape[0]
    if p_young == 0:
        axs[0].text(.7, .9, 'p < 0.001', transform=axs[0].transAxes)
    else: 
        axs[0].text(.7, .9, 'p={:.2g}'.format(p_young), transform=axs[0].transAxes)

    axs[1].hist(null_dist['old_im'], bins = 20)
    axs[1].axvline(old_specific_im, color = 'red')
    axs[1].set_title('Old-specific intermingling') 
    axs[1].set_xlabel('Number of old-specific \n intermingling pixels')
    #axs[1].set_ylabel('Number of simulations')
    p_old = len([o_im for o_im in null_dist['old_im'] if old_specific_im < o_im]) / null_dist.shape[0]
    if p_old == 0:
        axs[1].text(.7, .9, 'p < 0.001', transform=axs[1].transAxes)
    else: 
        axs[1].text(.7, .9, 'p={:.2g}'.format(p_old), transform=axs[1].transAxes)

    axs[2].hist(null_dist['shared_im'], bins = 20)
    axs[2].axvline(shared_im, color = 'red')
    axs[2].set_title('Shared intermingling') 
    axs[2].set_xlabel('Number of shared \n intermingling pixels')
    #axs[2].set_ylabel('Number of simulations')
    p_shared = len([s_im for s_im in null_dist['shared_im'] if shared_im < s_im]) / null_dist.shape[0]
    if p_shared == 0:
        axs[2].text(.7, .9, 'p < 0.001', transform=axs[2].transAxes)
    else: 
        axs[2].text(.7, .9, 'p={:.2g}'.format(p_shared), transform=axs[2].transAxes)


    
def plot_diff_map_non_DE(diff_map, transition, data_dir, save_dir):
    """ Plots diff map with non DE genes with same numbers of genes per chromosome as in a transition
    
    Args:
        diff_map: (pd DataFrame) diff_map containing all loci
        transition: (string) name of the transition
        data_dir: (string) path of the data directory
        save_dir: (string) path of the save directory
    
    Returns:
        Plot of diff map for non DE genes
    """    
    diff_map_non_DE, im_len = get_diff_map_non_DE(diff_map, transition, data_dir, save_dir, 20230208)
    
    # define colors
    palette = sns.color_palette("tab10")
    colors = ['whitesmoke', 'blue', 'magenta', 'lightgrey']
    cmap = ListedColormap(colors)
    
    # plot heatmap
    plt.figure(figsize = (6,5))
    g = sns.heatmap(diff_map_non_DE, cmap = cmap, xticklabels=False, yticklabels=False)
    plt.ylabel('') 
    plt.xlabel('')
    # add lines defining the cell state specific intermingling region
    plt.plot([im_len, im_len], [0, diff_map_non_DE.shape[0]], 'k-', lw = 0.8)
    plt.plot([0, diff_map_non_DE.shape[0]], [im_len, im_len], 'k-', lw = 0.8)
    plt.title('Intermingling of random non-DE genes')    
               

def get_diff_map_TF_transition(diff_map, TF, transition, data_dir, save_dir):
    """ Creates diff map for a TF and all its DE target genes in a specific transition
    
    Args:
        diff_map: (pd DataFrame) diff_map containing all loci
        TF: (string) name of the TF of interest
        transition: (string) name of the transition
        data_dir: (string) path of the data directory
        save_dir: (string) path of the save directory
    
    Returns:
        Sorted diff map with DE genes
    """
    # load DE genes
    DE_genes = pd.read_csv(save_dir+'DE_genes/DE_updown.csv')
    DE_genes = DE_genes[DE_genes['transition'] == transition]
    DE_genes = DE_genes[((DE_genes['transition'] == 'fc_16-26_27-60') & (DE_genes['updown'] == 'up')) | 
                        ((DE_genes['transition'] == 'fc_61-85_86-96') & (DE_genes['updown'] == 'down'))]
    
    # load DE targets of selected TF
    TF_targets = pd.read_csv(save_dir+'TF_targets/TF_targets_anno.csv')
    TF_targets = TF_targets[TF_targets['TF'] == TF]
    TF_targets = TF_targets[TF_targets['target'].isin(DE_genes['gene'])]
    TF_targets = TF_targets[['target', 'locus']]
    # add locus of the TF itself
    all_gene_loci = pd.read_csv(data_dir+'genome_data/all_gene_loci.csv')
    TF_targets = pd.concat([TF_targets, pd.DataFrame({'target': [TF], 
                                                     'locus': all_gene_loci.loc[all_gene_loci['gene'] == TF, 'locus'].item()})])
    # merge genes from the same locus
    TF_targets['target'] = TF_targets.groupby(['locus'])['target'].transform(lambda x: ', '.join(x))
    TF_targets = TF_targets.drop_duplicates()
    print(TF_targets.shape)
    
    # save order of the genes according the corresponding loci
    locus_order = [loc for loc in diff_map.columns if loc in TF_targets['locus'].unique()]
    TF_targets = TF_targets.set_index('locus', drop = False)
    TF_targets = TF_targets.loc[locus_order, :]
    gene_order = TF_targets['target']

    # create diff_map for DE targets
    diff_map_long = diff_map.copy()
    diff_map_long['loc1'] = diff_map_long.index
    diff_map_long = pd.melt(diff_map_long, id_vars = 'loc1', var_name = 'loc2')
    diff_map_long = diff_map_long[diff_map_long['loc1'].isin(TF_targets['locus']) & diff_map_long['loc2'].isin(TF_targets['locus'])]
    diff_map_long = TF_targets.rename(columns={"target": "target1", "locus": "loc1"}).merge(diff_map_long, on = 'loc1')
    diff_map_long = TF_targets.rename(columns={"target": "target2", "locus": "loc2"}).merge(diff_map_long, on = 'loc2')
    diff_map_long = diff_map_long[['target1', 'target2', 'value']]
    TF_diff_map = diff_map_long.pivot_table(index='target1', columns='target2', values='value')

    # sorting according to chromosomes
    TF_diff_map = TF_diff_map.loc[gene_order, gene_order]
    TF_diff_map.to_csv(save_dir + 'processed_hic_data/difference_maps/diff_DE_' + TF + '_' + transition + '.csv', index = False)
    return(TF_diff_map)
    
    
def diff_map_TF_transition(diff_map, TF, transition, name, data_dir, save_dir):
    """ Plots diff map for TF and all its DE target genes in a specific transition
    
    Args:
        diff_map: (pd DataFrame) diff_map containing all loci
        TF: (string) name of the TF of interest
        transition: (string) name of the transition
        name: (string) name that is used for the plot title (young, old)
        data_dir: (string) path of the data directory
        save_dir: (string) path of the save directory
    
    Returns:
        Sorted diff map with DE genes
    """
    TF_diff_map = get_diff_map_TF_transition(diff_map, TF, transition, data_dir, save_dir)
    
    # plot
    palette = sns.color_palette("tab10")
    colors = ['whitesmoke', 'blue', 'magenta', 'lightgrey']
    cmap = ListedColormap(colors)

    plt.figure(figsize = (6,5))
    g = sns.heatmap(TF_diff_map, cmap = cmap, xticklabels=False, yticklabels=False)
    plt.ylabel('') 
    plt.title('Intermingling differences for ' + TF + ' in the ' + name + ' network')
    

def diff_map_TF(diff_map, TF, transition, name, data_dir, save_dir):
    """ Creates diff map for all DE genes in a specific transition sorting the TF's targets to the front
    
    Args:
        diff_map: (pd DataFrame) diff_map containing all loci
        TF: (string) name of the TF of interest
        transition: (string) name of the transition
        name: (string) name that is used for the plot title (young, old)
        data_dir: (string) path of the data directory
        save_dir: (string) path of the save directory
    
    Returns:
        Sorted diff map with DE genes
    """
    TF_diff_map = get_diff_map_DE(diff_map, transition, data_dir, save_dir)
    gene_order = TF_diff_map.columns
    
    TF_targets = pd.read_csv(save_dir+'TF_targets/TF_targets_anno.csv')
    TF_targets = TF_targets[TF_targets['TF'] == TF]

    TF_order = [target for target in gene_order if target in list(TF_targets['target'])]
    target_len = len(TF_order)
    TF_order = TF_order + [target for target in gene_order if target not in TF_order]
    
    palette = sns.color_palette("tab10")
    colors = ['whitesmoke', 'blue', 'magenta', 'lightgrey']
    cmap = ListedColormap(colors)

    sns.heatmap(TF_diff_map.loc[TF_order, TF_order], cmap = cmap, xticklabels=False, yticklabels=False)
    # add lines defining the target region
    plt.plot([target_len, target_len], [0, len(TF_order)], 'k-', lw = 0.8)
    plt.plot([0, len(TF_order)], [target_len, target_len], 'k-', lw = 0.8)
    plt.ylabel('') 
    plt.title(TF)
   
 
def diff_map_young_old(diff_map, TF, data_dir, save_dir, ax, return_data = False):
    """ Creates diff map with all DE targets of a TF sorted by young, shared and old DE
    
    Args:
        diff_map: (pd DataFrame) diff_map containing all loci
        TF: (string) name of the TF of interest
        data_dir: (string) path of the data directory
        save_dir: (string) path of the save directory
        ax: ax to plot the subplot
        return_data: (boolean) specify whether instead of the plot only the diff map in long format should be returned
    
    Returns:
        Sorted diff map with young, shared and old DE genes
    """
    # load DE genes
    DE_genes = pd.read_csv(save_dir+'DE_genes/DE_updown.csv')
    DE_genes = DE_genes[DE_genes['transition'].isin(['fc_16-26_27-60','fc_61-85_86-96'])]
    DE_genes = DE_genes[((DE_genes['transition'] == 'fc_16-26_27-60') & (DE_genes['updown'] == 'up')) | 
                        ((DE_genes['transition'] == 'fc_61-85_86-96') & (DE_genes['updown'] == 'down'))]
    
    # load loci of all DE targets
    TF_targets = pd.read_csv(save_dir+'TF_targets/TF_targets_anno.csv')
    TF_targets = TF_targets[TF_targets['target'].isin(DE_genes['gene'])]
    TF_targets = TF_targets[TF_targets['TF'] == TF]
    TF_targets = TF_targets[['target', 'locus']].drop_duplicates()

    # save order of the genes according the corresponding loci
    locus_order = [loc for loc in diff_map.columns if loc in TF_targets['locus'].unique()]
    TF_targets = TF_targets.set_index('locus', drop = False)
    TF_targets = TF_targets.loc[locus_order, :]
    
    young = set(DE_genes.loc[DE_genes['transition'] == 'fc_16-26_27-60', 'gene']).difference(DE_genes.loc[DE_genes['transition'] == 'fc_61-85_86-96', 'gene'])
    old = set(DE_genes.loc[DE_genes['transition'] == 'fc_61-85_86-96', 'gene']).difference(DE_genes.loc[DE_genes['transition'] == 'fc_16-26_27-60', 'gene'])
    shared = set(DE_genes.loc[DE_genes['transition'] == 'fc_61-85_86-96', 'gene']).intersection(DE_genes.loc[DE_genes['transition'] == 'fc_16-26_27-60', 'gene'])
    gene_order = [gene for gene in TF_targets['target'] if gene in young]
    young_len = len(gene_order)
    gene_order = gene_order + [gene for gene in TF_targets['target'] if gene in shared]
    ys_len = len(gene_order)
    gene_order = gene_order + [gene for gene in TF_targets['target'] if gene in old]
    
    # create diff_map for DE targets
    diff_map_long = diff_map.copy()
    diff_map_long['loc1'] = diff_map_long.index
    diff_map_long = pd.melt(diff_map_long, id_vars = 'loc1', var_name = 'loc2')
    diff_map_long = diff_map_long[diff_map_long['loc1'].isin(TF_targets['locus']) & diff_map_long['loc2'].isin(TF_targets['locus'])]
    diff_map_long = TF_targets.rename(columns={"target": "target1", "locus": "loc1"}).merge(diff_map_long, on = 'loc1')
    diff_map_long = TF_targets.rename(columns={"target": "target2", "locus": "loc2"}).merge(diff_map_long, on = 'loc2')
    diff_map_long = diff_map_long[['target1', 'target2', 'value']]
    TF_diff_map = diff_map_long.pivot_table(index='target1', columns='target2', values='value')

    # sorting according to chromosomes and transitions
    TF_diff_map = TF_diff_map.loc[gene_order, gene_order]
    
    # color palette
    palette = sns.color_palette("tab10")
    colors = ['whitesmoke', 'blue', 'magenta', 'lightgrey']
    cmap = ListedColormap(colors)
    
    if return_data:
        return(TF_diff_map)
    else:
        sns.heatmap(TF_diff_map, cmap = cmap, xticklabels=False, yticklabels=False, ax = ax, cbar=False)
        # add lines defining the target region
        ax.plot([young_len, young_len], [0, len(gene_order)], 'k-', lw = 0.8)
        ax.plot([ys_len, ys_len], [0, len(gene_order)], 'k-', lw = 0.8)
        ax.plot([0, len(gene_order)], [young_len, young_len], 'k-', lw = 0.8)
        ax.plot([0, len(gene_order)], [ys_len, ys_len], 'k-', lw = 0.8)
        ax.set_ylabel('') 
        ax.set_xlabel('')
        ax.set_title(TF)
    

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
    blacklist_dir = config['BLACKLIST_DIR']
    tf_dir = config['TF_DIR']
    genome_dir = config['GENOME_DIR']
    resol_str = config['HIC_RESOLUTION_STR']
    resol = config['HIC_RESOLUTION']
    chr_list = config['chrs']
    cell_type = config['CELL_TYPE']
    subset = config['SUBSET']
    inter_threshold = config['LAS_THRESHOLD_INTER']
    intra_threshold = config['LAS_THRESHOLD_INTRA']
    
    print("Cell type: ", cell_type)
    
    # Select loci
    if subset == "all_gene_loci":
        selected_loci = pd.read_csv(genome_dir+'all_gene_loci.csv')['locus'].unique().tolist()
    else:
        selected_loci = pd.read_csv(tf_dir + subset + '_targets_loci.csv')['locus'].unique().tolist()
    
    # Remove blacklisted loci
    blacklist = get_blacklist_all_samples(blacklist_dir)
    selected_loci = [loc for loc in selected_loci if loc not in set(blacklist)]
    print("Number of selected loci: ", len(selected_loci))
    
    # Create binarized map for selected loci
    print("Binarize hic data for selected loci:")
    hic_binarized = get_binarized_hic(selected_loci, chr_list, hic_dir, cell_type, resol, inter_threshold, intra_threshold)
    
    # Convert from long to wide format and save results
    hic_wide = long_to_wide(hic_binarized)   
    hic_wide = hic_wide.loc[selected_loci, selected_loci]
    hic_wide.to_csv(hic_dir + cell_type + '/binarized_maps/' + 
                    subset + '_subset.csv')
    
    # Create binarized map for random loci (same amount per chromosome) as a comparison
    if subset != "all_gene_loci": 
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
        hic_wide.to_csv(hic_dir + cell_type + '/binarized_maps/' + 
                        subset + '_random.csv')
    

if __name__ == "__main__":
    main()
    
