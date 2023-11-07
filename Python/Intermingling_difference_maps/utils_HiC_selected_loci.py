import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import itertools
import random

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

    # merge genes from the same locus
    TF_targets['target'] = TF_targets.groupby(['locus'])['target'].transform(lambda x: ', '.join(x))
    TF_targets = TF_targets.drop_duplicates()
    
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
    diff_map_long = diff_map_long[['loc1', 'loc2', 'value']]
    TF_diff_map = diff_map_long.pivot_table(index='loc1', columns='loc2', values='value')

    # sorting according to chromosomes
    TF_diff_map = TF_diff_map.loc[locus_order, locus_order]
    return(TF_diff_map)


def get_normalized_hic(loci, age, processed_hic_dir):
    norm_hic = pd.DataFrame(0, columns=loci, index=loci)
    if age == 'young':
        dir_r1 = processed_hic_dir + 'Young_B1R1/final_BP250000_intraKR_interINTERKR/'
        dir_r2 = processed_hic_dir + 'Young_B1R2/final_BP250000_intraKR_interINTERKR/'
    else:
        dir_r1 = processed_hic_dir + 'Old_B1R1/final_BP250000_intraKR_interINTERKR/'
        dir_r2 = processed_hic_dir + 'Old_B2R2/final_BP250000_intraKR_interINTERKR/'
    
    chr_list = set([int(loc.split('_')[1]) for loc in loci])
    chr_pairs = list(itertools.combinations(chr_list, 2))
    chr_pairs = chr_pairs + [(chrom, chrom) for chrom in chr_list]
    
    for pair in chr_pairs:
        chr1, chr2 = pair
        # load normalized Hi-C data for selected chromosome pair for both replicate
        r1 = pd.read_csv(dir_r1 + 'hic_chr' + str(chr1) + '_chr' + str(chr2) + '_zscore.txt', index_col = 0)
        r2 = pd.read_csv(dir_r2 + 'hic_chr' + str(chr1) + '_chr' + str(chr2) + '_zscore.txt', index_col = 0)
        
        for chr1_loc in [locus for locus in loci if locus.split("_")[1] == str(chr1)]:
            for chr2_loc in [locus for locus in loci if locus.split("_")[1] == str(chr2)]:
                r1_value = r1.loc[int(chr1_loc.split('_')[3]), chr2_loc.split('_')[3]]
                r2_value = r2.loc[int(chr1_loc.split('_')[3]), chr2_loc.split('_')[3]]
                avg_hic = (r1_value + r2_value) / 2
                norm_hic.loc[chr1_loc, chr2_loc] = avg_hic
                norm_hic.loc[chr2_loc, chr1_loc] = avg_hic
    return norm_hic

def map_rownum2pos(df, row_num):
    positions = df.index.values
    return positions[row_num]

def map_colnum2pos(df, col_num):
    positions = df.columns.values
    return float(positions[col_num])

def plot_hic(hic_data, TF, age):
    plt.rc('font', family='serif')
    sns.set_style("dark", {'axes.grid':False})
    numclust = 50

    data = hic_data.to_numpy()

    plt.figure()
    plt.imshow(data, cmap = 'Reds', vmin = -5, vmax = 16) 
    plt.xticks([])
    plt.yticks([])
    
    cbar = plt.colorbar()
    cbar.set_label('Transformed Hi-C contacts', fontsize = 12)
    cbar.solids.set_rasterized(True) 

    plt.xlabel('DE genes targeted by ' + TF, fontsize = 14)
    plt.ylabel('DE genes targeted by ' + TF, fontsize = 14)
    plt.title('DE targets of ' + TF + ' in ' + age, fontsize = 16)
    
def plot_hic_random(hic_data, age):
    plt.rc('font', family='serif')
    sns.set_style("dark", {'axes.grid':False})
    numclust = 50

    data = hic_data.to_numpy()

    plt.figure()
    plt.imshow(data, cmap = 'Reds', vmin = -5, vmax = 16) 
    plt.xticks([])
    plt.yticks([])
    
    cbar = plt.colorbar()
    cbar.set_label('Transformed Hi-C contacts', fontsize = 12)
    cbar.solids.set_rasterized(True) 

    plt.xlabel('Random non-DE gene loci', fontsize = 14)
    plt.ylabel('Random non-DE gene loci', fontsize = 14)
    plt.title('Random non-DE genes in ' + age, fontsize = 16)
    
def get_random_loci(ref_loci, genome_dir, save_dir):
    random.seed(2023)
    # get list of chromosome for the loci set of interest
    chromosome_list = [locus.split("_")[1] for locus in ref_loci]
    
    # Load available loci per chromosome
    all_gene_loci = pd.DataFrame({'locus': pd.read_csv(genome_dir+'all_gene_loci.csv')['locus'].unique()})
    all_gene_loci[['chr','chr_number', 'loc', 'loc_number']]=all_gene_loci.locus.str.split('_',expand=True)
    
    # Remove DE gene loci
    DE_genes = pd.read_csv(save_dir+'DE_genes/DE_updown.csv')
    TF_targets = pd.read_csv(save_dir+'TF_targets/TF_targets_anno.csv')
    TF_targets = TF_targets[TF_targets['target'].isin(DE_genes['gene'])]
    all_gene_loci = all_gene_loci[~all_gene_loci['locus'].isin(TF_targets['locus'])]
    
    # Select random loci (same amount per chromosome as in set of interest)
    random_loci = []
    for chrom in range(1,23):
        n = chromosome_list.count(str(chrom))
        loci = all_gene_loci[all_gene_loci['chr_number'] == str(chrom)].sample(n, random_state=202208).sort_index()['locus']
        random_loci.append(loci)
    # flatten list 
    random_loci = list(itertools.chain(*random_loci))
    return random_loci

def calc_entropy(hic):
    observations = hic.values.flatten()

    # Calculate the probability distribution
    total_samples = len(observations)
    counter = Counter(observations)
    probabilities = [count / total_samples for count in counter.values()]

    # Compute the entropy
    entropy = -sum(p * np.log2(p) for p in probabilities)
    return entropy