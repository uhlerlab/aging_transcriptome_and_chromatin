import sys, getopt
import json
import os, os.path
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import random
from scipy import sparse
import scipy.stats
import csv
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import pickle
from collections import defaultdict
import operator
from scipy.sparse import csr_matrix
import itertools

import math
import pybedtools


def get_chrom_sizes(genome_dir,resol):
    '''
    Constructs dataframe of chromosome sizes (in bp and in loci counts at the chosen resolution)
    Args:
        genome_dir: (string) directory of genome data
        resol: (int) hic resolution
    Returns:
        A pandas datafrawith columns chr (chromosome), size, size_loci, size_roundup
    '''
    sizes_filename = genome_dir+'GRCh38-filter-ordered.chrom.sizes'
    df_sizes = pd.read_csv(sizes_filename, sep = '\t', header = None, names=['chr','size'])
    #df_sizes['chr'] = df_sizes['chr'].str.split('chr',expand=True)[1]
    df_sizes['size_loci'] = np.ceil(df_sizes['size']/resol).astype(int)
    df_sizes['size_roundup'] = np.ceil(df_sizes['size']/resol).astype(int)*resol
    df_sizes = df_sizes[~df_sizes['chr'].str.contains('hap|alt|Un|rand')]
    return(df_sizes)


def get_centromere_locations(genome_dir):
    '''
    Constructs dataframe with centromere location for all chromosomes
    Args:
        genome_dir: (string) directory of genome data
    Returns:
    A pandas DataFrame with centromere information
    '''
    centrom_filename = genome_dir+'GRCh38_merged_centromeres'
    df_centrom = pd.read_csv(centrom_filename, sep = ',')
    return(df_centrom)


def get_normed_hic_sparse(processed_hic_dir, cell_type, resol, quality, chr0, norm):
    '''
    Constructs dataframe of normalized HiC data for a given cell type, resolution, mapping quality and
    chromosome, including Knight-Ruiz balancing coefficients.
    Args:
        processed_hic_dir: (string) directory containing all processed hic data
        cell_type: (string) cell type of interest
        resol: (int) HiC resolution
        quality: (str) maping quality, one of MAPQ0 or MAPQ30
        chr0: (int) chromosome of interest
        norm: (str) the normalization to apply, one of NONE, KR, GWKR
    Returns:
        A sparse dataframe including normalized contact values for pairs of loci
    '''
    # Construct data directory
    normed_hic_dir = processed_hic_dir+cell_type+'/normalized_BP'+str(resol)+'/'
    normed_mat_path = normed_hic_dir+'chr'+str(chr0)+'_chr'+str(chr0)+'_'+norm+'.txt'

    # Load raw data
    data = np.loadtxt(normed_mat_path, delimiter = '\t')
    row_ind = data[:,0]/resol
    row_ind = row_ind.astype(int)
    col_ind = data[:,1]/resol
    col_ind = col_ind.astype(int)
    contact_values = data[:,2]
    # Convert sparse Hi-C data to pandas dataframe
    normed_hic_data = pd.DataFrame(data={'locus_chr1':row_ind,'locus_chr2':col_ind,'norm_value':contact_values})

    return(normed_hic_data)


def get_dense_hic_dataframe(normalized_hic_data, chr0_size, resol):
    '''
    Constructs dense HiC dataframe from HiC data in sparse format
    Args:
        normalized_hic_data: (pandas DataFrame) HiC data in sparse format
        chr0_size: (int) size of chromosome of interest
        resol: (int) HiC resolution
    Returns:
        A dense HiC dataframe
    '''
    hic_sparse_matrix = csr_matrix((normalized_hic_data['norm_value'], 
                             (normalized_hic_data['locus_chr1'], normalized_hic_data['locus_chr2'])), 
                            shape = (chr0_size, chr0_size))
    hic_dense = np.asarray(hic_sparse_matrix.todense())
    row_labels = np.arange(chr0_size)*resol
    col_labels = row_labels
    df = pd.DataFrame(hic_dense, index = row_labels, columns = col_labels)
    df = df.fillna(0)
    return(df)


def filter_centromeres(df, chrom, row_or_col, df_centrom, filter_size, resol):
    '''
    Filter out loci corresponding to centromeric and pericentromeric regions (defined by
    filter_size) from dense HiC dataframe
    Args:
        df: (pandas DataFrame) dense HiC dataframe
        chrom: (int) chromosome of interest
        row_or_col: (string) whether the chromosome chrom corresponds to the rows or columns of df
        df_centrom: (pandas DataFrame) centromere information
        filter_size: (int) size of filter for the pericentromeric regions
        resol: (int) HiC resolution
    Returns:
        A pandas DataFrame where centromeric and pericentromeric loci have been zeroed out
    '''
    chr_centrom_data = df_centrom[df_centrom['chrom'] == 'chr' + str(chrom)]
    centrom_start = int((chr_centrom_data['chromStart']//resol)*resol)
    centrom_end = int((chr_centrom_data['chromEnd']//resol)*resol)
    centrom_start = centrom_start - filter_size
    centrom_end = centrom_end + filter_size
    if (row_or_col == 'row'):
        df.loc[centrom_start:centrom_end, :] = 0
    if (row_or_col == 'col'):
        df.loc[:, centrom_start:centrom_end] = 0
    return(df)
    

def load_repeats_data(genome_dir):
    '''
    Load information of repeats
    Args:
        genome_dir: (str) directory of genome data
    Returns:
        A pandas DataFrame with information on repeats
    '''
    # Load repeats data
    repeats_filename = 'rmsk.txt'
    df_repeats0 = pd.read_csv(genome_dir+repeats_filename, sep= '\t')

    # Select relevant columns and add column with length of repeat
    df_repeats = df_repeats0[['genoName','genoStart','genoEnd']]
    df_repeats.insert(3,'repLength', df_repeats['genoEnd']-df_repeats['genoStart'], True)

    # Filter out alt, fix, random, Un chromosomes
    df_repeats = df_repeats[~df_repeats['genoName'].str.contains('alt|fix|rand|Un|hap')]
    
    return(df_repeats)

def find_repeat_locations(df_repeats, chr_list, df_sizes, resol):
    '''
    Find repeat locations (loci that intersect with repeats) to filter out from data (filtering is done
    for loci whose coverage is higher than the 95th percentile of coverage values across the genome)
    Args:
        df_repeats: (pandas DataFrame) dataframe with information on repeats
        chr_list: (list of ints) list of all chromosomes under consideration
        df_sizes: (pandas DataFrame) a dataframe with chromosome size information
        resol: (int) HiC resolution
    Returns:
        A dictionary with chromosomes as keys and loci to filter out as values
    '''
    # Dataframe including all coverage values across all chromosomes
    df_coverage = pd.DataFrame()
    # Convert df_repeats to BED format for genome arithmetics
    df_repeats_bed = pybedtools.BedTool.from_dataframe(df_repeats)
    
    # Compute coverage values for all start sites for all chromosomes
    for chrom in chr_list:
        # Create dataframe of all possible start sites for this chr
        chrom_size = int(df_sizes[df_sizes['chr']==str(chrom)]['size'])
        start_regions = range(0, chrom_size, resol)
        df_chr = pd.DataFrame({'chr':['chr' + str(chrom)]*len(start_regions), 'start':start_regions})
        df_chr['stop'] = df_chr['start'] + resol
        df_chr_bed = pybedtools.BedTool.from_dataframe(df_chr)
        
        # Compute coverage
        coverage_chr = df_chr_bed.coverage(df_repeats_bed).to_dataframe(names = ['chr', 'start', 'end', 'bases covered', 
                                                                                 'length intersection', 'length locus', 
                                                                                 'coverage'])
        df_coverage = pd.concat([df_coverage, coverage_chr])
    
    # Determine coverage threshold for filtering out (95th percentile)
    df_coverage.columns = ['chr','start','end','bases covered','length intersection','length locus','coverage']
    threshold = np.quantile(a=df_coverage['coverage'],q=0.95)
    
    # Dictionary with chromosomes as keys and loci with repeat coverage information
    dic_repeats_tofilter = {}
    for chrom in chr_list:
        coverage_chr = df_coverage[df_coverage['chr']=='chr'+str(chrom)]
        tofilter = coverage_chr[coverage_chr['coverage'] >= threshold]['start'].values
        dic_repeats_tofilter[str(chrom)] = tofilter
        
    return(dic_repeats_tofilter)


def filter_repeats(df, chrom, dic_repeats_tofilter, row_or_col):
    '''
    Filters out repeat-covered loci for chromsome chrom from the dense HiC dataframe df
    Args:
        df: (pandas DataFrame) dense HiC dataframe
        chrom: (int) chromosome of interest
        dic_repeats_tofilter: (dict) dictionary with chromosomes as keys and loci to filter out as values
        row_or_col: (string) whether the chromosome chrom corresponds to the rows or columns of df
    Returns:
        The dense HiC dataframe where repeat-covered loci for chrom have been zeroed out
    '''
    regions2filter = dic_repeats_tofilter[str(chrom)]
    if (row_or_col == 'row'):
        df.loc[regions2filter,:] = 0
    if (row_or_col == 'col'):
        df.loc[:,regions2filter] = 0
    return df


def detect_upper_outliers(array):
    '''
    Function to detect the upper outliers of an array, defined as all elements that are bigger than 
    QR3+1.5*IQR where QR3 is the 3rd quartile and IQR is the interquartile range
    Args:
        array: (numpy array) array of interest
    Returns:
        Threshold to determine upper outlier values
    '''
    p25 = np.percentile(array, 25)
    p75 = np.percentile(array, 75)
    upper = p75 + 1.5*(p75-p25)
    return upper


def log_transform(df):
    '''
    Log-transform dense HiC dataframe
    Args:
        df: (pandas Dataframe) dense HiC dataframe
    Returns:
        The log-transformed dense HiC dataframe
    '''
    return(np.log(1+df))


def filter_outliers(df_transformed):
    '''
    Filters out outliers of the log-transformed dense HiC dataframe, where outliers for chr1 are
    defined as loci of chr1 whose total contact values with chr2 are bigger than QR3+1.5*IQR where QR3 is 
    the 3rd quartile and IQR is the interquartile range of the total contact values of all chr1 loci with chr2
    (mutatis mutandis for chr2 and chr1)
    Args:
        df_transformed: (pandas Dataframe) log-transformed dense HiC dataframe
    Returns:
        The dense HiC dataframe where outliers have been zeroed out
    '''
    # Get all the row and column sums
    row_orig = np.sum(df_transformed, axis = 1).values
    col_orig = np.sum(df_transformed, axis = 0).values
    
    # Get rid of zeros (in order to compute row and col thresholds)
    row = row_orig[np.nonzero(row_orig)]
    col = col_orig[np.nonzero(col_orig)]

    # Compute a threshold for outliers for chr1 and chr2
    threshold_row = detect_upper_outliers(row)
    threshold_col = detect_upper_outliers(col)
    
    # Filter out outliers for chr1 (rows)
    ind_row = np.arange(len(row_orig))[row_orig>threshold_row]
    df_transformed.iloc[ind_row,:] = 0
    # Filter out outliers for chr2 (columns)
    ind_col = np.arange(len(col_orig))[col_orig>threshold_col]
    df_transformed.iloc[:,ind_col] = 0
    
    return(df_transformed, ind_row, ind_col)
    

def map_rownum2pos(df, row_num):
    '''
    Helper function for plot_dense_hic_dataframe, for a given row number 
    returns the index which is equal to the genome location of that locus on chr1
    Args:
        df: (pandas DataFrame) dense HiD dataframe
        row_num: (int) row number
    Returns:
        An int equal to the genome location of the locus of interest on chr1
    '''
    positions = df.index.values
    return positions[row_num]


def map_colnum2pos(df, col_num):
    '''
    Helper function for plot_dense_hic_dataframe, for a given column number 
    returns the index which is equal to the genome location of that locus on chr2
    Args:
        df: (pandas DataFrame) dense HiD dataframe
        col_num: (int) column number
    Returns:
        An int equal to the genome location of the locus of interest on chr2
    '''
    positions = df.columns.values
    return float(positions[col_num])


def plot_dense_hic_dataframe(df, chr0, plotname, hic_plots_dir, show=False):
    '''
    Plot a dense HiC dataframe as a heatmap
    Args:
        df: (pandas DataFrame) dense HiC dataframe
        chr0: (int) chromosome of interest (rows and columns)
        plotname: (str) name of plot
        hic_plots_dir: (str) directory where to save HiC plots
        show: (Boolean) whether to display the plot
    Returns:
        void
    '''
    # Plot heatmap with colorbar
    data = df.values
    plt.figure()
    plt.imshow(data, cmap = 'Reds', extent = [0, data.shape[0], 0, data.shape[1]], origin='lower')
    cbar = plt.colorbar()
    cbar.set_label('balanced contacts')
    cbar.solids.set_rasterized(True)

    # label ticks with genomic position (Mb)
    plt.tick_params(axis='x', which='major', labelbottom = False, bottom=False, top = True, labeltop=True)
    xaxis = np.arange(0, data.shape[0], 100)
    xlabels = [str(map_colnum2pos(df, x)/1000000.0) for x in xaxis]
    plt.xticks(xaxis, xlabels)
    yaxis = np.arange(0, data.shape[1], 100)
    ylabels = [str(map_rownum2pos(df, y)/1000000.0) for y in yaxis]
    plt.yticks(yaxis, ylabels)
    plt.gca().invert_yaxis()

    # Add axis labels
    plt.xlabel('chr' + str(chr0) + ' (Mb)', fontsize = 14)
    plt.ylabel('chr' + str(chr0) + ' (Mb)', fontsize = 14)

    # Save and close
    if plotname != '':
        plt.savefig(hic_plots_dir+plotname, format='eps')
    if show == True:
        plt.show()
    else:
        plt.close()


def whole_genome_mean_std(nonzero_entries):
    '''
    Computes mean and standard deviation of nonzero_entries
    Args:
        nonzero_entries: (Numpy array) array f nonzero entries from all Hi-C matrices
    Returns:
        The mean and standard deviation of the input array
    '''
    mean = np.mean(nonzero_entries)
    std = np.std(nonzero_entries)
    return mean, std


def z_score_hic_matrix(mean, std, chr_list, processed_hic_data_dir):
    '''
    Z-scores all Hi-C matrices for pairs of chromosomes in chr_pairs and pickles the result
    Args:
        mean: (float) mean for centering
        std: (float) standard deviation for scaling
        chr_list: (list) list of all chromosomes
        processed_hic_data_dir: (string) directory of processed Hi-C data
    Returns:
    '''
    for chr0 in chr_list:
        # Read in
        hic_filename = processed_hic_data_dir+'hic_'+'chr'+str(chr0)+'_'+'chr'+str(chr0)+'_norm1_filter3'+'.pkl'
        df = pickle.load(open(hic_filename, 'rb'))
        # z-score matrix
        df  = (df - mean)/std
        # save new matrix
        df.to_csv(processed_hic_data_dir+'hic_'+'chr'+str(chr0)+'_'+'chr'+str(chr0)+'_zscore.txt')
        #with open(processed_hic_data_dir+'hic_'+'chr'+str(chr0)+'_'+'chr'+str(chr0)+'_zscore.pkl', 'wb') as f:
         #   pickle.dump(df, f)

            
def output_blacklist_locations(chr_list, processed_hic_data_dir):
    '''
    Output a file with locations that have been removed (these locations will have an observed value of 0) 
    for each chromosome
    Args:
        chr_list: (list) list of chromosomes
        processed_hic_data_dir: (string) directory of processed Hi-C data
    Returns:
    '''
    # initialize empty dictionary
    blacklist = defaultdict(list)
    for chr0 in chr_list:
        # read in
        hic_filename = processed_hic_data_dir+'hic_'+'chr'+str(chr0)+'_'+'chr'+str(chr0)+'_norm1_filter3'+'.pkl'
        df = pickle.load(open(hic_filename, 'rb'))
        # find out which loci of chr2 (columns) have all zeros in them
        zero_cols = df.columns[(df == 0).all(axis=0)]
        blacklist[chr0].append(zero_cols)
        # find out which loci of chr1 (rows) have all zeros in them
        zero_rows = df.index[(df == 0).all(axis=1)]
        blacklist[chr0].append(zero_rows)
    # process the list
    for chrom in blacklist.keys():
        values_list = blacklist[chrom]
        blacklist[chrom] = set(map(int, list(itertools.chain.from_iterable(values_list))))
    # pickle this dictionary
    with open(processed_hic_data_dir+'blacklist.pickle', 'wb') as f:
        pickle.dump(blacklist, f)


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
    genome_dir = config['GENOME_DIR']
    hic_plots_dir = config['HIC_PLOTS_DIR']
    processed_hic_data_dir = config['PROCESSED_HIC_DATA_DIR']
    hic_celltype = config['HIC_CELLTYPE']
    resol = config['HIC_RESOLUTION']
    inter_norm = config['HIC_INTER_NORM']
    intra_norm = config['HIC_INTRA_NORM']
    quality = config['HIC_QUALITY']
    chr_list = config['chrs']
    filter_size = config['CENTROMERE_FILTER_SIZE']
    final_hic_data_dir = processed_hic_data_dir+hic_celltype+'/final_BP'+str(resol)+'_intra'+intra_norm+'_inter'+inter_norm+'/'
    
    # We first find repeat locations to filter out (used later)
    print('Load locations of repeats to filter out later...')
    df_repeats = load_repeats_data(genome_dir)
    df_sizes = get_chrom_sizes(genome_dir,resol)
    dic_repeats_tofilter = find_repeat_locations(df_repeats, chr_list, df_sizes, resol)
    with open(processed_hic_data_dir+'dic_repeats_tofilter.pkl', 'wb') as f:
        pickle.dump(dic_repeats_tofilter, f)
    blacklisted_repeats = np.array(list(itertools.chain.from_iterable([['chr_'+chrom+'_loc_'+str(loc) 
                                                                        for loc in dic_repeats_tofilter[chrom]] 
                                                                       for chrom in dic_repeats_tofilter.keys()])))
    
    # We also load centromere information (used later)
    print('Load centromere information to be used later...')
    df_centrom = get_centromere_locations(genome_dir)
    print(df_centrom.head())
    pericentromeric_dict = {}
    for chrom in np.arange(1, 22+1):
        perictm_start = (int(df_centrom[df_centrom['chrom']=='chr'+str(chrom)]['chromStart'])//resol)*resol-filter_size
        perictm_end = (int(df_centrom[df_centrom['chrom']=='chr'+str(chrom)]['chromEnd'])//resol)*resol+filter_size
        pericentromeric_dict[str(chrom)] = np.arange(perictm_start, perictm_end+resol, resol)
    with open(processed_hic_data_dir+'pericentromeric_dict.pkl', 'wb') as f:
        pickle.dump(pericentromeric_dict, f)
    blacklisted_centrom = np.array(list(itertools.chain.from_iterable([['chr_'+chrom+'_loc_'+str(loc) 
                                                                        for loc in pericentromeric_dict[chrom]] 
                                                                       for chrom in pericentromeric_dict.keys()])))
    
    # Record all values over all Hi-C matrices
    all_entries = []
    # Record all outlier loci
    outliers_list = []
    
    # List all chromosomes and process the corresponding intrachromosomal Hi-C data
    print("Process HiC data for all "+str(len(chr_list))+" chromosomes")
    for chr0 in chr_list:
        print('Process HiC data for chromosome '+str(chr0))
        print('---------------------------------------------------------------')
        
        
        # Load normalized HiC data
        print('Load normalized HiC data')
        normalized_hic_data = get_normed_hic_sparse(processed_hic_data_dir, hic_celltype, resol, quality, chr0, intra_norm)
        # Get chromosome sizes (in number of loci)
        chr0_size = int(df_sizes[df_sizes['chr']==str(chr0)]['size_loci'])
        # Create dense dataframe
        df = get_dense_hic_dataframe(normalized_hic_data, chr0_size, resol)
        # Symmetrize dense HiC dataframe
        df = df+df.transpose()
        df.values[[np.arange(df.shape[0])]*2] = df.values[[np.arange(df.shape[0])]*2]/2
        # Plot normalized dense HiC dataframe
        plotname = 'hic_'+'chr'+str(chr0)+'_'+'chr'+str(chr0)+'_norm1_filter0'+'.eps'
        plot_dense_hic_dataframe(df, chr0, plotname, hic_plots_dir)
        
        # Filter out centromeric and pericentromeric regions
        print('Filter out centromeres ')
        df = filter_centromeres(df, chr0, 'row', df_centrom, filter_size, resol)
        df = filter_centromeres(df, chr0, 'col', df_centrom, filter_size, resol)
        # Plot HiC data after filtering out centromeres
        plotname = 'hic_'+'chr'+str(chr0)+'_'+'chr'+str(chr0)+'_norm1_filter1'+'.eps'
        plot_dense_hic_dataframe(df, chr0, plotname, hic_plots_dir)
        
        # Filter out repeats 
        print('Filter out repeats for chromsome '+str(chr0))
        df = filter_repeats(df, chr0, dic_repeats_tofilter, 'row')
        df = filter_repeats(df, chr0, dic_repeats_tofilter, 'col')
        # Plot HiC data after filtering out repeats
        plotname = 'hic_'+'chr'+str(chr0)+'_'+'chr'+str(chr0)+'_norm1_filter2'+'.eps'
        plot_dense_hic_dataframe(df, chr0, plotname, hic_plots_dir)
        
        # Log-transform dataframe
        print('Log-transform HiC data')
        df_transformed = log_transform(df)
        
        # Filter out outliers
        print('Filter out outliers')
        df_transformed, ind_row, ind_col = filter_outliers(df_transformed)
        outliers_list = outliers_list+[f'chr_{chr0}_loc_{i*resol}' for i in ind_row]
        outliers_list = outliers_list+[f'chr_{chr0}_loc_{i*resol}' for i in ind_col]

        # Plot and save to pickle HiC data after filtering out centromeres, repeats and outliers
        plotname = 'hic_'+'chr'+str(chr0)+'_'+'chr'+str(chr0)+'_norm1_filter3'+'.eps'
        plot_dense_hic_dataframe(df_transformed, chr0, plotname, hic_plots_dir)
        picklename = final_hic_data_dir+'hic_'+'chr'+str(chr0)+'_'+'chr'+str(chr0)+'_norm1_filter3'+'.pkl'
        with open(picklename, 'wb') as f:
            pickle.dump(df_transformed, f)
        
        # Record all entries
        data = df_transformed.values
        data = np.asarray(list(itertools.chain.from_iterable(data)))
        all_entries.append(data)

    # Center and scale every Hi-C matrix by the mean and standard deviation across all matrices
    print('Z-score Hi-C matrices using mean and standard deviation')
    all_entries = np.asarray(list(itertools.chain.from_iterable(all_entries)))
    np.savetxt(final_hic_data_dir + 'whole_genome.logtrans_intra.txt', all_entries)
    mean, std = whole_genome_mean_std(all_entries)
    z_score_hic_matrix(mean, std, chr_list, final_hic_data_dir)
    
    # Output pickle of outlier loci
    outliers_list = np.array(outliers_list)
    with open(final_hic_data_dir+'outliers_list_'+hic_celltype+'_'+inter_norm+'.pkl', 'wb') as f:
            pickle.dump(outliers_list, f)
    
    # Output a file with locations that have been removed
    print('Output blacklisted loci file')
    # output_blacklist_locations(chr_pairs, final_hic_data_dir)
    blacklist = np.unique(np.concatenate([blacklisted_repeats, blacklisted_centrom, outliers_list]))
    with open(final_hic_data_dir+'blacklist_'+hic_celltype+'_'+inter_norm+'.pickle', 'wb') as f:
        pickle.dump(blacklist, f)

if __name__ == "__main__":
    main()
    
    




