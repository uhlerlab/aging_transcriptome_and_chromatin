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
import hicstraw
from sklearn.decomposition import PCA

''' This script calculates the first two principal components of the observed/expected intrachromosomal hic data.'''

        
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


def get_matrix(chrom, resol, hic):
    '''
    Reads config file
    Args:
        chrom: (int) chromosome number
        resol: (int) resolution of the matrix
        hic: (hicstraw.HiCFile) hic file
    Returns:
        A numpy array with the observed/expected intrachromosomal hic data for one chromosome
    '''
    matrix_object = hic.getMatrixZoomData(str(chrom), str(chrom), "oe", "NONE", "BP", resol)
    numpy_matrix = matrix_object.getRecordsAsMatrix(0, hic.getChromosomes()[chrom].length, 0, hic.getChromosomes()[chrom].length)
    return(numpy_matrix)


def get_df(hic_numpy, chrom, resol, hic):
    '''
    Reads config file
    Args:
        hic_numpy: (np.array) numpy array with the observed/expected intrachromosomal hic data for one chromosome
        chrom: (int) chromosome number
        resol: (int) resolution of the matrix
        hic: (hicstraw.HiCFile) hic file
    Returns:
        A pd DataFrame with the observed/expected intrachromosomal hic data for one chromosome
    '''
    hic_intra = pd.DataFrame(hic_numpy)
    # add bin numbers as row- and colnames
    names = np.arange(0, hic.getChromosomes()[chrom].length, resol)
    hic_intra.index = names
    hic_intra.columns = names
    return(hic_intra)


def get_PCs(hic_intra):
    '''
    Reads config file
    Args:
        hic_intra: (pd DataFrame) observed/expected intrachromosomal hic data for one chromosome
    Returns:
        A pd DataFrame with the first two principal components
    '''
    pca = PCA(n_components=2)
    PCs = pca.fit_transform(hic_intra)
    PCs = pd.DataFrame({'PC1': PCs[:,0], 
                        'PC2': PCs[:,1]})
    return(PCs)


def normalize_PCs(PCs):
    '''
    Reads config file
    Args:
        PCs: (pd DataFrame) first two principal components
    Returns:
        A pd DataFrame with the normalized first two principal components
    '''
    mean = np.mean(PCs, axis=0)
    std = np.std(PCs, axis=0)
    PCs_norm = (PCs - mean)/std
    return(PCs_norm)
    
    
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
    hic_filename = config['HIC_FILENAME']
    processed_hic_PCs_dir = config['PROCESSED_HIC_PC_DIR']
    hic_celltype = config['HIC_CELLTYPE']
    resol_str = config['HIC_RESOLUTION_STR']
    resol = config['HIC_RESOLUTION']
    chr_list = config['chrs']
    input_type = config['input_type']
    
    # Load hic file with straw
    hic = hicstraw.HiCFile(hic_dir+hic_filename)
    
    # Get PCs for every chromosome in chr_list
    print('Get PCs for all chromosomes')
    for chrom in chr_list:
        print('Get PCs for chr'+str(chrom))
        
        # Extract data from hic for chrom into a np matrix
        hic_numpy = get_matrix(chrom, resol, hic)
        
        # Create pandas DataFrame with correct col/rownames
        hic_intra = get_df(hic_numpy, chrom, resol, hic)
        
        # Calculate correlation matrix for input_type = "correlation"
        if input_type == "correlation":
            hic_intra = hic_intra.corr()
            hic_intra = np.nan_to_num(hic_intra)
        
        # PCA
        PCs = get_PCs(hic_intra)
        
        #Normalize PCs to use as features
        PCs_norm = normalize_PCs(PCs)
        
        # write PCs to file
        if input_type == "correlation":
            PCs_norm.to_csv(processed_hic_PCs_dir + 'PCs_corr_chr' + str(chrom) + '.csv')
        else:
            PCs_norm.to_csv(processed_hic_PCs_dir + 'PCs_chr' + str(chrom) + '.csv')
  
    


if __name__ == "__main__":
    main()
    




