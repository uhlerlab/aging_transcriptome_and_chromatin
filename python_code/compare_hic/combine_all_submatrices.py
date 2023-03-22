import sys, getopt
import json
import os, os.path
import numpy as np 
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
import time

def get_submatrix_scores(sample, hic_r1, hic_r2, hic_dir, threshold, chr1, chr2, resol):
    '''
    Calculates the submatrix scores for both replicates
    Args:
        sample: (string) sample name that is used for identified submatrices
        hic_r1: (pd DataFrame) normalized HiC values for chr1, chr2 for replicate 1
        hic_r2: (pd DataFrame) normalized HiC values for chr1, chr2 for replicate 2
        hic_dir: (string) directory of the processed hic data files
        threshold: (int) threshold of the LAS algorithm
        chr1: (int) chromosome 1
        chr2: (int) chromosome 2
        resol: (int) HiC resolution
    Returns:
        A pd DataFrame with the submatrices positions and scores in both replicates
    '''
    # load positions of submatrices for specified sample
    fname = hic_dir + sample + '/LAS-'+ str(threshold) + '/intermingling_regions.chr' + str(chr1) + '_chr' + str(chr2) + '.avg_filt.csv'
    submatrices = pd.read_csv(fname, index_col = 0)
    
    results = pd.DataFrame({'score_r1': [], 'score_r2': []})

    # iterate over all identified submatrices
    for ix in range(submatrices.shape[0]):

        # get rows and cols for submatrix
        rows, cols = get_rows_cols(submatrices, ix, resol)

        # calculate submatrix score for both replicates
        score_r1 = get_score(hic_r1, rows, cols)
        score_r2 = get_score(hic_r2, rows, cols)
        
        results = pd.concat([results, 
                             pd.DataFrame({'score_r1': [score_r1], 'score_r2': [score_r2]})], ignore_index = True)
        
    # add submatrix scores to the file with the submatrix positions
    submatrices['score_r1'] = results['score_r1']
    submatrices['score_r2'] = results['score_r2']
    return(submatrices)

        
def get_rows_cols(submatrices, ix, resol):
    '''
    Returns a list of rows and columns that are part of the submatrix
    Args:
        submatrices: (pd DataFrame) df with the start and stop positions of all submatrices for a chr pair
        ix: (int) index of the submatrix that should be considered
        resol: (int) HiC resolution
    Returns:
        A list of rows and columns that define a submatrix
    '''
    start_row = int(submatrices.loc[ix, 'start row'])
    stop_row = int(submatrices.loc[ix, 'stop row'])
    rows = [row for row in range(start_row, stop_row + resol, resol)]
    start_col = int(submatrices.loc[ix, 'start col'])
    stop_col = int(submatrices.loc[ix, 'stop col'])
    cols = [str(col) for col in range(start_col, stop_col + resol, resol)]
    return rows, cols


def get_score(hic, rows, cols):
    '''
    Returns the score for a submatrix
    Args:
        hic: (pd DataFrame) normalized HiC values for chr1, chr2
        rows: (list) list of rows for the submatrix
        cols: (list) list of cols for the submatrix
    Returns:
        Score for the submatrix
    '''
    submatrix = hic.loc[rows, cols]
    num_rows, num_cols = submatrix.shape
    score = np.sqrt(num_rows*num_cols)*np.average(submatrix)
    return score
    

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
    save_dir = config['SAVE_DIR']
    resol_str = config['HIC_RESOLUTION_STR']
    resol = config['HIC_RESOLUTION']
    chr_list = config['chrs']
    sample_r1 = config['SAMPLE_R1']
    sample_r2 = config['SAMPLE_R2']
    threshold = config['LAS_THRESHOLD']
    
    chr_pairs = list(itertools.combinations(chr_list, 2))
    
    print("Replicates: ", sample_r1, " and ", sample_r2)
    print("Calculate submatrix scores for both replicates on all chromosome pairs: ")
    
    all_scores = pd.DataFrame()
    
    for pair in tqdm(chr_pairs):
        time.sleep(0.01)
        chr1, chr2 = pair

        # load HiC matrices for both replicates
        fname1 = hic_dir + sample_r1 + '/final_BP' + str(resol) + '_intraKR_interINTERKR/hic_chr' + str(chr1) + '_chr' + str(chr2) + '_zscore.txt'
        fname2 = hic_dir + sample_r2 + '/final_BP' + str(resol) + '_intraKR_interINTERKR/hic_chr' + str(chr1) + '_chr' + str(chr2) + '_zscore.txt'
        hic_r1 = pd.read_csv(fname1, index_col = 0)
        hic_r2 = pd.read_csv(fname2, index_col = 0)
        
        scores = pd.DataFrame()
        
        # iterate over found submatrices in all samples
        for sample in ['Young_B1R1', 'Young_B1R2', 'Old_B1R1', 'Old_B2R2']:
            # get scores for the submatrices in the two replicates
            scores_sample = get_submatrix_scores(sample, hic_r1, hic_r2, hic_dir, threshold, chr1, chr2, resol)
            scores = pd.concat([scores, scores_sample], ignore_index = True).drop_duplicates()
        
        # filter to submatrices that are above the threshold in both replicates
        scores_filtered = scores[scores['score_r1'] > threshold]
        scores_filtered = scores_filtered[scores_filtered['score_r2'] > threshold]
        
        # save combined results
        LAS_dir = save_dir + sample_r1.split("_")[0] + '_combi/LAS-'+ str(threshold) 
        fname = LAS_dir  + '/intermingling_regions.chr' + str(chr1) + '_chr' + str(chr2) +'.avg_filt.csv'
        scores_filtered.to_csv(fname)
        
        # create df with unfiltered scores over all chr pairs
        scores['chr1'] = chr1
        scores['chr2'] = chr2
        all_scores = pd.concat([all_scores, scores], ignore_index = True)
        
    all_scores = all_scores[(all_scores.score_r1 > threshold) | (all_scores.score_r2 > threshold)]
    all_scores.to_csv(LAS_dir + '/intermingling_regions_all_chr_unfiltered.csv', index = False)
        
        
        
if __name__ == "__main__":
    main()

