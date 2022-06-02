# Imports
import sys, getopt
import json
import numpy as np
import pandas as pd
import networkx as nx
import OmicsIntegrator as oi
import tqdm
import time
import csv
import pickle


def get_prizes(transition, data_dir):
    """ Gets prizes for differentially expressed genes (t -> t+1) and highly expressed TFs (at time t)
    
    Args:
        transition: (string) information about which transition to use e.g. "0_10"
        data_dir: (string) directory that contains the data files
    
    Returns:
        df with the protein prizes
    """
    # Load DE data and filter df to the specified transition (t -> t+1)
    de_genes = pd.read_csv(data_dir+'de_data/fc_top10.csv')
    de_genes = de_genes[de_genes['transition'] == "fc_" + transition]
    de_genes.columns = ['transition', 'name', 'FPKM_t0', 'FPKM_t1', 'log2_fc', 'prize']
    de_genes = de_genes[['name', 'prize']]
    
    return(de_genes)


def get_prizes_combi(tr0, tr1, data_dir):
    """ Gets prizes for differentially expressed genes (t -> t+1) and highly expressed TFs (at time t)
    
    Args:
        tr0: (string) transition from t to t+1 e.g. "0_10"
        tr1: (string) transition from t+1 to t+2 e.g. "10_20"
        data_dir: (string) directory that contains the data files
    
    Returns:
        df with the protein prizes
    """
    # Load DE data for the two transitions
    de_genes_0 = get_prizes(tr0, data_dir)
    de_genes_0['name'] = de_genes_0['name'] + "_tr0"
    de_genes_1 = get_prizes(tr1, data_dir)
    de_genes_1['name'] = de_genes_1['name'] + "_tr1"
    de_genes = pd.concat([de_genes_0, de_genes_1])
    
    # Load TF data and filter df to time t + 1
    tfs = pd.read_csv(data_dir+'tf_data/TF_prizes.csv')
    tfs = tfs[tfs['age'] == int(tr0.split("_")[1])]
    tfs.columns = ['name', 'age', 'FPKM', 'log2_FPKM', 'prize']
    tfs = tfs[['name', 'prize']]
    tfs['name'] = tfs['name'] + "_tr0"
    
    # Combine DE prizes and TF prizes and save results
    prizes_data = pd.concat([de_genes, tfs])
    
    return(prizes_data)


def get_PPI(data_dir, augmented_forest_0, augmented_forest_1):
    """ Gets PPI from STRING (subsetted by the augmented forests per transition) and TF-target interactions
    
    Args:
        data_dir: (string) directory that contains the data files
        augmented_forest_0: augmented forest solution for transition 0
        agmented_forest_1: augmented forest solution for transition 1
    
    Returns:
        pandas df with all protein-protein interactions
    """
    
    # Load STRING PPI
    PPI = pd.read_csv(data_dir + 'ppi_data/PPI_string_processed.csv', sep = '\t')
    
    # Transition 0: Subset PPI to proteins in the augmented forest or TFs and add suffix
    PPI_0 = PPI.copy()
    PPI_0 = PPI_0[PPI_0['protein1'].isin(augmented_forest_0.nodes)]
    PPI_0 = PPI_0[PPI_0['protein2'].isin(augmented_forest_0.nodes)]
    PPI_0['protein1'] = PPI_0['protein1'] + "_tr0"
    PPI_0['protein2'] = PPI_0['protein2'] + "_tr0"
    
    # Transition 1: Subset PPI to proteins in the pcsf solution and add suffix
    PPI_1 = PPI.copy()
    PPI_1 = PPI_1[PPI_1['protein1'].isin(augmented_forest_1.nodes)]
    PPI_1 = PPI_1[PPI_1['protein2'].isin(augmented_forest_1.nodes)]
    PPI_1['protein1'] = PPI_1['protein1'] + "_tr1"
    PPI_1['protein2'] = PPI_1['protein2'] + "_tr1"
                                                       
    # Combine results of the two transitions
    interactome = pd.concat([PPI_0, PPI_1]) 
    
    # Load TF-target interactions
    tf_targets = pd.read_csv(data_dir + 'tf_data/tftarget_full_processed.csv', sep = '\t')
    tf_targets = tf_targets[['tf', 'target_gene']].drop_duplicates()
    tf_targets.columns = ['protein1', 'protein2']
    tf_targets['cost'] = np.median(interactome['cost'])
                                                       
    # filter TF-target interactions to the ones included in STRING
    # since hTF also includes non-protein-coding genes
    tf_targets['protein1'] = tf_targets['protein1'] + "_tr0"  
    tf_targets['protein2'] = tf_targets['protein2'] + "_tr1"                                                   
    proteins = set(interactome['protein1']).union(set(interactome['protein2']))
    tf_targets = tf_targets[tf_targets['protein1'].isin(proteins)]
    tf_targets = tf_targets[tf_targets['protein2'].isin(proteins)]                                                   
    
    # Combine STRING with TF-targets and save results
    full_PPI = pd.concat([interactome, tf_targets])
    return(full_PPI)
    

def run_pcst(graph_params, interactome_file_name, prize_file_name):
    """ Runs pcst 
    
    Args:
        graph_params: (dict) graph hyperparameters
        interactome_file_name: (string) path to the interactome file
        prize_file_name: (string) path to the prize file name
    
    Returns:
        augmented_forest
    """ 
    
    # Build graph
    print("Build the graph")
    graph = oi.Graph(interactome_file_name, graph_params)
    graph.prepare_prizes(prize_file_name)
    
    # Run PCST algorithm
    print("Run PCST")
    vertex_indices, edge_indices = graph.pcsf()
    forest, augmented_forest = graph.output_forest_as_networkx(vertex_indices, edge_indices)
    print("Done")
    
    return(augmented_forest)


def run_pcst_combi(step, augmented_forest_0, augmented_forest_1, graph_params, data_dir, save_dir):
    """ Runs pcst for the combination of two transitions
    
    Args:
        step: (string) information about which time step to use e.g. "0_10_20"
        graph_params: (dict) graph hyperparameters
        data_dir: (string) directory that contains the data files
        save_dir: (string) directory that contains the processed prize data
    
    Returns:
        augmented_forest
    """
    # Get names of the transitions
    tr0 = step.split("_")[0]+"_"+step.split("_")[1]
    tr1 = step.split("_")[1]+"_"+step.split("_")[2]
    
    # Get prizes
    prizes_data = get_prizes_combi(tr0, tr1, data_dir)
    prizes_data.to_csv(save_dir+'de_tf_terminals_'+step+'.tsv', header=True, index=None, sep='\t', quoting = csv.QUOTE_NONE, escapechar = '\t')
    prize_file_name = save_dir+'de_tf_terminals_'+step+'.tsv' 
    
    # Get PPI: STRING interactome subsetted by augmented forests & TF-target interactions
    interactome = get_PPI(data_dir, augmented_forest_0, augmented_forest_1)
    interactome.to_csv(data_dir + 'ppi_data/PPI_string_TFs.csv', sep='\t')
    interactome_file_name = data_dir + 'ppi_data/PPI_string_TFs.csv'     
    
    augmented_forest = run_pcst(graph_params, interactome_file_name, prize_file_name)
    
    return(augmented_forest)


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
    config = json.load(open(config_filename))
    data_dir = config['DATA_DIR']
    save_dir = config['SAVE_DIR']
    transition_list = config['transitions']
    steps_list = config['steps']
    graph_params = config['graph_params']
    
    # PCST for each transition
    print("Run pcst for each transition:")
    for transition in transition_list:
        print("Transition "+transition)
        
        # Prize data
        prizes_data = get_prizes(transition, data_dir)
        prizes_data.to_csv(save_dir+'de_terminals_'+transition+'.tsv', header=True, 
                           index=None, sep='\t', quoting = csv.QUOTE_NONE, escapechar = '\t')
        prize_file_name = save_dir+'de_terminals_'+transition+'.tsv'
        
        # PPI data
        interactome_file_name = data_dir + 'ppi_data/PPI_string_processed.csv'
        
        # PCST
        augmented_forest = run_pcst(graph_params, interactome_file_name, prize_file_name)
        oi.output_networkx_graph_as_pickle(augmented_forest, filename= save_dir + "network_"+transition+".pickle")
        
    # Run PCST combined for two following transitions
    print("Run combined pcst for each step:")
    for step in steps_list:
        print("Step " + step)
        
        # Get names of the transitions
        tr0 = step.split("_")[0]+"_"+step.split("_")[1]
        tr1 = step.split("_")[1]+"_"+step.split("_")[2]
    
        # load augmented forest corresponding to the two transitions
        augmented_forest_0 = pickle.load(open(save_dir + "network_" + tr0 + ".pickle", "rb"))
        augmented_forest_1 = pickle.load(open(save_dir + "network_" + tr1 + ".pickle", "rb"))
        
        augmented_forest = run_pcst_combi(step, augmented_forest_0, augmented_forest_1, 
                                          graph_params, data_dir, save_dir)
        oi.output_networkx_graph_as_pickle(augmented_forest, filename= save_dir + "network_"+step+".pickle")
    

if __name__ == "__main__":
    main()
    

    