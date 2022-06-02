# Imports
import sys, getopt
import json
import numpy as np
import pandas as pd
import scipy.stats as ss
import networkx as nx
import OmicsIntegrator as oi
import tqdm
import time
import csv
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


def get_prizes(transition, data_dir, top_n):
    """ Gets prizes for differentially expressed genes (t -> t+1) and highly expressed TFs (at time t)
    
    Args:
        transition: (string) information about which transition to use e.g. "0_10"
        data_dir: (string) directory that contains the data files
        top_n: (string) amount of DE genes that should be prized
    
    Returns:
        df with the protein prizes
    """
    # Load DE data and filter df to the specified transition (t -> t+1)
    de_genes = pd.read_csv(data_dir+'de_data/fc_'+top_n+'.csv')
    de_genes = de_genes[de_genes['transition'] == "fc_" + transition]
    de_genes.columns = ['transition', 'name', 'FPKM_t0', 'FPKM_t1', 'log2_fc', 'prize']
    de_genes = de_genes[['name', 'prize']]
    
    return(de_genes)


def get_prizes_combi(tr0, tr1, data_dir, interactome, top_n, cell_type):
    """ Gets prizes for differentially expressed genes (t -> t+1) and highly expressed TFs (at time t)
    
    Args:
        tr0: (string) transition from t to t+1 e.g. "0_10"
        tr1: (string) transition from t+1 to t+2 e.g. "10_20"
        data_dir: (string) directory that contains the data files
        top_n: (string) amount of DE genes that should be prized
    
    Returns:
        df with the protein prizes
    """
    # Load DE data for the two transitions
    de_genes_0 = get_prizes(tr0, data_dir, top_n)
    de_genes_0['name'] = de_genes_0['name'] + "_tr0"
    de_genes_1 = get_prizes(tr1, data_dir, top_n)
    de_genes_1['name'] = de_genes_1['name'] + "_tr1"
    
    # Filter DE genes of transition 1 to the ones targeted by the TFs
    de_genes_1 = de_genes_1[de_genes_1['name'].isin(interactome['protein2'])]
    
    # Prize TFs based on their expression at time t+1
    tfs = pd.read_csv(data_dir+'tf_data/TF_prizes_' +cell_type +'.csv')
    tfs = tfs[tfs['age'] == int(tr0.split("_")[1])]
    tfs.columns = ['name', 'age', 'FPKM', 'log2_FPKM', 'prize']
    tfs = tfs[['name', 'prize']]
    tfs['name'] = tfs['name'] + "_tr0"
    
    # Combine all prized genes
    prizes = pd.concat([de_genes_0, de_genes_1, tfs])
    
    return(prizes)


def get_PPI(data_dir, tr1, cell_type, top_n):
    """ Gets PPI from STRING (subsetted by the augmented forests per transition) and TF-target interactions
    
    Args:
        data_dir: (string) directory that contains the data files
        tr1: (string) transition t+1 -> t+2
        cell_type: (string) whether to use only tf-targets from fibroblasts or all cell_types
        top_n: (string) amount of DE genes that should be prized
    
    Returns:
        pandas df with all protein-protein interactions
    """
    
    # Load STRING PPI
    PPI = pd.read_csv(data_dir + 'ppi_data/PPI_string_physical.csv', sep = '\t')
    
    # Transition 0: Add suffix
    interactome = PPI.copy()
    interactome['protein1'] = interactome['protein1'] + "_tr0"
    interactome['protein2'] = interactome['protein2'] + "_tr0"
    
    # Load TF-target interactions (fibroblasts specific or all cell-types)
    if cell_type == "fibroblasts":
        tf_targets = pd.read_csv(data_dir + 'tf_data/tftarget_full_processed.csv', sep = '\t')
        tf_targets = tf_targets[['tf', 'target_gene']].drop_duplicates()
    else:
        tf_targets = pd.read_csv(data_dir + 'tf_data/tf-target-information.txt', sep = '\t')
        tf_targets = tf_targets[['TF', 'target']].drop_duplicates()
    tf_targets.columns = ['protein1', 'protein2']
    tf_targets['cost'] = 0
                                                       
    # filter TF-target interactions to the nodes included in the network
    tf_targets['protein1'] = tf_targets['protein1'] + "_tr0"  
    tf_targets['protein2'] = tf_targets['protein2'] + "_tr1"                                                   
    tf_targets = tf_targets[tf_targets['protein1'].isin(interactome['protein1'])]
    de_genes = get_prizes(tr1, data_dir, top_n)
    de_genes['name'] = de_genes['name'] + "_tr1"
    tf_targets = tf_targets[tf_targets['protein2'].isin(de_genes['name'])]                                                   
    
    return(interactome, tf_targets)
    

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


def run_pcst_combi(step, graph_params, data_dir, save_dir, cell_type, top_n):
    """ Runs pcst for the combination of two transitions
    
    Args:
        step: (string) information about which time step to use e.g. "0_10_20"
        graph_params: (dict) graph hyperparameters
        data_dir: (string) directory that contains the data files
        save_dir: (string) directory that contains the processed prize data
        cell_type: (string) whether to use only tf-targets from fibroblasts or all cell_types
        top_n: (string) amount of DE genes that should be prized
    
    Returns:
        augmented_forest
    """
    # Get names of the transitions
    tr0 = step.split("_")[0]+"_"+step.split("_")[1]
    tr1 = step.split("_")[1]+"_"+step.split("_")[2]
    
    # Get PPI: STRING interactome & TF-target interactions
    STRING, tf_targets = get_PPI(data_dir, tr1, cell_type, top_n)
    interactome = pd.concat([STRING, tf_targets])
    interactome.to_csv(data_dir + 'ppi_data/PPI_string_TFs.csv', sep='\t')
    interactome_file_name = data_dir + 'ppi_data/PPI_string_TFs.csv'    
    
    # Get prizes
    prizes_data = get_prizes_combi(tr0, tr1, data_dir, interactome, top_n, cell_type)
    prizes_data.to_csv(save_dir+'de_terminals_'+step+'.tsv', header=True, index=None, sep='\t', quoting = csv.QUOTE_NONE, escapechar = '\t')
    prize_file_name = save_dir+'de_terminals_'+step+'.tsv' 
    
    augmented_forest = run_pcst(graph_params, interactome_file_name, prize_file_name)
    
    return(augmented_forest)


def compare_networks(net_dict, data_dir, fig_dir, save_dir):
    """ Gets statistics for a list of networks
    
    Args:
        net_dict: (dictionary) networks to compare
        data_dir: (string) name of the data directory
        fig_dir: (string) name of the figure directory
        save_dir: (string) name of the save directory
    
    Returns:
        Pandas DataFrame with the comparison results and dict of the target counts per TF
    """
    results = pd.DataFrame(columns = ['n_nodes','n_edges', 'n_prized_nodes', 
                                      'n_Steiner_nodes', 'n_TFs', 'n_incl_TFs', 
                                      'n_prized_TFs', 'n_significant_TFs', 'incl_TFs', 'significant_TFs'])
    target_counts_dict = {}
    
    fig, axs = plt.subplots(nrows=2, ncols=int(np.ceil(len(net_dict) / 2)), figsize=(20, 7))
    plt.subplots_adjust(hspace=0.5)
    
    for (name, net), ax in zip(net_dict.items(), axs.ravel()):
        top_n, cell_type, step = name.split("-")
        # Get names of the transitions
        tr0 = step.split("_")[0]+"_"+step.split("_")[1]
        tr1 = step.split("_")[1]+"_"+step.split("_")[2]

        # Number of nodes and edges
        n_nodes = net.number_of_nodes()
        n_edges = net.number_of_edges()
        
        # Save nodes for analysis on STRING website (supports only networks up to 2000 nodes)
        nodes = pd.DataFrame(net.nodes)[0].str.split("_", expand=True)
        nodes.columns = ['protein', 'transition']
        nodes['protein'].to_csv(save_dir+"nodes_"+cell_type+"_"+top_n+"_step_"+step+".txt", 
                                sep = "\n", index = False, header = False)
        
        # Number of prized nodes and Steiner nodes
        network_df = oi.get_networkx_graph_as_dataframe_of_nodes(net)
        n_included_terminals = np.sum(network_df['terminal'])
        n_Steiner = n_nodes-n_included_terminals
        
        # TF-target interactions
        if cell_type == "fibroblasts":
            tf_targets = pd.read_csv(data_dir + 'tf_data/tftarget_full_processed.csv', sep = '\t')
            tf_targets = tf_targets[['tf', 'target_gene']].drop_duplicates()
        else:
            tf_targets = pd.read_csv(data_dir + 'tf_data/tf-target-information.txt', sep = '\t')
            tf_targets = tf_targets[['TF', 'target']].drop_duplicates()
        tf_targets.columns = ['protein1', 'protein2']
        
        # Included TFs
        STRING, tf_targets_subset = get_PPI(data_dir, tr1, cell_type, top_n)
        TFs = set(tf_targets_subset['protein1'])
        incl_TFs = set(net.nodes()).intersection(TFs)
        
        # Number of targets in the genome
        proteins = [prot[:-4] for prot in set(STRING['protein1']).union(set(STRING['protein2']))]
        tf_targets = tf_targets[tf_targets['protein1'].isin(proteins)]
        tf_targets = tf_targets[tf_targets['protein2'].isin(proteins)]
        tf_targets['protein1'] = tf_targets['protein1'] + "_tr0"  
        tf_targets['protein2'] = tf_targets['protein2'] + "_tr1" 
        TF_counts = pd.DataFrame(tf_targets['protein1'].value_counts())
        
        # Number of prized TFs
        network_df = oi.get_networkx_graph_as_dataframe_of_nodes(net)
        TF_df = network_df[network_df.index.isin(TFs)]
        n_prized_TFs = TF_df[TF_df['terminal'] == True].shape[0]
        
        # Significant TFs
        incl_tf_targets = tf_targets_subset[tf_targets_subset['protein1'].isin(incl_TFs)]
        incl_tf_targets = incl_tf_targets[incl_tf_targets['protein2'].isin(net.nodes)]
        target_counts = pd.DataFrame(incl_tf_targets['protein1'].value_counts()).join(TF_counts, lsuffix = "_incl_targets",
                                                                                      rsuffix = "_all_targets")
        target_counts['all_targets_percent'] = target_counts['protein1_all_targets'] / 18384
        n = nodes[nodes['transition'] == "tr1"].shape[0]
        target_counts['p_value'] = target_counts.apply(lambda x: ss.binomtest(k = int(x['protein1_incl_targets']), 
                                                n = n, 
                                                p = x['all_targets_percent'], 
                                                alternative= 'greater').pvalue, axis=1)
        
        # Scatterplot with coloring according to significance
        target_counts['significance'] = np.where(target_counts['p_value'] < 0.05, 'p < 0.05', 'p > 0.05')
        
        sns.scatterplot(x='protein1_all_targets', y='protein1_incl_targets', data=target_counts, 
                        hue='significance', hue_order = ['p > 0.05', 'p < 0.05'], ax = ax)
        ax.set_title(name)
        ax.set_xlabel('Number of targets in the genome')
        ax.set_ylabel('Number of targets in the result network')       

        significant_TFs = target_counts[target_counts['significance'] == 'p < 0.05'].index.tolist()
        results.loc[name] = [n_nodes, n_edges, n_included_terminals, n_Steiner, 
                             len(TFs), len(incl_TFs), n_prized_TFs, len(significant_TFs), list(incl_TFs), significant_TFs]
        target_counts_dict[name] = target_counts
    plt.savefig(fig_dir + 'significant_TFs.png')
    plt.show()
    return(results, target_counts_dict)


def get_net_dir_all_stages(save_dir, cell_type, top_n):
    """ Gets directory of networks to compare
    
    Args:
        save_dir: (string) name of the save directory
        cell_type: (string) whether to use only tf-targets from fibroblasts or all cell_types
        top_n: (string) amount of DE genes that should be prized
    
    Returns:
        Dictionary including the augmented forests
    """
    steps = ["0_10_20", "10_20_30", "20_30_40", "30_40_50", "40_50_60", "50_60_70", "60_70_80", "70_80_90"]
    net_dir = {}
    
    for step in steps:
        augmented_forest = pickle.load(open(save_dir + "network_"+cell_type+"_"+top_n+"_step_"+step+".pickle", "rb"))
        net_dir[top_n+"-"+cell_type+ "-" + step] = augmented_forest
    
    return(net_dir)


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
    steps_list = config['steps']
    graph_params = config['graph_params']
    cell_type = config['cell_type']
    top_n = config['top_n']
    
    # Run PCST combined for two following transitions
    print("Run combined pcst for each step:")
    for step in steps_list:
        print("Step " + step)
        
        augmented_forest = run_pcst_combi(step, graph_params, data_dir, save_dir, cell_type, top_n)
        oi.output_networkx_graph_as_pickle(augmented_forest, 
                                           filename= save_dir + "network_"+cell_type+"_"+top_n+"_step_"+step+".pickle")
    

if __name__ == "__main__":
    main()
    

    