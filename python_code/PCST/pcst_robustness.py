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
import gseapy
from gseapy.plot import barplot, dotplot


def get_prizes(group, data_dir):
    """ Gets prizes for differentially expressed genes 
    
    Args:
        group: (string) information about which DE genes to use
        data_dir: (string) directory that contains the data files
    
    Returns:
        df with the protein prizes
    """
    # Load DE data and filter df to the specified transition (t -> t+1)
    de_genes = pd.read_csv(data_dir+'de_data/DE_'+group+'.csv')
    de_genes.columns = ['name', 'fc', 'padj', 'prize', 'updown']
    #if group == "Group5":
    #    de_genes = de_genes[de_genes['updown'] == 'up']
    de_genes = de_genes[['name', 'prize']]
    
    return(de_genes)


def get_prizes_combi(tr0, tr1, data_dir, save_dir, interactome, p_thr, cell_type, step_list, design):
    """ Gets prizes for differentially expressed genes (t -> t+1) and highly expressed TFs (at time t)
    
    Args:
        tr0: (string) transition from t to t+1 e.g. "0_10"
        tr1: (string) transition from t+1 to t+2 e.g. "10_20"
        data_dir: (string) directory that contains the data files
        save_dir: (string) save directory 
        interactome: (pd DataFrame) STRING interactions to filter for included proteins
        p_thr: (string) threshold for the p-values to prize DE genes
        cell_type: (string) whether to use only tf-targets from fibroblasts or all cell_types
        design: (int) which design to choose (1-3)
    
    Returns:
        df with the protein prizes
    """
    # Load DE data for the two transitions
    de_genes_0 = get_prizes('Group1', data_dir)
    de_genes_1 = get_prizes('Group5', data_dir)
    
    # Filter DE genes to the ones included in STRING 
    proteins = [prot[:-4] for prot in set(interactome['protein1']).union(set(interactome['protein2']))]
    de_genes_0 = de_genes_0[de_genes_0['name'].isin(proteins)]
    de_genes_1 = de_genes_1[de_genes_1['name'].isin(proteins)]
          
    # Add suffix
    de_genes_0['name'] = de_genes_0['name'] + "_tr0"
    de_genes_1['name'] = de_genes_1['name'] + "_tr1"
    
    prizes = pd.concat([de_genes_0, de_genes_1])
    return(prizes)


def get_TF_targets(data_dir, cell_type):
    """ Loads TF-target interactions from hTFtarget
    
    Args:
        data_dir: (string) directory that contains the data files
        cell_type: (string) whether to use only tf-targets from fibroblasts or all cell_types
    
    Returns:
        pandas df with all TF-target interactions
    """
    if cell_type == "fibroblasts":
        tf_targets = pd.read_csv(data_dir + 'tf_data/tftarget_full_processed.csv', sep = '\t')
        tf_targets = tf_targets[['tf', 'target_gene']].drop_duplicates()
    else:
        tf_targets = pd.read_csv(data_dir + 'tf_data/tf-target-information.txt', sep = '\t')
        tf_targets = tf_targets[['TF', 'target']].drop_duplicates()
    tf_targets.columns = ['protein1', 'protein2']
    return(tf_targets)


def get_PPI(step, data_dir, save_dir, cell_type, p_thr, step_list, design):
    """ Gets PPI from STRING (subsetted by the augmented forests per transition) and TF-target interactions
    
    Args:
        step: (string) information about which time step to use e.g. "0-9_10-19_20-29"
        data_dir: (string) directory that contains the data files
        save_dir: (string) directory that contains the processed prize data
        cell_type: (string) whether to use only tf-targets from fibroblasts or all cell_types
        p_thr: (string) threshold for the p-values to prize DE genes
        step_list: (list) list containing all the time steps
        design: (int) which design to choose (1-3)
    
    Returns:
        pandas df with all protein-protein interactions
    """
    # Get names of the transitions
    tr0 = step.split("_")[0]+"_"+step.split("_")[1]
    tr1 = step.split("_")[1]+"_"+step.split("_")[2]
    
    # Load STRING PPI
    PPI = pd.read_csv(data_dir + 'ppi_data/PPI_string_physical.csv', sep = '\t')
    
    # Filter for active genes in transition 0 and add suffix for transition 0
    interactome_tr0 = PPI.copy()
    gene_activity = pd.read_csv(save_dir + 'gene_activity.csv')
    gene_activity = gene_activity[gene_activity['transition'] == 'fc_' + tr0]
    interactome_tr0 = interactome_tr0[(interactome_tr0['protein1'].isin(gene_activity['gene']) &
                                       interactome_tr0['protein2'].isin(gene_activity['gene']))]
    interactome_tr0['protein1'] = interactome_tr0['protein1'] + "_tr0"
    interactome_tr0['protein2'] = interactome_tr0['protein2'] + "_tr0"
    
    # Load TF-target interactions (fibroblasts specific or all cell-types)
    tf_targets = get_TF_targets(data_dir, cell_type)
    tf_targets['cost'] = np.min(PPI['cost'])
    print("TF target cost: ", np.min(PPI['cost']))
                                                       
    # filter TF-target interactions to the nodes included in the network
    tf_targets['protein1'] = tf_targets['protein1'] + "_tr0"  
    tf_targets['protein2'] = tf_targets['protein2'] + "_tr1"                                                   
    tf_targets = tf_targets[tf_targets['protein1'].isin(interactome_tr0['protein1'])]
    
    nodes_right = get_prizes_combi(tr0, tr1, data_dir, save_dir, interactome_tr0, 
                                   p_thr, cell_type, step_list, design)['name'].tolist()
    nodes_right = [node for node in nodes_right if node.endswith('_tr1')]
    tf_targets = tf_targets[tf_targets['protein2'].isin(nodes_right)]
    
    return(interactome_tr0, tf_targets)
    

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


def run_pcst_combi(step, graph_params, data_dir, save_dir, cell_type, p_thr, step_list, design):
    """ Runs pcst for the combination of two transitions
    
    Args:
        step: (string) information about which time step to use e.g. "0-9_10-19_20-29"
        graph_params: (dict) graph hyperparameters
        data_dir: (string) directory that contains the data files
        save_dir: (string) directory that contains the processed prize data
        cell_type: (string) whether to use only tf-targets from fibroblasts or all cell_types
        p_thr: (string) threshold for the p-values to prize DE genes
        step_list: (list) list containing all the time steps
        design: (int) which design to choose (1-3)
    
    Returns:
        augmented_forest
    """
    # Get names of the transitions
    tr0 = step.split("_")[0]+"_"+step.split("_")[1]
    tr1 = step.split("_")[1]+"_"+step.split("_")[2]
    
    # Get PPI: STRING interactome & TF-target interactions
    interactome_tr0, tf_targets = get_PPI(step, data_dir, save_dir, cell_type, p_thr, step_list, design)
    interactome = pd.concat([interactome_tr0, tf_targets])
    interactome.to_csv(data_dir + 'ppi_data/PPI_string_TFs.csv', sep='\t')
    interactome_file_name = data_dir + 'ppi_data/PPI_string_TFs.csv'    
    
    # Get prizes
    prizes_data = get_prizes_combi(tr0, tr1, data_dir, save_dir, interactome_tr0, p_thr, cell_type, step_list, design)
    prizes_data.to_csv(save_dir+'de_terminals_'+step+"design_" + str(design) +'.tsv', 
                       header=True, index=None, sep='\t', quoting = csv.QUOTE_NONE, escapechar = '\t')
    prize_file_name = save_dir+'de_terminals_'+step+"design_" + str(design) +'.tsv' 
    
    augmented_forest = run_pcst(graph_params, interactome_file_name, prize_file_name)
    
    return(augmented_forest)


def save_net_html(net, incl_TFs, save_dir, name, design):
    """ Saves the net with node attributes as interactive html
    
    Args:
        net: augmented forest solution from OmicsIntegrator2
        incl_TFs: (set) TFs that are included in the network
        save_dir: (string) directory that contains the processed prize data
        name: (string) name including the p_thr (1%, 5% or 10%), cell_type (fibroblasts or allTFs) and step (e.g. 0_10_20)
        design: (int) which design to choose (1-3)
    
    Returns:
        nodes: pd DataFrame including the category for each node
    """
    # Get nodes and assign attributes (Steiner node, DE_tr0, TF, DE_tr1)
    network_df = oi.get_networkx_graph_as_dataframe_of_nodes(net)
    nodes = pd.DataFrame(net.nodes)[0].str.split("_", expand=True)
    nodes.columns = ['protein', 'transition']
    nodes['name'] = nodes['protein'] + "_" + nodes['transition']
    nodes['category'] = 'Steiner node'
    nodes['category'][nodes['name'].isin(network_df.loc[network_df['terminal'] == True,:].index.tolist())] = "DE_tr0"
    nodes['category'][nodes['transition'] == 'tr1'] = "DE_tr1"
    nodes['TF'] = "No bridge TF"
    nodes['TF'][nodes['name'].isin(incl_TFs)] = "Bridge TF"
    group_dict = {node: nodes.loc[nodes['name'] == node, 'category'].values[0] for node in nodes['name']}
    TF_dict = {node: nodes.loc[nodes['name'] == node, 'TF'].values[0] for node in nodes['name']}
    label_dict = {node: '' for node in nodes['name']}
    nx.set_node_attributes(net, group_dict, name='groups')
    nx.set_node_attributes(net, TF_dict, name='TFs')
    nx.set_node_attributes(net, label_dict, name= 'labels')
    
    # Save the results
    p_thr, cell_type, step = name.split(".")
    oi.output_networkx_graph_as_interactive_html(net, filename=save_dir+"net_robustness_G1_G45.html")
    
    return(group_dict)


def compare_networks(net_dict, data_dir, fig_dir, save_dir, step_list, TFs_with_targets, design):
    """ Gets statistics for a list of networks
    
    Args:
        net_dict: (dictionary) networks to compare
        data_dir: (string) name of the data directory
        fig_dir: (string) name of the figure directory
        save_dir: (string) name of the save directory
        step_list: (list) list containing all the time steps
        TFs_with_targets: (boolean) if true only the TFs that target genes on the right are included
        design: (int) which design to choose (1-3)
    
    Returns:
        Pandas DataFrame with the comparison results and dict of the target counts per TF
    """
    results = pd.DataFrame(columns = ['n_nodes','n_edges', 'n_incl_terminals', 'percent_incl_terminals',
                                      'n_Steiner_nodes', 'n_TFs', 'n_incl_TFs', 
                                      'n_prized_TFs', 'n_significant_TFs', 'incl_TFs', 
                                      'significant_TFs', 'Steiner_nodes'])
    target_counts_dict = {}
    
    for name, net in net_dict.items():       
        p_thr, cell_type, step = name.split(".")
        
        # Get names of the transitions
        tr0 = step.split("_")[0]+"_"+step.split("_")[1]
        tr1 = step.split("_")[1]+"_"+step.split("_")[2]

        # Number of nodes and edges
        n_nodes = net.number_of_nodes()
        n_edges = net.number_of_edges()
        
        # Get nodes
        nodes = pd.DataFrame(net.nodes)[0].str.split("_", expand=True)
        nodes.columns = ['protein', 'transition']

        # Number of prized nodes and Steiner nodes
        network_df = oi.get_networkx_graph_as_dataframe_of_nodes(net)
        n_included_terminals = np.sum(network_df['terminal'])
        n_Steiner = n_nodes-n_included_terminals
        
        # TF-target interactions
        tf_targets = get_TF_targets(data_dir, cell_type)
        
        # Number of targets in the genome
        STRING, tf_targets_subset = get_PPI(step, data_dir, save_dir, cell_type, p_thr, step_list, design)
        proteins = [prot[:-4] for prot in set(STRING['protein1']).union(set(STRING['protein2']))]
        tf_targets = tf_targets[tf_targets['protein1'].isin(proteins)]
        tf_targets = tf_targets[tf_targets['protein2'].isin(proteins)]
        tf_targets['protein1'] = tf_targets['protein1'] + "_tr0"  
        tf_targets['protein2'] = tf_targets['protein2'] + "_tr1" 
        TF_counts = pd.DataFrame(tf_targets['protein1'].value_counts())
        
        # Percentage of included prized genes
        prizes_data = pd.read_csv(save_dir+'de_terminals_'+step+"design_" + str(design) +'.tsv', sep='\t')
        terminals = set(prizes_data['name'].tolist())
        terminals = [node for node in terminals if node[:-4] in proteins]
        percent_included_terminals = n_included_terminals/len(terminals) * 100
        
        # Included TFs
        if TFs_with_targets == False:
            TFs = set(tf_targets['protein1'])
        else: 
            TFs = set(tf_targets_subset['protein1'])
        incl_TFs = set(net.nodes()).intersection(TFs)
        
        # Number of prized TFs
        network_df = oi.get_networkx_graph_as_dataframe_of_nodes(net)
        TF_df = network_df[network_df.index.isin(TFs)]
        n_prized_TFs = TF_df[TF_df['terminal'] == True].shape[0]
        
        # Number of targets that are in the network per TF
        incl_tf_targets = tf_targets[tf_targets['protein1'].isin(incl_TFs)]
        incl_tf_targets = incl_tf_targets[incl_tf_targets['protein2'].isin(net.nodes)]
        target_counts = pd.DataFrame(index = incl_TFs)
        target_counts = target_counts.join(incl_tf_targets['protein1'].value_counts()).fillna(0)
        target_counts = target_counts.join(TF_counts, lsuffix = "_incl_targets", rsuffix = "_genome_targets")
        target_counts['percent_genome_targets'] = target_counts['protein1_genome_targets'] / 18384
        
        # how many percent of the differentially expressed genes are targeted by each TF:
        target_counts['percent_DE_targeted'] = target_counts['protein1_incl_targets'] / nodes[nodes['transition'] == "tr1"].shape[0]
        
        # Significant TFs
        n = nodes[nodes['transition'] == "tr1"].shape[0]
        target_counts['p_value'] = target_counts.apply(lambda x: ss.binomtest(k = int(x['protein1_incl_targets']), 
                                                n = n, 
                                                p = x['percent_genome_targets'], 
                                                alternative= 'greater').pvalue, axis=1)
        target_counts['significance'] = np.where(target_counts['p_value'] < 0.05, 'p < 0.05', 'p > 0.05')
        significant_TFs = target_counts[target_counts['significance'] == 'p < 0.05'].index.tolist()
        
        # Numbers of Protein-Protein-Interactions per TF
        degrees = pd.DataFrame(net.degree()).set_index(0)
        degrees.columns = ['degree']
        target_counts = target_counts.join(degrees)
        target_counts['PPI'] = target_counts['degree'] - target_counts['protein1_incl_targets']
        
        # Save network as interactive html for visualizations with node attributes
        nodes = save_net_html(net, incl_TFs, save_dir, name, design)
        
        # Add results for the current network to the others
        results.loc[step] = [n_nodes, n_edges, n_included_terminals, percent_included_terminals, n_Steiner, 
                             len(TFs), len(incl_TFs), n_prized_TFs, len(significant_TFs), list(incl_TFs), 
                             significant_TFs, nodes.loc[nodes['category'] == 'Steiner node', 'name'].tolist()]
        target_counts_dict[step] = target_counts        
    
    return(results, target_counts_dict)


def get_net_dir_all_stages(save_dir, cell_type, p_thr, design):
    """ Gets directory of networks to compare
    
    Args:
        save_dir: (string) name of the save directory
        cell_type: (string) whether to use only tf-targets from fibroblasts or all cell_types
        p_thr: (string) threshold for the p-values to prize DE genes
        design: (int) which design to choose (1-3)
    
    Returns:
        Dictionary including the augmented forests
    """
    steps = ["1-15_16-26_27-60"]
    net_dir = {}
    
    for step in steps:
        augmented_forest = pickle.load(open(save_dir + "network_robustness_G1_G45.pickle", "rb"))
        net_dir[p_thr+"."+cell_type+ "." + step] = augmented_forest
    
    return(net_dir)


def save_net_html(net, incl_TFs, save_dir, name, design):
    """ Saves the net with node attributes as interactive html
    
    Args:
        net: augmented forest solution from OmicsIntegrator2
        incl_TFs: (set) TFs that are included in the network
        save_dir: (string) directory that contains the processed prize data
        name: (string) name including the p_thr (1%, 5% or 10%), cell_type (fibroblasts or allTFs) and step (e.g. 0_10_20)
        design: (int) which design to choose (1-3)
    
    Returns:
        nodes: pd DataFrame including the category for each node
    """
    # Get nodes and assign attributes (Steiner node, DE_tr0, TF, DE_tr1)
    network_df = oi.get_networkx_graph_as_dataframe_of_nodes(net)
    nodes = pd.DataFrame(net.nodes)[0].str.split("_", expand=True)
    nodes.columns = ['protein', 'transition']
    nodes['name'] = nodes['protein'] + "_" + nodes['transition']
    nodes['category'] = 'Steiner node'
    nodes['category'][nodes['name'].isin(network_df.loc[network_df['terminal'] == True,:].index.tolist())] = "DE_tr0"
    nodes['category'][nodes['transition'] == 'tr1'] = "DE_tr1"
    nodes['TF'] = "No bridge TF"
    nodes['TF'][nodes['name'].isin(incl_TFs)] = "Bridge TF"
    group_dict = {node: nodes.loc[nodes['name'] == node, 'category'].values[0] for node in nodes['name']}
    TF_dict = {node: nodes.loc[nodes['name'] == node, 'TF'].values[0] for node in nodes['name']}
    label_dict = {node: '' for node in nodes['name']}
    nx.set_node_attributes(net, group_dict, name='groups')
    nx.set_node_attributes(net, TF_dict, name='TFs')
    nx.set_node_attributes(net, label_dict, name= 'labels')
    
    # Save the results
    p_thr, cell_type, step = name.split(".")
    oi.output_networkx_graph_as_interactive_html(net, filename=save_dir+"net_"+cell_type+"_"+p_thr+"_step_"+step+
                                                 "_design_" + str(design) +".html")
    
    return(nodes)


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
    step_list = config['steps']
    graph_params = config['graph_params']
    cell_type = config['cell_type']
    p_thr = config['p_thr']
    design = config['design']
    
    # Run PCST combined for two following transitions
    print("Run pcst with Group1-specfic genes as source DE and Group4-5 specific genes as target DE genes")
    for step in step_list:
        
        augmented_forest = run_pcst_combi(step, graph_params, data_dir, save_dir, cell_type, p_thr, step_list, design)
        oi.output_networkx_graph_as_pickle(augmented_forest, 
                                           filename= save_dir + "network_robustness_G1_G45.pickle")
    

if __name__ == "__main__":
    main()
    

    