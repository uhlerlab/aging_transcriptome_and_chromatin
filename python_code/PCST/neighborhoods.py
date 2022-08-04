# Imports

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pickle
import logging
logger = logging.getLogger(__name__)
import OmicsIntegrator as oi
from upsetplot import from_contents, UpSet


def add_edge_confidence(network_selected):
    """ Adds edge confidence to network attribute
    
    Args:
        network_selected: NetworkX network with cost information
    
    Returns:
        void
    """
    cost = nx.get_edge_attributes(network_selected,'cost')
    edge_costs = [cost[edge] for edge in list(network_selected.edges)]
    edge_confidences = {edge: (max(edge_costs)-cost[edge])/(max(edge_costs)-min(edge_costs)) for edge in list(network_selected.edges)}
    nx.set_edge_attributes(network_selected, edge_confidences, 'confidence')


def get_neighborhood_intersections(TFs, net_dict, cthreshold = 1):
    
    TFs = [TF + '_tr0' for TF in TFs]
    neighborhood_dict = {}
    
    for name, net in net_dict.items():
        network_selected_new = net.copy()

        # Filter network to the nodes on the left
        right_side = [node for node in list(network_selected_new.nodes()) if node.endswith('_tr1')]
        network_selected_new.remove_nodes_from(right_side)

        # Apply cost threshold
        cost = nx.get_edge_attributes(network_selected_new,'cost')
        cost_threshold = cthreshold
        expensive_edges = list([edge for edge in list(network_selected_new.edges) if cost[edge]>cost_threshold])
        network_selected_new.remove_edges_from(expensive_edges)

        # Find neighbors of TFs of interest
        neighborhood_set = set()
        for TF in TFs:
            for node in network_selected_new.neighbors(TF):
                neighborhood_set = neighborhood_set.union(set({node}))
        
        neighborhood_set = [protein[:-4] for protein in neighborhood_set]
        neighborhood_dict[name.split(".")[2]] = neighborhood_set
            
    intersections = from_contents(neighborhood_dict)
    upset = UpSet(intersections, subset_size='count', show_counts = True,  
                  sort_by="cardinality", min_subset_size = 1).plot()
    return(neighborhood_dict)


def plot_neighborhood_subnetwork(TFs,
                                 network_selected,
                                 ax,
                                 vmin_edges = 0,
                                 vmax_edges = 1,
                                 cthreshold = 1, 
                                 nodesize = 1000):
    
    TFs = [TF + '_tr0' for TF in TFs]
    network_selected_new = network_selected.copy()
        
    # Filter network to the nodes on the left
    right_side = [node for node in list(network_selected_new.nodes()) if node.endswith('_tr1')]
    network_selected_new.remove_nodes_from(right_side)

    # Apply cost threshold
    cost = nx.get_edge_attributes(network_selected_new,'cost')
    cost_threshold = cthreshold
    expensive_edges = list([edge for edge in list(network_selected_new.edges) if cost[edge]>cost_threshold])
    network_selected_new.remove_edges_from(expensive_edges)

    # Find neighbors of TFs of interest
    neighborhood_set = set()
    for TF in TFs:
        for node in network_selected_new.neighbors(TF):
            neighborhood_set = neighborhood_set.union(set({node}))
    not_in_neighborhood = set(network_selected_new.nodes()).difference(neighborhood_set.union(set(TFs)))

    # Create subnetwork
    neighborhood_net = network_selected_new.copy()
    neighborhood_net.remove_nodes_from(not_in_neighborhood)
    neighborhood_df = oi.get_networkx_graph_as_dataframe_of_nodes(neighborhood_net)
    
    # Define type of nodes
    node_terminals = set(neighborhood_df[neighborhood_df['prize']>0.001].index)
    steiner_nodes = set(neighborhood_net.nodes()).difference(node_terminals.union(TFs))

    # Draw
    pos = nx.layout.kamada_kawai_layout(neighborhood_net,
                                        weight=None,
                                        scale=10)
    # Draw terminals 
    terminals = nx.draw_networkx_nodes(neighborhood_net, 
                                       pos,
                                       nodelist=sorted(list(node_terminals)),
                                       node_color= 'blue', 
                                       node_size=nodesize,
                                       node_shape='o',
                                       alpha=0.4, ax=ax)
    
    # Draw all other nodes
    nx.draw_networkx_nodes(neighborhood_net, pos,
                           nodelist=steiner_nodes,
                           node_color='grey',
                           node_size=nodesize,
                           node_shape='o',
                           alpha=0.4, ax=ax)
    
    # Draw TFs 
    nx.draw_networkx_nodes(neighborhood_net, pos,
                           nodelist=TFs,
                           node_color='r',
                           node_size=nodesize,
                           node_shape='h',
                           alpha=0.8, ax=ax)
    
    # Draw edges
    edges = nx.draw_networkx_edges(neighborhood_net, 
                                   pos, 
                                   width=1.0, 
                                   alpha=0.4, 
                                   edge_color='grey', ax=ax)
                                  
    # Draw labels
    nx.draw_networkx_labels(neighborhood_net,
                            pos,
                            font_size=14,
                            font_weight = 'bold', ax=ax)
    
    ax.axis('off')
    
    
    
def plot_neighborhood_all_networks(TFs,
                                 net_dict,
                                 vmin_edges = 0,
                                 vmax_edges = 1,
                                 cthreshold = 1, 
                                 nodesize = 1000,
                                 save=False):
    
    fig, axes = plt.subplots(nrows=1, ncols=len(net_dict), figsize = (40,15))
    
    for (name, net), ax in zip(net_dict.items(), axes.ravel()):
        plot_neighborhood_subnetwork(TFs, net, ax, cthreshold = cthreshold, nodesize = nodesize)
        ax.set_title(name.split(".")[2], fontsize = 24)
    
    if save == True:
        plt.savefig('neighborhood'+str(TFs)+'_cthresh'+str(cthreshold)+'.pdf', format='pdf')
    
    plt.show()
    