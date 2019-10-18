import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from collections import OrderedDict

# params
infomap_path = '~/infomap/Infomap'
workspace_path = './workspace/'
filename = 'test'


def infomap(net_path, altmap=False, additional_args=''):
    workspace_path = './workspace/'
    args = ' -2 -u -vvv'
    if altmap:
        args += ' --altmap --to-nodes -p0.15'

    args += additional_args

    os.system(infomap_path + ' ' + net_path + ' ' + workspace_path + ' ' + args)


def read_tree(tree_path):
    df = pd.read_csv(tree_path, sep=' ', header=1)
    df.columns = ['community', 'flow', 'name', 'node', 'trash']
    df = df.drop(['flow', 'trash'], axis=1)

    for i, path in enumerate(df['community']):
        df.iloc[i, 0] = path.split(':')[0]

    return df


def plogq(p, q):
    if q < 1e-18:
        print(f'Unexpected zero operand in plogq: p={p}, q={q}\n.')
        return 0.0

    return p * np.log2(q)


def plogp(p):
    if p < 1e-18:
        return 0.0

    return p * np.log2(p)


def drawNetwork(G, communities, labels=True):
    # position map
    pos = nx.spring_layout(G)
    # community ids
    communities = [v for v in communities.values()]

    # Draw edges
    nx.draw_networkx_edges(G, pos)

    # Draw nodes
    nodeCollection = nx.draw_networkx_nodes(G,
                                            pos=pos,
                                            node_color=communities,
                                            cmap=plt.get_cmap('Set3')
                                            )

    # Draw node labels
    if labels:
        for n in G.nodes():
            plt.annotate(n,
                         xy=pos[n],
                         textcoords='offset points',
                         horizontalalignment='center',
                         verticalalignment='center',
                         xytext=[0, 0],
                         color='k'
                         )

    plt.axis('off')
    plt.show()

# compute altmap cost
def altmap_cost(G, communities):
    # compute stationary and conditional distribution for the nodes
    pagerank = nx.pagerank_numpy(G, alpha=0.95)
    p_nodes = np.array([[val] for val in pagerank.values()])
    p_node_transitions = nx.google_matrix(G, alpha=1.0).T

    # print (f'Stationary distribution = {p_nodes}')
    # print (f'Transition matrix = {p_node_transitions}')
    # if we dont trust the page rank results (works for undir networks)
    # p_nodes = np.linalg.matrix_power(p_node_transitions, 100000).dot(p_nodes)
    # p_nodes /= np.sum(p_nodes)

    # compute stationary and joint distribution for the communities
    num_communities = max(communities.values()) - min(communities.values()) + 1
    p_comm = np.zeros(num_communities)
    p_comm_stay = np.zeros(num_communities)
    H_x = 0
    H_nodes = 0
    for alpha, node in enumerate(G.nodes):
        comm_idx = communities[node] - 1
        p_comm[comm_idx] += p_nodes[alpha]
        H_nodes -= plogp(p_nodes[alpha])

        neighbors = nx.all_neighbors(G, node)
        for neighbor in neighbors:
            beta = neighbor - 1
            H_x += p_nodes[alpha] * plogp(p_node_transitions[beta, alpha])
            if communities[node] == communities[neighbor]:
                p_comm_stay[comm_idx] += p_nodes[alpha] * p_node_transitions[beta, alpha]

    p_comm_leave = p_comm - p_comm_stay

    # print (f'P_comm is {p_comm}.\n')
    # print (f'P_comm_leave is {p_comm_leave}.\n')

    # compute altmap cost
    epsilon = 1e-18  # vicinity threshold for numerical stability
    cost_per_module = np.zeros((num_communities, 1))
    for i in range(num_communities):

        # check for edge cases
        if (p_comm[i] <= epsilon) or (p_comm[i] + epsilon >= 1.0):
            continue

        cost_per_module[i] -= plogp(p_comm_stay[i])
        cost_per_module[i] += 2.0 * plogq(p_comm_stay[i], p_comm[i])
        cost_per_module[i] -= plogp(p_comm_leave[i])
        cost_per_module[i] += plogq(p_comm_leave[i], p_comm[i] * (1.0 - p_comm[i]))
        # print (f'Cost for module {i+1} is {cost_per_module[i]}.\n')

    cost = np.sum(cost_per_module)
    max_cost = H_x + H_nodes
    # print (f'Maximum cost is {max_cost}.\n')
    # print(f'AltMap cost is {cost}.')
    # print (f'Total cost would be {max_cost + cost}.\n')
    # print (f'Node entropy H(x) =  {H_x}.\n')
    return cost


# create initial partition file (init.tree)
def create_initfile(G, N_partitions=None, randomized=True):
    N = len(G.nodes())
    node_ids = np.asarray(range(1, N + 1))
    if randomized:
        np.random.shuffle(node_ids)  # randomize node order

    pagerank = nx.pagerank_numpy(G, alpha=0.95)
    p_nodes = np.array([[val] for val in pagerank.values()])

    num_partitions = N_partitions
    if num_partitions == None:
        num_partitions = int(np.sqrt(N))

    partition_size = int(N / num_partitions)
    communities = {}

    with open(workspace_path + 'init.tree', "w+") as init_file:
        init_file.write('# path flow name node:\n')
        n = 0
        for partition in range(1, num_partitions + 1):
            for node_rank in range(1, partition_size + 1):
                if n >= N:
                    break
                node_id = node_ids[n]
                n += 1

                node_flow = p_nodes[node_id - 1, 0]
                init_file.write(str(partition) + ':' + str(node_rank) + ' ' + str(node_flow))
                init_file.write(' ' + '\"' + str(node_id) + '\"' + ' ' + str(node_id) + '\n')
                communities[node_id] = partition

        while n < N:
            node_rank += 1
            node_id = node_ids[n]
            n += 1

            node_flow = p_nodes[node_id - 1, 0]
            init_file.write(str(partition) + ':' + str(node_rank) + ' ' + str(node_flow))
            init_file.write(' ' + '\"' + str(node_id) + '\"' + ' ' + str(node_id) + '\n')
            communities[node_id] = partition

    return communities

def read_communities_from_tree_file():
    df = read_tree(workspace_path + filename + '.tree')

    communities_found = {}
    for index, row in df.iterrows():
        node = int(row['node'])
        communities_found[node] = int(row['community'])

    communities_found = OrderedDict(sorted(communities_found.items()))
    num_communities = max(communities_found.values()) - min(communities_found.values()) + 1

    return communities_found, num_communities