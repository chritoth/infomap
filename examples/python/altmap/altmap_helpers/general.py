import os
import time
from collections import OrderedDict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering

# paths
infomap_path = '~/infomap/Infomap'
workspace_path = './workspace/'
filename = 'test'


def init_plt_params():
    plt.rcParams.update({'font.size': 20})
    plt.rcParams.update({'text.usetex': True})
    plt.rcParams.update({'font.family': 'sans-serif'})
    plt.rcParams.update({'lines.linewidth': 3})
    plt.rcParams.update({'lines.markersize': 10})
    plt.rcParams.update({'lines.markeredgewidth': 3})
    plt.rcParams.update({'axes.labelpad': 20})
    plt.rcParams['text.latex.preamble'] = [
        r'\usepackage{amsmath,amssymb,amsfonts,amsthm}',
        r'\usepackage{siunitx}',  # i need upright \micro symbols, but you need...
        r'\sisetup{detect-all}',  # ...this to force siunitx to actually use your fonts
        r'\usepackage{helvet}',  # set the normal font here
        r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
        r'\sansmath'  # <- tricky! -- gotta actually tell tex to use!
    ]


def read_tree(tree_path):
    df = pd.read_csv(tree_path, sep=' ', header=1)
    df.columns = ['community', 'flow', 'name', 'node', 'trash']
    df = df.drop(['flow', 'trash'], axis=1)
    df['community'] = df['community'].apply(lambda x: x.split(':')[0])
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


def get_num_communities(communities):
    comm_ids = np.unique([v for v in communities.values()])
    return len(comm_ids)


# get nodes grouped per community
def nodes_per_community(communities):
    comm_ids = np.unique([v for v in communities.values()])
    Nc = len(comm_ids)

    comm_sizes = np.zeros((Nc,), dtype=int)
    comm_nodes = []
    for i, comm_id in enumerate(comm_ids):
        community_nodes = [node for (node, comm) in communities.items() if comm == comm_id]
        comm_nodes.append(community_nodes)
        comm_sizes[i] = len(community_nodes)

    return comm_ids, comm_nodes, comm_sizes


# compute node positions for communities placed in a circular manner
def community_layout(communities):
    N = len(communities)

    comm_ids, comm_nodes, comm_sizes = nodes_per_community(communities)
    Nc = len(comm_ids)

    comm_angles = 2.0 * np.pi / N * comm_sizes
    comm_radii = 1.0 / N * comm_sizes

    prev_axis = 0
    comm_axis = np.zeros((Nc,))
    distances_from_origin = np.zeros((Nc,))
    for i, comm_angle in enumerate(comm_angles):
        axis = comm_angle / 2 + prev_axis
        prev_axis += comm_angle
        comm_axis[i] = axis

        distances_from_origin[i] = comm_radii[i]
        if comm_angle < np.pi:
            # distances_from_origin[i] /= np.abs(np.sin(comm_angle / 2))
            distances_from_origin[i] *= 0.9 * np.pi / comm_angle

    comm_centers = [[d * np.cos(axis), d * np.sin(axis)] for d, axis in zip(distances_from_origin, comm_axis)]
    comm_centers = np.asarray(comm_centers)

    node_positions = dict()
    for radius, size, center, node_ids in zip(comm_radii, comm_sizes, comm_centers, comm_nodes):
        node_radii = 1.0 * radius * np.sqrt(np.random.uniform(0, 1, size))
        node_angles = np.random.uniform(0, 2 * np.pi, size)

        for node_id, r, phi in zip(node_ids, node_radii, node_angles):
            node_positions[node_id] = [r * np.cos(phi) + center[0], r * np.sin(phi) + center[1]]

    return OrderedDict(sorted(node_positions.items()))


def get_node_shapes(communities):
    N = len(communities)

    comm_ids, comm_nodes, comm_sizes = nodes_per_community(communities)
    Nc = len(comm_ids)

    node_shapes = dict()
    for nodes, comm_size in zip(comm_nodes, comm_sizes):
        node_shapes.update(zip(nodes, ['s'] * comm_size))

    return list(OrderedDict(sorted(node_shapes.items())).values())


def drawNetwork(G, communities, labels=True, ax=None):
    node_size = 1200
    edge_width = 1.2
    font_size = 25

    # position map
    # pos = nx.spring_layout(G)
    pos = nx.kamada_kawai_layout(G)
    # pos = nx.planar_layout(G)
    # pos = community_layout(communities)

    # community ids
    color_idc = [v for v in communities.values()]

    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_width)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos=pos, node_color=color_idc, cmap=plt.get_cmap('Set3'), ax=ax, node_size=node_size)

    # Draw node labels
    if labels:
        nx.draw_networkx_labels(G, pos, ax=ax, labels=communities, font_weight='bold', font_size=font_size)
        # for n in G.nodes():
        #     plt.annotate(n, xy=pos[n], textcoords='offset points',
        #                  horizontalalignment='center',verticalalignment='center', xytext=[0, 0],color='k')

    if ax:
        ax.set_frame_on(False)
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_xticks([])
    else:
        plt.axis('off')
        plt.show()


# compute altmap cost
def altmap_cost(G, communities):
    # compute stationary and conditional distribution for the nodes
    pagerank = nx.pagerank_numpy(G, alpha=0.95)
    p_nodes = np.array([[val] for val in pagerank.values()])
    p_node_transitions = nx.google_matrix(G, alpha=1.0).T

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


def generate_initfile(G, method='random'):
    # random is default method

    N = len(G.nodes())
    node_ids = np.asarray(range(1, N + 1))

    if method == 'sc':
        # use spectral clustering to determine initial communities
        time_start = time.clock()
        adj_mat = nx.to_numpy_matrix(G)
        sc = SpectralClustering(n_clusters=int(np.sqrt(N)), affinity='precomputed', n_init=1, assign_labels='kmeans')
        sc.fit(adj_mat)
        elapsed_time = time.clock() - time_start
        print(f'Spectral clustering finished in {elapsed_time} seconds.')
        labels = sc.labels_ + 1
        num_communities = np.max(labels)
    elif method == 'twomodule':
        num_communities = 2
        labels = [1] * int(N / 2) + [2] * (N - int(N / 2))
    else:
        num_communities = int(np.sqrt(N))
        labels = np.random.randint(1, num_communities + 1, node_ids.shape)

    communities = dict(zip(node_ids, labels))

    # compute stationary distribution
    # time_start = time.clock()
    # pagerank = nx.pagerank_numpy(G, alpha=0.85)
    # p_nodes = np.array([val for val in pagerank.values()])
    # elapsed_time = time.clock() - time_start
    # print(f'Page rank finished in {elapsed_time} seconds.')
    p_nodes = np.ones_like(node_ids) / N

    with open(workspace_path + 'init.tree', "w+") as init_file:
        init_file.write('# path flow name node:\n')
        for community_id in range(1, num_communities + 1):
            community = [key for (key, value) in communities.items() if value == community_id]
            n = 0
            for node_id in community:
                n += 1

                node_flow = p_nodes[node_id - 1]  # , 0]
                init_file.write(str(community_id) + ':' + str(n) + ' ' + str(node_flow))
                init_file.write(' ' + '\"' + str(node_id) + '\"' + ' ' + str(node_id) + '\n')

    return communities, num_communities


def read_communities_from_tree_file():
    df = read_tree(workspace_path + filename + '.tree')

    node_ids = df['name'].tolist()
    labels = df['community'].apply(lambda x: int(x)).tolist()
    communities_found = dict(zip(node_ids, labels))

    communities_found = OrderedDict(sorted(communities_found.items()))
    num_communities = max(labels) - min(labels) + 1

    return communities_found, num_communities


def infomap(G, altmap=False, init='std', update_inputfile=True, additional_args=''):
    if not os.path.exists(workspace_path):
        os.mkdir(workspace_path)

    # write graph to input file
    input_path = workspace_path + filename + '.net'
    if update_inputfile:
        nx.write_pajek(G, input_path)

    # construct argument string
    args = ' -2 -u '  # two-level, undirected network, verbose output
    if altmap:
        args += ' --altmap '  # use altmap cost, teleport to nodes rather than edges

    if init not in {'std', 'random', 'twomodule', 'sc'}:
        init = 'std'

    args += additional_args

    # generate init file
    communities_init = num_communities_init = None
    if init != 'std':
        communities_init, num_communities_init = generate_initfile(G, method=init)
        args += ' --cluster-data ./workspace/init.tree '

    os.system(infomap_path + ' ' + input_path + ' ' + workspace_path + ' ' + args)

    communities_found, num_communities_found = read_communities_from_tree_file()
    return communities_found, num_communities_found, communities_init, num_communities_init


# compute altmap module cost
def altmap_module_cost(p_comm, p_comm_leave):
    # check for edge cases
    epsilon = 1e-18  # vicinity threshold for numerical stability
    if (p_comm <= epsilon) or (p_comm + epsilon >= 1.0):
        return 0.0

    p_comm_stay = p_comm - p_comm_leave
    cost_per_module = -plogp(p_comm_stay)
    cost_per_module += 2.0 * plogq(p_comm_stay, p_comm)
    cost_per_module -= plogp(p_comm_leave)
    cost_per_module += plogq(p_comm_leave, p_comm * (1.0 - p_comm))
    return cost_per_module