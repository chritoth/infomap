# imports
from networkx.generators.community import LFR_benchmark_graph
from sklearn.metrics import adjusted_mutual_info_score as ami_score

from altmap.altmap_helpers.general import *

# generate LFR benchmark graph + extract ground truth communities
def generate_LFR_benchmark(N=500, mu=0.1):
    # LFR params generic
    max_community = int(0.2 * N)
    min_community = int(max_community * 0.25)
    max_degree = int(max_community * 0.3)
    min_degree = int(min_community * 0.4)
    gamma = 3.5  # Power law exponent for the degree distribution
    beta = 1.1  # Power law exponent for the community size distribution

    # generate LFR benchmark graph
    G = LFR_benchmark_graph(N, gamma, beta, mu, min_degree=min_degree, max_degree=max_degree,
                            max_community=max_community, min_community=min_community)
    G = nx.convert_node_labels_to_integers(G, first_label=1)

    # extract ground truth communities from networkx graph object
    communities_true = {}
    num_communities = 0
    for n in range(1, N + 1):
        if n in communities_true:
            continue

        num_communities = num_communities + 1
        community = G.nodes[n]['community']
        node_ids = np.asarray(list(community))
        node_ids = node_ids + 1  # have node labels >= 1
        communities_true.update(dict.fromkeys(node_ids, num_communities))

    communities_true = OrderedDict(sorted(communities_true.items()))
    num_communities_true = max(communities_true.values()) - min(communities_true.values()) + 1

    return G, communities_true, num_communities_true


# compute normalized mutual information between two partitions
def compute_score(communities_true, communities_found):
    labels_true = list(communities_true.values())
    labels_found = list(communities_found.values())

    # return nmi_score(labels_true,labels_found, average_method='arithmetic')
    return ami_score(labels_true, labels_found, average_method='arithmetic')


def avg_community_clusterings(G, communities):
    _, comm_nodes, _ = nodes_per_community(communities)
    num_comms = len(comm_nodes)
    comm_clusterings = []
    for i in range(num_comms):
        tmp_list = comm_nodes.copy()
        tmp_list.pop(i)
        tmp_list = [node for comm in tmp_list for node in comm]

        G_tmp = G.copy()
        G_tmp.remove_nodes_from(tmp_list)
        comm_clusterings.append(nx.average_clustering(G_tmp))

    return np.mean(comm_clusterings), np.min(comm_clusterings)


class BenchmarkResults:
    def __init__(self, var_list, num_realizations):
        self.var_list = var_list
        self.num_datapoints = len(var_list)
        self.num_realizations = num_realizations
        self.actual_realizations = np.zeros((self.num_datapoints,), dtype=int)
        self.scores = np.zeros((num_realizations, self.num_datapoints))
        self.errors = np.zeros((num_realizations, self.num_datapoints))

        self.mean_scores = np.empty((self.num_datapoints,))
        self.mean_scores[:] = np.nan
        self.std_scores = np.empty((self.num_datapoints,))
        self.std_scores[:] = np.nan
        self.mean_errors = np.empty((self.num_datapoints,))
        self.mean_errors[:] = np.nan
        self.std_errors = np.empty((self.num_datapoints,))
        self.std_errors[:] = np.nan

    def evaluate_results(self):
        for i in range(self.num_datapoints):
            nr = self.actual_realizations[i]
            if nr == 0:
                continue

            self.std_scores[i] = np.std(self.scores[:nr, i], ddof=1)
            self.std_errors[i] = np.std(self.errors[:nr, i], ddof=1)
            self.mean_scores[i] = np.mean(self.scores[:nr, i])
            self.mean_errors[i] = np.mean(self.errors[:nr, i])

    def write_csv(self, path):
        self.evaluate_results()  # make sure we have sth to write

        df = pd.DataFrame()
        df['var_list'] = self.var_list
        df['actual_realizations'] = self.actual_realizations
        df['mean_scores'] = self.mean_scores
        df['std_scores'] = self.std_scores
        df['mean_errors'] = self.mean_errors
        df['std_errors'] = self.std_errors

        df.to_csv(path, index_label='id')

    @classmethod
    def read_csv(cls, path):
        df = pd.read_csv(path, index_col=0)

        num_realizations = int(np.max(df['actual_realizations'].values))
        results = cls(df['var_list'].values, num_realizations)

        results.actual_realizations = df['actual_realizations'].values
        results.mean_scores = df['mean_scores'].values
        results.std_scores = df['std_scores'].values
        results.mean_errors = df['mean_errors'].values
        results.std_errors = df['std_errors'].values

        return results


# LFR Benchmark
# num_realizations .. number of network realizations for each parameter pair (mu, N)
def run_benchmark(N_list: list, mu_list: list, num_realizations=10):
    N = N_list[0]
    var_list = mu_list
    vary_mu = True
    if len(N_list) > 1:
        mu = mu_list[0]
        var_list = N_list
        vary_mu = False

    num_benchmarks = len(var_list) * num_realizations
    benchmark_id = 0

    infomap_results = BenchmarkResults(var_list, num_realizations)
    altmap_results = BenchmarkResults(var_list, num_realizations)
    altmap_sci_results = BenchmarkResults(var_list, num_realizations)
    sci_results = BenchmarkResults(var_list, num_realizations)
    acc_results = BenchmarkResults(var_list, num_realizations)
    for var_idx, var in enumerate(var_list):
        if vary_mu:
            mu = var
        else:
            N = var

        realization_idx = -1
        for _ in range(num_realizations):
            benchmark_id = benchmark_id + 1
            print(f'Starting benchmark {benchmark_id}/{num_benchmarks} for (N,mu) = ({N},{mu})\n')
            try:
                G, communities_true, num_communities_true = generate_LFR_benchmark(N, mu)
            except nx.ExceededMaxIterations as err:
                print(f'Failed to generate network for (N,mu) = ({N},{mu}): ', err)
                continue

            realization_idx += 1

            # test infomap
            communities_found, num_communities_found, _, _ = infomap(G, altmap=False)
            print(f'Infomap found {num_communities_found} communities vs. {num_communities_true} ground truth '
                  f'communities.\n')

            score = compute_score(communities_true, communities_found)
            infomap_results.scores[realization_idx, var_idx] = score
            error = num_communities_found / num_communities_true - 1.0
            infomap_results.errors[realization_idx, var_idx] = error

            # test altmap
            communities_found, num_communities_found, _, _ = infomap(G, altmap=True, update_inputfile=False)
            print(f'Altmap found {num_communities_found} communities vs. {num_communities_true} ground truth '
                  f'communities.\n')

            score = compute_score(communities_true, communities_found)
            altmap_results.scores[realization_idx, var_idx] = score
            error = num_communities_found / num_communities_true - 1.0
            altmap_results.errors[realization_idx, var_idx] = error

            # test altmap with SCI
            communities_found, num_communities_found, \
                communities_init, num_communities_init = infomap(G, altmap=True, init='sc', update_inputfile=False)
            print(f'Altmap with SCI ({num_communities_init}) found {num_communities_found} communities vs. '
                  f'{num_communities_true} ground truth communities.\n')

            score = compute_score(communities_true, communities_found)
            altmap_sci_results.scores[realization_idx, var_idx] = score
            error = num_communities_found / num_communities_true - 1.0
            altmap_sci_results.errors[realization_idx, var_idx] = error

            score = compute_score(communities_true, communities_init)
            sci_results.scores[realization_idx, var_idx] = score
            error = num_communities_init / num_communities_true - 1.0
            sci_results.errors[realization_idx, var_idx] = error

            # compute community clustering
            acc_mean, acc_min = avg_community_clusterings(G, communities_true)
            acc_results.scores[realization_idx, var_idx] = acc_mean
            acc_results.errors[realization_idx, var_idx] = acc_min
            print(f'Avg community clustering is {acc_mean}, minimum is {acc_min}.')

        # store actual number of realizations
        infomap_results.actual_realizations[var_idx] = realization_idx + 1
        altmap_results.actual_realizations[var_idx] = realization_idx + 1
        altmap_sci_results.actual_realizations[var_idx] = realization_idx + 1
        sci_results.actual_realizations[var_idx] = realization_idx + 1
        acc_results.actual_realizations[var_idx] = realization_idx + 1

    print(f'Finished benchmark successfully!\n')
    return infomap_results, altmap_results, altmap_sci_results, sci_results, acc_results


def plot_benchmark_results(results: BenchmarkResults, type='scores', color='blue', marker=None, label=None,
                           lower_bound=None, upper_bound=None, plot_uncertainty=True):
    xdata = results.var_list
    data = results.mean_scores
    data_std = results.std_scores

    if type == 'errors':
        data = results.mean_errors
        data_std = results.std_errors

    lw = 2
    ms = 10
    line = plt.plot(xdata, data, '--', marker=marker, color=color, linewidth=lw, markersize=ms, label=label)

    if plot_uncertainty:
        upper = data + data_std
        lower = data - data_std
        if lower_bound is not None:
            lower[lower < lower_bound] = lower_bound
        if upper_bound is not None:
            upper[upper > upper_bound] = upper_bound

        plt.fill_between(xdata, upper, lower, color=color, alpha=0.25)

    return line
