import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter


def fknn(data, u_best, k, c, m):
    nodes_dist = {}
    null_dist = []
    for i in range(len(data)):
        null_dist.append([i, 0])
    for i in range(len(data)):
        nodes_dist[i] = {'node': data[i], 'dist': null_dist, 'cluster': 0}
        for j in range(len(data)):
            nodes_dist[i]['dist'][j][1] = np.linalg.norm(data[i] - data[j])
        nodes_dist[i]['dist'] = sorted(nodes_dist[i]['dist'], key=itemgetter(1), reverse=False)
        nodes_dist[i]['dist'] = nodes_dist[i]['dist'][0:k]
    for node in nodes_dist:
        node_memebership_to_clust = []
        for cluster in range(c):
            node_memebership_to_clust.append(
                [cluster, node_cluster_membership(nodes_dist, cluster, u_best, node, k, m)])
        node_memebership_to_clust = sorted(node_memebership_to_clust, key=itemgetter(1), reverse=True)
        nodes_dist[node]['cluster'] = node_memebership_to_clust[0][0]

    all_clusters = []
    for cluster in range(c):
        clust = []
        for node in nodes_dist:
            if nodes_dist[node]['cluster'] == cluster:
                clust.append(nodes_dist[node]['node'])
        all_clusters.append(clust)
    plot_clusters(all_clusters)
    return 0


def plot_clusters(all_clusters):
    rgb_colors = ['brown', 'red', 'orange', 'purple', 'gray', 'yellow', 'black', 'green', 'blue']
    counter = 0
    for cluster in all_clusters:
        for node in cluster:
            plt.scatter(node[0], node[1], color=rgb_colors[counter])
        counter += 1
    plt.show()
    return 0


def node_cluster_membership(nodes_dist, cluster, u_best, node, k, m):
    numerator = 0
    denominator = 0
    for j in range(k):
        numerator += u_best[cluster][nodes_dist[node]['dist'][j][0]] * (
                (1 / nodes_dist[node]['dist'][j][1]) ** (2 / (m - 1)))
        denominator += (1 / nodes_dist[node]['dist'][j][1]) ** (2 / (m - 1))
    return numerator / denominator
