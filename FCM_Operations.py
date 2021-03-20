import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from operator import itemgetter

best_u = []


def fcm(data, data_train, Y_train, data_test, Y_test, classes_count=2, n_clusters=1, landa=.1, n_init=30, m=2,
        max_iter=300, tol=1e-16):
    min_cost = np.inf
    centers = data[np.random.choice(
        data.shape[0], size=n_clusters, replace=False
    ), :]
    dist = np.fmax(
        cdist(centers, data, metric='sqeuclidean'),
        np.finfo(np.float64).eps
    )
    entropy = 0
    best_u = []
    for iter1 in range(max_iter):
        # Compute memberships
        u_prim = (1 / dist) ** (2 / (m - 1))
        u = u_prim / u_prim.sum(axis=0)
        um = u ** m
        # Recompute centers
        prev_centers = centers
        centers = um.dot(data) / um.sum(axis=1)[:, None]
        dist = cdist(centers, data, metric='sqeuclidean')
        if np.linalg.norm(centers - prev_centers) < tol:
            best_u = u
            break
        # Entropy
        cost = np.sum(u * np.log2(u))
        if cost < min_cost:
            min_cost = entropy
            min_centers = centers
            # best_u = u
        if iter1 == max_iter - 1:
            best_u = u
    for i in range(len(best_u)):
        for j in range(len(best_u[i])):
            entropy += best_u[i][j] * np.log2(best_u[i][j])
    entropy = -1 * entropy
    # entropy = -1 * np.sum(best_u * np.log2(best_u))
    # entropy = 1 / entropy
    # print('Entropy : ')
    # print(entropy)
    plt.plot(data[:, 0], data[:, 1], 'go', min_centers[:, 0], min_centers[:, 1], 'bs')
    plt.show()
    entropy = calculate_entropy(best_u, data, min_centers)
    return min_centers, entropy, best_u


def calculate_entropy(u, data, centers):
    # u
    # print('u')
    # print(u)
    all_centers = np.array(centers)
    all_clusters = []
    for i in range(len(u)):
        all_clusters.append(list())
    for i in range(len(u[0])):
        cluster = []
        for j in range(len(u)):
            cluster.append([j, u[j][i]])
        which_center = sorted(cluster, key=itemgetter(1), reverse=True)[0][0]
        all_clusters[which_center].append(
            np.linalg.norm(data[i] - all_centers[which_center]))
    all_variances = []
    all_clusters = np.array(all_clusters)
    for i in range(len(all_clusters)):
        all_variances.append(np.var(all_clusters[i]))
    # print(all_variances)
    # entropy = np.max(all_variances)
    entropy = np.sum(all_variances)
    print('Entropy : ')
    print(entropy)
    return entropy