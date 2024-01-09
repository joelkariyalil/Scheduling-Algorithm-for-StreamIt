import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def enter_bandwidths():
    print('\n\nEnter the Bandwidths')
    try:
        arr = list(map(int, input().split()))
    except ValueError:
        print("Invalid input. Please enter integers.")
        return []
    arr.sort()
    speeds = np.array(arr)
    return speeds


def cluster_initialization(speeds):
    try:
        min_transfers = int(input('\nEnter Minimum Transfers: '))
        max_transfers = int(input('Enter Maximum Transfers: '))
    except ValueError:
        print("Invalid input. Please enter integers.")
        return 1
    k = min(max(len(speeds) // max_transfers, 1), len(speeds) // min_transfers)
    print('K Value is: ', k)
    return k


def normalization_and_clustering(speeds, k):
    Point = (speeds - np.mean(speeds)) / np.std(speeds)
    kmeans_cluster = KMeans(n_clusters=k)
    kmeans = kmeans_cluster.fit(Point.reshape(-1, 1))
    label = np.sort(kmeans.labels_)
    return label, kmeans


def print_clusters_and_bandwidths(arr, labels):
    print('\nThe Clusters (Using KMeans Clustering)!\n')
    for i, bandwidth in enumerate(arr):
        print(f"{bandwidth}\t\tGroup: {labels[i]}")


def calculate_clstr_speeds_averages(speeds, labels, kmeans):
    clstr_speeds = [np.mean(speeds[labels == i]) for i in range(kmeans.n_clusters)]
    return clstr_speeds


def display_kmeans_clustering_graph(speeds, labels, clstr_speeds):
    centers = clstr_speeds

    plt.scatter(speeds, np.zeros_like(speeds), c=labels)
    plt.scatter(centers, np.zeros_like(centers), marker='x', s=200, linewidths=3, color='r')
    plt.xlabel('Network Speeds')
    plt.title('KMeans Clustering of Network Speeds')
    plt.show()


def main():
    # Step 1: Collect data on network speeds/bandwidths of all users
    speeds = enter_bandwidths()

    if len(speeds) == 0:
        print("No valid input. Exiting.")
    else:
        # Step 2: Initialize number of clusters
        k = cluster_initialization(speeds)

        # Steps 3-4: Normalize and cluster data
        labels, kmeans = normalization_and_clustering(speeds, k)

        # Step 5: Assign users to nearest cluster
        # Sorting the Labels
        labels = np.sort(labels)

        # Step 6: Calculate average network speed/bandwidth for each cluster
        clstr_speeds = calculate_clstr_speeds_averages(speeds, labels, kmeans)
        print_clusters_and_bandwidths(speeds, labels)
        print('\n\n\nCluster Speeds:\n', clstr_speeds)

        # Displaying the KMeans Clustering Graphically
        display_kmeans_clustering_graph(speeds, labels, clstr_speeds)


if __name__ == "__main__":
    main()
