from cuml.cluster import HDBSCAN

clusterer = HDBSCAN(min_cluster_size=10, min_samples=5)
labels = clusterer.fit_predict(candidates[['dm', 'time', 'snr']].values)
