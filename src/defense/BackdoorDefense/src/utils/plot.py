import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np


def pcaScatterPlot(X, labels, output_file):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        mask = labels == label
        color = "red" if label == 1 else "blue"
        if label:
            label_str = "poisoned"
        else:
            label_str = "clean"
        plt.scatter(
            X_pca[mask, 0], X_pca[mask, 1], label=label_str, color=color, alpha=0.5
        )
    plt.legend()
    plt.savefig(output_file)
    plt.close()
