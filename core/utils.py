import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def scatter_matrix(df, colors, labels):
  c = df['class'].apply(lambda idx: colors[idx])
  size = df.shape[1] - 1

  plots = pd.plotting.scatter_matrix(df, c=c, marker='o', alpha=0.8, s=20, figsize=(10, 7), hist_kwds={'alpha': 0})
  for i in range(size):
    for j in range(3):
      plots[i, i].hist(df.iloc[50 * j:50 * j + 50, i], color=colors[j], alpha=0.8)

  handles = [plt.plot([], [], color=colors[i], ls="", marker=".")[0] for i in range(3)]
  plt.legend(handles, labels, loc='lower left', bbox_to_anchor=(-size + 1, size + 0.02), ncol=3, borderaxespad=0, frameon=False)
  plt.show()

def draw_centroids(ax, X, y, centroids, idx, labels, features, colors):
  x1_min, x1_max = X[:, 0].min() - 0.25, X[:, 0].max() + 0.25
  x2_min, x2_max = X[:, 1].min() - 0.25, X[:, 1].max() + 0.25
  ax.set_xlim(x1_min, x1_max)
  ax.set_ylim(x2_min, x2_max)

  size = centroids.shape[0]
  markers = np.array(['o', 'x', '+', 'v', 's', '^', '*'])
  for i in range(size):
    ax.scatter(X[idx == i, 0], X[idx == i, 1], color=colors[y[idx == i].astype(int)], marker=markers[i], alpha=0.8, s=20)
  ax.set_xlabel(features[0])
  ax.set_ylabel(features[1])

  for i in range(size):
    ax.scatter(centroids[i, 0], centroids[i, 1], color='black', marker=markers[i], alpha=0.5, s=60)
