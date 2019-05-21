import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core.utils import scatter_matrix, draw_centroids
from core.kmeans import random_initialisation, find_closest_centroids, compute_centroids, cost_function

def run_iris(small=False, n_iter=10, K=3, random_state=None):
  features = np.array(['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'class'])
  df = pd.read_csv("./data/iris.data", header=None, names=features)
  if small:
    df = df.drop(df.columns[[1, 3]], axis=1)
    features = np.array(['sepal length (cm)', 'petal length (cm)', 'class'])

  labels = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
  y = df.iloc[:, 2].values if small else df.iloc[:, 4].values
  for key, value in labels.items():
    y[y == key] = value
  X = df.iloc[:, 0:2].values if small else df.iloc[:, 0:4].values

  colors = np.array(['#F44336', '#03A9F4', '#4CAF50'])
  scatter_matrix(df, colors, labels.keys())

  centroids = random_initialisation(X, K, random_state=random_state)
  prev_cost = -1
  for _ in range(n_iter):
    idx = find_closest_centroids(X, centroids)
    centroids = compute_centroids(X, idx, K)
    cost = cost_function(X, centroids, idx)
    print(f"# Iteration: {_} -- Cost: {cost}")
    if prev_cost == cost:
      print("Cannot optimize more.")
      break
    prev_cost = cost

    if small:
      _, ax = plt.subplots(1, 1, figsize=(12.5, 7))
      draw_centroids(ax, X, y, centroids, idx, labels.keys(), features, colors)
    else:
      _, axs = plt.subplots(2, 3, figsize=(12.5, 7))
      draw_centroids(axs[0, 0], X[:, [0, 1]], y, centroids[:, [0, 1]], idx, labels.keys(), features[[0, 1]], colors)
      draw_centroids(axs[0, 1], X[:, [0, 2]], y, centroids[:, [0, 2]], idx, labels.keys(), features[[0, 2]], colors)
      draw_centroids(axs[0, 2], X[:, [0, 3]], y, centroids[:, [0, 3]], idx, labels.keys(), features[[0, 3]], colors)
      draw_centroids(axs[1, 0], X[:, [1, 2]], y, centroids[:, [1, 2]], idx, labels.keys(), features[[1, 2]], colors)
      draw_centroids(axs[1, 1], X[:, [1, 3]], y, centroids[:, [1, 3]], idx, labels.keys(), features[[1, 3]], colors)
      draw_centroids(axs[1, 2], X[:, [2, 3]], y, centroids[:, [2, 3]], idx, labels.keys(), features[[2, 3]], colors)

    handles = [plt.plot([], [], color=colors[i], ls="", marker=".")[0] for i in range(3)]
    plt.legend(handles, labels, loc='lower left', bbox_to_anchor=(0, 1.02) if small else (-2.4, 2.2), ncol=3, borderaxespad=0, frameon=False)
    plt.show()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("iter", metavar="iter", type=int, help="maximum iteration")
  parser.add_argument("K", metavar="K", type=int, help="number of clusters")
  parser.add_argument("--rand", metavar="rand", type=int, help="random state")
  parser.add_argument("--full", action="store_true", help="enable full mode")

  args = parser.parse_args()
  run_iris(small=not args.full, n_iter=args.iter, K=args.K, random_state=args.rand)
