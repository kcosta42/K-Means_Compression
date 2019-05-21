import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from core.kmeans import random_initialisation, find_closest_centroids, compute_centroids, cost_function

def run_pixels(X, n_iter=10, K=16, random_state=None):
  print("Applying K-Means to compress an image.\n")
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

  idx = find_closest_centroids(X, centroids)
  X_recovered = centroids[idx.astype(int)].astype(int)
  return X_recovered

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("iter", metavar="iter", type=int, help="maximum iteration")
  parser.add_argument("K", metavar="K", type=int, help="number of clusters")
  parser.add_argument("image", type=str, help="path to image")
  parser.add_argument("--rand", metavar="rand", type=int, default=None, help="random state")

  args = parser.parse_args()
  original = np.array(Image.open(args.image))
  size = original.shape[0]

  original = original.reshape(size * size, 3)
  _, axs = plt.subplots(1, 2, figsize=(10, 7))
  axs[0].title.set_text("Original")
  axs[0].imshow(original.reshape(size, size, 3))

  compressed = run_pixels(original, n_iter=args.iter, K=args.K, random_state=args.rand)
  axs[1].title.set_text(f"Compressed, with {args.K} colors.")
  axs[1].imshow(compressed.reshape(size, size, 3))

  plt.show()
