import numpy as np

def random_initialisation(X, K, random_state=None):
  if random_state:
    np.random.seed(random_state)

  indices = np.random.choice(X.shape[0], K, replace=False)
  centroids = X[indices]
  return centroids

def find_closest_centroids(X, centroids):
  K = centroids.shape[0]
  idx = np.zeros([X.shape[0], 1])

  idx_val = np.linalg.norm(X - centroids[0], axis=1) ** 2
  idx = np.zeros(X.shape[0])
  for i in range(1, K):
    new_val = np.linalg.norm(X - centroids[i], axis=1) ** 2
    idx[new_val < idx_val] = i
    idx_val[new_val < idx_val] = new_val[new_val < idx_val]
  return idx

def compute_centroids(X, idx, K):
  centroids = np.zeros([K, X.shape[1]])

  for i in range(K):
    if len(X[idx == i]) != 0 :
      centroids[i] = np.sum(X[idx == i], axis=0) / len(X[idx == i])
  return centroids

def cost_function(X, centroids, idx):
  return (sum(np.linalg.norm(X - centroids[idx.astype(int)], axis=1) ** 2) / X.shape[0])
