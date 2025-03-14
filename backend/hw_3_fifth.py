import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class DBSCAN:
    def __init__(self, eps=0.05, min_samples=10):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit(self, X):
        n_points = X.shape[0]
        self.labels_ = np.full(n_points, -1)
        visited = np.zeros(n_points, dtype=bool)
        cluster_id = 0

        for point_idx in range(n_points):
            if not visited[point_idx]:
                visited[point_idx] = True
                neighbors = self._region_query(X, point_idx)
                if len(neighbors) < self.min_samples:
                    self.labels_[point_idx] = -1
                else:
                    self._expand_cluster(X, point_idx, neighbors, cluster_id, visited)
                    cluster_id += 1
        return self

    def _region_query(self, X, point_idx):
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        return np.where(distances <= self.eps)[0]

    def _expand_cluster(self, X, point_idx, neighbors, cluster_id, visited):
        self.labels_[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            if not visited[neighbor_idx]:
                visited[neighbor_idx] = True
                neighbor_neighbors = self._region_query(X, neighbor_idx)
                if len(neighbor_neighbors) >= self.min_samples:
                    neighbors = np.concatenate((neighbors, neighbor_neighbors))
            if self.labels_[neighbor_idx] == -1:
                self.labels_[neighbor_idx] = cluster_id
            i += 1


# Генерация данных с двумя лунными структурами и шумом в 2D, одной луной и шумом в 3D
def generate_data():
    X1, _ = make_moons(n_samples=500, noise=0.1, random_state=42)
    X2, _ = make_moons(n_samples=500, noise=0.1, random_state=43)
    X2[:, 0] += 2.5  # Смещение второй луны вправо
    X2[:, 1] += 2.5  # Смещение второй луны вверх

    noise = np.random.uniform(low=-1, high=4, size=(200, 2))  # Шум в 2D
    X_2d = np.vstack((X1, X2, noise))

    X3, _ = make_moons(n_samples=500, noise=0.1, random_state=42)
    z = np.random.normal(scale=0.3, size=(X3.shape[0], 1))  # Добавляем третью ось (высоту)
    noise_3d = np.random.uniform(low=-1, high=1, size=(200, 3))  # Шум в 3D
    X_3d = np.vstack((np.hstack((X3, z)), noise_3d))

    return X_2d, X_3d


# Метод локтя и силуэтный анализ
def find_optimal_k(X):
    distortions = []
    silhouette_scores = []
    K = range(2, 10)

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
        distortions.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))

    best_k = K[np.argmax(silhouette_scores)]
    print(f'Оптимальное число кластеров: {best_k}')
    return best_k


# Применение кластеризации
def clustering(X_2d, X_3d, optimal_k):
    dbscan_2d = DBSCAN(eps=0.15, min_samples=10).fit(X_2d)
    labels_2d = dbscan_2d.labels_
    dbscan_3d = DBSCAN(eps=0.15, min_samples=10).fit(X_3d)
    labels_3d = dbscan_3d.labels_

    kmeans_2d = KMeans(n_clusters=optimal_k, random_state=42).fit(X_2d)
    labels_kmeans_2d = kmeans_2d.labels_
    kmeans_3d = KMeans(n_clusters=optimal_k, random_state=42).fit(X_3d)
    labels_kmeans_3d = kmeans_3d.labels_

    return labels_2d, labels_3d, labels_kmeans_2d, labels_kmeans_3d


# Визуализация кластеров
def plot_clusters(X, labels, title, is_3d=False):
    plt.figure(figsize=(7, 6))
    if is_3d:
        ax = plt.axes(projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis', s=30)
    else:
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30)
    plt.title(title)
    plt.show()


# Основной вызов функций
X_2d, X_3d = generate_data()
optimal_k = find_optimal_k(X_2d)
labels_2d, labels_3d, labels_kmeans_2d, labels_kmeans_3d = clustering(X_2d, X_3d, optimal_k)

plot_clusters(X_2d, labels_2d, "DBSCAN 2D")
plot_clusters(X_3d, labels_3d, "DBSCAN 3D", is_3d=True)
plot_clusters(X_2d, labels_kmeans_2d, "KMeans 2D")
plot_clusters(X_3d, labels_kmeans_3d, "KMeans 3D", is_3d=True)
