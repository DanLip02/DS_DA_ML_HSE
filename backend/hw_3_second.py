import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.datasets import make_moons
from sklearn.metrics.pairwise import euclidean_distances
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors


class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit(self, X):
        n_samples = X.shape[0]
        visited = np.zeros(n_samples, dtype=bool)
        labels = np.full(n_samples, -1, dtype=int)
        cluster_id = 0

        for i in range(n_samples):
            if not visited[i]:
                visited[i] = True
                neighbors = self.region_query(X, i)
                if len(neighbors) < self.min_samples:
                    labels[i] = -1
                else:
                    self.expand_cluster(X, i, neighbors, cluster_id, visited, labels)
                    cluster_id += 1
        self.labels_ = labels
        return self

    def expand_cluster(self, X, core_point_idx, neighbors, cluster_id, visited, labels):
        labels[core_point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            point_idx = neighbors[i]
            if not visited[point_idx]:
                visited[point_idx] = True
                new_neighbors = self.region_query(X, point_idx)
                if len(new_neighbors) >= self.min_samples:
                    neighbors = np.concatenate([neighbors, new_neighbors])
            if labels[point_idx] == -1:
                labels[point_idx] = cluster_id
            i += 1

    def region_query(self, X, point_idx):
        distances = euclidean_distances([X[point_idx]], X).ravel()
        return np.where(distances <= self.eps)[0]


# Генерация данных с явными кластерами
def generate_2d_data():
    # Первый набор лун
    X1, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

    # Второй набор лун с большим смещением
    X2, _ = make_moons(n_samples=300, noise=0.05, random_state=52)
    X2 = X2 @ np.array([[0.7, -0.7], [0.7, 0.7]]) + [3.5, 1.0]

    return np.vstack([X1, X2])


def generate_3d_data():
    # Создаем два четких кластера
    X1, _ = make_moons(n_samples=500, noise=0.05)
    X2 = X1 + [2.5, 1.0, 0.0]  # Сдвигаем второй кластер

    X_3d = np.vstack([X1, X2])
    X_3d[:, 2] = np.sin(X_3d[:, 0] * 3)  # Добавляем третье измерение
    return X_3d


def plot_data(X_2d, X_3d):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], s=30, edgecolors='k', alpha=0.7)
    plt.title("2D данные (2 пары лун)")

    ax = plt.subplot(1, 2, 2, projection='3d')
    ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], s=30, edgecolors='k', alpha=0.7)
    plt.title("3D данные (2 кластера)")
    plt.show()


def interactive_dbscan(X, max_eps=1.0, min_samples_range=(3, 15)):
    plt.figure(figsize=(12, 6))

    eps_slider = plt.Slider(
        ax=plt.axes([0.2, 0.02, 0.6, 0.03]),
        label='Eps',
        valmin=0.1,
        valmax=max_eps,
        valinit=0.5
    )

    min_samples_slider = plt.Slider(
        ax=plt.axes([0.2, 0.06, 0.6, 0.03]),
        label='Min Samples',
        valmin=min_samples_range[0],
        valmax=min_samples_range[1],
        valinit=5,
        valstep=1
    )

    def update(val):
        eps = eps_slider.val
        min_samples = int(min_samples_slider.val)

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(X)
        labels = dbscan.labels_

        plt.clf()
        if X.shape[1] == 2:
            plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab20',
                        s=50, edgecolors='k', alpha=0.8)
        else:
            ax = plt.subplot(111, projection='3d')
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels,
                       cmap='tab20', s=50, edgecolors='k', alpha=0.8)

        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        plt.title(f'Eps: {eps:.2f}, Min Samples: {min_samples}\nClusters: {n_clusters}')
        plt.draw()

    eps_slider.on_changed(update)
    min_samples_slider.on_changed(update)
    update(None)
    plt.show()


# Основной блок выполнения
X_2d = generate_2d_data()
X_3d = generate_3d_data()
plot_data(X_2d, X_3d)

print("Интерактивная настройка для 2D данных:")
interactive_dbscan(X_2d, max_eps=0.8, min_samples_range=(3, 10))

print("\nИнтерактивная настройка для 3D данных:")
interactive_dbscan(X_3d, max_eps=1.2, min_samples_range=(5, 15))


# Финализация параметров после ручной настройки
def final_clustering():
    # Параметры для 2D (подобраны интерактивно)
    print("\nФинальная кластеризация 2D:")
    dbscan_2d = DBSCAN(eps=0.45, min_samples=6)
    dbscan_2d.fit(X_2d)
    plot_results(X_2d, dbscan_2d.labels_, "2D данные")

    # Параметры для 3D (подобраны интерактивно)
    print("\nФинальная кластеризация 3D:")
    dbscan_3d = DBSCAN(eps=0.85, min_samples=8)
    dbscan_3d.fit(X_3d)
    plot_results(X_3d, dbscan_3d.labels_, "3D данные")


def plot_results(X, labels, title):
    plt.figure(figsize=(10, 6))

    if X.shape[1] == 2:
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab20',
                    s=50, edgecolors='k', alpha=0.8)
    else:
        ax = plt.subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels,
                   cmap='tab20', s=50, edgecolors='k', alpha=0.8)

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    plt.title(f'{title}\nНайдено кластеров: {n_clusters}')
    plt.colorbar(boundaries=np.unique(labels))
    plt.show()

    if n_clusters > 1:
        mask = labels != -1
        score = silhouette_score(X[mask], labels[mask])
        print(f"Силуэт: {score:.2f}")


final_clustering()