import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.datasets import make_moons
from sklearn.metrics.pairwise import euclidean_distances
from mpl_toolkits.mplot3d import Axes3D


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


# Генерация данных
# 2D данные: 4 кластера
X1, y1 = make_moons(n_samples=200, noise=0.05, random_state=42)
X2, y2 = make_moons(n_samples=200, noise=0.05, random_state=52)
X2 += np.array([2.5, 0])
X_2d = np.vstack([X1, X2])

# 3D данные: 2 кластера
X_circles, y_circles = make_moons(n_samples=500, noise=0.05, random_state=42)
X_3d = np.zeros((X_circles.shape[0], 3))
X_3d[:, :2] = X_circles
X_3d[:, 2] = np.sin(X_circles[:, 0] * 3)  # Усиленное нелинейное преобразование

# Визуализация данных
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_2d[:, 0], X_2d[:, 1], s=50)
plt.title("2D данные (4 кластера)")

ax = plt.subplot(1, 2, 2, projection='3d')
ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], s=50)
plt.title("3D данные (2 кластера)")
plt.show()


def apply_dbscan(X, eps, min_samples, title):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(X)
    labels = dbscan.labels_

    # Визуализация
    if X.shape[1] == 2:
        plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
    else:
        ax = plt.figure().add_subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, s=50, cmap='viridis')
    plt.title(title)
    plt.show()

    # Проверка кластеров
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    if n_clusters > 1:
        mask = labels != -1
        score = silhouette_score(X[mask], labels[mask])
        print(f"Силуэт: {score:.2f}")
    else:
        print("Недостаточно кластеров для оценки силуэтта")


# Кластеризация 2D
print("Обработка 2D данных:")
apply_dbscan(X_2d, eps=0.3, min_samples=5, title="DBSCAN 2D")

# Кластеризация 3D
print("\nОбработка 3D данных:")
apply_dbscan(X_3d, eps=0.5, min_samples=5, title="DBSCAN 3D")


def compare_with_kmeans(X, true_clusters):
    # Метод локтя
    wcss = []
    max_k = 10
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, max_k + 1), wcss)
    plt.title("Метод локтя")
    plt.xlabel("Количество кластеров")
    plt.show()

    # Оптимальный k по силуэту
    best_score = -1
    best_k = 1
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_k = k

    print(f"Лучший KMeans: k={best_k}, силуэт={best_score:.2f}")

    # Сравнение с DBSCAN
    dbscan = DBSCAN(eps=0.3 if X.shape[1] == 2 else 0.5,
                    min_samples=5)
    dbscan.fit(X)
    dbscan_labels = dbscan.labels_

    unique_labels = set(dbscan_labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    if n_clusters > 1:
        mask = dbscan_labels != -1
        dbscan_score = silhouette_score(X[mask], dbscan_labels[mask])
        print(f"DBSCAN силуэт: {dbscan_score:.2f}")
    else:
        print("DBSCAN не нашёл кластеры")


print("\nСравнение для 2D данных:")
compare_with_kmeans(X_2d, 4)

print("\nСравнение для 3D данных:")
compare_with_kmeans(X_3d, 2)