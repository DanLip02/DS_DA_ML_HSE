import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, DBSCAN as SklearnDBSCAN
from sklearn.datasets import make_moons
from sklearn.metrics.pairwise import euclidean_distances
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors


# Реализация DBSCAN (собственная версия)
class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit(self, X):
        n_samples = X.shape[0]
        visited = np.zeros(n_samples, dtype=bool)
        labels = np.full(n_samples, -1, dtype=int)  # -1 = шум
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
def generate_data():
    X1, _ = make_moons(n_samples=200, noise=0.03, random_state=42)
    X2, _ = make_moons(n_samples=200, noise=0.03, random_state=52)
    X2 += np.array([4.0, 1.0])  # Сдвигаем второй набор
    X_2d = np.vstack([X1, X2])

    X_circles, _ = make_moons(n_samples=500, noise=0.05, random_state=42)
    X_3d = np.zeros((X_circles.shape[0], 3))
    X_3d[:, :2] = X_circles
    X_3d[:, 2] = np.sin(X_circles[:, 0] * 3.5)
    return X_2d, X_3d


# Визуализация данных
def plot_original_data(X_2d, X_3d):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], s=30)
    plt.title("Исходные 2D данные")
    plt.xlabel("X")
    plt.ylabel("Y")

    ax = plt.subplot(1, 2, 2, projection='3d')
    ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], s=30)
    ax.set_title("Исходные 3D данные")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()


# Подбор параметров DBSCAN методом силуэта
def tune_dbscan(X, eps_values, min_samples_values):
    best_score = -1
    best_params = None
    for eps in eps_values:
        for min_samples in min_samples_values:
            db = SklearnDBSCAN(eps=eps, min_samples=min_samples).fit(X)
            labels = db.labels_
            mask = labels != -1
            if len(np.unique(labels[mask])) > 1:
                score = silhouette_score(X[mask], labels[mask])
                if score > best_score:
                    best_score = score
                    best_params = (eps, min_samples)
    return best_params, best_score


# Функция для сравнения DBSCAN (собственная и sklearn)
def compare_with_sklearn_dbscan(X, eps, min_samples, title):
    dbscan_custom = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels_custom = dbscan_custom.labels_

    dbscan_sklearn = SklearnDBSCAN(eps=eps, min_samples=min_samples)
    labels_sklearn = dbscan_sklearn.fit_predict(X)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].scatter(X[:, 0], X[:, 1], c=labels_custom, cmap='tab20', s=50, edgecolors='k')
    axs[0].set_title(f'Собственная DBSCAN\nКластеров: {len(set(labels_custom)) - (1 if -1 in labels_custom else 0)}')

    axs[1].scatter(X[:, 0], X[:, 1], c=labels_sklearn, cmap='tab20', s=50, edgecolors='k')
    axs[1].set_title(f'Sklearn DBSCAN\nКластеров: {len(set(labels_sklearn)) - (1 if -1 in labels_sklearn else 0)}')

    plt.show()

    for name, labels in zip(["Собственная", "Sklearn"], [labels_custom, labels_sklearn]):
        unique_labels = np.unique(labels)
        mask = labels != -1
        if len(np.unique(labels[mask])) > 1:
            score = silhouette_score(X[mask], labels[mask])
            print(f'{name} DBSCAN - Силуэт: {score:.2f}')
        else:
            print(f'{name} DBSCAN - Недостаточно кластеров для силуэта')


# Функция для сравнения с KMeans
def compare_with_kmeans(X):
    wcss = []
    max_k = 10
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, max_k + 1), wcss, 'bo-')
    plt.title('Метод локтя для KMeans')
    plt.xlabel('Число кластеров')
    plt.ylabel('WCSS')
    plt.show()

    best_score = -1
    best_k = 1
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_k = k

    print(f'Оптимальное K: {best_k} (силуэт: {best_score:.2f})')


# Основной код
X_2d, X_3d = generate_data()
plot_original_data(X_2d, X_3d)

best_params_2d, _ = tune_dbscan(X_2d, np.linspace(0.15, 0.30, 8), range(3, 6))
compare_with_sklearn_dbscan(X_2d, eps=best_params_2d[0], min_samples=best_params_2d[1], title="Сравнение DBSCAN")

print("Сравнение с KMeans:")
compare_with_kmeans(X_2d)
