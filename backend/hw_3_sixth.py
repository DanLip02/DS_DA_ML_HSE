import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, DBSCAN as SklearnDBSCAN
from mpl_toolkits.mplot3d import Axes3D
import warnings
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


class DBSCAN:
    def __init__(self, eps=0.2, min_samples=5, scale_data=True):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        self.scale_data = scale_data  # Флаг для масштабирования
        self.scaler = StandardScaler() if scale_data else None

    def fit(self, X):
        if self.scale_data:
            X = self.scaler.fit_transform(X)  # Масштабируем данные

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

def generate_data():
    # Генерация первой выборки "moons" (2 кластера)
    X1, y1 = make_moons(n_samples=700, noise=0.01, random_state=42)

    # Поворачиваем первую выборку на 180° и сдвигаем её, получая вторую выборку
    theta = np.pi / 2  # 180 градусов
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    X2 = (X1 @ R.T) + np.array([1.5, 3])
    y2 = y1 + 1  # новые метки: 2 и 3

    # Объединяем обе выборки в одну
    X_2d = np.vstack((X1, X2))
    y_true_2d = np.concatenate((y1, y2))

    # %% [markdown]
    # **2.2. 3D выборка с 2 кластерами:**
    # Используем данные "moons" и добавляем к ним третье измерение с небольшим шумом.

    # %%
    # Генерация данных "moons" для 3D выборки
    X_temp, y_temp = make_moons(n_samples=700, noise=0.01, random_state=42)
    # Добавляем случайное третье измерение
    z = np.random.normal(scale=0.1, size=(X_temp.shape[0], 1))
    X_3d = np.hstack((X_temp, z))
    y_true_3d = y_temp

    return X_2d, y_true_2d, X_3d, y_true_3d

#
# def generate_data():
#     # --- Оригинальная генерация ---
#     X1, y1 = make_moons(n_samples=300, noise=0.05, random_state=42)
#
#     # Добавляем небольшой шум
#     X1 += np.random.normal(scale=0.05, size=X1.shape)
#
#     # Поворот и сдвиг второй выборки
#     theta = np.pi / 2  # 180 градусов
#     R = np.array([[np.cos(theta), -np.sin(theta)],
#                   [np.sin(theta), np.cos(theta)]])
#     X2 = (X1 @ R.T) + np.array([1.5, 3])
#
#     # Добавляем шум ко второй выборке
#     X2 += np.random.normal(scale=0.05, size=X2.shape)
#
#     y2 = y1 + 1  # Смещаем метки
#
#     # Объединение исходных данных
#     X_2d_original = np.vstack((X1, X2))
#     y_2d_original = np.concatenate((y1, y2))
#
#     # --- Альтернативная генерация (с трансформациями) ---
#     X1_alt, y1_alt = make_moons(n_samples=700, noise=0.03)
#     X2_alt, y2_alt = make_moons(n_samples=(560, 350), noise=0.03)
#
#     # Трансформация X1
#     X1_alt[:, 0] *= -1
#     X1_alt[:, 0][y1_alt == 1] -= 0.3
#
#     # Трансформация X2
#     X2_alt[y2_alt == 0] *= 1.6
#     X2_alt[:, 0][y2_alt == 1] += 0.3
#
#     # Добавляем шум к альтернативным данным
#     X1_alt += np.random.normal(scale=0.05, size=X1_alt.shape)
#     X2_alt += np.random.normal(scale=0.05, size=X2_alt.shape)
#
#     # Объединение альтернативных данных
#     X_2d_alt = np.vstack([X1_alt, X2_alt])
#     y_2d_alt = np.concatenate([y1_alt, y2_alt + 2])  # Смещение меток для X2_alt
#
#     # --- 3D выборка ---
#     X_temp, y_temp = make_moons(n_samples=700, noise=0.05, random_state=42)
#
#     # Добавляем случайное третье измерение
#     z = np.random.normal(scale=0.12, size=(X_temp.shape[0], 1))  # Увеличенный шум для 3D
#
#     X_3d = np.hstack((X_temp, z))
#     y_3d = y_temp
#
#     return X_2d_alt, y_2d_alt, X_3d, y_3d

# Подбор оптимального числа кластеров
def find_optimal_k(X):
    distortions = []
    silhouette_scores = []
    K = range(2, 10)

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
        distortions.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax2 = ax1.twinx()
    ax1.plot(K, distortions, 'bo-', label='Инерция')
    ax2.plot(K, silhouette_scores, 'ro-', label='Силуэтный коэффициент')

    ax1.set_xlabel('Число кластеров')
    ax1.set_ylabel('Инерция', color='b')
    ax2.set_ylabel('Силуэт', color='r')
    plt.title('Метод локтя и силуэтный анализ')
    plt.show()

    best_k = K[np.argmax(silhouette_scores)]
    print(f'Оптимальное число кластеров: {best_k}')
    return best_k


# Применение кластеризации
def clustering(X_2d, X_3d, optimal_k):
    dbscan_2d = DBSCAN(eps=0.15, min_samples=5).fit(X_2d)
    labels_2d = dbscan_2d.labels_
    dbscan_3d = DBSCAN(eps=0.05, min_samples=20).fit(X_3d)
    labels_3d = dbscan_3d.labels_

    kmeans_2d = KMeans(n_clusters=optimal_k, random_state=42).fit(X_2d)
    labels_kmeans_2d = kmeans_2d.labels_
    kmeans_3d = KMeans(n_clusters=optimal_k, random_state=42).fit(X_3d)
    labels_kmeans_3d = kmeans_3d.labels_

    def safe_silhouette(X, labels, name):
        if len(set(labels)) > 1:
            score = silhouette_score(X, labels)
            print(f'Силуэт для {name}: {score:.3f}')
        else:
            print(f'Силуэт для {name}: невозможно вычислить (один кластер)')

    safe_silhouette(X_2d, labels_2d, 'DBSCAN 2D')
    safe_silhouette(X_3d, labels_3d, 'DBSCAN 3D')
    safe_silhouette(X_2d, labels_kmeans_2d, 'KMeans 2D')
    safe_silhouette(X_3d, labels_kmeans_3d, 'KMeans 3D')

    return labels_2d, labels_3d, labels_kmeans_2d, labels_kmeans_3d


# Функция сравнения с DBSCAN из scikit-learn
def compare_sklearn_dbscan(X, eps, min_samples):
    dbscan_sklearn = SklearnDBSCAN(eps=eps, min_samples=min_samples)
    labels_sklearn = dbscan_sklearn.fit_predict(X)

    plt.figure(figsize=(7, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels_sklearn, cmap='tab10', s=20)
    plt.title(f'Sklearn DBSCAN (eps={eps}, min_samples={min_samples})')
    plt.show()

    if len(set(labels_sklearn)) > 1:
        score = silhouette_score(X, labels_sklearn)
        print(f'Sklearn DBSCAN - Силуэт: {score:.3f}')
    else:
        print('Sklearn DBSCAN - Недостаточно кластеров для силуэта')


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
X_2d, y_true_2d, X_3d, y_true_3d = generate_data()
scaler = StandardScaler()

optimal_k = find_optimal_k(X_2d)
labels_2d, labels_3d, labels_kmeans_2d, labels_kmeans_3d = clustering(X_2d, X_3d, optimal_k)

plot_clusters(X_2d, labels_2d, "DBSCAN 2D")
plot_clusters(X_3d, labels_3d, "DBSCAN 3D", is_3d=True)
plot_clusters(X_2d, labels_kmeans_2d, "KMeans 2D")
plot_clusters(X_3d, labels_kmeans_3d, "KMeans 3D", is_3d=True)

# Сравнение с DBSCAN из scikit-learn
compare_sklearn_dbscan(X_2d, eps=0.15, min_samples=5)
compare_sklearn_dbscan(X_3d, eps=0.15, min_samples=5)

X_2d_filtered = X_2d[labels_2d != -1]
labels_2d_filtered = labels_2d[labels_2d != -1]

X_3d_filtered = X_3d[labels_3d != -1]
labels_3d_filtered = labels_3d[labels_3d != -1]

plot_clusters(X_2d_filtered, labels_2d_filtered, "DBSCAN FIL 2D")
plot_clusters(X_3d_filtered, labels_3d_filtered, "DBSCAN FIL 3D", is_3d=True)
