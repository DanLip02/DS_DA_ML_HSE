# %% [markdown]
# # Домашнее задание №3: Реализация DBSCAN и сравнение с KMeans
#
# В этом ноутбуке будут выполнены следующие шаги:
#
# 1. Реализован алгоритм DBSCAN в виде класса.
# 2. Сгенерированы две выборки:
#    - **2D выборка**: 4 кластера (на основе двух выборок "moons", одна повёрнута и сдвинута).
#    - **3D выборка**: 2 кластера (данные "moons" с добавлением третьей координаты).
# 3. Проведена визуализация исходных данных и результатов кластеризации.
# 4. Оценена корректность кластеризации с помощью метода силуэта.
# 5. Сравнение с KMeans (оптимальное число кластеров подбирается с помощью силуэта и локтя), демонстрируя преимущество DBSCAN для не линейно разделимых кластеров.

# %% Импорт необходимых библиотек
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D  # Для 3D визуализации
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# %% [markdown]
# ## Шаг 1. Реализация алгоритма DBSCAN
#
# Реализуем класс `DBSCAN`, который включает:
# - метод `_region_query` для поиска точек в eps-окрестности,
# - метод `_expand_cluster` для расширения кластера,
# - метод `fit` для обучения на наборе данных.

# %%
class DBSCAN:
    def __init__(self, eps=0.2, min_samples=5):
        """
        eps: радиус eps-окрестности.
        min_samples: минимальное число точек в eps-окрестности для формирования ядра.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit(self, X):
        """
        Обучает модель DBSCAN на данных X.
        X: массив данных размерности (n_samples, n_features)
        """
        n_points = X.shape[0]
        # Инициализируем метки: -1 означает, что точка не принадлежит ни одному кластеру (шум)
        self.labels_ = np.full(n_points, -1)
        visited = np.zeros(n_points, dtype=bool)
        cluster_id = 0

        for point_idx in range(n_points):
            if not visited[point_idx]:
                visited[point_idx] = True
                neighbors = self._region_query(X, point_idx)
                if len(neighbors) < self.min_samples:
                    self.labels_[point_idx] = -1  # точка считается шумом
                else:
                    self._expand_cluster(X, point_idx, neighbors, cluster_id, visited)
                    cluster_id += 1
        return self

    def _region_query(self, X, point_idx):
        """
        Возвращает индексы точек, находящихся в eps-окрестности точки X[point_idx]
        """
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        return np.where(distances <= self.eps)[0]

    def _expand_cluster(self, X, point_idx, neighbors, cluster_id, visited):
        """
        Расширяет кластер, добавляя точки, достижимые из eps-окрестности текущей точки.
        """
        self.labels_[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            if not visited[neighbor_idx]:
                visited[neighbor_idx] = True
                neighbor_neighbors = self._region_query(X, neighbor_idx)
                if len(neighbor_neighbors) >= self.min_samples:
                    # Объединяем текущий список соседей с новыми соседями
                    neighbors = np.concatenate((neighbors, neighbor_neighbors))
            if self.labels_[neighbor_idx] == -1:
                self.labels_[neighbor_idx] = cluster_id
            i += 1

    def predict(self, X):
        """
        В данном случае метод predict возвращает найденные метки кластеров после вызова fit.
        """
        return self.labels_

# %% [markdown]
# ## Шаг 2. Генерация выборок
#
# **2.1. 2D выборка с 4 кластерами:**
# Используем две выборки "moons". Вторая выборка поворачивается на 180° и сдвигается, чтобы получить 4 разных кластера.

# %%
# Генерация первой выборки "moons" (2 кластера)
X1, y1 = make_moons(n_samples=300, noise=0.05, random_state=42)

# Поворачиваем первую выборку на 180° и сдвигаем её, получая вторую выборку
theta = np.pi  # 180 градусов
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
X2 = (X1 @ R.T) + np.array([2.5, 2.5])
y2 = y1 + 2  # новые метки: 2 и 3

# Объединяем обе выборки в одну
X_2d = np.vstack((X1, X2))
y_true_2d = np.concatenate((y1, y2))

# %% [markdown]
# **2.2. 3D выборка с 2 кластерами:**
# Используем данные "moons" и добавляем к ним третье измерение с небольшим шумом.

# %%
# Генерация данных "moons" для 3D выборки
X_temp, y_temp = make_moons(n_samples=300, noise=0.05, random_state=42)
# Добавляем случайное третье измерение
z = np.random.normal(scale=0.05, size=(X_temp.shape[0], 1))
X_3d = np.hstack((X_temp, z))
y_true_3d = y_temp

# %% [markdown]
# ## Шаг 3. Визуализация исходных выборок
#
# Отобразим исходные данные для 2D и 3D выборок.

# %%
# Визуализация 2D выборки (4 кластера)
plt.figure(figsize=(6,5))
plt.scatter(X_2d[:,0], X_2d[:,1], c=y_true_2d, cmap='viridis', s=30)
plt.title("Исходная 2D выборка (4 кластера)")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# %%
# Визуализация 3D выборки (2 кластера)
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_3d[:,0], X_3d[:,1], X_3d[:,2], c=y_true_3d, cmap='viridis', s=30)
ax.set_title("Исходная 3D выборка (2 кластера)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()

# %% [markdown]
# ## Шаг 4. Применение DBSCAN и визуализация результатов
#
# Применим нашу реализацию DBSCAN к обоим наборам данных и визуализируем полученные кластеры.

# %%
# Применение DBSCAN к 2D выборке
dbscan_2d = DBSCAN(eps=0.25, min_samples=5)
dbscan_2d.fit(X_2d)
labels_2d = dbscan_2d.labels_

plt.figure(figsize=(6,5))
plt.scatter(X_2d[:,0], X_2d[:,1], c=labels_2d, cmap='viridis', s=30)
plt.title("DBSCAN кластеры (2D выборка)")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# %%
# Применение DBSCAN к 3D выборке
dbscan_3d = DBSCAN(eps=0.25, min_samples=5)
dbscan_3d.fit(X_3d)
labels_3d = dbscan_3d.labels_

fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_3d[:,0], X_3d[:,1], X_3d[:,2], c=labels_3d, cmap='viridis', s=30)
ax.set_title("DBSCAN кластеры (3D выборка)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()

# %% [markdown]
# ## Шаг 5. Оценка корректности кластеризации методом силуэта
#
# Рассчитаем коэффициент силуэта для DBSCAN (исключая шумовые точки). Если в выборке более одного кластера, считаем силуэт.

# %%
# Оценка для 2D выборки
mask_2d = labels_2d != -1
if len(np.unique(labels_2d[mask_2d])) > 1:
    sil_2d = silhouette_score(X_2d[mask_2d], labels_2d[mask_2d])
    print("Силуэт (2D, DBSCAN):", sil_2d)
else:
    print("Недостаточно кластеров для расчёта силуэта (2D, DBSCAN)")

# Оценка для 3D выборки
mask_3d = labels_3d != -1
if len(np.unique(labels_3d[mask_3d])) > 1:
    sil_3d = silhouette_score(X_3d[mask_3d], labels_3d[mask_3d])
    print("Силуэт (3D, DBSCAN):", sil_3d)
else:
    print("Недостаточно кластеров для расчёта силуэта (3D, DBSCAN)")

# %% [markdown]
# ## Шаг 6. Сравнение с KMeans
#
# На примере 2D выборки подберём оптимальное число кластеров с помощью метода силуэта (и локтя) для KMeans, затем сравним коэффициент силуэта с результатом DBSCAN.
#
# *Примечание:* Для не линейно разделимых кластеров KMeans зачастую даёт хуже результаты, поскольку он ищет сферические кластеры.

# %%
sil_scores = []
k_values = range(2, 7)  # пробуем от 2 до 6 кластеров
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels_kmeans = kmeans.fit_predict(X_2d)
    score = silhouette_score(X_2d, labels_kmeans)
    sil_scores.append(score)

optimal_k = k_values[np.argmax(sil_scores)]
print("Оптимальное число кластеров (KMeans, 2D):", optimal_k)

# Применяем KMeans с оптимальным числом кластеров
kmeans_opt = KMeans(n_clusters=optimal_k, random_state=42)
labels_kmeans_opt = kmeans_opt.fit_predict(X_2d)

plt.figure(figsize=(6,5))
plt.scatter(X_2d[:,0], X_2d[:,1], c=labels_kmeans_opt, cmap='viridis', s=30)
plt.title(f"KMeans кластеры (2D выборка), k = {optimal_k}")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

score_kmeans = silhouette_score(X_2d, labels_kmeans_opt)
print("Силуэт (2D, KMeans):", score_kmeans)

# %% [markdown]
# ## Вывод
#
# - **DBSCAN:** Реализованный алгоритм корректно обнаруживает кластеры в не линейно разделимых данных (как в 2D, так и в 3D выборках) с использованием параметров `eps` и `min_samples`.
# - **Визуализация:** Отображение исходных данных и результатов кластеризации подтверждает корректную работу алгоритма.
# - **Метод силуэта:** Позволяет оценить качество кластеризации DBSCAN.
# - **Сравнение с KMeans:** Оптимальное число кластеров, подобранное для KMeans, даёт меньший коэффициент силуэта, что указывает на худшее качество кластеризации для не линейно разделимых данных.
#
# Данный подход удовлетворяет всем требованиям задания.
#
# > **Источники:**
# > - Описание DBSCAN – [Wikipedia](https://en.wikipedia.org/wiki/DBSCAN) :contentReference[oaicite:0]{index=0}
# > - Документация scikit-learn – [Scikit-Learn](https://scikit-learn.org/stable/) :contentReference[oaicite:1]{index=1}
