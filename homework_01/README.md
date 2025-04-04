# DS_DA_ML_HSE
Репозиторий создан для демонстрации домашних работ и семинаров в рамках курса "Анализ данных и машинное обучение".

## Структура репозитория

- **data_pandas**: Задача по анализу данных о игре **Murder in SQL City** с использованием библиотеки `pandas`.
  - **csv файлы**: Исходные данные для анализа.
  - **solution_hw_1.ipynb**: Решение задачи в формате Jupyter Notebook, которое включает в себя использование pandas (merge, filters, etc.) для обнаружения необходимых данных.
- **data_statistic**: Задача по статистическому анализу.
  - **csv файл**: Таблица для статистического анализа.
  - **solution_hw_2.ipynb**: Решение задачи в формате Jupyter Notebook, включающее статистические тесты и анализ данных.

## Задача 1: Анализ данных по игре **Murder in SQL City**

### Описание задачи
В этой задаче нужно проанализировать данные, связанные с игрой **Murder in SQL City**. Данные включают в себя информацию о различных событиях и действиях в игре, таких как их crime_report, interview, person другие таблицы.

Задача включает в себя:
- Загрузку и предварительную обработку данных.
- Использование встроенных методов для фильтрации, сортировки и мерджа таблиц
### Используемые инструменты:
- `pandas`: для работы с данными.

### Структура папки:

data_pandas/ \
├── crime_scene_report.csv # Исходные данные об игре "Murder in SQL City" \
├── drivers_license.csv \
├── facebook_event_checkin.csv \
├── get_fit_now_check_in.csv \
├── get_fit_now_member.csv \
├── income.csv \
├── interview.csv \
├── person.csv \
└── solution_hw_1.ipynb # Решение задачи

## Задача 2: Статистический анализ данных

### Описание задачи
В этой задаче необходимо провести статистический анализ с использованием данных, представленных в таблице CSV. Задача включает в себя:
- Применение статистических тестов для оценки значимости различий между группами.
- Оценку метрик удержания игроков.
- Использование тестов на нормальность, а также хи-квадрат тестов и Z-тестов для проверки гипотез.

### Используемые инструменты:
- `pandas`: для работы с данными.
- `scipy`: для статистических тестов.
- `seaborn`: для визуализации данных
- `matplotlib.pyplot`: для визуализации данных

### Структура папки:

data_statistics/\
├── cookie_cats.csv\
└── solution_hw_2.ipynb # Решение задачи