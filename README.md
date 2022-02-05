# Comptech: Предсказание спроса и планирование рабочих смен

Продукт представляет собой реализацию проекта «Предсказание спроса и планирование рабочих смен» в рамках зимней школы [CompTech School 2022](https://comptechschool.com/).

## Структура репозитория

```
|    ├── config
|       └── config.yaml -- Конфигурационный файл
|    ├── data
|       ├── X_test_preprocessed.csv -- Тестовые данные после нормализации
|       ├── X_train_preprocessed.csv -- Тренировочные данные после нормализации
|       ├── X_val_preprocessed.csv -- Валидационные данные после нормализации
|       ├── celebrations.txt -- Даты неофициальных праздников
|       ├── dayoffs.txt -- Даты нерабочих дней
|       ├── orders.csv -- Исходные данные о количестве заказов
|       ├── partners_delays.csv -- Исходные данные об опозданиях курьеров
|    ├── models
|       └── lgbm_model_1.pkl -- Сериализованная обученная модель
|    ├── notebooks
|       ├── comptech_linprog.ipynb -- Расчет рабочих смен (task 3)
|       ├── comptech_task2_statistical_approach.ipynb -- Статистический подход к оптимальному распределению курьеров (task 2)
|       ├── comptech_prophet.ipynb -- Эксперименты с моделью Prophet (task 1)
|       ├── test_scripts.ipynb -- Тесты для функций извлечения признаков
|       └── сomptech_lgbm.ipynb -- Эксперименты с моделями LightGBM (task 1)
|    ├── src
|       ├── bootstrap.py -- Получение медианы количества курьеров на один заказ
|       ├── inference.py -- Функции для предсказания
|       ├── lgbm_model.py -- Класс модели
|       ├── linprog.py -- Расчет рабочих смен симплекс методом
|       ├── preprocessing.py -- Функции для предобработки данных
|       └── utils.py -- Вспомогательные функции
|    ├── start.py -- Скрипт для запуска пайплайна инференса
│    └── requirements.txt -- Файл зависимостей 
```

## Первая задача

**Цель**: спрогнозировать количество заказов по часам на 7 дней вперед. 

**Выбранный подход**: собственная модель из нескольких базовых моделей LightGBM (по одной на каждый день).

**Метрика**: MAPE (Mean absolute percentage error)

### Запуск пайплайна предсказания

Для запуска модуля необходимо запустить команду установки зависимостей из корня проекта:

`pip install -r requirements.txt`

Затем запустить из терминала основной скрипт `start.py`, при необходимости скорректировав параметры работы пайплайна в файле `config/config.yaml`

По умолчанию для предсказания рассчитываются для семи дней от 2021-11-18. Для предсказания с другой даты (диапазон тестовой выборки 2021-11-17 - 2021-11-30), поддерживается аргумент date: `start.py --date 2021-11-20`

### Результат

После выполнения скрипта start.py создается файл с почасовым распределением заказов на определенную дату *date* с именем `hours_distribution_from_{date}.csv`, а также файл 
`couriers_by_hours_from_{date}.csv` с почасовым распределением курьеров.

Значения MAPE для валидационной / тестовой выборок:


|Days         | LGBMWeekModel | LGBMWeekModelTopFeatures|
|-------------|:-------------:|:-----------------------:|
|1            | 0.213 / 0.183 |      0.215 / 0.181      |   
|2            | 0.219 / 0.181 |      0.219 / 0.181      |
|3            | 0.222 / 0.182 |      0.222 / 0.182      |
|4            | 0.224 / 0.181 |      0.224 / 0.182      |
|5            | 0.228 / 0.190 |      0.227 / 0.192      |
|6            | 0.232 / 0.196 |      0.231 / 0.196      |
|7            | 0.235 / 0.184 |      0.237 / 0.184      |


## Вторая задача

**Цель**: поиск оптимального количества курьеров на каждый час.

**Выбранный подход**: константное решение (в качестве множителя взята медиана количества партнеров в расчете на заказ).

## Третья задача

**Цель**: получения оптимального распределения курьеров по сменам.

**Выбранный подход**: симплекс-метод.

## Команда

- Мельникова Маргарита - Data Analyst
- Желтова Кристина - ML Engineer
- Сарыглар Орлан - технический писатель
