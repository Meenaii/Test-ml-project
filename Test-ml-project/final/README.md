# 📌 Описание
В данном проекте решается задача бинарной классификации: предсказать, совершит ли пользователь целевое действие на сайте сервиса по аренде автомобилей.
[Данные](https://drive.google.com/drive/folders/1rA4o6KHH-M2KMvBLHp5DZ5gioF2q7hZw) представляют собой логи из Google Analytics: ga_sessions и ga_hits. Объём данных — ~4.3 ГБ.

# Структура проекта

```bash
.
├── data/                    # Сырые входные данные (.csv/.pickle) 
│                            # ВАЖНО: данной директории нет в проекте из-за огромных размеров, создать её нужно самостоятельно
├── model/
│   ├── metada.json          # Данные, собранные из входных данных, использующиеся при обработке данных для предсказания
│   ├── classifier_pipe.pkl  # Cериализованная модель
│   ├── preprocess.py        # Функции по очистке и извлечению признаков
│   ├── preprocess_predict.py # Обработка данных для предсказания
│   └── train.py             # Обучение и оценка модели
├── predict_data/            # Данные для предсказаний, а так же файл с результатом работы модели после исполнения
├── predict.py               # Применение обученной модели к новым данным
└── README.md                # Этот файл

```
# Обработка данных

* Объединение таблиц sessions и hits по session_id.

* Извлечение признаков:
    * поведенческие: посещали ли страницу авто, число просмотренных авто, глубина сессии, последнее просмотренное авто;
    * временные: час визита, месяц, день, день недели;
    * технические: устройство, ОС, рекламные признаки.
* Удаление выбросов (например, visit_number > 100).
* Обработка пропусков.
* Кодирование категориальных признаков и численных признаков через OneHot и Robust соответсвенно

# 🧪 Обучение
Использованы две модели: LogisticRegression и CatBoostClassifier.
Дисбаланс классов учтён с помощью class_weight='balanced'.
Оценка производилась по метрике ROC-AUC по кросс-валидации.

📈 Результаты

| Модель             | ROC-AUC |
|--------------------|---------|
| LogisticRegression | 81.2%   |
| CatBoostClassifier | 85.6%   |

# 🚀 Применение модели
Для получения предсказаний на новых данных:

```bash
predict.py
```
predict.py применяет модель к новым ga_sessions.csv и ga_hits.csv в директории predict_data/ и сохраняет результат в predictions.csv,
если в файле более одной строки, при одной строке также выводит результат и в терминал.

```python
import pandas as pd
import dill
from model.preprocess_predict import prepare_data

with open('model/classifier_pipe.pkl', 'rb') as file:
   model = dill.load(file)
#Загрузка данных
def main():
    df_hits = pd.read_csv('predict_data/ga_hits_session1.csv').drop(['event_value',
                                                    'hit_referer',
                                                    'hit_time',
                                                    'event_label'], axis=1)

    df_sessions =  pd.read_csv('predict_data/ga_sessinons_session1.csv', low_memory=False).drop(['device_model',
                                                                          'utm_keyword'], axis = 1)
```
# 📎 Примечания
Все признаки и фильтрации были зафиксированы на обучающем датасете.
При применении модели используется тот же список популярных городов, границы выбросов и параметры кодирования, что и при обучении (сохраняются в metadata.json).
