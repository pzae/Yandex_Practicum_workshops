# Трекинг движущихся объектов на конвейере

## Введение
**Описание проекта:** 
Renue – IT-компании из Екатеринбурга, которая разрабатывает высоконагруженные и отказоустойчивые решения для крупных российских заказчиков, требуется создать решение для отслеживания и сортировки мусора на конвейере – выделять пластиковые бутылки в общем потоке предметов.

**Цель проекта:**  
Создать модель потокового трекинга пластиковой тары на конвейере, которая выдает координаты центра обнаруженных объектов для каждого видеокадра.
Скорость обработки потока должна быть не более 100 мс.
Метрика оценки модели –  *MOTA* (Multiple Object Tracking Accuracy).

**Данные:**
- Предобученная модель детекции пластиковых бутылок и пример кода для ее запуска.
- Датасет с изображениями и разметкой в нескольких форматах: MOT, COCO, CVAT.
- Примеры видеозаписей работы конвейера.

## Методы

**Библиотеки:**

- Python, Ultralytics, Motmetrics, OpenCV, PyTorch, Numpy, SciPy, time, timedelta,  PIL, OS, Glob.

**Модели и алгоритмы:**

- *Ultralytics YOLO*(версия YOLOv10): популярная одноэтапная **модель обнаружения объектов** и сегментации изображений в реальном времени с высокой производительностью и точностью. Модель сохраняет уникальный идентификатор каждого обнаруженного в кадре объекта по мере продвижения видео [`Ultralitics Yolo`](https://docs.ultralytics.com/ru/modes/track/) . 


- *BoT-Sort* (Robust Associations Multi-Pedestrian Tracking): надежный **современный трекер**, сочетающий преимущества информации о движении и внешнем виде вместе с компенсацией движения камеры и более точным вектором состояния фильтра Калмана  [`BoT-Sort`](https://github.com/NirAharon/BoT-SORT) .


- *ByteTrack*: современный **алгоритм трекинга** c высокой производительностью и точностью [`ByteTrack`](https://github.com/ifzhang/ByteTrack) .


**Оценка качества:**

- Для оценки качества моделей мы использовали метрику ***MOTA*** (Multiple Object Tracking Accuracy) из библиотеки py-motmetrics [`py-motmetrics`](https://github.com/cheind/py-motmetrics):  метрика рассчитывает правильность обнаружения объектов в кадре на основе данных об истинных значениях и результатах отслеживания.  

$$
\displaystyle MOTA=1-\frac{\sum_t FN_t+FP_t+IDS_t}{\sum_t GT_t}\
$$

$$
\displaystyle\textrm {Вычитаем из единицы отношение количества ошибок к количеству истинных объектов, рассчитанное по кадрам.}
$$

- Из вспомогательных метрик мы использовали Precision, Recall и **MOTP** (Multiple Object Tracking Precision) из библиотеки py-motmetrics [`py-motmetrics`](https://github.com/cheind/py-motmetrics): MOTP – это отношение между суммой расстояний между сопоставленными объектами и количеством таких сопоставлений.

$$
\displaystyle MOTP=\frac{\sum_{i,t} d_t^i}{\sum_t c_t}\
$$


МОТР хорошо оценивает ошибки определения расположения объекта в кадре, учитывая в результате только те объекты, которые удалось засечь трекеру и в значительной степени зависит от качества детектора.


## Структура исследования

Сложность задачи заключалась в подборе алгоритма, который, с одной стороны, покажет хорошую метрику сопровождения объектов, причем, не только на статичных кадрах, но и в реальном времени, а с другой, сделает это быстро – за время не превышающее 100 мс.    

В начале исследования мы создали и опробовали базовую модель (baseline) детектора объектов со встроенным трекером. В качестве Baseline мы выбрали предобученную модель YOLOv10 c трекером BoT-SORT по умолчанию. Мы проверили работу модели на датасете со 100 изображениями в фрмате JPG и на 8-секундном фрагменте видео в формате mp4. 

Модель сразу же показала хорошую метрику (MOTA = 0.912409), но скорость обработки одного кадра оставляла желать лучшего (125.75 мс). Результаты работы модели мы занесли в таблицу. 

На следующем шаге мы исследовали и применяли различные варианты улучшения работы модели: тестировали модель с разными сочетаниями гиперпараметров и алгоритмов трекинга, визуализируя результаты.  


Одновременно, мы анализировали и сохраняли полученные результаты.

Также мы запускали модель обнаруживать и сопровождать объекты на потоковом видео. Причем, мы выбирали видео в разном разрешении.  

На финальном шаге мы аккумулировали полезные материалы и код в репозиторий, а результаты – в итоговую таблицу, сформулировали и оформили выводы. 

Создали приложение на платформе [`streamlit`](https://streamlit.io), в котором можно увидеть работу модели.  


## Результаты

**Таблица с результатами:**

|№| Модель | Трекер |Формат входного изображения| Количество кадров| Параметр imgsz| Recall | Precision | MOTA | MOTP | Время обработки кадра, ms |Общее время вывода, min:s|
|:---|:---- |:------ |:---------|:------- | :---------|:------ |:--------- | :------- |:------ |:--------- | :------- |
|1|**YOLOv10(baseline)**|**BoT-SORT**|**jpg**|**100**|**640**|0.928401|0.997436|**0.921241**|0.09007|**113.86**|00:11.385932|
|2|YOLOv10|BoT-SORT |jpg|100|480|0.918854|1.0|0.914081|0.088913|116.88|00:11.687704|
|3|YOLOv10|BoT-SORT|jpg|100|352|0.909308|1.0|0.906921|0.098021|107.23|00:10.722746|
|||||||||||||
|4|YOLOv10|**BoT-SORT**|**jpg**|**9000**|**480**|0.921665|0.998998|**0.917861**|0.081014|**120.85**|18:07.656098|
|5|YOLOv10|BoT-SORT|jpg|9000|352|0.912445|0.999553|0.909286|0.092041|106.51|15:58.631334|
|||||||||||||
|6|YOLOv10|**ByteTrack**|**jpg**|**100**|**640**|0.704057|0.760309|**0.477327**|0.257375|**59.02**| 00:05.901877|
|7|YOLOv10|ByteTrack|jpg|100|480|0.735084|0.8|0.546539|0.259656|46.07|00:04.606776|
|8|YOLOv10|**ByteTrack**|**jpg**|**100**|**352**|0.75895|0.834646|**0.603819**|0.259355|**50.36**|00:05.035684|
|||||||||||||
|9|YOLOv10|**ByteTrack**|jpg|**9000**|**480**|0.728826|0.798723|**0.535557**|0.251003|**55.15**|08:16.354860|
|10|YOLOv10|ByteTrack|jpg|9000|352|0.717414|0.795472|0.524339|0.258531|	53.76|08:03.796729|
|||||||||||||
|11|YOLOv10|**BoT-SORT**|**mp4**|**99**|**640**|0.924574|0.997375|**0.917275**|0.090119|**107.57**| 00:10.649339|
|12|YOLOv10|BoT-SORT|mp4|99|480|0.917275|1.0|0.912409|0.089113|104.56|00:10.351197|
|13|YOLOv10|BoT-SORT|mp4|99|352|0.907543|1.0|0.905109|0.098528|99.12|00:09.812414|
|||||||||||||
|14|YOLOv10|ByteTrack|mp4|99|640|0.70073|0.761905|0.476886|0.258298|59.10| 00:05.850563|
|15|YOLOv10|ByteTrack|mp4|99|480|0.73236|0.798408|0.545012|0.260442|44.61|00:04.416384|
|16|YOLOv10|**ByteTrack**|**mp4**|**99**|**352**|0.759124|0.836461|**0.605839**|0.259733|**35.43**|00:03.507705|
|||||||||||||
|17|YOLOv10|BoT-SORT|mp4|21565|480|0.921751|0.998905|0.917861|0.080872|104.42|37:31.719363|
|18|YOLOv10|**BoT-SORT**|**mp4**|**21565**|**352**|0.912338|0.999576|**0.909136**|0.092135|**94.91**|34:06.678558|
|||||||||||||
|19|YOLOv10|ByteTrack|mp4|21565|480|0.728031|0.797721|0.533816|0.251346|48.11|17:17.492632|
|20|YOLOv10|**ByteTrack**|**mp4**|**21565**|**352**|0.715566|0.793385|**0.52075**|0.259103|**45.86**|16:29.013377|



**Анализ результатов:**

- Судя по результатам, модель очень хорошо справляется с разными форматами и разным количеством кадров, показывая высокую метрику.

- При уменьшении размера изображения для вывода (imgsz) метрика снижалась: модели удавалось обнаружить меньше объектов. При этом, скорость обработки кадра, предсказуемо, росла.     

- При работе с большим количеством кадров модель показывала чуть меньшую метрику: видимо, чем крупнее датасет, тем  больше накапливается ошибок. 

- Трекер BoT-SORT показал более высокие результаты метрик, а ByteTrack был значительно быстрее.  


## Выводы

**Выбор модели для заявленных целей:** 

Для заявленной цели – создать модель потокового трекинга пластиковой тары на конвейере, которая выдает координаты центра обнаруженных объектов для каждого видеокадра со скоростью обработки потока не более 100 мс – больше подойдет предобученная модель YOLOv10 с трекером BoT-SORT.


**Возможные направления развития проекта:**

- **Использовать другие модели и трекеры:** Deep SORT, FairMOT, DEVA, CSRT.


- **Обучить модель на датасете большего размера:** Обучение моделей на большем датасете может улучшить их качество и обобщающую способность.


- **Изменить положение записывающей камеры:** 
   - Переместить камеру дальше от места подачи мусора, чтобы исключить "влетание" мусора в кадр и дальнейшее его неконтролируемое передвижение по конвейерной ленте.
   - Поднять камеру выше над транспортерной лентой. При том же угле обзора это расширит поле зрения камеры и позволит модели дольше сопровождать  объекты, что, в свою очередь, даст выигрыш в качестве трекинга.


- **Использование Streamlit/FastAPI для более удобного трекинга и детекции объектов и анализа результатов**

**Структура репозитория:**

| #    | Наименование файла                | Описание   |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1.   | [README.md](https://github.com/FedorSafonov/tracking-objects-on-a-conveyor-belt/blob/main/README.md) | Представлена основная информация по проекту и его результатах   |
| 2.   | [config.py](https://github.com/FedorSafonov/tracking-objects-on-a-conveyor-belt/blob/main/config.py) | Класс, в котором задаются константы и пути к файлам   |
| 3.   | [init_model.py](https://github.com/FedorSafonov/tracking-objects-on-a-conveyor-belt/blob/main/init_model.py) | Класс, который инициализирует работу модели детекции и трекинга объектов и всех функций необходимых для визуализации их работы и расчета метрик   |
| 4.   | [main.py](https://github.com/FedorSafonov/tracking-objects-on-a-conveyor-belt/blob/main/main.py) | Код запуска всех методов имеющихся в классах    |
| 5.   | [requirements.txt](https://github.com/FedorSafonov/tracking-objects-on-a-conveyor-belt/blob/main/requirements.txt) | Список всех библиотек и их версии, необходимых для установки в виртуальной среде для запуска кода проекта   |
| 6.   | [streamlit_yolo.py](https://github.com/FedorSafonov/tracking-objects-on-a-conveyor-belt/blob/main/streamlit_yolo.py) | Веб приложение в стримлит с моделью заказчика YOLA |
| 7.   | [report.md](https://github.com/FedorSafonov/tracking-objects-on-a-conveyor-belt/blob/main/report.md) | Отчёт о проделанной работе |
| 8.   | [Renue_group_5.ipynb](https://github.com/FedorSafonov/tracking-objects-on-a-conveyor-belt/blob/main/Renue_group_5.ipynb) | Тетрадка с ходом исследования |
| 9.   | [Presentation_Renue.pdf](https://github.com/FedorSafonov/tracking-objects-on-a-conveyor-belt/blob/main/Presentation_Renue.pdf) | Презентация проекта |

**Ссылки:**

- [`Модель Ultralitics Yolo`](https://docs.ultralytics.com/ru/modes/track/) 
- [`Трекер BoT-Sort`](https://github.com/NirAharon/BoT-SORT) 
- [`Трекер ByteTrack`](https://github.com/ifzhang/ByteTrack) 
- [`Библиотека py-motmetrics`](https://github.com/cheind/py-motmetrics) 


## Команда проекта

- [Федор Сафонов (TeamLead)](https://github.com/FedorSafonov)
- [Анна Йорданова](https://github.com/A-Yordanova)
- [Юрий Кашин](https://github.com/yakashin)
- [Александр Вотинов](https://github.com/VotinovAlS)
- [Гульшат Зарипова](https://github.com/gulshart)
- [Сергей Пашкин](https://github.com/DrSartoriuss)
- [Александр Глазунов](https://github.com/pzae)

