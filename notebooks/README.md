# Содержание

* `custom_nms` - mns на основе кластеризации для Yolo
* `kpoints_regression_fastrcnn` - нахождение kpoint с использованием faster_rcnn
* `kpoints_regression_yolo` - нахождение kpoint с использованием yolo
* `pipeline_maskrcnn` - пайплайн для выделения лучших кропов по текстовому запросу ViLD -> Clip -> Annoy по valid LVIS
* `pipeline_yolo (train)` - пайплайн для выделения лучших кропов по текстовому запросу Yolo -> Clip -> Annoy по train LVIS
* `pipeline_yolo (valid)` - пайплайн для выделения лучших кропов по текстовому запросу Yolo -> Clip -> Annoy по valid LVIS
* `training_coco` - дообучение faster_rcnn модели на LVIS
* `training_lvis` - дообучение mask_rcnn модели на LVIS
* `assh_utils` - Batch iterator, доп. функции. Проверить все ли функции нужны
* `yolo_utils` - CustomWrapper для Yolo с раскрученными порожками 