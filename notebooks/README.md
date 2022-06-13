# Содержание

* `custom_nms` - nms на основе кластеризации для Yolo
* `kpoints_regression_yolo_batch` - нахождение kpoint с использованием yolo
* `pipeline_maskrcnn` - пайплайн для выделения лучших кропов по текстовому запросу ViLD -> Clip -> Annoy по valid LVIS
* `pipeline_yolo (valid)` - пайплайн для выделения лучших кропов по текстовому запросу Yolo -> Clip -> Annoy по valid LVIS
* `training_coco` - дообучение faster_rcnn модели на LVIS
* `training_lvis` - дообучение mask_rcnn модели на LVIS
* `yolo_demo_full` - ДЕМО для нахождения кропов по тексту, доразметки кропов и дообучения модели
* `assh_utils` - Batch iterator, доп. функции
* `yolo_utils` - CustomWrapper для Yolo с раскрученными порожками 
