# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 10:40:34 2019

@author: Zonsor
"""
import os
import cv2
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
import re
numbers = re.compile(r'(\d+)')
setup_logger()


def show_example(dataset_dicts, num=3)  :
    for d in random.sample(dataset_dicts, num):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=SVHN_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        plt.imshow(vis.get_image()[:, :, ::-1])
        plt.show()


def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def get_digits_dicts(dataset_dicts):
    for i in range(len(dataset_dicts)):
        for j in range(len(dataset_dicts[i]['annotations'])):
            dataset_dicts[i]['annotations'][j]['bbox_mode'] = BoxMode.XYWH_ABS
    return dataset_dicts


with open('SVHN_train.json', 'r') as f:
    dataset_dicts = json.load(f)

DatasetCatalog.register("SVHN_dataset", lambda: get_digits_dicts(dataset_dicts))
MetadataCatalog.get("SVHN_dataset").set(thing_classes=["0", "1", "2", "3", "4",
                                                       "5", "6", "7", "8", "9"])
SVHN_metadata = MetadataCatalog.get("SVHN_dataset")

dataset_dicts = get_digits_dicts(dataset_dicts)


# ============ train ===========
cfg = get_cfg()
cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("SVHN_dataset",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = "model_final_280758.pkl"  # pre-trained model file location
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 100000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# ============= test ===========
cfg = get_cfg()
cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10
cfg.DATASETS.TEST = ("SVHN_dataset", )
predictor = DefaultPredictor(cfg)

test_dir = 'test'
result_dir = 'test_result'
output_dicts = []
for filename in sorted(os.listdir('test'), key=numericalSort):
    if filename.endswith(".mat"):
        continue
    filepath = os.path.join(test_dir, filename)
    output_path = os.path.join(result_dir, filename)
    print('predicting ' + filepath)

    im = cv2.imread(filepath)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=SVHN_metadata,
                   scale=3,
                   )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite(output_path, v.get_image()[:, :, ::-1])

    record = {}
    anno = outputs["instances"].to("cpu").get_fields()
    box = list(anno['pred_boxes'])
    npbox = np.zeros([len(box), 4])
    for i in range(len(box)):
        npbox[i] = box[i]
        npbox[i][0], npbox[i][1] = npbox[i][1], npbox[i][0]
        npbox[i][2], npbox[i][3] = npbox[i][3], npbox[i][2]

    record['bbox'] = np.round(npbox).astype(int).tolist()
    record['score'] = anno['scores'].tolist()
    anno['pred_classes'][np.where(anno['pred_classes'] == 0)] = 10
    record['label'] = anno['pred_classes'].tolist()

    output_dicts.append(record)

with open('Submission.json', 'w') as outfile:
    json.dump(output_dicts, outfile)
