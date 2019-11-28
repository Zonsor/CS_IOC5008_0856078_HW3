# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:01:44 2019

@author: Zonsor
"""

import os
import h5py
import cv2
import json


def get_name(index, hdf5_data):
    name = hdf5_data['/digitStruct/name']
    return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]].value])


def get_bbox(index, hdf5_data):
    attrs = {}
    item = hdf5_data['digitStruct']['bbox'][index].item()
    for key in ['label', 'left', 'top', 'width', 'height']:
        attr = hdf5_data[item][key]
        values = [hdf5_data[attr.value[i].item()].value[0][0]
                  for i in range(len(attr))] if len(attr) > 1 else [attr.value[0][0]]
        attrs[key] = values
    return attrs


# write a function that loads the dataset into detectron2's standard format
def h5_to_json(img_dir):
    h5_file_path = os.path.join(img_dir, "digitStruct.mat")
    f = h5py.File(h5_file_path, 'r')

    dataset_dicts = []
    for j in range(f['/digitStruct/bbox'].shape[0]):
        record = {}

        img_name = get_name(j, f)
        row_dict = get_bbox(j, f)
        print("processing" + img_name)
        filename = os.path.join(img_dir, img_name)
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = j
        record["height"] = height
        record["width"] = width

        objs = []
        for index in range(len(row_dict['label'])):
            if row_dict['label'][index] == 10:
                row_dict['label'][index] = 0
            obj = {
                "bbox": [row_dict['left'][index], row_dict['top'][index],
                         row_dict['width'][index], row_dict['height'][index]],
                "bbox_mode": 'BoxMode.XYWH_ABS',
                "category_id": int(row_dict['label'][index]),
                "iscrowd": 0
            }
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


dataset_dicts = h5_to_json("train")
with open('SVHN_train.json', 'w') as outfile:
    json.dump(dataset_dicts, outfile)
