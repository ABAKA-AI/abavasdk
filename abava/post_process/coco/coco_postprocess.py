#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import random
from pathlib import Path

from tqdm import tqdm

from ..process import PostProcess
from ...utils import load_json
from ...abava_data import ABAVA
from ...export_format.coco.coco import COCO
# from .coco import COCO
from datetime import datetime


class CocoProcess(PostProcess):
    def __init__(self, data_path, out_path=None):
        super(CocoProcess, self).__init__(data_path, out_path)
        COCO.info = {
            'description': 'Convert from ABAVA dataset to COCO dataset',
            'url': 'https://github.com/ABAKA-AI/abavasdk',
            'version': 'ABAVA SDK V1.0',
            'year': f"{datetime.utcnow().year}",
            'contributor': '',
            'date_created': datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        }

    def split(self, test_size=0.3, train_size=None, shuffle=True):
        if train_size is None:
            train_size = 1 - test_size

        coco_data = ABAVA(load_json(self.data_path))
        train_annotations = []
        test_annotations = []

        if shuffle:
            random.shuffle(coco_data.images)

        test_images = coco_data.images[:int(len(coco_data.images) * test_size)]
        for image in tqdm(test_images):
            test_id = image.id
            test_selected_annotations = [item for item in coco_data.annotations if item['image_id'] == test_id]
            test_annotations.extend(test_selected_annotations)
        COCO.images = sorted(test_images, key=lambda item: item['id'])
        sorted_test_annotations = sorted(test_annotations, key=lambda item: item['image_id'])
        COCO.annotations = [{**self.abava2dict(item), 'id': idx + 1} for idx, item in enumerate(sorted_test_annotations)]
        test_file_name = Path(self.data_path).parts[-1].replace('.json', '_test.json')
        self.save_labels(self.abava2dict(COCO), 'split', test_file_name)

        train_images = coco_data.images[int(len(coco_data.images) * test_size):int(len(coco_data.images) * (test_size + train_size))]
        for image in tqdm(train_images):
            train_id = image.id
            train_selected_annotations = [item for item in coco_data.annotations if item['image_id'] == train_id]
            train_annotations.extend(train_selected_annotations)
        COCO.images = sorted(train_images, key=lambda x: x['id'])
        sorted_train_annotations = sorted(train_annotations, key=lambda x: x['image_id'])
        COCO.annotations = [{**self.abava2dict(item), 'id': idx + 1} for idx, item in enumerate(sorted_train_annotations)]
        train_file_name = Path(self.data_path).parts[-1].replace('.json', '_train.json')
        self.save_labels(self.abava2dict(COCO), 'split', train_file_name)

        eval_images = coco_data.images[int(len(coco_data.images) * (test_size + train_size)):]
        if len(eval_images) > 0:
            eval_annotations = [item for item in coco_data.annotations if
                                item not in test_annotations and item not in train_annotations]
            sorted_eval_annotations = sorted(eval_annotations, key=lambda x: x['image_id'])
            COCO.images = sorted(eval_images, key=lambda x: x['id'])
            COCO.annotations = [{**self.abava2dict(item), 'id': idx + 1} for idx, item in enumerate(sorted_eval_annotations)]
            eval_file_name = Path(self.data_path).parts[-1].replace('.json', '_val.json')
            self.save_labels(self.abava2dict(COCO), 'split', eval_file_name)

    def merge(self):
        merged_images = []
        merged_annotations = []
        coco_paths = glob.glob(self.data_path + '/*')
        for coco_path in coco_paths:
            coco_data = ABAVA(load_json(coco_path))
            image_length = len(merged_images)
            images_id = [item['id'] for item in coco_data.images]
            mapping = {id: i + 1 + image_length for i, id in enumerate(images_id)}
            merged_images.extend([{**self.abava2dict(item), 'id': mapping[item['id']]} for item in coco_data.images])
            merged_annotations.extend([{**self.abava2dict(item), 'image_id': mapping[item['image_id']]} for item in coco_data.annotations])

        sorted_merged_images = sorted(merged_images, key=lambda item: item['id'])
        sorted_merged_annotations = sorted(merged_annotations, key=lambda item: item['image_id'])

        COCO.images = sorted_merged_images
        COCO.annotations = sorted_merged_annotations

        merged_file_name = 'merged_data.json'
        self.save_labels(self.abava2dict(COCO), 'merge', merged_file_name)

