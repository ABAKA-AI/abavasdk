#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import abstractmethod
from ..check.statistics import Statistic
from ..export_format.coco.export_coco import ExportCoco
from ..export_format.kitti.export_kitti import ExportKitti
from ..export_format.labelme.export_labelme import ExportLabelme
from ..export_format.voc.export_voc import ExportVoc
from ..export_format.yolo.export_yolo import ExportYolo
from ..export_format.mask.generate_mask import ExportMask
from ..visualize.coco.visual_coco import VisualCoco
from ..visualize.labelme.visual_labelme import VisualLabelme
from ..visualize.source.visual_source import VisualSource
from ..visualize.voc.visual_voc import VisualVoc
from ..visualize.yolo.visual_yolo import VisualYolo


class DataFactory():
    """
    abstract factory
    """
    type = ""

    # def createProcess(self,porcessClass):
    #     print(self.type," process has been created.")
    #     foodIns=porcessClass()
    #     return foodIns

    @abstractmethod
    def export_coco_product(self, source_data, out_path=None, mapping=None):
        pass

    @abstractmethod
    def export_kitti_product(self, source_data, out_path=None, mapping=None):
        pass

    @abstractmethod
    def export_labelme_product(self, source_data, out_path=None, mapping=None):
        pass

    @abstractmethod
    def export_voc_product(self, source_data, out_path=None, mapping=None):
        pass

    @abstractmethod
    def export_yolo_product(self, source_data, out_path=None, mapping=None):
        pass

    @abstractmethod
    def export_mask_product(self, source_data, out_path=None, mapping=None):
        pass

    @abstractmethod
    def visual_coco_product(self, source_data, data_path, out_path=None):
        pass

    @abstractmethod
    def visual_labelme_product(self, source_data, data_path, out_path=None):
        pass

    @abstractmethod
    def visual_source_product(self, source_data, out_path=None):
        pass

    @abstractmethod
    def visual_voc_product(self, source_data, data_path, image_path, out_path=None):
        pass

    @abstractmethod
    def visual_yolo_product(self, source_data, data_path, label_path, image_path, out_path=None):
        pass

    @abstractmethod
    def statistics_product(self, source_data):
        pass


class ExportFactory(DataFactory):
    def __init__(self):
        self.type = "EXPORT"

    def export_coco_product(self, source_data, out_path=None, mapping=None):
        print(self.type, "process has been created.")
        return ExportCoco(source_data, out_path, mapping)

    def export_kitti_product(self, source_data, out_path=None, mapping=None):
        print(self.type, "process has been created.")
        return ExportKitti(source_data, out_path, mapping)

    def export_labelme_product(self, source_data, out_path=None, mapping=None):
        print(self.type, "process has been created.")
        return ExportLabelme(source_data, out_path, mapping)

    def export_voc_product(self, source_data, out_path=None, mapping=None):
        print(self.type, "process has been created.")
        return ExportVoc(source_data, out_path, mapping)

    def export_yolo_product(self, source_data, out_path=None, mapping=None):
        print(self.type, "process has been created.")
        return ExportYolo(source_data, out_path, mapping)

    def export_mask_product(self, source_data, out_path=None, mapping=None):
        print(self.type, "process has been created.")
        return ExportMask(source_data, out_path, mapping)


class ImportFactory(DataFactory):
    def __init__(self):
        self.type = "IMPORT"


class VisualFactory(DataFactory):
    def __init__(self):
        self.type = "VISUAL"

    def visual_coco_product(self, source_data, data_path, out_path=None):
        print(self.type, "process has been created.")
        return VisualCoco(source_data, data_path, out_path)

    def visual_labelme_product(self, source_data, data_path, out_path=None):
        print(self.type, "process has been created.")
        return VisualLabelme(source_data, data_path, out_path)

    def visual_source_product(self, source_data, out_path=None):
        print(self.type, "process has been created.")
        return VisualSource(source_data, out_path)

    def visual_voc_product(self, source_data, data_path, image_path, out_path=None):
        print(self.type, "process has been created.")
        return VisualVoc(source_data, data_path, image_path, out_path)

    def visual_yolo_product(self, source_data, data_path, label_path, image_path, out_path=None):
        print(self.type, "process has been created.")
        return VisualYolo(source_data, data_path, label_path, image_path, out_path)


class CheckFactory(DataFactory):
    def __init__(self):
        self.type = "CHECK"

    def statistics_product(self, source_data):
        print(self.type, "process has been created.")
        return Statistic(source_data)
