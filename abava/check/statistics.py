# -*-coding:utf-8 -*-
from pathlib import *
from .check import CheckData


class Statistic(CheckData):
    def __init__(self, source_data):
        super(Statistic, self).__init__(source_data)

    def count_labels(self):
        """
        Total number of labels counted
        :return: int
        """
        count = 0
        for i in self.source_data.data:
            count += len(i.labels)
        return count

    def count_aim_labels(self, label):
        """
        Count the total number of specified labels
        :return: int
        """
        count = 0
        for i in self.source_data.data:
            for j in i.labels:
                if j.data.label == label:
                    count += 1
        return count

    def count_drawtype(self, drawtype):
        """
        Count the total number of specified drawtypes
        :param drawtype: str
        :return: int
        """
        count = 0
        for i in self.source_data.data:
            for j in i.labels:
                if j.data.drawType == drawtype:
                    count += 1
        return count

    def count_files(self):
        """
        Count the number of images in each folder
        :return: dict
        """
        files = {}
        for i in self.source_data.data:
            urls = i.info.info.url
            for url in urls:
                folder = '/'.join(Path(url).parts[3:len(Path(url).parts)-1])
                if folder not in files:
                    files[folder] = 1
                else:
                    files[folder] += 1
        return files

    def count_images(self):
        """
        Counting the number of images
        :return: int
        """
        count = 0
        for i in self.source_data.data:
            urls = i.info.info.url
            count += len(urls)
        return count

    def unlabeld_images(self):
        """
        Count the total number of unlabeled images
        :return: int
        """
        count = 0
        for i in self.source_data.data:
            urls = i.info.info.url
            url_index = [i for i in range(len(urls))]
            for j in i.labels:
                frameIndex = j.data.frameIndex
                if frameIndex in url_index:
                    url_index.remove(frameIndex)
            count += len(url_index)

        return count

    def labeled_images(self):
        """
        Count the total number of tagged images
        :return: int
        """
        count = 0
        for i in self.source_data.data:
            frame_list = []
            for j in i.labels:
                frameIndex = j.data.frameIndex
                if frameIndex not in frame_list:
                    frame_list.append(frameIndex)
            count += len(frame_list)

        return count
