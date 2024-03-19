#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os
import urllib.parse
from os.path import join


class PostProcess():
    def __init__(self, data_path, out_path=None):
        self.data_path = data_path
        self.out_path = out_path

    def abava2dict(self, data):
        """
        Converting ABAVA types to dict
        :param data:
        :return:
        """
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, list):
                    data[f'{k}'] = [self.abava2dict(d) for d in v]
                elif repr(v) == 'molar_ai_format':
                    data[f'{k}'] = self.abava2dict(v)
        elif repr(data) == 'molar_ai_format':
            data = self.abava2dict(vars(data))
        return data

    def save_labels(self, final_data, mode, filename):
        """
        :param final_data: Final converted data
        :param filename: Saved file name
        :param sub_folder: Subcategories
        :return: json file
        """
        filename = urllib.parse.unquote(filename)
        if mode == 'split':
            file_path = 'split_labels'
        elif mode == 'merge':
            file_path = 'merge_labels'

        if self.out_path is None:
            out_root = './' + file_path
        else:
            out_root = join(self.out_path, file_path)

        print(out_root)
        if not os.path.exists(out_root):
            os.makedirs(out_root)
        with open(join(out_root, filename), 'w', encoding='utf-8') as write_f:
            json.dump(final_data, write_f, indent=2, ensure_ascii=False)