#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time         : 2023/11/22 17:52
# @Author       : Wu Xinjun
# @Site         : 
# @File         : check.py
# @Project      : AbavaSDK
# @Software     : PyCharm
# @Description  : 
"""
from ..abava_data import ABAVA


class CheckData():
    def __init__(self, source_data):
        self.source_data = ABAVA(source_data)