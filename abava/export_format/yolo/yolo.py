# -*- coding: utf-8 -*-
from ...abava_data import ABAVA
from dynaconf import settings

DEFAULT_VALUE = settings.get("VALUE_TO_BE_FILLED_IN", "None")
'''
yolo is a list that include some objects.
example:
    {
    "index" : int,
    "center_x": float,      // The x-coordinate of the centre point of the label, which is the relative scale with respect to the whole image
    "center_y": float,      // The y-coordinate of the centre point of the label, which is the relative scale with respect to the whole image
    "weight": float,        
    "height": float         // The width and height are also relative to the whole image
    }
'''

YOLO = ABAVA({
    "index": int,
    "center_x": float,
    "center_y": float,
    "weight": float,
    "height": float
})
