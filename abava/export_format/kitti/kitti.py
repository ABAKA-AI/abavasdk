# -*-coding:utf-8 -*-
from ...abava_data import ABAVA
from dynaconf import settings

DEFAULT_VALUE = settings.get("VALUE_TO_BE_FILLED_IN", "None")
'''
kitti is a list that include some objects.
example:
    {
    "label": str,
    "truncated": int,       // Degree of truncation
    "occluded": float,      // Obscuration rate
    "alpha": float,         // Viewing Angle
    "2DBB_min_xy": float,   // 2DBB top left corner coordinates
    "2DBB_max_xy": float,   // 2DBB lower right hand corner coordinates
    "3DBB_hwl": float,      // 3DBB's height, width, length
    "3DBB_camera": float,   // Coordinates of the 3DBB position in the camera
    "angle_Y": float,       // Angle of rotation relative to the Y-axis
    "score": float
    }
'''
KITTI = ABAVA({
    "label": str,
    "truncated": int,
    "occluded": float,
    "alpha": float,
    "2DBB_min_xy": float,
    "2DBB_max_xy": float,
    "3DBB_hwl": float,
    "3DBB_camera": float,
    "angle_Y": float,
    "score": float
})
