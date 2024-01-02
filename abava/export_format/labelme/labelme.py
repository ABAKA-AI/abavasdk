# -*- coding: utf-8 -*-
from ...abava_data import ABAVA
from dynaconf import settings

DEFAULT_VALUE = settings.get("VALUE_TO_BE_FILLED_IN", "None")

# labelme
'''
example:
        {
        "version": "5.1.1",
        "flags": {},
        "shapes": [],
        "imagePath": str,
        "imageData": str,       // Base64 encoding of images
        "imageHeight": int,
        "imageWidth": int
        }
'''
LABELME = ABAVA({
    "version": "5.1.1",
    "flags": DEFAULT_VALUE,
    "shapes": DEFAULT_VALUE,
    "imagePath": DEFAULT_VALUE,
    "imageData": DEFAULT_VALUE,
    "imageHeight": DEFAULT_VALUE,
    "imageWidth": DEFAULT_VALUE
})

# labelme.shapes
'''
labelme.shapes is a list which include some objects.
example :

    rectangle:
    {
    "label": label,
    "points": points_labelme.tolist(),
    "group_id": None,
    "shape_type": "rectangle",
    "flags": {}
    }
    
    polygon:
    {
    "label": label,
    "points": points_labelme.tolist(),
    "group_id": None,
    "shape_type": "polygon",
    "flags": {}
    }
    
    point:
    {
    "label": point_label,
    "points": [points_labelme[p].tolist()],
    "group_id": None,
    "shape_type": "point",
    "flags": {}
    }
'''
LABELME_SHAPE = ABAVA({
    "label": DEFAULT_VALUE,
    "points": DEFAULT_VALUE,
    "group_id": None,
    "shape_type": DEFAULT_VALUE,
    "flags": {}
})

LABELME.shapes = []
