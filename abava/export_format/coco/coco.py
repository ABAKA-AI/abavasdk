# -*- coding: utf-8 -*-
from ...abava_data import ABAVA
from dynaconf import settings

DEFAULT_VALUE = settings.get("VALUE_TO_BE_FILLED_IN", "None")

COCO = ABAVA({})

# coco.info
'''
example : 
        {
          "year" : int
          "version" : str
          "description" : str
          "contributor" : str
          "url" : str
          "date_created" : datetime
        }
'''
COCO.info = ABAVA({
      "year": DEFAULT_VALUE,
      "version": DEFAULT_VALUE,
      "description": DEFAULT_VALUE,
      "contributor": DEFAULT_VALUE,
      "url": DEFAULT_VALUE,
      "date_created": DEFAULT_VALUE,
})

# coco.license
'''
example :
        {
          "id" : int
          "name" : str
          "url" : str
        }
'''
COCO_LICENSE = ABAVA({
      "id": DEFAULT_VALUE,
      "name": DEFAULT_VALUE,
      "url": DEFAULT_VALUE,
})
COCO.license = [COCO_LICENSE]

# coco.images
'''
coco.images is a list which include some object.
the object example : 
        {
          "id" : int
          "width" : int
          "height" : int
          "file_name" : str
          "license" : int
          "flickr_url" : str
          "coco_url" : str
          "date_captured" : datetime
        }
'''
COCO_IMAGE = ABAVA({
      "id": DEFAULT_VALUE,
      "width": DEFAULT_VALUE,
      "height": DEFAULT_VALUE,
      "file_name": DEFAULT_VALUE,
      "license": DEFAULT_VALUE,
      "flickr_url": DEFAULT_VALUE,
      "coco_url": DEFAULT_VALUE,
      "date_captured": DEFAULT_VALUE,
})
COCO.images = []

# coco.annotations
'''
coco.annotations is a list which include some object.
the object example:

    instances_polygons:
    {
      "segmentation": [[510.66, 423.01, 420.03, 510.45]], // Horizontal and vertical coordinates of polygon boundary vertices
      "area": 702.1057499999998, // Area of the marked area
      "iscrowd": 0, // 0: polygon, 1: RLE
      "image_id": 289343, // Corresponds to the serial number of the picture
      "bbox": [473.07, 395.93, 38.65, 28.67], // The horizontal and vertical coordinates of the upper left corner of the 
                                                 bbox rectangle and the length and width of the rectangle.
      "category_id": 18, // Serial number of the corresponding category
      "id": 1768
    },

    instances_rle:
    {
        "segmentation": 
            {
            "counts": [8214,6,629,17,2,6,614,28,611,29,610,31,609,31,609,32,608,32,608,32,608,31,609,31,610,29,612,27], // RLE
            "size": [640, 549] // Width and height of the image
            },
        "area": 962,
        "iscrowd": 1, // 0: polygon, 1: RLE
        "image_id": 374545,
        "bbox": [12, 524, 381, 33],
        "category_id": 1,
        "id": 900100374545
    }

    key_points:
    {
        "segmentation": [[125.12, 539.69, 140.94, 522.43...]],
        "num_keypoints": 10, // Indicates the number of key points marked on this target
        "area": 47803.27955,
        "iscrowd": 0,
        "keypoints": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,142,309,1,177,320,2,191,398...],  
        // The new keypoints are an array of length 3*k, where k is the total number of keypoints in the category. 
           The first and second elements of each keypoint are the x and y coordinate values respectively, 
           and the third element is a flag v. A v of 0 means that the keypoint is not marked (in which case x=y=v=0), 
           a v of 1 means that the keypoint is marked but not visible (obscured), and a v of 2 means that the keypoint is marked and also visible. 
           Translated with www.DeepL.com/Translator (free version)
        "image_id": 425226,
        "bbox": [73.35, 206.02, 300.58, 372.5],
        "category_id": 1,
        "id": 183126
    }
'''
COCO_INSTANCES_POLYGONS = ABAVA({
      "segmentation": DEFAULT_VALUE,
      "area": DEFAULT_VALUE,
      "iscrowd": DEFAULT_VALUE,
      "image_id": DEFAULT_VALUE,
      "bbox": DEFAULT_VALUE,
      "category_id": DEFAULT_VALUE,
      "id": DEFAULT_VALUE
})
COCO_INSTANCES_RLE = ABAVA({
      "segmentation":
            {
                  "counts": DEFAULT_VALUE,
                  "size": DEFAULT_VALUE
            },
      "area": DEFAULT_VALUE,
      "iscrowd": DEFAULT_VALUE,
      "image_id": DEFAULT_VALUE,
      "bbox": DEFAULT_VALUE,
      "category_id": DEFAULT_VALUE,
      "id": DEFAULT_VALUE
})
COCO_KEY_POINTS = ABAVA({
      "segmentation": DEFAULT_VALUE,
      "num_keypoints": DEFAULT_VALUE,
      "area": DEFAULT_VALUE,
      "iscrowd": DEFAULT_VALUE,
      "keypoints": DEFAULT_VALUE,
      "image_id": DEFAULT_VALUE,
      "bbox": DEFAULT_VALUE,
      "category_id": DEFAULT_VALUE,
      "id": DEFAULT_VALUE
})
COCO.annotations = []

# coco.categories
'''
coco.categories is a list which include some object.
the object example :

     {
     "id": 1, 
     "name": "name", 
     "supercategory": "name"
     } 

'''
COCO_CATEGORY = ABAVA({
      "id": DEFAULT_VALUE,
      "name": DEFAULT_VALUE,
      "supercategory": DEFAULT_VALUE
})
COCO.categories = []
