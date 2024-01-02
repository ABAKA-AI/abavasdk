<div align="center">
  <img src="resources/ABAKA-AI-logo.png" width="600"/>
  <div>&nbsp;</div>
  <div align="center">
    <span><font size="150">ABAVA website</font></span>
    <sup>
      <a href="https://www.abaka.ai/">
        <i><font size="5">HOT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

[![PyPI](https://img.shields.io/pypi/v/mmdet3d)](https://pypi.org/project/mmdet3d)
[![docs](https://img.shields.io/badge/docs-latest-blue)](README.md)
[![license](https://img.shields.io/github/license/mashape/apistatus)](LICENSE)

</div>

# ABAVA Data SDK | For Python

Welcome to `ABAVA SDK`, an open-source Software Development Kit that forms the backbone of the ABAVA platform. Designed to convert data between ABAVA’s native format and widely-used, universal data structures such as COCO, YOLO, LABELME, KITTI, VOC, ABAVA SDK helps to streamline and simplify your data operations.

The SDK is more than just a converter. It’s a swiss army knife of data processing tools. It comes loaded with an assortment of handy utility functions often used in data processing workflows, such as Calculate the area of a polygon or determine if a point is inside a polygon.

Whether you’re moving data, cleaning data, transforming data, or just managing it, the ABAVA SDK has got you covered with powerful features that make your work simpler and easier. Built for developers, engineers and data scientists, this SDK aims to make your data-heavy processes more seamless and efficient.

Stay tuned to get more details about the features, capabilities, and the simplicity ABAVA SDK brings to your data operations.

Learn more about ABAVA [here](https://www.abaka.ai/)!

## Overview

- [Requirements](#requirements)
- [Installation](#installation)
- [What can we do](#what-can-we-do)
- [Usage](#usage)
- [Changelog](#changelog)
- [Contact us](#contact-us)
- [License](#license)

## requirements

```sh
python==3.9
lxml~=4.9
numpy~=1.25
opencv_python~=4.8
Pillow~=10.0
Requests~=2.31
tqdm~=4.66
dynaconf~=3.2
```

## Installation

### Install with PyPi (pip)

```sh
pip install abava_sdk
```

### Install with Anaconda (conda)

```sh
conda install -c conda-forge abava_sdk
```

## What can we do

### Data Format

- [ABAVA data -> COCO data](http://github.molardata.com/open/abavasdk/-/blob/main/abava/export_format/coco/export_coco.py)
- [ABAVA data -> LABELME data](http://github.molardata.com/open/abavasdk/-/blob/main/abava/export_format/labelme/export_labelme.py)
- [ABAVA data -> VOC data](http://github.molardata.com/open/abavasdk/-/blob/main/abava/export_format/voc/export_voc.py)
- [ABAVA data -> YOLO data](http://github.molardata.com/open/abavasdk/-/blob/main/abava/export_format/yolo/export_yolo.py)
- [ABAVA data -> KITTI data](http://github.molardata.com/open/abavasdk/-/blob/main/abava/export_format/kitti/export_kitti.py)
- [ABAVA data -> MASK](http://github.molardata.com/open/abavasdk/-/blob/main/abava/export_format/mask/generate_mask.py)

### Data Check

- [count labels number](http://github.molardata.com/open/abavasdk/-/blob/main/abava/check/statistics.py#L11)
- [count specific labels number](http://github.molardata.com/open/abavasdk/-/blob/main/abava/check/statistics.py#L21)
- [count drawtype number](http://github.molardata.com/open/abavasdk/-/blob/main/abava/check/statistics.py#L33)
- [count file number](http://github.molardata.com/open/abavasdk/-/blob/main/abava/check/statistics.py#L46)
- [count image number](http://github.molardata.com/open/abavasdk/-/blob/main/abava/check/statistics.py#L62)
- [count unlabeled image number](http://github.molardata.com/open/abavasdk/-/blob/main/abava/check/statistics.py#L73)
- [count labeled image number](http://github.molardata.com/open/abavasdk/-/blob/main/abava/check/statistics.py#L90)

### Data Visualization

- [visual ABAVA data](http://github.molardata.com/open/abavasdk/-/blob/main/abava/visualize/source/visual_source.py)
- [visual COCO data](http://github.molardata.com/open/abavasdk/-/blob/main/abava/visualize/coco/visual_coco.py)
- [visual LABELME data](http://github.molardata.com/open/abavasdk/-/blob/main/abava/visualize/labelme/visual_labelme.py)
- [visual VOC data](http://github.molardata.com/open/abavasdk/-/blob/main/abava/visualize/voc/visual_voc.py)
- [visual YOLO data](http://github.molardata.com/open/abavasdk/-/blob/main/abava/visualize/yolo/visual_yolo.py)

### Computer Vision tools

- [image data -> base64](http://github.molardata.com/open/abavasdk/-/blob/main/abava/utils/cv_tools.py#L17)
- [base64 -> image data](http://github.molardata.com/open/abavasdk/-/blob/main/abava/utils/cv_tools.py#L31)
- [read url image](http://github.molardata.com/open/abavasdk/-/blob/main/abava/utils/cv_tools.py#L43)
- [get url image size](http://github.molardata.com/open/abavasdk/-/blob/main/abava/utils/cv_tools.py#L56)
- [hexadecimal color values -> RGB](http://github.molardata.com/open/abavasdk/-/blob/main/abava/utils/cv_tools.py#L69)
- [generate random RGB values](http://github.molardata.com/open/abavasdk/-/blob/main/abava/utils/cv_tools.py#L82)
- [drawing boxes on the image](http://github.molardata.com/open/abavasdk/-/blob/main/abava/utils/cv_tools.py#L89)
- [drawing points on the image](http://github.molardata.com/open/abavasdk/-/blob/main/abava/utils/cv_tools.py#L112)
- [drawing polygons on the image](http://github.molardata.com/open/abavasdk/-/blob/main/abava/utils/cv_tools.py#L129)
- [plate mode MASK -> POLYGON](http://github.molardata.com/open/abavasdk/-/blob/main/abava/utils/cv_tools.py#L218)
- [MASK -> RLE](http://github.molardata.com/open/abavasdk/-/blob/main/abava/utils/cv_tools.py#L233)
- [RLE -> MASK](http://github.molardata.com/open/abavasdk/-/blob/main/abava/utils/cv_tools.py#L246)
- [POLYGON -> MASK](http://github.molardata.com/open/abavasdk/-/blob/main/abava/utils/cv_tools.py#L264)
- [determine if the point is inside the outer rectangle](http://github.molardata.com/open/abavasdk/-/blob/main/abava/utils/cv_tools.py#L280)
- [determine if the point is inside the polygon](http://github.molardata.com/open/abavasdk/-/blob/main/abava/utils/cv_tools.py#L336)
- [calculate the polygon area](http://github.molardata.com/open/abavasdk/-/blob/main/abava/utils/cv_tools.py#L416)
- [determining the containment of polygons](http://github.molardata.com/open/abavasdk/-/blob/main/abava/utils/cv_tools.py#L433)
- [skeleton polygon](http://github.molardata.com/open/abavasdk/-/blob/main/abava/utils/cv_tools.py#L561)
- [image de-distortion](http://github.molardata.com/open/abavasdk/-/blob/main/abava/utils/cv_tools.py#L623)

### Point Cloud tools

- [read PCD format point clouds](http://github.molardata.com/open/abavasdk/-/blob/main/abava/utils/pc_tools.py#L15)
- [write PCD format point clouds](http://github.molardata.com/open/abavasdk/-/blob/main/abava/utils/pc_tools.py#L49)
- [PCD -> BIN](http://github.molardata.com/open/abavasdk/-/blob/main/abava/utils/pc_tools.py#L67)
- [BIN -> PCD](http://github.molardata.com/open/abavasdk/-/blob/main/abava/utils/pc_tools.py#L74)
- [removing points from the point cloud 3D box](http://github.molardata.com/open/abavasdk/-/blob/main/abava/utils/pc_tools.py#L95)
- [voxel subsampling for points outside the intensity range](http://github.molardata.com/open/abavasdk/-/blob/main/abava/utils/pc_tools.py#L126)
- [random subsampling for points outside the intensity range](http://github.molardata.com/open/abavasdk/-/blob/main/abava/utils/pc_tools.py#L146)
- [the pnp method computes rotation matrices and translation vectors](http://github.molardata.com/open/abavasdk/-/blob/main/abava/utils/pc_tools.py#L166)
- [calculate the number of points in the 3D box](http://github.molardata.com/open/abavasdk/-/blob/main/abava/utils/pc_tools.py#L187)
- [rotation matrix -> quaternion](http://github.molardata.com/open/abavasdk/-/blob/main/abava/utils/pc_tools.py#L216)
- [rotation matrix -> euler](http://github.molardata.com/open/abavasdk/-/blob/main/abava/utils/pc_tools.py#L230)
- [euler -> rotation matrix](http://github.molardata.com/open/abavasdk/-/blob/main/abava/utils/pc_tools.py#L243)
- [quaternion -> rotation matrix](http://github.molardata.com/open/abavasdk/-/blob/main/abava/utils/pc_tools.py#L269)
- [euler -> quaternion](http://github.molardata.com/open/abavasdk/-/blob/main/abava/utils/pc_tools.py#L291)
- [quaternion -> euler](http://github.molardata.com/open/abavasdk/-/blob/main/abava/utils/pc_tools.py#L306)

## Usage

### get source data

```python
import abava

"""
You can get your ak/sk in the platform's "Team Space" -> "Setting"
"""
ak = Access Key
sk = Secret Key
client = abava.Client(ak, sk)

"""
After creating an export task, you can see the export task id of the corresponding task
in "Import/Export"->"Data Export".
"""
source_data = client.get_data(export_task_id)
```

### data format

```python
from abava import Export

mapping = {"背景background": 'background', "草地lawn": 'lawn', "道路road": 'road'}
# coco
out_path = "./output"
Export.abava_json2coco(source_data=source_data, out_path=out_path)

```

### visualize

```python
from abava import Visual

data_path = "./data.json"
out_path = "./output"
Visual.visual_coco(source_data, data_path, out_path)
```

### utils

```python
def test_isin_external_rectangle():
    point = [55, 100]
    vertex_lst = [[50,50], [200,200], [200,50], [50,50]]
    tag = abava.isin_external_rectangle(point, vertex_lst)
    return tag


def test_to_string():
    test_string = b'example_string'
    string = abava.b2string(test_string)
    print(string)


def test_pcd2bin():
    bin_path = './bin'
    pcd_path = './pcd'
    abavadata.pcd2bin(pcd_path, bin_path)
```

Please refer to [examples.md](example/examples.md) to learn more usage about ABAVA SDK.

## Changelog

[2023-11-29] Updated

- Optimize mask export

[2023-10-12] New features:

- Support for interconversion of Euler angles, quaternions and rotation matrices

[2023-08-31] New features:

- Support pinhole camera image de-distortion and fisheye camera image de-distortion
- Support point cloud random subsampling and voxel subsampling
- Support for removing points in the 3D box of the point cloud
- Support Quaternion to Euler angle
- Support PNP <br>

[2023-07-21] ABAVA SDK v1.0.0 is released. <br>

## Contact Us

- wxj@molardata.com

## License

ABAVA SDK is released under the MIT license.
