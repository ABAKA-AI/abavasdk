# -*- coding: utf-8 -*-
from dynaconf import settings
from lxml import objectify

DEFAULT_VALUE = settings.get("VALUE_TO_BE_FILLED_IN", "None")

VOC_MAKER = objectify.ElementMaker(annotate=False)
'''
VOC is a xml that include some objects.
example:

    {
    VOC_MAKER.folder(),
    VOC_MAKER.filename(filename),
    VOC_MAKER.path(self.source_data[i]['info'][idx + 1:]),
    VOC_MAKER.source(
        VOC_MAKER.database('Unknown')),
    VOC_MAKER.size(
        VOC_MAKER.width(width),
        VOC_MAKER.height(height),
        VOC_MAKER.depth(3)),
    VOC_MAKER.segmented(0),
    VOC_MAKER.object(
        VOC_MAKER.name(name),
        VOC_MAKER.pose('Unspecified'),          // Shooting angles: front, rear, left, right, unspecified 
        VOC_MAKER.truncated('Unspecified'),     // Whether the target is truncated (e.g. outside the picture), or obscured (more than 15%)
        VOC_MAKER.difficult(0),                 // The ease of detection, this is mainly based on the size of the target, the variation in lighting and the quality of the image
        VOC_MAKER.bndbox(
            VOC_MAKER.xmin(xmin),
            VOC_MAKER.ymin(ymin),
            VOC_MAKER.xmax(xmax),
            VOC_MAKER.ymax(ymax)))
    }
'''
VOC = VOC_MAKER.annotation()
VOC_FORMAT = VOC_MAKER.annotation(
    VOC_MAKER.folder(),
    VOC_MAKER.filename(DEFAULT_VALUE),
    VOC_MAKER.path(DEFAULT_VALUE),
    VOC_MAKER.source(
        VOC_MAKER.database(DEFAULT_VALUE)),
    VOC_MAKER.size(
        VOC_MAKER.width(DEFAULT_VALUE),
        VOC_MAKER.height(DEFAULT_VALUE),
        VOC_MAKER.depth(DEFAULT_VALUE)),
    VOC_MAKER.segmented(DEFAULT_VALUE),
    VOC_MAKER.object(
        VOC_MAKER.name(DEFAULT_VALUE),
        VOC_MAKER.pose(DEFAULT_VALUE),
        VOC_MAKER.truncated(DEFAULT_VALUE),
        VOC_MAKER.difficult(DEFAULT_VALUE),
        VOC_MAKER.bndbox(
            VOC_MAKER.xmin(DEFAULT_VALUE),
            VOC_MAKER.ymin(DEFAULT_VALUE),
            VOC_MAKER.xmax(DEFAULT_VALUE),
            VOC_MAKER.ymax(DEFAULT_VALUE))))
