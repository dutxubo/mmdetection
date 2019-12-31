import numpy as np
from pycocotools.coco import COCO

from .coco import CocoDataset
from .registry import DATASETS


@DATASETS.register_module
class MyPlateDataset(CocoDataset):

    CLASSES = ('plate')

    