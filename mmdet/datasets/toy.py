
import cv2
import numpy as np
import logging
import os.path as osp
import tempfile

import mmcv
from torch.utils.data import Dataset

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from mmdet.core import eval_recalls

from mmdet.core import eval_map, eval_recalls
from .pipelines import Compose
from .registry import DATASETS
from .coco import CocoDataset

@DATASETS.register_module
class ToyDataset(Dataset):
    """Car dataset."""
    CLASSES = ('circle')
    def __init__(self,
                 pipeline,
                 img_shape=(256, 256),
                 max_radius=64,
                 num_classes=1,
                 max_objects=1,
                 test_mode=False):
        super(ToyDataset).__init__()
        self.img_shape = np.array(img_shape)
        self.num_classes = num_classes
        self.max_width = 64
        self.max_height = 64
        self.max_radius = min(img_shape) // 4
        self.max_objects = max_objects
        self.test_mode = test_mode
        self.test_gt_bboxes = np.zeros((100,4))

        w, h = self.img_shape // 4
        # prepare mesh center points
        x_arr = np.arange(w) + 0.5
        y_arr = np.arange(h) + 0.5
        self.xy_mesh = np.stack(np.meshgrid(x_arr, y_arr))  # [2, h, w]

        # set group flag for the sampler
        self._set_group_flag()

        # processing pipeline
        self.pipeline = Compose(pipeline)

        self.test_folder = '/home/songbai.xb/detection/mmdetection/mmdet/datasets/tmp'

    def __len__(self):
        return 1000

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            self.flag[i] = 0

    def __getitem__(self, idx):

        im = np.zeros(self.img_shape, dtype=np.float32)
        heatmap = np.zeros((self.num_classes + 4, self.img_shape[0] // 4,
                            self.img_shape[1] // 4),
                           dtype=np.float32)
        gt_bboxes = []
        gt_labels = []
        for _ in range(np.random.randint(1, self.max_objects + 1 )):
            x, y = np.random.randint(0, self.img_shape[0]), np.random.randint(
                0, self.img_shape[1])
            radius = np.random.randint(10, self.max_radius)
            label = np.random.randint(1, self.num_classes + 1)
            im = np.maximum(
                im, cv2.circle(im, (x, y),
                               radius=radius,
                               color=1,
                               thickness=-1))

            x1 = max(0, x-radius)
            y1 = max(0, y-radius)
            x2 = min(self.img_shape[0], x + radius)
            y2 = min(self.img_shape[1], y + radius)
            gt_bboxes.append([x1, y1, x2, y2])
            gt_labels.append(label)
            
            
        # 1 channel convert to 3 channel
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
            
        
        results = dict()
        results['img'] = im
        results['img_shape'] = im.shape
        results['scale_factor'] = 1
        results['gt_bboxes'] = np.array(gt_bboxes).astype(np.float32) 
        results['gt_labels'] = gt_labels
        results['bbox_fields'] = []
        results['bbox_fields'].append('gt_bboxes')
        data = self.pipeline(results)

        if self.test_mode:
            np.save(self.test_folder+'/{}.npy'.format(idx), results['gt_bboxes'][0])
            # self.test_gt_bboxes中的值无法传递到evaluate函数 ???
            #self.test_gt_bboxes[idx] = results['gt_bboxes'][0]
            #print('***************', idx, results['gt_bboxes'], self.test_gt_bboxes[idx])

        return data

    



    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.arange(0.5, 0.96, 0.05)):
        """Evaluation in COCO protocol.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float]): IoU threshold used for evaluating
                recalls. If set to a list, the average recall of all IoUs will
                also be computed. Default: 0.5.

        Returns:
            dict[str: float]
        """
        import common
        import numpy as np
        from common.eval_utils.detect_eval import eval_detection, generate_report

        groundtruth_dict = dict()
        pred_dict = dict()
        #import pdb
        #pdb.set_trace()
        #cur_gt_start = len(self.test_gt_bboxes) - 100
        score_thres = 0.1
        for i in range(len(results)):
            cur_gt = np.load(self.test_folder+'/{}.npy'.format(i))
            single_groundtruth = dict()
            single_groundtruth['bboxes_list'] = cur_gt.reshape(-1, 4)
            single_groundtruth['labels'] = np.zeros(len(single_groundtruth['bboxes_list'] ))

            single_pred = dict()
            pred_idx = results[i][0][:, 4] >  score_thres
            single_pred['boxes'] = results[i][0][pred_idx]
            single_pred['labels'] = np.zeros(len(single_pred['boxes'] ))
            groundtruth_dict[i] = single_groundtruth
            pred_dict[i] = single_pred

        # 评估record
        global_record = eval_detection(groundtruth_dict, pred_dict)
    
        # 生成报告
        report = generate_report(global_record)

        # mmdet/core/evaluation/eval_hooks.py
        # 需要传递回一个字典
        return dict()