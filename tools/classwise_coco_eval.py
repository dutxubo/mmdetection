from argparse import ArgumentParser

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def classwise_coco_eval(gtFile, resFile, annType, maxDets=[1, 10, 100], iouThrs=-1, scoreThrs=-1, classwise=False):
    '''
    gtFile : coco格式的groundtruth.json
    resFile：mmdetection的json结果文件，格式如下
            [{image_id:0, bbox:[x, y, w, h], score:0.98, category_id:1},
             {image_id:0, bbox:[x, y, w, h], score:0.88, category_id:1},
             {image_id:1, bbox:[x, y, w, h], score:0.80, category_id:0},
             ...]
    annType: 'bbox', 'segm' or 'keypoint'
    maxDets: 默认 [1, 10, 100]
    iouThrs: -1 or value in (0,1)
    scoreThrs: -1 or value in (0,1)
    classwise: True or False
    '''
        
    cocoGt=COCO(gtFile)
    if scoreThrs < 0:
        cocoDt=cocoGt.loadRes(resFile)
    else:
        annos = json.load(open(resFile))
        filter_anns = [anno for anno in annos if anno['score'] > score_thres]
        cocoDt=cocoGt.loadRes(filter_anns)  #如果出现 unicode问题 直接在源码中删掉这个判断
    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    
    catIds = cocoEval.params.catIds 
    cocoEval.params.maxDets = maxDets
    if iouThrs>=0.0 and iouThrs<=1.0:
        cocoEval.params.iouThrs = [iouThrs]
    
    print('\n')
    print('*********')
    print('all ')
    print('*********')
    cocoEval.evaluate()
    cocoEval.accumulate()
    eva_result = cocoEval.summarize()
    
    if classwise:
        for catId in catIds:
            print('\n')
            print('*********')
            print('class: ', catId)
            print('*********')
            cocoEval.params.catIds = [catId]
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
    
def main():
    parser = ArgumentParser(description='Classwise COCO Evaluation')
    parser.add_argument('ann', help='annotation file path')
    parser.add_argument('result', help='result file path')
    parser.add_argument(
        '--types',
        type=str,
        choices=['bbox', 'segm', 'keypoint'],
        default='bbox',
        help='result types')
    parser.add_argument(
        '--max-dets',
        type=int,
        nargs='+',
        default=[1, 10, 100],
        help='proposal numbers, only used for recall evaluation')
    parser.add_argument('--iou_thrs', type=float, default=-1.0, help='每个类单独计算recall的阈值, (0, 1)之间, 如果为负值则默认为(0.5: 0.95 :0.0.5)')
    parser.add_argument('--score_thrs', type=float, default=-1.0, help='得分阈值, (0, 1)之间, 如果为负值不通过阈值筛选')
    parser.add_argument('--classwise',  action='store_true', help='是否对每个类分别评估，默认为Fasle')
    
    args = parser.parse_args()
    classwise_coco_eval(args.ann, args.result, args.types, args.max_dets, args.iou_thrs, args.score_thrs, args.classwise )


if __name__ == '__main__':
    main()