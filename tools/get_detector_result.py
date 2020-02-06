import os
from tqdm import tqdm
import pickle
from mmdet.apis import inference_detector, init_detector

def get_detector_result(model, img_names, save_pkl_file=None, img_root=None):
    '''
    将mmdetection的结果转为dict存储
    '''
    
    detector_results = dict()
    for img_name in tqdm(img_names):
        img_path = img_name if img_root is None else (img_root + img_name)
        if not os.path.exists(img_path):
            continue
        
        if model.with_mask:
            bbox_results, segm_results = inference_detector(model, img_path) 
        else:   
            bbox_results = inference_detector(model, img_path) 
            segm_results = None
            
        if hasattr(model, 'keypoint_results'):
            keypoint_results = model.keypoint_results    
            bbox_keypoint_results = [ np.concatenate((bbox_results[i],keypoint_results[i]), axis=1) for i in range(len(bbox_results))]    
            bbox_results = bbox_keypoint_results
        detector_results[img_name] = dict(bbox_results = bbox_results, segm_results = segm_results)

    
    if save_pkl_file is not None:
        pickle.dump(detector_results, open(save_pkl_file, 'wb'))
    
    return detector_results


if __name__ == '__main__':
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    
    config_file = '/home/songbai.xb/detection/mmdetection/myprojects/dengxiang/configs/repkeypoints_partial_minmax_r50_fpn.py'
    checkpoint_file = '/home/songbai.xb/detection/mmdetection/myprojects/dengxiang//work_dirs/20191209/repkeypoints_partial_minmax_r50_4img2gpu/latest.pth'
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    
    detector_results = get_detector_result(model, img_names, save_pkl_file='detector_result/20191209_repkeypoints_result.pkl', img_root=img_root)