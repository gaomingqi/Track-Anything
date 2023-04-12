from time import time

import numpy as np
import torch

from ..inference import utils
from ..inference.clicker import Clicker

try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm


def evaluate_dataset(dataset, predictor, oracle_eval=False, **kwargs):
    all_ious = []

    start_time = time()
    for index in tqdm(range(len(dataset)), leave=False):
        sample = dataset.get_sample(index)
        item = dataset[index]

        if oracle_eval:
            gt_mask = torch.tensor(sample['instances_mask'], dtype=torch.float32)
            gt_mask = gt_mask.unsqueeze(0).unsqueeze(0)
            predictor.opt_functor.mask_loss.set_gt_mask(gt_mask)
        _, sample_ious, _ = evaluate_sample(item['images'], sample['instances_mask'], predictor, **kwargs)
        all_ious.append(sample_ious)
    end_time = time()
    elapsed_time = end_time - start_time

    return all_ious, elapsed_time


def evaluate_sample(image_nd, instances_mask, predictor, max_iou_thr,
                    pred_thr=0.49, max_clicks=20):
    clicker = Clicker(gt_mask=instances_mask)
    pred_mask = np.zeros_like(instances_mask)
    ious_list = []

    with torch.no_grad():
        predictor.set_input_image(image_nd)

        for click_number in range(max_clicks):
            clicker.make_next_click(pred_mask)
            pred_probs = predictor.get_prediction(clicker)
            pred_mask = pred_probs > pred_thr

            iou = utils.get_iou(instances_mask, pred_mask)
            ious_list.append(iou)

            if iou >= max_iou_thr:
                break

        return clicker.clicks_list, np.array(ious_list, dtype=np.float32), pred_probs
