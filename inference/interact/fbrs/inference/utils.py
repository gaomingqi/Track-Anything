from datetime import timedelta
from pathlib import Path

import torch
import numpy as np

from ..model.is_deeplab_model import get_deeplab_model
from ..model.is_hrnet_model import get_hrnet_model


def get_time_metrics(all_ious, elapsed_time):
    n_images = len(all_ious)
    n_clicks = sum(map(len, all_ious))

    mean_spc = elapsed_time / n_clicks
    mean_spi = elapsed_time / n_images

    return mean_spc, mean_spi


def load_is_model(checkpoint, device, backbone='auto', **kwargs):
    if isinstance(checkpoint, (str, Path)):
        state_dict = torch.load(checkpoint, map_location='cpu')
    else:
        state_dict = checkpoint

    if backbone == 'auto':
        for k in state_dict.keys():
            if 'feature_extractor.stage2.0.branches' in k:
                return load_hrnet_is_model(state_dict, device, backbone, **kwargs)
        return load_deeplab_is_model(state_dict, device, backbone, **kwargs)
    elif 'resnet' in backbone:
        return load_deeplab_is_model(state_dict, device, backbone, **kwargs)
    elif 'hrnet' in backbone:
        return load_hrnet_is_model(state_dict, device, backbone, **kwargs)
    else:
        raise NotImplementedError('Unknown backbone')


def load_hrnet_is_model(state_dict, device, backbone='auto', width=48, ocr_width=256,
                        small=False, cpu_dist_maps=False, norm_radius=260):
    if backbone == 'auto':
        num_fe_weights = len([x for x in state_dict.keys() if 'feature_extractor.' in x])
        small = num_fe_weights < 1800

        ocr_f_down = [v for k, v in state_dict.items() if 'object_context_block.f_down.1.0.bias' in k]
        assert len(ocr_f_down) == 1
        ocr_width = ocr_f_down[0].shape[0]

        s2_conv1_w = [v for k, v in state_dict.items() if 'stage2.0.branches.0.0.conv1.weight' in k]
        assert  len(s2_conv1_w) == 1
        width = s2_conv1_w[0].shape[0]

    model = get_hrnet_model(width=width, ocr_width=ocr_width, small=small,
                            with_aux_output=False, cpu_dist_maps=cpu_dist_maps,
                            norm_radius=norm_radius)

    model.load_state_dict(state_dict, strict=False)
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    model.eval()

    return model


def load_deeplab_is_model(state_dict, device, backbone='auto', deeplab_ch=128, aspp_dropout=0.2,
                          cpu_dist_maps=False, norm_radius=260):
    if backbone == 'auto':
        num_backbone_params = len([x for x in state_dict.keys()
                                   if 'feature_extractor.backbone' in x and not('num_batches_tracked' in x)])

        if num_backbone_params <= 181:
            backbone = 'resnet34'
        elif num_backbone_params <= 276:
            backbone = 'resnet50'
        elif num_backbone_params <= 531:
            backbone = 'resnet101'
        else:
            raise NotImplementedError('Unknown backbone')

        if 'aspp_dropout' in state_dict:
            aspp_dropout = float(state_dict['aspp_dropout'].cpu().numpy())
        else:
            aspp_project_weight = [v for k, v in state_dict.items() if 'aspp.project.0.weight' in k][0]
            deeplab_ch = aspp_project_weight.size(0)
            if deeplab_ch == 256:
                aspp_dropout = 0.5

    model = get_deeplab_model(backbone=backbone, deeplab_ch=deeplab_ch,
                              aspp_dropout=aspp_dropout, cpu_dist_maps=cpu_dist_maps,
                              norm_radius=norm_radius)

    model.load_state_dict(state_dict, strict=False)
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    model.eval()

    return model


def get_iou(gt_mask, pred_mask, ignore_label=-1):
    ignore_gt_mask_inv = gt_mask != ignore_label
    obj_gt_mask = gt_mask == 1

    intersection = np.logical_and(np.logical_and(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()
    union = np.logical_and(np.logical_or(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()

    return intersection / union


def compute_noc_metric(all_ious, iou_thrs, max_clicks=20):
    def _get_noc(iou_arr, iou_thr):
        vals = iou_arr >= iou_thr
        return np.argmax(vals) + 1 if np.any(vals) else max_clicks

    noc_list = []
    over_max_list = []
    for iou_thr in iou_thrs:
        scores_arr = np.array([_get_noc(iou_arr, iou_thr)
                               for iou_arr in all_ious], dtype=np.int32)

        score = scores_arr.mean()
        over_max = (scores_arr == max_clicks).sum()

        noc_list.append(score)
        over_max_list.append(over_max)

    return noc_list, over_max_list


def find_checkpoint(weights_folder, checkpoint_name):
    weights_folder = Path(weights_folder)
    if ':' in checkpoint_name:
        model_name, checkpoint_name = checkpoint_name.split(':')
        models_candidates = [x for x in weights_folder.glob(f'{model_name}*') if x.is_dir()]
        assert len(models_candidates) == 1
        model_folder = models_candidates[0]
    else:
        model_folder = weights_folder

    if checkpoint_name.endswith('.pth'):
        if Path(checkpoint_name).exists():
            checkpoint_path = checkpoint_name
        else:
            checkpoint_path = weights_folder / checkpoint_name
    else:
        model_checkpoints = list(model_folder.rglob(f'{checkpoint_name}*.pth'))
        assert len(model_checkpoints) == 1
        checkpoint_path = model_checkpoints[0]

    return str(checkpoint_path)


def get_results_table(noc_list, over_max_list, brs_type, dataset_name, mean_spc, elapsed_time,
                      n_clicks=20, model_name=None):
    table_header = (f'|{"BRS Type":^13}|{"Dataset":^11}|'
                    f'{"NoC@80%":^9}|{"NoC@85%":^9}|{"NoC@90%":^9}|'
                    f'{">="+str(n_clicks)+"@85%":^9}|{">="+str(n_clicks)+"@90%":^9}|'
                    f'{"SPC,s":^7}|{"Time":^9}|')
    row_width = len(table_header)

    header = f'Eval results for model: {model_name}\n' if model_name is not None else ''
    header += '-' * row_width + '\n'
    header += table_header + '\n' + '-' * row_width

    eval_time = str(timedelta(seconds=int(elapsed_time)))
    table_row = f'|{brs_type:^13}|{dataset_name:^11}|'
    table_row += f'{noc_list[0]:^9.2f}|'
    table_row += f'{noc_list[1]:^9.2f}|' if len(noc_list) > 1 else f'{"?":^9}|'
    table_row += f'{noc_list[2]:^9.2f}|' if len(noc_list) > 2 else f'{"?":^9}|'
    table_row += f'{over_max_list[1]:^9}|' if len(noc_list) > 1 else f'{"?":^9}|'
    table_row += f'{over_max_list[2]:^9}|' if len(noc_list) > 2 else f'{"?":^9}|'
    table_row += f'{mean_spc:^7.3f}|{eval_time:^9}|'

    return header, table_row