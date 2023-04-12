import torch
from .fbrs.controller import InteractiveController
from .fbrs.inference import utils


class FBRSController:
    def __init__(self, checkpoint_path, device='cuda:0', max_size=800):
        model = utils.load_is_model(checkpoint_path, device, cpu_dist_maps=True, norm_radius=260)

        # Predictor params
        zoomin_params = {
                'skip_clicks': 1,
                'target_size': 480,
                'expansion_ratio': 1.4,
            }

        predictor_params = {
            'brs_mode': 'f-BRS-B',
            'prob_thresh': 0.5,
            'zoom_in_params': zoomin_params,
            'predictor_params': {
                'net_clicks_limit': 8,
                'max_size': 800,
            },
            'brs_opt_func_params': {'min_iou_diff': 1e-3},
            'lbfgs_params': {'maxfun': 20}
        }

        self.controller = InteractiveController(model, device, predictor_params)
        self.anchored = False
        self.device = device

    def unanchor(self):
        self.anchored = False

    def interact(self, image, x, y, is_positive):
        image = image.to(self.device, non_blocking=True)
        if not self.anchored:
            self.controller.set_image(image)
            self.controller.reset_predictor()
            self.anchored = True

        self.controller.add_click(x, y, is_positive)
        # return self.controller.result_mask
        # return self.controller.probs_history[-1][1]
        return (self.controller.probs_history[-1][1]>0.5).float()

    def undo(self):
        self.controller.undo_click()
        if len(self.controller.probs_history) == 0:
            return None
        else:
            return (self.controller.probs_history[-1][1]>0.5).float()