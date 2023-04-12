import torch

from ..fbrs.inference import clicker
from ..fbrs.inference.predictors import get_predictor


class InteractiveController:
    def __init__(self, net, device, predictor_params, prob_thresh=0.5):
        self.net = net.to(device)
        self.prob_thresh = prob_thresh
        self.clicker = clicker.Clicker()
        self.states = []
        self.probs_history = []
        self.object_count = 0
        self._result_mask = None

        self.image = None
        self.predictor = None
        self.device = device
        self.predictor_params = predictor_params
        self.reset_predictor()

    def set_image(self, image):
        self.image = image
        self._result_mask = torch.zeros(image.shape[-2:], dtype=torch.uint8)
        self.object_count = 0
        self.reset_last_object()

    def add_click(self, x, y, is_positive):
        self.states.append({
            'clicker': self.clicker.get_state(),
            'predictor': self.predictor.get_states()
        })

        click = clicker.Click(is_positive=is_positive, coords=(y, x))
        self.clicker.add_click(click)
        pred = self.predictor.get_prediction(self.clicker)
        torch.cuda.empty_cache()

        if self.probs_history:
            self.probs_history.append((self.probs_history[-1][0], pred))
        else:
            self.probs_history.append((torch.zeros_like(pred), pred))

    def undo_click(self):
        if not self.states:
            return

        prev_state = self.states.pop()
        self.clicker.set_state(prev_state['clicker'])
        self.predictor.set_states(prev_state['predictor'])
        self.probs_history.pop()

    def partially_finish_object(self):
        object_prob = self.current_object_prob
        if object_prob is None:
            return

        self.probs_history.append((object_prob, torch.zeros_like(object_prob)))
        self.states.append(self.states[-1])

        self.clicker.reset_clicks()
        self.reset_predictor()

    def finish_object(self):
        object_prob = self.current_object_prob
        if object_prob is None:
            return

        self.object_count += 1
        object_mask = object_prob > self.prob_thresh
        self._result_mask[object_mask] = self.object_count
        self.reset_last_object()

    def reset_last_object(self):
        self.states = []
        self.probs_history = []
        self.clicker.reset_clicks()
        self.reset_predictor()

    def reset_predictor(self, predictor_params=None):
        if predictor_params is not None:
            self.predictor_params = predictor_params
        self.predictor = get_predictor(self.net, device=self.device,
                                       **self.predictor_params)
        if self.image is not None:
            self.predictor.set_input_image(self.image)

    @property
    def current_object_prob(self):
        if self.probs_history:
            current_prob_total, current_prob_additive = self.probs_history[-1]
            return torch.maximum(current_prob_total, current_prob_additive)
        else:
            return None

    @property
    def is_incomplete_mask(self):
        return len(self.probs_history) > 0

    @property
    def result_mask(self):
        return self._result_mask.clone()
