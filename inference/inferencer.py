import pickle
import torch
import numpy as np
from inference.utils import *


best_dfc_3_2_threshold = 0.74
best_dfc_3_3_threshold = 0.57
best_dfc_3_4_threshold = 0.82


class DFC_Activator:
    def __init__(self, model_type: str = 'dfc-3.2'):
        model_type = model_type.lower()

        if not valid(model_type):
            raise ValueError(f"Неправильная версия модели: {model_type}")

        self.model_type = model_type

        if self.model_type in {'dfc-3.2', 'dfc-3.3'}:
            self.models = load_models(self.model_type)
        else:
            paths = get_model_weights_path('dfc-3.4')

            self.models_3_2 = load_models('dfc-3.2')
            self.models_3_3 = load_models('dfc-3.3')

            meta_model_path = paths[2]
            with open(meta_model_path, 'rb') as f:
                self.meta_model = pickle.load(f)


    def _get_probs(self, image_path, models):
        probs = []
        for model in models:
            _, prob = predict(model, image_path)
            probs.append(float(prob))
        return probs


    @torch.inference_mode()
    def activate(self, image):
        image_path = str(image)

        if self.model_type in {'dfc-3.2', 'dfc-3.3'}:
            probs = self._get_probs(image_path, self.models)
            mean_prob = sum(probs) / len(probs)

            thr = best_dfc_3_2_threshold if self.model_type == 'dfc-3.2' else best_dfc_3_3_threshold
            pred = mean_prob >= thr

            return pred, mean_prob

        probs_3_2 = self._get_probs(image_path, self.models_3_2)
        probs_3_3 = self._get_probs(image_path, self.models_3_3)

        prob_dfc32 = sum(probs_3_2) / len(probs_3_2)
        prob_dfc33 = sum(probs_3_3) / len(probs_3_3)

        probs = np.array([prob_dfc32, prob_dfc33], dtype=np.float32)

        prob_mean = probs.mean()
        prob_std = probs.std()
        prob_min = probs.min()
        prob_max = probs.max()
        prob_median = np.median(probs)
        prob_range = prob_max - prob_min
        prob_12_gap = abs(prob_dfc32 - prob_dfc33)

        features = np.array([[
            prob_dfc32,
            prob_dfc33,
            prob_mean,
            prob_std,
            prob_min,
            prob_max,
            prob_median,
            prob_range,
            prob_12_gap
        ]], dtype=np.float32)

        meta_prob = float(self.meta_model.predict_proba(features)[0, 1])
        pred = meta_prob >= best_dfc_3_4_threshold

        return pred, meta_prob

    def __call__(self, image):
        return self.activate(image)