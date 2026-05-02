import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
from pathlib import Path
from typing import Union, Optional
from PIL import Image
from torch.optim.swa_utils import AveragedModel
from collections import OrderedDict
from inference.dfc_versions.DFC3_2 import DFC_3_2
from inference.dfc_versions.DFC3_3 import DFC_3_3



def find_best_threshold(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.01)

    best_thr = 0.5
    best_f1 = -1.0

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)

        if score > best_f1:
            best_f1 = score
            best_thr = thr

    return best_thr, best_f1


@torch.inference_mode()
def predict_test(model, loader, 
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
                 use_tta=True, verbose=True):
    model.eval()
    model.to(device)

    all_probs = []
    all_ids = []

    ranger = tqdm(loader, desc="Predict test") if verbose else loader

    for images, ids in ranger:
        images = images.to(device, non_blocking=True)

        logits = model(images).squeeze(1)

        if use_tta:
            images_flip = torch.flip(images, dims=[3]) 
            logits_flip = model(images_flip).squeeze(1)

            logits = (logits + logits_flip) / 2.0 # Среднее по логитам
        
        probs = torch.sigmoid(logits)
        probs = probs.detach().cpu().numpy().reshape(-1)

        all_probs.extend(probs.tolist())
        all_ids.extend([int(x) for x in ids])

    pred_df = pd.DataFrame({
        'Id': all_ids,
        "probability": all_probs
    }).sort_values('Id').reset_index(drop=True)

    return pred_df


def get_base_transform():
    from torchvision.transforms.v2 import (
        Compose, ToImage, ToDtype, Normalize, Resize
    )
    MEAN = [0.519, 0.428, 0.384]
    STD = [0.286, 0.264, 0.264]

    return Compose([
        Resize((256, 256), antialias=True),
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=MEAN, std=STD),
    ])

@torch.inference_mode()
def predict(
    model: nn.Module,
    image_path: Union[str, Path],
    transform=None,
    threshold: Optional[float] = 0.5,
    device: Optional[torch.device] = None,
    use_tta: bool = True,
    return_logit: bool = False,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Файл не найден: {image_path}")

    model.eval()
    model.to(device)

    image = Image.open(image_path).convert("RGB")

    if transform is None:
        transform = get_base_transform()

    image = transform(image)

    if not isinstance(image, torch.Tensor):
        raise TypeError("После transform должен получиться torch.Tensor")

    image = image.unsqueeze(0).to(device, non_blocking=True)
    logits = model(image).squeeze(1)

    if use_tta:
        image_flip = torch.flip(image, dims=[3])
        logits_flip = model(image_flip).squeeze(1)
        logits = (logits + logits_flip) / 2.0

    prob = torch.sigmoid(logits).item()
    pred = int(prob >= threshold)

    return (pred, prob) if not return_logit else (pred, prob, logits.item())


# Функция для проверки, поддерживается ли указанный тип модели
def valid(model_type):
    return model_type.lower() in {'dfc-3.2', 'dfc-3.3', 'dfc-3.4'}


# Функция для создания ema-модели
def build_ema_model(model, device, decay=0.9995):
    def ema_avg_fn(averaged_model_parameter, model_parameter, num_averaged):
        return decay * averaged_model_parameter + (1.0 - decay) * model_parameter

    ema_model = AveragedModel(model, avg_fn=ema_avg_fn).to(device)
    return ema_model


# Функция для загрузки модели, лучшего порога, лучшего f1-score и конфигов
def load_model(model, path, use_ema=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ckpt = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    raw_sd = ckpt["model_state_dict"]

    clean_sd = OrderedDict()
    for k, v in raw_sd.items():
        if k == "n_averaged":
            continue
        if k.startswith("module."):
            k = k[len("module."):]
        clean_sd[k] = v

    model.load_state_dict(clean_sd)
    model.eval()

    if use_ema:
        model = build_ema_model(model, device)
    
    return model, ckpt['best_threshold'], ckpt['best_f1'], ckpt['config']


# Получение пути к весам модели в зависимости от типа
def get_model_weights_path(model_type):
    current_dir = os.getcwd()

    if 'YL_Project_Solution' not in current_dir:
        raise RuntimeError(f"Текущая директория должна быть внутри папки 'YL_Project_Solution', но сейчас: {current_dir}")
    else:
        model_weights_path = os.path.join(current_dir.split('YL_Project_Solution')[0], 'YL_Project_Solution/model_weights')

        if model_type == 'dfc-3.2':
            return os.path.join(model_weights_path, 'dfc-3.2-weights')
        if model_type == 'dfc-3.3':
            return os.path.join(model_weights_path, 'dfc-3.3-weights')
        if model_type == 'dfc-3.4':
            return [
                os.path.join(model_weights_path, 'dfc-3.2-weights'),
                os.path.join(model_weights_path, 'dfc-3.3-weights'),
                os.path.join(model_weights_path, 'dfc-3.4-weights', 'meta_model.pkl')
            ]

        raise ValueError(f"Неизвестный model_type: {model_type}")
    

# Загрузка моделей всех фолдов
def load_models(model_type):
    dir_path = get_model_weights_path(model_type)

    if model_type == 'dfc-3.2':
        models = []
        files = sorted([f for f in os.listdir(dir_path) if f.endswith('.pt')])

        for file_name in tqdm(files, desc='Downloading DFC-3.2'):
            model_path = os.path.join(dir_path, file_name)
            model, _, _, _ = load_model(DFC_3_2(), model_path, use_ema=False)
            models.append(model)
        
        return models
    elif model_type == 'dfc-3.3':
        models = []
        files = sorted([f for f in os.listdir(dir_path) if f.endswith('.pt')])

        for file_name in tqdm(files, desc='Downloading DFC-3.3'):
            model_path = os.path.join(dir_path, file_name)
            model, _, _, _ = load_model(DFC_3_3(), model_path, use_ema=False)
            models.append(model)
        
        return models
    else:
        models = [[], []]

        len1 = len(os.listdir(dir_path[0]))
        len2 = len(os.listdir(dir_path[1]))

        assert len1 == len2, "Количество моделей в dfc-3.2 и dfc-3.3 должно совпадать для dfc-3.4"

        files1 = sorted([f for f in os.listdir(dir_path[0]) if f.endswith('.pt')])
        files2 = sorted([f for f in os.listdir(dir_path[1]) if f.endswith('.pt')])

        for f1, f2 in tqdm(zip(files1, files2), total=len(files1), desc='Downloading DFC-3.4'):
            model_path1 = os.path.join(dir_path[0], f1)
            model_path2 = os.path.join(dir_path[1], f2)

            model1, _, _, _ = load_model(DFC_3_2(), model_path1, use_ema=False)
            model2, _, _, _ = load_model(DFC_3_3(), model_path2, use_ema=False)

            models[0].append(model1)
            models[1].append(model2)

        return models