import numpy as np
from pathlib import Path
import pandas as pd
import torch
from loguru import logger


def read_file(path):
    label_path = Path(path)
    if label_path.suffix == ".csv":
        try:
            label = pd.read_csv(label_path)
        except FileNotFoundError as e:
            logger.error(f"label file not found:{e}")
            return
        return label
    else:
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError as e:
            logger.error(f"label file not found:{e}")
            return
        return lines


def get_class_number(label_file, list):
    action_count = [0 for _ in range(len(action_label))]
    file = read_file(label_file)
    for id in list:
        labels = file[file["video id"] == id]
        start = labels['start']
        end = labels['end']
        act = labels['action label']

        for i, (s, e, a) in enumerate(zip(start, end, act)):
            time = e * 30 - s * 30
            action = action_label[a]
            action_count[action] += time
    
    return action_count

def get_class_weight(list, label_file):
    video_list = [int(file.stem) for file in list]
    count = get_class_number(label_file, video_list)
    class_num = torch.tensor(count)
    total = class_num.sum().item()
    frequency = class_num.float() / total
    median = torch.median(frequency)
    class_weight = median / frequency

    return class_weight

def get_pos_weight(list, label_file):
    n_classes = 2
    video_list = [int(file.stem)  for file in list]
    nums = [0 for _ in range(n_classes)]
    file = read_file(label_file)

    for id in video_list:
        labels = file[file["video id"] == id]
        start = labels['start']
        end = labels['end']
        act = labels['action label']

        boundary = boundary_label(start, end, act)
        for b in boundary:
            nums[b] += 1
    pos_ratio = nums[1] / sum(nums)
    pos_weight = 1 / pos_ratio

    return torch.tensor(pos_weight)
