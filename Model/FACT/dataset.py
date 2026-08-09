import numpy as np
from pathlib import Path
from eval import read_file, framewise_label
from action_dict import action_label
from collections import Counter
import torch

def load_feature(feature_name, transpose = True):
    feature = np.load(feature_name)
    if transpose:
        feature = feature.T
    return feature  #(T, H)

class Dataset():
    """
    self.features[video]: the feature array of the given video (frames x dimension)
    self.input_dimension: dimension of video features
    self.n_classes: number of classes
    """
    def __init__(self, video_list, nclasses, load_video_func, bg_class):
        self.video_list = video_list
        self.load_video = load_video_func

        self.nclasses = nclasses
        self.bg_class = bg_class
        self.data = {}

        self.data[video_list[0]] = load_video_func(video_list[0])
        if isinstance(self.data[video_list[0]], tuple):
            # training / evaluation mode:
            # first_item = (feature, train_label, eval_label)
            self.input_dimension = self.data[video_list[0]][0].shape[1]
        else:
            # inference mode:
            # first_item = feature
            self.input_dimension = self.data[video_list[0]].shape[1]

    
    def __str__(self):
        string = "< Dataset %d videos, %d feat-size, %d classes >"
        string = string % (len(self.video_list), self.input_dimension, self.nclasses)
        return string
    
    def __repr__(self):
        return str(self)

    def get_vnames(self):
        return self.video_list[:]

    def __getitem__(self, video):
        if video not in self.video_list:
            raise ValueError(video)

        if video not in self.data:
            self.data[video] = self.load_video(video)

        return self.data[video]

    def __len__(self):
        return len(self.video_list)


class DataLoader():
    def __init__(self, dataset: Dataset, batch_size, shuffle=False):

        self.num_video = len(dataset)
        self.dataset = dataset
        self.videos = list(dataset.get_vnames())
        self.shuffle = shuffle
        self.batch_size = batch_size

        self.num_batch = int(np.ceil(self.num_video/self.batch_size))
        self.selector = list(range(self.num_video))
        self.index = 0
        if self.shuffle:
            np.random.shuffle(self.selector)
            # self.selector = self.selector.tolist()

    def __len__(self):
        return self.num_batch

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.num_video:
            if self.shuffle:
                np.random.shuffle(self.selector)
                # self.selector = self.selector.tolist()
            self.index = 0
            raise StopIteration

        else:
            video_idx = self.selector[self.index : self.index+self.batch_size]
            if len(video_idx) < self.batch_size:
                video_idx = video_idx + self.selector[:self.batch_size-len(video_idx)]
            videos = [self.videos[i] for i in video_idx]
            self.index += self.batch_size

            batch_sequence = []
            batch_train_label = []
            batch_eval_label = []
            for vname in videos:
                sequence, train_label, eval_label = self.dataset[vname]
                batch_sequence.append(torch.from_numpy(sequence))
                batch_train_label.append(torch.LongTensor(train_label))
                batch_eval_label.append(eval_label)

            return videos, batch_sequence, batch_train_label, batch_eval_label

def load_action_mapping(label2index):
    index2label = dict()
    for k,v in label2index.items():
        index2label[v] = k

    return index2label

def shrink_frame_label(label, clip_len) -> list:
    num_clip = ((len(label) - 1) // clip_len) + 1
    new_label = []
    for i in range(num_clip):
        s = i * clip_len
        e = s + clip_len
        l = label[s:e]
        ct = Counter(l)
        l = ct.most_common()[0][0]
        new_label.append(l)

    return new_label

def create_dataset(video_list, label_path, sample_rate = None):
    index2label = load_action_mapping(action_label)
    nclasses = len(index2label)
    label_file = read_file(label_path)
    bg_class = [7]
    """
    load video interface:
        Input: video name
        Output:
            feature, label_for_training, label_for_evaluation
    """
    def load_video(feature_path : Path):
        video_id = int(feature_path.stem)
        feature = load_feature(feature_path, transpose=False) #(T,H)

        actions = label_file[label_file["video id"] == int(video_id)]
        start = np.array(actions["start"])
        end = np.array(actions["end"])
        acts = np.array(actions["action label"])
        gt_label = framewise_label(start, end, acts)  

        if feature.shape[1] != len(gt_label):
            l = min(feature.shape[0], len(gt_label))
            feature = feature[:l]
            gt_label = gt_label[:l]  #(T)

        
        if sample_rate:
            feature = feature[::sample_rate]
            gt_label_sampled = shrink_frame_label(gt_label, sample_rate)
        else:
            gt_label_sampled = gt_label

        return feature, gt_label_sampled, gt_label

    ################################################
    dataset = Dataset(video_list, nclasses, load_video, bg_class)
        
    dataset.label2index = action_label
    dataset.index2label = index2label

    test_dataset.label2index = action_label
    test_dataset.index2label = index2label

    return dataset


class InferenceDataLoader():
    def __init__(self, dataset: Dataset, batch_size, shuffle=False):
        self.num_video = len(dataset)
        self.dataset = dataset
        self.videos = list(dataset.get_vnames())
        self.shuffle = shuffle
        self.batch_size = batch_size

        self.num_batch = int(np.ceil(self.num_video / self.batch_size))
        self.selector = list(range(self.num_video))
        self.index = 0

        if self.shuffle:
            np.random.shuffle(self.selector)

    def __len__(self):
        return self.num_batch

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.num_video:
            if self.shuffle:
                np.random.shuffle(self.selector)
            self.index = 0
            raise StopIteration

        video_idx = self.selector[self.index:self.index + self.batch_size]
        videos = [self.videos[i] for i in video_idx]
        self.index += self.batch_size

        batch_sequence = []
        for vname in videos:
            sequence = self.dataset[vname]
            batch_sequence.append(torch.from_numpy(sequence))

        return videos, batch_sequence, None, None

def create_inference_dataset(video_list, sample_rate=None):
    index2label = load_action_mapping(action_label)
    nclasses = len(index2label)
    bg_class = [5]

    def load_video(feature_path: Path):
        feature = load_feature(feature_path, transpose=False)  # (T, H)

        if sample_rate:
            feature = feature[::sample_rate]

        return feature

    dataset = Dataset(video_list, nclasses, load_video, bg_class)

    dataset.label2index = action_label
    dataset.index2label = index2label

    return dataset