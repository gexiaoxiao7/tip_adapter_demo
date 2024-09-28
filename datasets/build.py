import pandas as pd
import os
from torch.utils.data import DataLoader
from PIL import Image
import torch
import clip
import cv2
class ImageDataset():
    def __init__(self, config, preprocess, ann_file ,device):
        self.root_path = config['root_path']
        self.labels_file = config['labels_file']
        self.preprocess = preprocess
        self.ann_file = ann_file
        self.shots = config['shots']
        self.device = device
        self.num_classes = config['num_classes']
        self.image_info = self.load_annotations()
    @property
    def classes(self):
        classes_all = pd.read_csv(self.labels_file)
        return classes_all.values.tolist()

    def prepare_image(self, path):
        path = os.path.join(self.root_path, path)
        if not os.path.exists(path):
            print(f"File {path} not found.")
            return None
        # 读取图片
        image = cv2.imread(path)
        if image is None:
            print(f"File {path} is broken.")
            return None
        return self.preprocess(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(self.device)

    def load_annotations(self):
        image_info = []
        with open(self.ann_file, 'r') as fin:
            lines = fin.readlines()
            for (idx, line) in enumerate(lines):
                if (idx >= self.num_classes * self.shots):
                    break
                line_split = line.strip().split()
                filename, label = line_split
                label = int(label)
                data = self.prepare_image(filename)
                if data is not None:
                    # 将形状转化为(channels, height, width), 去掉第一维数据
                    data = data.squeeze(0)
                    image_info.append((data, label))
        return image_info

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        return self.image_info[idx]

class SubsetRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.epoch = 0
        self.indices = indices
    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))
    def __len__(self):
        return len(self.indices)
    def set_epoch(self, epoch):
        self.epoch = epoch


def build_dataloader(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, preprocess = clip.load(config['backbone'], device=device)
    test_data = ImageDataset(config, preprocess=preprocess, device=device, ann_file=config['test_file'])
    test_loader = DataLoader(test_data, batch_size=config['batch_size'], sampler=SubsetRandomSampler(list(range(len(test_data)))))
    val_data = ImageDataset(config, preprocess=preprocess, device=device, ann_file=config['val_file'])
    val_loader = DataLoader(val_data, batch_size=config['batch_size'], sampler=SubsetRandomSampler(list(range(len(val_data)))))
    train_cache = ImageDataset(config, preprocess=preprocess, device=device, ann_file=config['cache_file'])
    train_loader_cache = DataLoader(train_cache, batch_size=config['batch_size'], sampler=SubsetRandomSampler(list(range(len(train_cache)))))
    train_F = ImageDataset(config, preprocess=preprocess, device=device, ann_file=config['train_file'])
    train_loader_F = DataLoader(train_F, batch_size=config['batch_size'], sampler=SubsetRandomSampler(list(range(len(train_F)))))
    return train_loader_cache, train_loader_F, val_loader, test_loader

