# ------------------------------------------------------------------------------------
# Enhancing Transformers
# Copyright (c) 2022 Thuan H. Nguyen. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import PIL
from typing import Any, Tuple, Union, Optional, Callable
from torch.utils.data import DataLoader, Dataset
CROP_SIZE = 32
CROPS_PER_IMG = 60
CROPS_PER_IMG = 9999999999999

import random
import torch
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from pathlib import Path
import os
from PIL import Image

class MyImageFolder(Dataset):
    def __init__(self):
        super().__init__()
        #paths = list(Path(folder).rglob(ext))

        paths = []
        for p in Path('/home/brans/repos/spaces_dataset-master/data/800').rglob('*.JPG'):
            if '2k' not in str(p):
                paths.append(p)
        random.shuffle(paths)
        self.paths = paths
        self.pathes_processed = 0
        #self.rndCrop = T.RandomCrop(CROP_SIZE)
        #self.rndFlip = T.RandomHorizontalFlip()
        self.transform = T.Compose([
            T.CenterCrop(80),
            T.RandomCrop(CROP_SIZE),
            #T.RandomHorizontalFlip(),
            T.ToTensor()
        ])
        self.img = None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = os.path.join(self.paths[index])
        path = '/home/brans/repos/spaces_dataset-master/data/800/scene_036/cam_09/image_008.JPG'
        if self.pathes_processed > CROPS_PER_IMG or self.img is None:
            self.img = Image.open(path)
            self.pathes_processed = 0

        img = self.transform(self.img)
        self.pathes_processed += 1
        return img



class ImageNetBase(MyImageFolder):
    def __init__(self, root: str, split: str,
                 transform: Optional[Callable] = None) -> None:
        #self.transform = transform
        #self.split = split
        #self.root = root
        super().__init__()
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        sample = super().__getitem__(index)

        return {'image': sample, 'class': torch.tensor([0])}


class ImageNetTrain(ImageNetBase):
    def __init__(self, root: str,
                 resolution: Union[Tuple[int, int], int] = 256,
                 resize_ratio: float = 0.75) -> None:

        # transform = T.Compose([
        #     T.Resize(resolution),
        #     T.RandomCrop(resolution),
        #     T.RandomHorizontalFlip(),
        #     T.ToTensor()
        # ])

        super().__init__(root=root, split='train')
        

class ImageNetValidation(ImageNetBase):
    def __init__(self, root: str,
                 resolution: Union[Tuple[int, int], int] = 256,) -> None:

        # if isinstance(resolution, int):
        #     resolution = (resolution, resolution)
        #
        # transform = T.Compose([
        #     T.Resize(resolution),
        #     T.CenterCrop(resolution),
        #     T.ToTensor()
        # ])

        super().__init__(root=root, split='val')
