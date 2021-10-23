import os

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

import nibabel as nib

import matplotlib.pyplot as plt

from skimage.transform import resize
from skimage.util import montage
import torch
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader

from albumentations import Compose, HorizontalFlip, CenterCrop

from config import *

from einops import rearrange

class BratsDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 phase: str='test',
                 is_resize: bool=False,
                 with_drop: bool=False,
                 frac=0.2,
                 all_sequence: bool=True):
        self.with_drop = with_drop
        self.all_sequence = all_sequence

        if self.with_drop:
            self.df = df.sample(frac=self.frac).sort_index()
        else:
            self.df = df

        if self.all_sequence:
            self.data_types = ['_flair.nii.gz', '_t1ce.nii.gz', '_t1.nii.gz', '_t2.nii.gz']
        else:
            self.data_types_t1 = ['_t1ce.nii.gz', '_t1.nii.gz']
            self.data_types_t2 = ['_flair.nii.gz', '_t2.nii.gz']
        self.phase = phase
        self.augmentations = get_augmentations(phase)
        self.data_types = ['_flair.nii.gz', '_t1.nii.gz', '_t1ce.nii.gz', '_t2.nii.gz']
        self.is_resize = is_resize
        self.frac = frac

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        id_ = self.df.loc[idx, 'Brats20ID']
        root_path = self.df.loc[self.df['Brats20ID'] == id_]['path'].values[0]
        if self.all_sequence:
            images = []
            for data_type in self.data_types:
                img_path = os.path.join(root_path, id_ + data_type)
                img = self.load_img(img_path)

                if self.is_resize:
                    img = self.resize(img)

                img = self.normalize(img)
                images.append(img)

            img = np.stack(images)
            img = np.moveaxis(img, (0, 1, 2, 3), (0, 3, 2, 1))
        else:
            images_t1 = []
            images_t2 = []
            for data_type_t1, data_type_t2 in self.data_types_t1, self.data_types_t2:
                img_path_t1 = os.path.join(root_path, id_ + data_type_t1)
                img_path_t2 = os.path.join(root_path, id_ + data_type_t2)
                img_t1= self.load_img(img_path_t1)
                img_t2 = self.load_img(img_path_t2)

                if self.is_resize:
                    img_t1 = self.resize(img_t1)
                    img_t2 = self.resize(img_t2)

                img_t1 = self.normalize(img_t1)
                img_t2 = self.normalize(img_t2)
                images_t1.append(img_t1)
                images_t2.append(img_t2)


            img_t1 = np.stack(images_t1)
            img_t2 = np.stack(images_t2)

            img_t1 = np.moveaxis(img_t1, (0, 1, 2, 3), (0, 3, 2, 1))
            img_t2 = np.moveaxis(img_t2, (0, 1, 2, 3), (0, 3, 2, 1))

        if self.phase != 'test':
            mask_path = os.path.join(root_path, id_ + '_seg.nii.gz')
            mask = self.load_img(mask_path)

            if self.is_resize:
                mask = self.resize(mask)
                mask = np.clip(mask.astype(np.uint8), 0, 1).astype(np.float32)
                mask = np.clip(mask, 0, 1)
            mask = self.prepocess_mask_labels(mask)

            if self.all_sequence:
                augmented = self.augmentations(image=img.astype(np.float32), mask=mask.astype(np.float32))
                img = augmented['image']
                mask = augmented['mask']
            else:
                augmented = self.augmentations(image_t1=img_t1.astype(np.float32), image_t2=img_t2.astype(np.float32), mask=mask.astype(np.float32))
                img_t1 = augmented['image_t1']
                img_t2 = augmented['image_t2']
                mask = augmented['mask']

            if self.all_sequence:
                return {
                    'Id': id_,
                    'image':img,
                    'mask':mask,
                }
            else:
                return {
                    'Id': id_,
                    'image_t1':img_t1,
                    'image_t2':img_t2,
                    'mask':mask,
                }
        return {
            "Id": id_,
            'image':img,
        }

    def load_img(self, file_path):
        data = nib.load(file_path)
        data = np.asarray(data.dataobj)
        return data

    def normalize(self, data: np.ndarray):
        data_min = np.min(data)
        return (data - data_min) / (np.max(data) - data_min)

    def resize(self, data: np.ndarray):
        data = resize(data, (78, 120, 120), preserve_range=True)
        return data

    def prepocess_mask_labels(self, mask: np.ndarray):

        mask_WT = mask.copy()
        mask_WT[mask_WT == 1] = 1
        mask_WT[mask_WT == 2] = 1
        mask_WT[mask_WT == 4] = 1

        mask_TC = mask.copy()
        mask_TC[mask_TC == 1] = 1
        mask_TC[mask_TC == 2] = 0
        mask_TC[mask_TC == 4] = 1

        mask_ET = mask.copy()
        mask_ET[mask_ET == 1] = 0
        mask_ET[mask_ET == 2] = 0
        mask_ET[mask_ET == 4] = 1

        mask = np.stack([mask_WT, mask_TC, mask_ET])
        mask = np.moveaxis(mask, (0, 1, 2, 3), (0, 3, 2, 1))

        return mask


def get_augmentations(phase):
    list_transforms = []
    list_trfms = Compose(list_transforms)
    return list_trfms


def get_dataloader(
        dataset: torch.utils.data.Dataset,
        path_to_csv: str,
        phase: str,
        fold: int = 0,
        batch_size: int = 1,
        num_workers: int = 4,
        with_drop: bool = False,
        all_sequence: bool = True
):
    df = pd.read_csv(path_to_csv)
    train_df = df.loc[df['fold'] != fold].reset_index(drop=True)
    val_df = df.loc[df['fold'] == fold].reset_index(drop=True)

    df = train_df if phase == 'train' else val_df
    dataset = dataset(df, phase, is_resize=False, with_drop=with_drop, frac=0.2, all_sequence=all_sequence)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )
    return dataloader


if __name__ == '__main__':
    dataloader = get_dataloader(dataset=BratsDataset, path_to_csv='log/train_data.csv', phase='train', fold=0, all_sequence=False)
    print(len(dataloader))

    data = next(iter(dataloader))
    print(data['Id'], data['image_t1'].shape, data['image_t2'].shape, data['mask'].shape)
    img1 = data['image_t1']
    img2 = data['image_t2']
    img = np.zip(img1, img2)
    print(img.shape)



    # img_tensor = data['image'].squeeze().cpu().detach().numpy()
    # mask_tensor = data['mask'].squeeze().squeeze().cpu().detach().numpy()

    # print('data[image].shape', data['image'].shape)
    # print('data[mask].shape', data['mask'].shape)
    # print('Num uniq Image valuses:', len(np.unique(img_tensor, return_counts=True)[0]))
    # print('Min/Max Image values:', img_tensor.min(), img_tensor.max())
    # print('Num uniq Mask values:', np.unique(mask_tensor, return_counts=True))

    # dataloader = get_paired_dataloader(dataset_t1=BratsDatasetT1,
    #                                    dataset_t2=BratsDatasetT2,
    #                                    path_to_csv='log/train_data.csv',
    #                                    phase="train",
    #                                    fold=0)
    #
    # for iter, data_batch in enumerate(dataloader):
    #
    #     img, mask = data_batch[0]['image'], data_batch[0]['mask']
    #     print(img.shape)
    #     print(mask.shape)










