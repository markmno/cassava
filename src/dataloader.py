import os
from typing import Literal, Optional

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset


class TestDataset(Dataset):
    def __init__(self, df, root, transform=None):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.file_names = df["image_id"].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        file_name = self.file_names[index]
        file_path = f"{self.root}/{file_name}"
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            res = self.transform(image=image)
            image = res["image"]

        return image


class ImageDataset(Dataset):
    def __init__(self, df, root, transforms=None, output_labels=True):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.root = root
        self.transforms = transforms
        self.output_labels = output_labels

    @staticmethod
    def get_img(path):
        img_bgr = cv2.imread(path)
        img = img_bgr[:, :, ::-1]
        return img

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.df.iloc[idx]["image_id"])
        image = self.get_img(path)
        if self.transforms:
            image = self.transforms(image=image)["image"]
        if self.output_labels:
            label = self.df.iloc[idx]["label"]
            return image, label
        return image


class LoadData:
    def __init__(self, df, root, img_size):
        self.df = df
        self.root = root
        self.img_size = img_size

    def get_train_transforms(self):
        return A.Compose(
            [
                A.Resize(self.img_size, self.img_size),
                A.Transpose(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.5),
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),  # type: ignore
                A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),  # type: ignore
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                A.CoarseDropout(p=0.8),
                ToTensorV2(p=1.0),
            ]
        )

    def get_test_transforms(self):
        return A.Compose(
            [
                A.Resize(self.img_size, self.img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(p=1.0),
            ]
        )

    def get_dataloader(
        self,
        idx: Optional[int],
        bs: int,
        shuffle: bool = False,
        mode: Literal["train", "test"] = "train",
        output_labels: bool = True,
    ):
        df = self.df.iloc[idx, :] if idx is not None else self.df
        transforms = (
            self.get_train_transforms if mode == "train" else self.get_test_transforms
        )
        dataset = ImageDataset(
            df=df, root=self.root, transforms=transforms(), output_labels=output_labels
        )
        return DataLoader(
            dataset=dataset,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
        )
