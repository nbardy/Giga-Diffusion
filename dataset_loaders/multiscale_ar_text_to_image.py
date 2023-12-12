
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets
import numpy as np
from PIL import Image
import os
import torch

# This is a pytorch data loader tha can be used as a tranform on top of huggingface datasets
# it should accept a list of dataset names, concat them map through them and create some buckets
# crops sizes.
#
# Then it should return a different fized bucket size each batch

buckets = {
    "128_128": 0.05,
    "256_256": 0.10,
    "512_512": 0.20,
    "1024_1024": 0.10,
    "512_1024": 0.10,
    "1024_512": 0.10,
}

bucket_aspect_ratios = {
    "128_128": 1,
    "256_256": 1,
    "512_512": 1,
    "1024_1024": 1,
    "512_1024": 0.5,
    "1024_512": 2,
}


class MultiScaleTextToImage(Dataset):
    def __init__(self, dataset_names, transform=None, bucket_sizes=[128, 256, 512, 1024], crop_types=['random', 'good', 'bad'], crop_percentages=[0.33, 0.33, 0.34]):
        self.datasets = []
        self.transform = transform
        self.bucket_sizes = bucket_sizes
        self.crop_types = crop_types
        self.crop_percentages = crop_percentages

        # Concat
        for name in dataset_names:
            dataset = load_dataset(name)
            self.datasets.append(dataset)


        self.dataset = np.concatenate(self.datasets)

        # shuffe
        self.dataset = np.random.shuffle(self.dataset)
        self.create_buckets()


    def create_buckets(self):
        # Create buckets of different crop sizes
        self.buckets = []
        for idx, data in enumerate(self.dataset):
            img, label = data['image'], data['label']
            for bucket in self.bucket_sizes:
                if img.size[0] <= bucket and img.size[1] <= bucket:
                    # Random crop
                    random_crop = self.crop_image(img, (bucket, bucket), 'random')
                    random_crop.save(f"{idx}_random_crop.webp", "WEBP", quality=98)
                    self.buckets.append((random_crop, label, 'random'))

                    # Good crop
                    aspect_ratio = img.size[0] / img.size[1]
                    closest_aspect_ratio = min(bucket_aspect_ratios, key=lambda x: (bucket_aspect_ratios[x] - aspect_ratio)**2) # [Change] Use MSE between aspect ratios
                    good_crop = self.crop_image(img, (int(closest_aspect_ratio.split('_')[0]), int(closest_aspect_ratio.split('_')[1])), 'good')
                    good_crop.save(f"{idx}_good_crop.webp", "WEBP", quality=98)
                    self.buckets.append((good_crop, label, 'good'))
                    break

    def crop_image(self, img, size, crop_type):
        # Implement your own logic for different crop types
        if crop_type == 'random':
            return self.random_crop(img, size)
        elif crop_type == 'good':
            return self.good_crop(img, size)
        else:
            return

    def random_crop(self, img, size):
        # Get the width and height of the image
        width, height = img.size

        # Calculate the starting points for the crop
        left = np.random.randint(0, width - size[0] + 1)
        top = np.random.randint(0, height - size[1] + 1)

        # Perform the crop
        img_cropped = img.crop((left, top, left + size[0], top + size[1]))

        return img_cropped

    def good_crop(self, img, size):
        # Get the width and height of the image
        width, height = img.size

        # Calculate the starting points for the crop
        left = (width - size[0]) // 2
        top = (height - size[1]) // 2

        # Perform the crop
        img_cropped = img.crop((left, top, left + size[0], top + size[1]))

        return img_cropped


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.dataset[idx]
        if self.transform:
            sample = self.transform(sample)

        return sample