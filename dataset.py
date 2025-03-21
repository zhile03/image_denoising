import os
import cv2
import glob
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

def augmentation(img, crop_size):  # custom augmentation function (random crop, flips, rotations)
    h, w = img.shape[:2]
    # check the crop size
    assert crop_size <= h and crop_size <= w, 'crop size is too large'

    y1 = random.randint(0, h - crop_size)
    x1 = random.randint(0, w - crop_size)
    img = img[y1:y1 + crop_size, x1:x1 + crop_size]

    # geometric transformation
    if random.random() < 0.5:  # hflip
        img = img[:, ::-1]
    if random.random() < 0.5:  # vflip
        img = img[::-1, :]
    # if random.random() < 0.5:  # rot90
        # img = img.transpose(1, 0, 2) # use this if the input is not the grayscale image
        # img = img.transpose(1, 0)
    rotate = random.randint(0, 3)
    img = np.rot90(img, rotate)

    return np.ascontiguousarray(img)


def add_gaussian_noise(img, mean, std):
    noise = np.random.normal(mean, std, img.shape)
    noisy_image = img + noise
    noisy_image = np.clip(noisy_image, 0, 1)
    return noisy_image

def add_salt_pepper_noise(img, noise_ratio=0.02):
    noisy_img = img.copy()
    h, w = noisy_img.shape[:2]
    num_noise_pixels = int(noise_ratio * h * w)
    
    rows = np.random.randint(0, h, num_noise_pixels)
    cols = np.random.randint(0, w, num_noise_pixels)
    noisy_img[rows, cols] = 0 if np.random.rand() < 0.5 else 1
    return noisy_img

# dataset train test 不通用
# train: augmentation
# test: no aug

##### 测试用的模拟数据，没有带噪声的图像，我们是在干净图像上加噪声来模拟输入的
##### 测试结果不一致，因为每次的噪声都是随机的
##### 每张图多随机几次，保存img1_noise1, img1_noise2, ..., img1_noiseN

class DenoisingDataset(Dataset):
    def __init__(self, image_dir, phase='train', patch_size=40, mean=0, sigma=25, num_patches=512, debug=False):
        self.image_paths = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png"))
        self.patch_size = patch_size
        self.phase = phase
        # normalize noise level
        self.mean = mean / 255.0
        self.sigma = sigma / 255.0
        self.num_patches = num_patches  # each image will generate multiple patches

        self.debug = debug

    def __len__(self):

        if self.phase == 'train':
            if self.debug:
                return len(self.image_paths) * 10
            else:
                return len(self.image_paths) * self.num_patches
        else:
            if self.debug:
                return min(10, len(self.image_paths))
            else:
                return len(self.image_paths)


    def __getitem__(self, index):
        # load and preprocess image
        img_index = index % len(self.image_paths)  # ensure index loops over images
        img_path = self.image_paths[img_index]
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        img = img.astype(np.float32) / 255.0  # normalize

        if self.phase == 'train':
            # apply augmentations
            patch = augmentation(img, self.patch_size)  # extract patch
            noise_type = random.choice(['gaussian', 'salt_pepper'])
            if noise_type == 'gaussian':
                noisy_patch = add_gaussian_noise(patch, self.mean, self.sigma)
            else:
                noisy_patch = add_salt_pepper_noise(patch, noise_ratio=0.02)
                
        else:
            patch = img

        # add Gaussian noise
        noisy_patch = add_gaussian_noise(patch, self.mean, self.sigma)

        # patch shape is [H,W], need convert to [C,H,W]
        noisy_patch = torch.from_numpy(np.expand_dims(noisy_patch, axis=0)).float()
        patch = torch.from_numpy(np.expand_dims(patch, axis=0)).float()

        return noisy_patch, patch, img_name  # img_name: list, len(img_name) = batchsize


if __name__ == '__main__':
    os.makedirs('./test_dataloader/', exist_ok=True)
    trainset = DenoisingDataset(image_dir='./BSDS500-master/BSDS500/data/images/train', phase='train')
    dataloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)

    noisy_patch, patch, img_name = next(iter(dataloader))
    print(f'noisy_patch shape: {noisy_patch.shape}')
    print(f'patch shape: {patch.shape}')

    noisy_patch = noisy_patch[0].numpy().transpose(1, 2, 0)
    patch = patch[0].numpy().transpose(1, 2, 0)

    cv2.imwrite('./test_dataloader/noisy_patch.png', np.uint8(noisy_patch * 255))
    cv2.imwrite('./test_dataloader/patch.png', np.uint8(patch * 255))