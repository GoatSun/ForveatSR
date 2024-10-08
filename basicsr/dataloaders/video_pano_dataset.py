import time

import cv2
import torch
import numpy as np

from torchvision.transforms import ToTensor
from torch.utils import data as data

from basicsr.utils.registry import DATASET_REGISTRY

from .crop_panorama import crop_panorama_image


@DATASET_REGISTRY.register()
class VideoPanoTrainData(data.Dataset):
    def __init__(self, opt):
        super(VideoPanoTrainData, self).__init__()
        self.video_gt = cv2.VideoCapture(opt['dataroot_gt'])
        self.video_lq = cv2.VideoCapture(opt['dataroot_lq'])
        self.count = self.video_gt.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = round(self.video_gt.get(cv2.CAP_PROP_FPS))
        self.time = self.count // self.fps
        self.fov = opt['fov']
        self.num_theta = 360 / self.fov
        self.num_phi = 180 / self.fov
        self.size_gt = opt['size_gt']
        self.size_lq = opt['size_lq']

    def __getitem__(self, idx):
        np.random.seed(idx)
        time_num = np.random.randint(self.time)
        frame_num = int(time_num * self.fps) + np.random.randint(self.fps)
        theta = np.random.randint(0, 360)
        phi = np.random.randint(-90, 90)
        self.video_gt.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        success, frame_gt = self.video_gt.read()
        self.video_lq.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        success, frame_ds = self.video_lq.read()

        image_gt = crop_panorama_image(frame_gt, theta, phi, size=self.size_gt, fov=self.fov)
        image_lq = crop_panorama_image(frame_ds, theta, phi, size=self.size_lq, fov=self.fov)
        image_gt = cv2.cvtColor(image_gt, cv2.COLOR_RGB2BGR)
        image_lq = cv2.cvtColor(image_lq, cv2.COLOR_RGB2BGR)
        filename = '%06d_%03d_%03d' % (frame_num, theta, phi)
        # lq = ToTensor()(image_lq.copy())
        # gt = ToTensor()(image_gt.copy())
        lq = torch.from_numpy(image_lq.transpose(2, 0, 1)).float()
        gt = torch.from_numpy(image_gt.transpose(2, 0, 1)).float()
        return {'gt': gt, 'lq': lq, 'name': filename}

    def __len__(self):
        return int(self.time*self.num_theta*self.num_phi)


@DATASET_REGISTRY.register()
class VideoPanoTestData(data.Dataset):
    def __init__(self, opt):
        super(VideoPanoTestData, self).__init__()
        self.batch = 8 
        self.video_gt = cv2.VideoCapture(opt['dataroot_gt'])
        self.video_lq = cv2.VideoCapture(opt['dataroot_lq'])
        self.count = self.video_gt.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = round(self.video_gt.get(cv2.CAP_PROP_FPS))
        self.time = self.count // self.fps
        self.fov = opt['fov']
        self.size_gt = opt['size_gt']
        self.size_lp = opt['size_lq']
        self.theta = [242, 121, 168, 121, 169, 251, 287, 138]  # np.random.randint(0, 360)
        self.phi = [5, 72, 43, 32, 2, 7, -15, 7]  # np.random.randint(-90, 90)

        self.val_frames_id, self.val_frames_lq, self.val_frames_gt = [], [], []
        np.random.seed(1)
        for i in np.random.randint(int(self.count), size=self.batch):
            self.val_frames_id.append(i)
            # save gt
            self.video_gt.set(cv2.CAP_PROP_POS_FRAMES, i)
            success, frame = self.video_gt.read()
            if success:
                self.val_frames_gt.append(frame)
            # save lq
            self.video_lq.set(cv2.CAP_PROP_POS_FRAMES, i)
            success, frame = self.video_lq.read()
            if success:
                self.val_frames_lq.append(frame)

    def __getitem__(self, index):
        frame_num = self.val_frames_id[index-1]
        frame_gt = self.val_frames_gt[index-1]
        frame_lq = self.val_frames_lq[index-1]
        theta = self.theta[(index-1) % len(self.theta)]  # np.random.randint(0, 360)
        phi = self.phi[(index-1) % len(self.phi)]  # np.random.randint(-90, 90)
        image_gt = crop_panorama_image(frame_gt, theta, phi, size=self.size_gt, fov=self.fov)
        image_lq = crop_panorama_image(frame_lq, theta, phi, size=self.size_lp, fov=self.fov)
        image_gt = cv2.cvtColor(image_gt, cv2.COLOR_RGB2BGR)
        image_lq = cv2.cvtColor(image_lq, cv2.COLOR_RGB2BGR)
        filename = '%06d_%03d_%03d' % (frame_num, theta, phi)
        # lq = ToTensor()(image_lq.copy())
        # gt = ToTensor()(image_gt.copy())
        lq = torch.from_numpy(image_lq.transpose(2, 0, 1)).float()
        gt = torch.from_numpy(image_gt.transpose(2, 0, 1)).float()
        return {'gt': gt, 'lq': lq, 'name': filename}

    def __len__(self):
        return len(self.val_frames_id)
