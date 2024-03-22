import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
from typing import Dict
import math
from spconv.pytorch.utils import PointToVoxel

DEBUG = False


class EtnaDataset(Dataset):
    """
    Dataset wrapper for Oxford laser scans dataset from PointNetVLAD project.
    """

    def __init__(self, dataset_path: str, query_filename: str,
                 transform=None, set_transform=None, image_transform=None,
                 use_cloud: bool = True, use_rgb: bool = True, cfg: dict = None):
        assert os.path.exists(
            dataset_path), 'Cannot access dataset path: {}'.format(dataset_path)
        self.dataset_path = dataset_path
        self.query_filepath = os.path.join(dataset_path, query_filename)
        assert os.path.exists(
            self.query_filepath), 'Cannot access query file: {}'.format(self.query_filepath)

        self.transform = transform
        self.set_transform = set_transform

        self.queries: Dict[int, TrainingTuple] = pickle.load(
            open(self.query_filepath, 'rb'))
        self.image_transform = image_transform
        self.n_points = 4096  # 4096    # pointclouds in the dataset are downsampled to 4096 points
        self.image_ext = '.png'
        self.use_cloud = use_cloud
        self.use_rgb = use_rgb

        self.point_cloud_range = cfg.model.point_cloud.range
        self.cfg = cfg
        print('{} queries in the dataset'.format(len(self)))

        self.queries = pickle.load(open(self.query_filepath, "rb"))
        self.dataset = pickle.load(
            open(os.path.join(dataset_path, "etna_complete_dataset2.pickle"), "rb"))
        self.images = use_rgb

        self.gen = PointToVoxel(vsize_xyz=cfg.model.point_cloud.voxel_size,
                        coors_range_xyz=cfg.model.point_cloud.range,
                        num_point_features=cfg.model.point_cloud.num_point_features,
                        max_num_voxels=cfg.model.point_cloud.max_num_voxels,
                        max_num_points_per_voxel=cfg.model.point_cloud.max_num_points_per_voxel)
        
    def __len__(self):
        return len(self.queries)

    def __getitem__(self, ndx):
        # Load point cloud and apply transform
        result = {'ndx': ndx}
        file = self.queries[ndx].timestamp
        value = self.dataset.loc[self.dataset['file'] == file]

        if self.use_cloud:
            # Load point cloud and apply transform
            query_pc= self.load_pc(value)

            if self.transform is not None:
                query_pc = self.transform(query_pc)
            result['cloud'] = query_pc

        if self.images:
            img_path = value["img_path"].values[0].split("/")[-3:]

            img_path = os.path.join(
                self.dataset_path, '/'.join(str(x) for x in img_path))
            img_path = img_path.replace("s3li_crater_inout", "s3li_zcrater_inout")
            img = Image.open(img_path).convert("L")
            # crop image to remobe black borders, 2% of image size
            img = img.crop((int(img.size[0]*0.02), int(img.size[1]*0.02), int(img.size[0]*0.98), int(img.size[1]*0.98)))
            if img is None:
                print("Image not found: {}".format(img_path))
            if self.image_transform is not None:
                img = self.image_transform(img)
            result['image'] = img

        return result

    def get_positives(self, ndx):
        return self.queries[ndx].positives

    def get_non_negatives(self, ndx):
        return self.queries[ndx].non_negatives

    def load_pc(self, file):
        # Load point cloud, does not apply any transform
        pc = np.array(file["point_cloud"].values[0])

        voxel_range = [-30, -10, 0, 30, 10, 40] # xmin, ymin, zmin, xmax, ymax, zmax
        pc = pc[(pc[:, 0] > voxel_range[0]) & (pc[:, 0] < voxel_range[3]) &
                (pc[:, 1] > voxel_range[1]) & (pc[:, 1] < voxel_range[4]) &
                (pc[:, 2] > voxel_range[2]) & (pc[:, 2] < voxel_range[5])]
        


        N = pc.shape[0]
        if N == 0:
            assert False, "Empty point cloud"
        subsample_idx = np.random.choice(N, self.n_points)
        pc = pc[subsample_idx, :]
        pc = pc[:, :3]

        pc = torch.tensor(pc, dtype=torch.float).contiguous()
        return pc



class TrainingTuple:
    # Tuple describing an element for training/validation
    def __init__(self, id: int, timestamp: int, positives: np.ndarray,
                 non_negatives: np.ndarray, position: np.ndarray):
        # id: element id (ids start from 0 and are consecutive numbers)
        # ts: timestamp
        # positives: sorted ndarray of positive elements id
        # negatives: sorted ndarray of elements id
        # position: x, y position in meters (northing, easting)
        assert position.shape == (2,)

        self.id = id
        self.timestamp = timestamp
        self.positives = positives
        self.non_negatives = non_negatives
        self.position = position
