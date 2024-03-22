# Author: Jacek Komorowski
# Warsaw University of Technology

import numpy as np
import torch
import dotsi
import yaml
from torch.utils.data import DataLoader
from datasets.oxford import OxfordDataset
from datasets.etna import EtnaDataset

from typing import List, Union

from datasets.augmentation import TrainTransform, TrainSetTransform, TrainRGBTransform, ValRGBTransform, TrainGreyTransform, ValGreyTransform
from datasets.samplers import BatchSampler
from misc.utils import UMFParams

def load_config(config_file):
    with open(config_file, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return cfg


def make_datasets(params: UMFParams, debug=False):
    # Create training and validation datasets
    datasets = {}
    train_transform = TrainTransform(params.aug_mode)
    train_set_transform = TrainSetTransform(params.aug_mode)
    cfg = load_config(params.model_params_path)
    cfg = dotsi.Dict(cfg)    

    if params.dataset == 'robotcar':
        image_train_transform = TrainRGBTransform(params.aug_mode)
        image_val_transform = ValRGBTransform()

        datasets['train'] = OxfordDataset(params.dataset_folder, params.train_file, image_path=params.image_path,
                                        lidar2image_ndx_path=params.lidar2image_ndx_path, transform=train_transform,
                                        set_transform=train_set_transform, image_transform=image_train_transform,
                                        use_cloud=params.use_cloud, cfg=cfg)
        val_transform = None
        if params.val_file is not None:
            datasets['val'] = OxfordDataset(params.dataset_folder, params.val_file, image_path=params.image_path,
                                            lidar2image_ndx_path=params.lidar2image_ndx_path, transform=val_transform,
                                            set_transform=None, image_transform=image_val_transform,
                                            use_cloud=params.use_cloud, cfg=cfg)

    elif params.dataset == 'etna':
        image_train_transform = TrainGreyTransform(params.aug_mode)
        image_val_transform = ValGreyTransform()

        datasets['train'] = EtnaDataset(params.dataset_folder, params.train_file, 
                                        transform=train_transform, set_transform=train_set_transform,
                                        image_transform=image_train_transform,
                                        use_cloud=params.use_cloud, use_rgb=params.use_rgb,  cfg=cfg)
        val_transform = None
        if params.val_file is not None:
            datasets['val'] = EtnaDataset(params.dataset_folder, params.val_file,
                                            transform=val_transform, set_transform=None,
                                            image_transform=image_val_transform,
                                            use_cloud=params.use_cloud, use_rgb=params.use_rgb,  cfg=cfg)
                                            
    return datasets



def gather_features_by_pc_voxel_id(seg_res_features: torch.Tensor, pc_voxel_id: torch.Tensor, invalid_value: Union[int, float] = 0):
    """This function is used to gather segmentation result to match origin pc.
    """
    if seg_res_features.device != pc_voxel_id.device:
        pc_voxel_id = pc_voxel_id.to(seg_res_features.device)
    res_feature_shape = (pc_voxel_id.shape[0], *seg_res_features.shape[1:])
    if invalid_value == 0:
        res = torch.zeros(res_feature_shape, dtype=seg_res_features.dtype, device=seg_res_features.device)
    else:
        res = torch.full(res_feature_shape, invalid_value, dtype=seg_res_features.dtype, device=seg_res_features.device)
    pc_voxel_id_valid = pc_voxel_id != -1
    pc_voxel_id_valid_ids = torch.nonzero(pc_voxel_id_valid).view(-1)
    seg_res_features_valid = seg_res_features[pc_voxel_id[pc_voxel_id_valid_ids]]
    res[pc_voxel_id_valid_ids] = seg_res_features_valid
    return res 

def make_collate_fn(dataset: OxfordDataset):
    # set_transform: the transform to be applied to all batch elements
    def collate_fn(data_list):
        # Constructs a batch object
        labels = [e['ndx'] for e in data_list]

        # Compute positives and negatives mask
        positives_mask = [[in_sorted_array(e, dataset.queries[label].positives) for e in labels] for label in labels]
        negatives_mask = [[not in_sorted_array(e, dataset.queries[label].non_negatives) for e in labels] for label in labels]
        positives_mask = torch.tensor(positives_mask)
        negatives_mask = torch.tensor(negatives_mask)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Returns (batch_size, n_points, 3) tensor and positives_mask and
        # negatives_mask which are batch_size x batch_size boolean tensors
        result = {'positives_mask': positives_mask, 'negatives_mask': negatives_mask}
        if 'cloud' in data_list[0]:
            clouds = [e['cloud'] for e in data_list]  
            clouds = torch.stack(clouds, dim=0)  
            partial_clouds = clouds
            if dataset.set_transform is not None:
                # Apply the same transformation on all dataset elements
                partial_clouds = dataset.set_transform(clouds)

        images = [e['image'] for e in data_list]
        result['images'] = torch.stack(images, dim=0)

        voxel_data = [dataset.gen(partial_clouds[batch_id]) for batch_id in range(len(data_list))]
        batch_voxel, batch_voxel_coords, batch_num_pts_in_voxels = zip(*voxel_data)
        batch_voxel_coords = list(batch_voxel_coords)


        for batch_id in range(len(batch_voxel_coords)):
            coordinates = batch_voxel_coords[batch_id]

            discrete_coords = torch.cat(
                (
                    torch.zeros(coordinates.shape[0], 1, dtype=torch.int32),
                    coordinates,
                ),
                1,
            )

            batch_voxel_coords[batch_id] = discrete_coords
            # Move batchids to the beginning
            batch_voxel_coords[batch_id][:, 0] = batch_id
        
        batch_voxel_coords = torch.cat(batch_voxel_coords, dim=0)


        num_points = torch.cat(batch_num_pts_in_voxels, 0)
        batch_voxel = torch.cat(batch_voxel, dim=0)
        feats_batch = torch.ones((batch_voxel.shape[0], 1), dtype=torch.float32, device=partial_clouds.device)

        result["coordinates"] = batch_voxel_coords
        result["voxel_features"] = feats_batch

        return result

    return collate_fn



def mean_vfe(voxel_features, voxel_num_points):
    points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
    normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
    points_mean = points_mean / normalizer
    voxel_features = points_mean.contiguous()
    return voxel_features


def make_dataloaders(params: UMFParams, debug=False):
    """
    Create training and validation dataloaders that return groups of k=2 similar elements
    :param train_params:
    :param model_params:
    :return:
    """
    datasets = make_datasets(params, debug=debug)

    dataloders = {}
    train_sampler = BatchSampler(datasets['train'], batch_size=params.batch_size,
                                 batch_size_limit=params.batch_size_limit,
                                 batch_expansion_rate=params.batch_expansion_rate)
    # Collate function collates items into a batch and applies a 'set transform' on the entire batch
    train_collate_fn = make_collate_fn(datasets['train'])
    dataloders['train'] = DataLoader(datasets['train'], batch_sampler=train_sampler, collate_fn=train_collate_fn,
                                     num_workers=params.num_workers, pin_memory=True, persistent_workers=True)

    if 'val' in datasets:
        val_sampler = BatchSampler(datasets['val'], batch_size=params.val_batch_size)
        # Collate function collates items into a batch and applies a 'set transform' on the entire batch
        # Currently validation dataset has empty set_transform function, but it may change in the future
        val_collate_fn = make_collate_fn(datasets['val'])
        dataloders['val'] = DataLoader(datasets['val'], batch_sampler=val_sampler, collate_fn=val_collate_fn,
                                       num_workers=params.num_workers, pin_memory=True, persistent_workers=True,
                                       shuffle=False)

    return dataloders


def in_sorted_array(e: int, array: np.ndarray) -> bool:
    pos = np.searchsorted(array, e)
    if pos == len(array) or pos == -1:
        return False
    else:
        return array[pos] == e
