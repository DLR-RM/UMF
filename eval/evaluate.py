import numpy as np
import pickle
import os
import sys
import argparse
import torch
import tqdm
from misc.utils import UMFParams
from models.model_factory import model_factory
from datasets.oxford import image4lidar
from datasets.augmentation import ValRGBTransform
import logging
from spconv.pytorch.utils import PointToVoxel
import dotsi
import yaml
from eval.utils import load_config, compute_and_log_stats

logging.basicConfig(filename='rerank_log.txt', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOAD_FEATURES = False


def load_config(config_file):
    with open(config_file, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return cfg


def evaluate(model, device, params, silent=True):
    assert len(params.eval_database_files) == len(params.eval_query_files)

    stats = {}
    stats_img = {}
    stats_pc = {}
    stats_combined = {}

    for database_file, query_file in zip(params.eval_database_files, params.eval_query_files):
        # Extract location name from query and database files
        location_name = database_file.split('_')[0]
        temp = query_file.split('_')[0]
        assert location_name == temp, 'Database location: {} does not match query location: {}'.format(database_file,
                                                                                                       query_file)

        p = os.path.join(params.dataset_folder, database_file)
        with open(p, 'rb') as f:
            database_sets = pickle.load(f)

        p = os.path.join(params.dataset_folder, query_file)
        with open(p, 'rb') as f:
            query_sets = pickle.load(f)

        temp, temp_img, temp_pc, temp_combined = evaluate_dataset(model, device, params, database_sets, query_sets, silent=silent)
        stats[location_name] = temp
        stats_img[location_name] = temp_img
        stats_pc[location_name] = temp_pc
        stats_combined[location_name] = temp_combined

    return stats, stats_img, stats_pc, stats_combined


def evaluate_dataset(model, device, params, database_sets, query_sets, silent=True):
    features_folder = 'features' 

    if not os.path.exists(features_folder):
        os.makedirs(features_folder)

    initialize_model(model, database_sets, features_folder, device, params)

    if not LOAD_FEATURES:
        # Process and save database sets
        for idx, set in enumerate(tqdm.tqdm(database_sets, disable=silent)):
            out = get_latent_vectors(model, set, device, params, dim_reduction=True, normalize=False)
            with open(os.path.join(features_folder, f'db_set_{idx}.pkl'), 'wb') as f:
                pickle.dump(out, f)

        # Process and save query sets
        for idx, set in enumerate(tqdm.tqdm(query_sets, disable=silent)):
            out = get_latent_vectors(model, set, device, params, dim_reduction=True, normalize=False)
            with open(os.path.join(features_folder, f'query_set_{idx}.pkl'), 'wb') as f:
                pickle.dump(out, f)

    stats = []
    mode = params.model_params.params['mode']

    modalities = ['base']
    if mode != 'fusion':
        modalities.extend(['img', 'pc', 'combined'])

    for modality in modalities:        
        stat = compute_and_log_stats(mode, modality, query_sets, database_sets, features_folder)
        stats.append(stat)
    
    return tuple(stats)

def load_data_item(file_name, params):
    # returns Nx3 matrix
    file_path = os.path.join(params.dataset_folder, file_name)

    result = {}
    if params.use_cloud:
        pc = np.fromfile(file_path, dtype=np.float64)
        # coords are within -1..1 range in each dimension
        assert pc.shape[0] == params.num_points * 3, "Error in point cloud shape: {}".format(file_path)
        pc = np.reshape(pc, (pc.shape[0] // 3, 3))
        pc = torch.tensor(pc, dtype=torch.float)
        result['cloud'] = pc

    if params.use_rgb:
        # Get the first closest image for each LiDAR scan
        assert os.path.exists(params.lidar2image_ndx_path), f"Cannot find lidar2image_ndx pickle: {params.lidar2image_ndx_path}"
        lidar2image_ndx = pickle.load(open(params.lidar2image_ndx_path, 'rb'))
        img = image4lidar(file_name, params.image_path, '.png', lidar2image_ndx, k=1)
        transform = ValRGBTransform()
        # Convert to tensor and normalize
        result['image'] = transform(img)
    return result


def process_cloud_data(clouds, gen, device):
    voxel_data = [gen(c) for c in clouds]
    batch_voxel_features, batch_voxel_coords, _ = zip(*voxel_data)
    # Prepare batch data
    batch_voxel_coords = [
        torch.cat((
            torch.full((coords.shape[0], 1), batch_id, dtype=torch.int32, device=device),
            coords.to(device)), 
            dim=1)
        for batch_id, coords in enumerate(batch_voxel_coords)
    ]
    return torch.cat(batch_voxel_coords, dim=0), torch.cat(batch_voxel_features, dim=0)

def process_data_item(item, device, params, gen):
    x = load_data_item(item["query"], params)
    batch = {}
    if params.use_cloud:
        clouds = [x['cloud'].to(device)]
        batch_voxel_coords, batch_voxel = process_cloud_data(clouds, gen, device)
        feats_batch = torch.ones((batch_voxel.shape[0], 1), dtype=torch.float32, device=device)
        batch['coordinates'] = batch_voxel_coords
        batch['voxel_features'] = feats_batch
    if params.use_rgb:
        batch['images'] = x['image'].unsqueeze(0).to(device)
    return batch


def get_latent_vectors(model, dataset, device, params, dim_reduction=True, normalize=False):
    cfg = load_config(params.model_params_path)
    cfg = dotsi.Dict(cfg)

    gen = PointToVoxel(vsize_xyz=cfg.model.point_cloud.voxel_size,
                        coors_range_xyz=cfg.model.point_cloud.range,
                        num_point_features=cfg.model.point_cloud.num_point_features,
                        max_num_voxels=cfg.model.point_cloud.max_num_voxels,
                        max_num_points_per_voxel=cfg.model.point_cloud.max_num_points_per_voxel, 
                        device=device)

    model.eval()
    embeddings_l = []

    feature_lists = {
        'img_fine_feat': [], 'img_attns': [], 'pc_fine_feat': [], 'pc_attns': [],
        'img_super_feat': [], 'img_strengths': [], 'pc_super_feat': [], 'pc_strengths': []
    }

    for elem_ndx in dataset:
        if not dim_reduction and np.random.rand() > 0.1:
            continue

        batch = process_data_item(dataset[elem_ndx], device, params, gen)
        with torch.no_grad():
            outputs = model(batch, dim_reduction=dim_reduction, normalize=normalize)
            if params.normalize_embeddings:
                outputs['embedding'] = torch.nn.functional.normalize(outputs['embedding'], p=2, dim=1)
        embedding = outputs['embedding'].detach().cpu().numpy()[0]
        embeddings_l.append(embedding)

        for key in feature_lists.keys():
            if key in outputs:
                feature_lists[key].append(outputs[key].cpu())

    out = {'embedding': torch.tensor(np.array(embeddings_l), dtype=torch.float32)}
    for key, value in feature_lists.items():
        if value:
            out[key] = torch.stack(value, dim=0) if 'feat' in key else torch.cat(value, dim=0)

    return out

def print_eval_stats(stats):
    for database_name in stats:
        print('Dataset: {}'.format(database_name))
        t = 'Avg. top 1% recall: {:.2f}   Avg. similarity: {:.4f}   Avg. recall @N:'
        print(t.format(stats[database_name]['ave_one_percent_recall'], 
                       stats[database_name]['average_similarity']))
        print(stats[database_name]['ave_recall'])


def initialize_model(model, train_set, features_folder, device, params):
    model.eval()
    mode = params.model_params.params['mode']
    des_img_train = []
    des_pc_train = []
    train_set = train_set[:]

    if not LOAD_FEATURES:
        for set in train_set:
            out = get_latent_vectors(model, set, device, params, dim_reduction=False, normalize=False)

            if 'img_fine_feat' in out:
                des_img_train.append(out['img_fine_feat'].detach().cpu())
                des_pc_train.append(out['pc_fine_feat'].detach().cpu())

            elif 'img_super_feat' in out:
                des_img_train.append(out['img_super_feat'].detach().cpu())
                des_pc_train.append(out['pc_super_feat'].detach().cpu())

        des_img_train = torch.cat(des_img_train, dim=0)
        des_pc_train = torch.cat(des_pc_train, dim=0)

        if mode == 'ransac':
            pc_feat_dim = des_pc_train.shape[2]
            img_feat_dim = des_img_train.shape[2]
        elif mode == 'superfeatures':
            pc_feat_dim = des_pc_train.shape[2]
            img_feat_dim = des_img_train.shape[2]
        
        max_num_samples = 50000
        if des_img_train.shape[0] > max_num_samples:
            indices_img = np.random.choice(des_img_train.shape[0], max_num_samples, replace=False)
            des_img_train_sampled = des_img_train[indices_img].reshape(-1, img_feat_dim)
        else:
            des_img_train_sampled = des_img_train.reshape(-1, img_feat_dim)
        
        
        if des_pc_train.shape[0] > max_num_samples:
            indices_pc = np.random.choice(des_pc_train.shape[0], max_num_samples, replace=False)
            des_pc_train_sampled = des_pc_train[indices_pc].reshape(-1, pc_feat_dim)
        else:
            des_pc_train_sampled = des_pc_train.reshape(-1, pc_feat_dim)

        if mode == 'ransac':
            mi, Pi = model.lt.reduction_layer.initialize_pca_whitening(des_img_train_sampled)
            mp, Pp = model.lt3d.reduction_layer.initialize_pca_whitening(des_pc_train_sampled)
        elif mode == 'superfeatures':
            mi, Pi = model.lit.reduction_layer.initialize_pca_whitening(des_img_train_sampled)
            mp, Pp = model.lit3d.reduction_layer.initialize_pca_whitening(des_pc_train_sampled)

        del des_img_train, des_pc_train, des_img_train_sampled, des_pc_train_sampled
        
        with open(f"{features_folder}/initialization.pkl", 'wb') as f:
            pickle.dump({'mi': mi, 'Pi': Pi, 'mp': mp, 'Pp': Pp}, f)
    
    # load initialization values
    with open(f"{features_folder}/initialization.pkl", 'rb') as f:
        init = pickle.load(f)
    mi, Pi, mp, Pp = init['mi'], init['Pi'], init['mp'], init['Pp']

    if hasattr(model, 'lit'):
        print('-> Loading initialization values for lit and lit3d')
        model.lit.reduction_layer.load_initialization(mi, Pi)
        model.lit3d.reduction_layer.load_initialization(mp, Pp)
    elif hasattr(model, 'lt'):
        print('-> Loading initialization values for lt and lt3d')
        model.lt.reduction_layer.load_initialization(mi, Pi)
        model.lt3d.reduction_layer.load_initialization(mp, Pp)
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model on RobotCar dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model-specific configuration file')
    parser.add_argument('--weights', type=str, required=False, help='Trained model weights')

    args = parser.parse_args()
    print('Config path: {}'.format(args.config))
    print('Model config path: {}'.format(args.model_config))
    if args.weights is None:
        sys.exit('Please provide a path to the trained model weights')

    params = UMFParams(args.config, args.model_config)
    params.print()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device: {}'.format(device))
    model = model_factory(params)
    if args.weights is not None:
        assert os.path.exists(args.weights), 'Cannot open network weights: {}'.format(args.weights)
        print('Loading weights: {}'.format(args.weights))

        pretrained_state_dict = torch.load(args.weights, map_location='cpu')
        dim_reduction_prefix = '.reduction_layer.'
        filtered_state_dict = {k: v for k, v in pretrained_state_dict.items() if dim_reduction_prefix not in k}
        model.load_state_dict(filtered_state_dict, strict=False)

    model.to(device)
    
    
    stats, stats_img, stats_pc, stats_combined = evaluate(model, device, params, silent=False)
    print("----- Without reranking -----")

    print_eval_stats(stats)

    print("----- With Img reranking -----")
    print_eval_stats(stats_img)


    print("----- With PC reranking -----")
    print_eval_stats(stats_pc)

    print("----- With Combined reranking -----")
    print_eval_stats(stats_combined)



