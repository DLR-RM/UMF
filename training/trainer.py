import os
from datetime import datetime
import numpy as np
import torch
import pickle
import tqdm
import pathlib

from torch.utils.tensorboard import SummaryWriter
from misc.lr_scheduler import LinearWarmupCosineAnnealingLR

from misc.utils import UMFParams, get_datetime
from models.loss import make_loss
from models.model_factory import model_factory
from models.UMF.UMFnet import UMFnet
import os
from matplotlib import pyplot as plt
import wandb


VERBOSE = False


def print_stats(stats, phase):
    if 'num_triplets' in stats:
        # For triplet loss
        s = '{} - Loss (mean/total): {:.4f} / {:.4f}    Avg. embedding norm: {:.4f}   Triplets per batch (all/non-zero): {:.1f}/{:.1f}'
        print(s.format(phase, stats['loss'], stats['total_loss'], stats['avg_embedding_norm'], stats['num_triplets'],
                       stats['num_non_zero_triplets']))
    elif 'num_pos' in stats:
        s = '{} - Mean loss: {:.6f}    Avg. embedding norm: {:.4f}   #positives/negatives: {:.1f}/{:.1f}'
        print(s.format(phase, stats['loss'], stats['avg_embedding_norm'], stats['num_pos'], stats['num_neg']))

    s = ''
    l = []
    if 'mean_pos_pair_dist' in stats:
        s += 'Pos dist (min/mean/max): {:.4f}/{:.4f}/{:.4f}   Neg dist (min/mean/max): {:.4f}/{:.4f}/{:.4f}'
        l += [stats['min_pos_pair_dist'], stats['mean_pos_pair_dist'], stats['max_pos_pair_dist'],
              stats['min_neg_pair_dist'], stats['mean_neg_pair_dist'], stats['max_neg_pair_dist']]
    if 'pos_loss' in stats:
        if len(s) > 0:
            s += '   '
        s += 'Pos loss: {:.4f}  Neg loss: {:.4f}'
        l += [stats['pos_loss'], stats['neg_loss']]
    if len(l) > 0:
        print(s.format(*l))

    if 'final_loss' in stats:
        # Multi loss
        s1 = '{} - Loss (total/final'.format(phase)
        s2 = '{:.4f} / {:.4f}'.format(stats['loss'], stats['final_loss'])
        s3 = 'Active triplets (final '
        s4 = '{:.1f}'.format(stats['final_num_non_zero_triplets'])
        if 'cloud_loss' in stats:
            s1 += '/cloud'
            s2 += '/ {:.4f}'.format(stats['cloud_loss'])
            s3 += '/cloud'
            s4 += '/ {:.1f}'.format(stats['cloud_num_non_zero_triplets'],)
        if 'image_loss' in stats:
            s1 += '/image'
            s2 += '/ {:.4f}'.format(stats['image_loss'])
            s3 += '/image'
            s4 += '/ {:.1f}'.format(stats['image_num_non_zero_triplets'],)

        s1 += '): '
        s3 += '): '
        print(s1 + s2)
        print(s3 + s4)


def tensors_to_numbers(stats):
    stats = {e: stats[e].item() if torch.is_tensor(stats[e]) else stats[e] for e in stats}
    return stats


def split_batch_pre_tensor(batch, minibatch_size):
    """Split a batch into minibatches before converting to tensors."""
    minibatches = []
    # Determine the size of the batch based on one of the elements
    batch_size = len(next(iter(batch.values())))
    for start_idx in range(0, batch_size, minibatch_size):
        end_idx = min(start_idx + minibatch_size, batch_size)  # Handle the case of the last minibatch being smaller
        minibatch = {}
        for key, value in batch.items():
            if key not in ['coordinates', 'voxel_features']:
                minibatch[key] = value[start_idx:end_idx]

            if key == 'coordinates':
                coords = value.cpu()
                indices = torch.where((coords[:, 0] >= start_idx) & (coords[:, 0] < end_idx))
                coords =  coords[indices[0], :]
                coords[:, 0] = coords[:, 0] - start_idx 
                coords[:, 0] = coords[:, 0].to(torch.int32)
                minibatch['coordinates'] = coords
                feats_batch = torch.ones((coords.shape[0], 1), dtype=torch.float32)
                minibatch['voxel_features'] = feats_batch

        minibatches.append(minibatch)
    return minibatches


def train_step(batch, model, phase, device, optimizer, scheduler, loss_fn, params):
    assert phase in ['train', 'val']
    if phase == 'train':
        model.train()
    else:
        model.eval()
    positives_mask = batch["positives_mask"]
    negatives_mask = batch["negatives_mask"]

    # Move batch to device
    batch = {e: torch.from_numpy(np.array(batch[e])).to(device) for e in batch}
    with torch.cuda.amp.autocast(enabled=phase == 'train'):
        with torch.set_grad_enabled(phase == 'train'):
            y = model(batch)
            loss, stats, _ = loss_fn(y, positives_mask, negatives_mask)

            stats = tensors_to_numbers(stats)
            stats['loss'] = loss.item()

            if phase == 'train':
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
                optimizer.step()
    if params.scheduler in ['OneCycleLR', 'WarmupCosineSchedule']:
        scheduler.step()
    optimizer.zero_grad()
    return stats, loss



def multistaged_training_step(batch, model, phase, device, optimizer, loss_fn, mode, dataset):
    # Training step using multistaged backpropagation algorithm as per:
    # "Learning with Average Precision: Training Image Retrieval with a Listwise Loss"
    # This method will break when the model contains Dropout, as the same mini-batch will produce different embeddings.
    # Make sure mini-batches in step 1 and step 3 are the same (so that BatchNorm produces the same results)
    # See some exemplary implementation here: https://gist.github.com/ByungSun12/ad964a08eba6a7d103dab8588c9a3774
    assert phase in ['train', 'val']
    minibatch_size =  20

    positives_mask = batch['positives_mask'].to(device)
    negatives_mask = batch['negatives_mask'].to(device)
                
    if phase == 'train':
        model.train()
    else:
        model.eval()

    # Stage 1 - calculate descriptors of each batch element (with gradient turned off)
    # In training phase network is in the train mode to update BatchNorm stats
    img_attn_l = []
    cloud_attn_l = []
    embeddings = []
    cloud_embeddings, image_embeddings = None, None
    embeddings_list, cloud_embeddings_list, image_embeddings_list = [], [], []
    with torch.cuda.amp.autocast(enabled=phase == 'train'):
        with torch.set_grad_enabled(False):
            minibatches = split_batch_pre_tensor(batch, minibatch_size)
            for minibatch in minibatches:
                minibatch = {e: minibatch[e].to(device) for e in minibatch}

                y = model(minibatch)
                embeddings_list.append(y['embedding'])
                if mode == 'superfeatures':
                    cloud_embeddings_list.append(y['pc_local_feat'])
                    image_embeddings_list.append(y['img_local_feat'])   
                    img_attn_l.append(y['img_attn'])
                    cloud_attn_l.append(y['pc_attn'])        
                elif mode == 'ransac':
                    cloud_embeddings_list.append(y['cloud_embedding'])
                    image_embeddings_list.append(y['image_embedding'])

    # Stage 2 - compute gradient of the loss w.r.t embeddings
    embeddings = torch.cat(embeddings_list, dim=0)
    if mode == 'superfeatures':
        cloud_embeddings = torch.cat(cloud_embeddings_list, dim=0)
        image_embeddings = torch.cat(image_embeddings_list, dim=0)   
        pc_attn = torch.cat(cloud_attn_l, dim=0)
        img_attn = torch.cat(img_attn_l, dim=0)
    elif mode == 'ransac':
        cloud_embeddings = torch.cat(cloud_embeddings_list, dim=0)
        image_embeddings = torch.cat(image_embeddings_list, dim=0)

    with torch.cuda.amp.autocast(enabled=phase == 'train'):
        with torch.set_grad_enabled(phase == 'train'):
            if phase == 'train':
                embeddings.requires_grad_(True)
                if mode == 'superfeatures':
                    cloud_embeddings.requires_grad_(True)
                    image_embeddings.requires_grad_(True)
                    pc_attn.requires_grad_(True)
                    img_attn.requires_grad_(True)
                elif mode == 'ransac':

                    cloud_embeddings.requires_grad_(True)
                    image_embeddings.requires_grad_(True)
                    
            out = {'embedding': embeddings}

            if mode == 'superfeatures' or mode == 'ransac':
                out.update({'cloud_embedding': cloud_embeddings, 'image_embedding': image_embeddings})
            
            if mode == 'superfeatures':
                out['pc_attn'] = pc_attn
                out['img_attn'] = img_attn
            loss, stats, _ = loss_fn(out, positives_mask=positives_mask, negatives_mask=negatives_mask)
            stats = tensors_to_numbers(stats)
            stats['loss'] = loss.item()
            if phase == 'train':
                loss.backward()
                embeddings_grad = embeddings.grad
                if mode == 'superfeatures':
                    cloud_embeddings_grad = cloud_embeddings.grad
                    image_embeddings_grad = image_embeddings.grad  
                    pc_attn_grad = pc_attn.grad
                    img_attn_grad = img_attn.grad            
                elif mode == 'ransac':
                    cloud_embeddings_grad = cloud_embeddings.grad
                    image_embeddings_grad = image_embeddings.grad  

    # Delete intermediary values
    del embeddings_list, cloud_embeddings_list, image_embeddings_list, embeddings, cloud_embeddings, image_embeddings, y, loss
    # Stage 3 - recompute descriptors with gradient enabled and compute the gradient of the loss w.r.t.
    # network parameters using cached gradient of the loss w.r.t embeddings
    if phase == 'train':
        optimizer.zero_grad()
        i = 0
        with torch.cuda.amp.autocast(enabled=phase == 'train'):
            with torch.set_grad_enabled(True):
                minibatches = split_batch_pre_tensor(batch, minibatch_size)
                for minibatch in minibatches:
                    minibatch = {e: minibatch[e].to(device) for e in minibatch}

                    y = model(minibatch)
                    embeddings = y['embedding']
                    minibatch_size = len(embeddings)
                    # Compute gradients of network params w.r.t. the loss using the chain rule (using the
                    # gradient of the loss w.r.t. embeddings stored in embeddings_grad)
                    # By default gradients are accumulated
                    # For all but the last minibatch, retain the graph
                    if mode == 'superfeatures':
                        grads = torch.cat([embeddings_grad[i: i+minibatch_size],
                                            cloud_embeddings_grad[i: i+minibatch_size], 
                                            image_embeddings_grad[i: i+minibatch_size],
                                            pc_attn_grad[i: i+minibatch_size],
                                            img_attn_grad[i: i+minibatch_size]], dim=1)

                        outputs = torch.cat([y['embedding'], y['pc_local_feat'], 
                                        y['img_local_feat'], y['pc_attns'], y['img_attns']], dim=1)

                    elif mode == 'ransac':
                        grads = torch.cat([embeddings_grad[i: i+minibatch_size], cloud_embeddings_grad[i: i+minibatch_size], image_embeddings_grad[i: i+minibatch_size]], dim=1)
                        outputs = torch.cat([y['embedding'], y['cloud_embedding'], y['image_embedding']], dim=1)

                    else:
                        grads = embeddings_grad[i: i+minibatch_size]
                        outputs = y['embedding']

                        outputs = y['embedding']
                    outputs.backward(gradient=grads)
                    i += minibatch_size
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
                optimizer.step()
    return stats


def do_train(dataloaders, params: UMFParams, debug=False):
    # Create model class
    s = get_datetime()
    model = model_factory(params)

    # Move the model to the proper device before configuring the optimizer
    if torch.cuda.is_available():
        device = "cuda"
        model.to(device)
    else:
        device = "cpu"

    print('Model device: {}'.format(device))
    
    if params.load_weights is not None:
        assert os.path.exists(params.load_weights), 'Cannot open network weights: {}'.format(params.load_weights)
        print('Loading weights: {}'.format(params.load_weights))
        model.load_state_dict(torch.load(params.load_weights, map_location=device))

    model_name = 'model_' + params.model_params.model + '_' + s
    mode = params.model_params.params.get('mode', 'ransac')
    
    print('Model name: {}'.format(model_name))
    weights_path = create_weights_folder()

    model_pathname = os.path.join(weights_path, model_name)
    if hasattr(model, 'print_info'):
        model.print_info()
    else:
        n_params = sum([param.nelement() for param in model.parameters()])
        print('Number of model parameters: {}'.format(n_params))

    loss_fn = make_loss(params)
    params_l = []
    if isinstance(model, UMFnet):
        # Different LR for image feature extractor (pretrained ResNet)
        if model.image_fe is not None:
            lower_lr = params.image_lr / 10
            num_top_layers = 0
            # Set lower lr for top layers of resnet
            for i, (name, param) in enumerate(model.image_fe.named_parameters()):
                if i >= len(list(model.image_fe.parameters())) - num_top_layers:  
                    params_l.append({'params': param, 'lr': lower_lr})  
                else:
                    params_l.append({'params': param, 'lr': params.image_lr})

        if model.cloud_fe is not None:
            params_l.append({'params': model.cloud_fe.parameters(), 'lr': params.lr})
        if model.final_block is not None:
            params_l.append({'params': model.fusion_encoder.parameters(), 'lr': params.lr})
    else:
        # All parameters use the same lr
        params_l.append({'params': model.parameters(), 'lr': params.lr})

    # Training elements
    if params.optimizer == 'Adam':
        if params.weight_decay is None or params.weight_decay == 0:
            optimizer = torch.optim.Adam(params_l)
        else:
            optimizer = torch.optim.Adam(params_l, weight_decay=params.weight_decay)
    elif params.optimizer == 'AdamW':
        if params.weight_decay is None or params.weight_decay == 0:
            optimizer = torch.optim.AdamW(params_l)
        else:
            optimizer = torch.optim.AdamW(params_l, weight_decay=params.weight_decay, eps=1e-4)
    elif params.optimizer == 'SGD':
        # SGD with momentum (default momentum = 0.9)
        if params.weight_decay is None or params.weight_decay == 0:
            optimizer = torch.optim.SGD(params_l, momentum=0.9 )
        else:
            optimizer = torch.optim.SGD(params_l, weight_decay=params.weight_decay, momentum=0.9)
    else:
        raise NotImplementedError('Unsupported optimizer: {}'.format(params.optimizer))
    
    
    if params.scheduler is None or params.optimizer == 'auto':
        scheduler = None
    else:
        if params.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.epochs+1,
                                                                   eta_min=params.min_lr)

        elif params.scheduler == 'LinearWarmupCosineAnnealingLR':
            scheduler = LinearWarmupCosineAnnealingLR(optimizer, 
            warmup_epochs=params.warmup_epochs,
            max_epochs=params.epochs)                                         

        elif params.scheduler == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=params.lr,
                                                            total_steps=params.epochs*len(dataloaders['train']))
 
        elif params.scheduler == 'ExpotentialLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        elif params.scheduler == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, params.scheduler_milestones, gamma=0.1)
        elif params.scheduler == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=6, cooldown=2)

        else:
            raise NotImplementedError('Unsupported LR scheduler: {}'.format(params.scheduler))


    ###########################################################################
    # Initialize TensorBoard writer
    ###########################################################################
    now = datetime.now()
    logdir = os.path.join("../tf_logs", now.strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(logdir)


    is_validation_set = 'val' in dataloaders
    if is_validation_set:
        phases = ['train', 'val']
    else:
        phases = ['train']

    # Training statistics
    stats = {'train': [], 'val': [], 'eval': []}
    best_val_loss = 1e10
    best_val_triplets = 1e10


    for epoch in tqdm.tqdm(range(1, params.epochs + 1)):
        metrics = {'train': {}, 'val': {}}    # Metrics for wandb reporting

        first_iter = True
        for phase in phases:
            running_stats = [] 
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_stats = []  # running stats for the current epoch
            for batch_idx, (batch) in enumerate(dataloaders[phase]):
                # Prepare data and masks for multi-staged training
                positives_mask = batch["positives_mask"]
                negatives_mask = batch["negatives_mask"]
                dataset = dataloaders[phase].dataset

                # Skip batches without positives or negatives
                if torch.sum(positives_mask) == 0 or torch.sum(negatives_mask) == 0:
                    print('WARNING: Skipping batch without positive or negative examples')
                    continue

                if params.train_step == 'multistaged':
                    # Multi-staged training step
                    batch_stats = multistaged_training_step(
                        batch, model, phase, device, optimizer, loss_fn, mode, dataset
                    )
                else:
                    # Standard training step
                    batch_stats, loss = train_step(
                        batch, model, phase, device, optimizer, scheduler, loss_fn, params
                    )

                running_stats.append(batch_stats)

                if params.debug and batch_idx > 10:
                    break

                if params.scheduler in ['OneCycleLR']:
                    scheduler.step()

            epoch_stats = {}
            for key in running_stats[0]:
                temp = [e[key] for e in running_stats]
                if type(temp[0]) is dict:
                    epoch_stats[key] = {key: np.mean([e[key] for e in temp]) for key in temp[0]}
                elif type(temp[0]) is np.ndarray:
                    # Mean value per vector element
                    epoch_stats[key] = np.mean(np.stack(temp), axis=0)
                else:
                    epoch_stats[key] = np.mean(temp)

            stats[phase].append(epoch_stats)
            print_stats(epoch_stats, phase)

            # Log metrics for wandb
            for key in epoch_stats:
                if type(epoch_stats[key]) is dict:
                    metrics[phase].update(epoch_stats[key])
                else:
                    metrics[phase][key] = epoch_stats[key]

        if wandb.run is not None:
            wandb.log(metrics)
             
        # ******* EPOCH END *******
        if params.scheduler not in ['OneCycleLR']:
            if params.scheduler == 'ReduceLROnPlateau':
                scheduler.step(epoch_stats['loss'])
            else:
                scheduler.step()

        model.zero_grad()
        loss_metrics = {'train': stats['train'][-1]['loss']}
        if 'val' in phases:
            loss_metrics['val'] = stats['val'][-1]['loss']

            print('Current lr: {:.7f}'.format(optimizer.param_groups[0]['lr']))
            if wandb.run is not None:
                wandb.log({"lr": optimizer.param_groups[0]['lr']})
            epoch_val_stats = stats['val'][-1]
            val_num_non_zero_triplets = epoch_val_stats.get("final_num_non_zero_triplets", 0) 

            if val_num_non_zero_triplets < best_val_triplets:
                if not params.debug:
                    torch.save(model.state_dict(), model_pathname + '_best_triplets.pth')

            if loss_metrics['val'] < best_val_loss:
                best_val_loss = loss_metrics['val']
                #best_model_wts = copy.deepcopy(model.state_dict())
                print('New best model found, loss: {:.6f}'.format(best_val_loss))
                print('Saving model to {}'.format(model_pathname))
                # Save model
                best_model_path = model_pathname + '_best.pth'
                torch.save(model.state_dict(), best_model_path)
                if not params.debug:
                    torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'best_model.pth'))


        writer.add_scalars('Loss', loss_metrics, epoch)

        if 'num_triplets' in stats['train'][-1]:
            nz_metrics = {'train': stats['train'][-1]['num_non_zero_triplets']}

            if 'val' in phases:
                nz_metrics['val'] = stats['val'][-1]['num_non_zero_triplets']
            writer.add_scalars('Non-zero triplets', nz_metrics, epoch)

        elif 'num_pairs' in stats['train'][-1]:
            nz_metrics = {'train_pos': stats['train'][-1]['pos_pairs_above_threshold'],
                          'train_neg': stats['train'][-1]['neg_pairs_above_threshold']}
            if 'val' in phases:
                nz_metrics['val_pos'] = stats['val'][-1]['pos_pairs_above_threshold']
                nz_metrics['val_neg'] = stats['val'][-1]['neg_pairs_above_threshold']
            writer.add_scalars('Non-zero pairs', nz_metrics, epoch)

            
        if params.batch_expansion_th is not None:
            # Dynamic batch expansion of the training batch
            epoch_train_stats = stats['train'][-1]
            if 'num_non_zero_triplets' in epoch_train_stats:
                # Ratio of non-zero triplets
                rnz = epoch_train_stats['num_non_zero_triplets'] / epoch_train_stats['num_triplets']
                if rnz < params.batch_expansion_th:
                    dataloaders['train'].batch_sampler.expand_batch()
            elif 'final_num_non_zero_triplets' in epoch_train_stats:
                rnz = []
                rnz.append(epoch_train_stats['final_num_non_zero_triplets'] / epoch_train_stats['final_num_triplets'])
                if 'image_num_non_zero_triplets' in epoch_train_stats:
                    rnz.append(epoch_train_stats['image_num_non_zero_triplets'] / epoch_train_stats['image_num_triplets'])
                if 'cloud_num_non_zero_triplets' in epoch_train_stats:
                    rnz.append(epoch_train_stats['cloud_num_non_zero_triplets'] / epoch_train_stats['cloud_num_triplets'])
                rnz = max(rnz)
                if rnz < params.batch_expansion_th:
                    dataloaders['train'].batch_sampler.expand_batch()
            else:
                print('WARNING: Batch size expansion is enabled, but the loss function is not supported')
    print('')

    # Save final model weights
    final_model_path = model_pathname + '_final.pth'
    torch.save(model.state_dict(), final_model_path)
    print('Final model saved to: {}'.format(final_model_path))


def export_eval_stats(file_name, prefix, eval_stats):
    s = prefix
    ave_1p_recall_l = []
    ave_recall_l = []
    # Print results on the final model
    with open(file_name, "a") as f:
        for ds in ['etna', 'university', 'residential', 'business']:
            if ds not in eval_stats:
                continue
            ave_1p_recall = eval_stats[ds]['ave_one_percent_recall']
            ave_1p_recall_l.append(ave_1p_recall)
            ave_recall = eval_stats[ds]['ave_recall'][0]
            ave_recall_l.append(ave_recall)
            s += ", {:0.2f}, {:0.2f}".format(ave_1p_recall, ave_recall)

        mean_1p_recall = np.mean(ave_1p_recall_l)
        mean_recall = np.mean(ave_recall_l)
        s += ", {:0.2f}, {:0.2f}\n".format(mean_1p_recall, mean_recall)
        f.write(s)


def create_weights_folder():
    # Create a folder to save weights of trained models
    this_file_path = pathlib.Path(__file__).parent.absolute()
    temp, _ = os.path.split(this_file_path)
    weights_path = os.path.join(temp, 'weights')
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    assert os.path.exists(weights_path), 'Cannot create weights folder: {}'.format(weights_path)
    return weights_path
