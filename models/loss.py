import numpy as np
import torch
from pytorch_metric_learning import losses, miners, reducers
from pytorch_metric_learning.distances import LpDistance
from models.losses.super_feature_losses import DecorrelationAttentionLoss, SuperfeatureLoss
from models.losses.truncated_smoothap import TruncatedSmoothAP

def make_loss(params):
    if params.loss == 'BatchHardTripletMarginLoss':
        # BatchHard mining with triplet margin loss
        # Expects input: embeddings, positives_mask, negatives_mask
        loss_fn = BatchHardTripletLossWithMasks(params.margin, params.normalize_embeddings)
    elif params.loss == 'MultiBatchHardTripletMarginLoss':
        # BatchHard mining with triplet margin loss
        # Expects input: embeddings, positives_mask, negatives_mask
        loss_fn = MultiBatchHardTripletLossWithMasks(params.margin, params.normalize_embeddings, params.weights)

    elif params.loss == 'MultiBatchHardTripletLossWithMasksAugmented':
        loss_fn = MultiBatchHardTripletLossWithMasksAugmented(params)
    
    elif params.loss == 'MultiBatchHardTripletLossWithMasksAP':
        loss_fn = MultiBatchHardTripletLossWithMasksAP(params)
    else:
        print('Unknown loss: {}'.format(params.loss))
        raise NotImplementedError
    return loss_fn


class HardTripletMinerWithMasks:
    # Hard triplet miner
    def __init__(self, distance):
        self.distance = distance
        # Stats
        self.max_pos_pair_dist = None
        self.max_neg_pair_dist = None
        self.mean_pos_pair_dist = None
        self.mean_neg_pair_dist = None
        self.min_pos_pair_dist = None
        self.min_neg_pair_dist = None

    def __call__(self, embeddings, positives_mask, negatives_mask):
        assert embeddings.dim() == 2
        d_embeddings = embeddings.detach()
        with torch.no_grad():
            hard_triplets = self.mine(d_embeddings, positives_mask, negatives_mask)
        return hard_triplets

    def mine(self, embeddings, positives_mask, negatives_mask):
        # Based on pytorch-metric-learning implementation
        dist_mat = self.distance(embeddings.float())
        (hardest_positive_dist, hardest_positive_indices), a1p_keep = get_max_per_row(dist_mat, positives_mask)
        (hardest_negative_dist, hardest_negative_indices), a2n_keep = get_min_per_row(dist_mat, negatives_mask)
        a_keep_idx = torch.where(a1p_keep & a2n_keep)
        a = torch.arange(dist_mat.size(0)).to(hardest_positive_indices.device)[a_keep_idx]
        p = hardest_positive_indices[a_keep_idx]
        n1 = hardest_negative_indices[a_keep_idx]

        # second hardest negative
        # first remove the hardest negative
        dist_mat[a, n1] = float('inf')
        (second_hardest_negative_dist, second_hardest_negative_indices), a2n_keep = get_min_per_row(dist_mat, negatives_mask)
        n2 = second_hardest_negative_indices[a_keep_idx]

        # third hardest negative
        # first remove the second hardest negative
        dist_mat[a, n2] = float('inf')
        (third_hardest_negative_dist, third_hardest_negative_indices), a2n_keep = get_min_per_row(dist_mat, negatives_mask)
        n3 = third_hardest_negative_indices[a_keep_idx]   

        #n = torch.stack([n1, n2, n3], dim=1)   
        n = torch.stack([n1, n2, n3], dim=1)   

        self.max_pos_pair_dist = torch.max(hardest_positive_dist).item()
        self.max_neg_pair_dist = torch.max(hardest_negative_dist).item()
        self.mean_pos_pair_dist = torch.mean(hardest_positive_dist).item()
        self.mean_neg_pair_dist = torch.mean(hardest_negative_dist).item()
        self.min_pos_pair_dist = torch.min(hardest_positive_dist).item()
        self.min_neg_pair_dist = torch.min(hardest_negative_dist).item()
        return a, p, n


def get_max_per_row(mat, mask):
    non_zero_rows = torch.any(mask, dim=1)
    mat_masked = mat.clone()
    mat_masked[~mask] = 0
    return torch.max(mat_masked, dim=1), non_zero_rows


def get_min_per_row(mat, mask):
    non_inf_rows = torch.any(mask, dim=1)
    mat_masked = mat.clone()
    mat_masked[~mask] = float('inf')
    return torch.min(mat_masked, dim=1), non_inf_rows



class MultiBatchHardTripletLossWithMasks:
    def __init__(self, margin, normalize_embeddings, weights):
        assert len(weights) == 3
        self.weights = weights
        self.final_loss = BatchHardTripletLossWithMasksHelper(margin[0], normalize_embeddings)
        self.cloud_loss = BatchHardTripletLossWithMasksHelper(margin[1], normalize_embeddings)
        self.image_loss = BatchHardTripletLossWithMasksHelper(margin[2], normalize_embeddings)
        print('MultiBatchHardTripletLossWithMasks')
        print('Weights (final/cloud/image): {}'.format(weights))
        print('Margins (final/cloud/image): {}'.format(margin))

    def __call__(self, x, positives_mask, negatives_mask):
        # Loss on the final global descriptor
        final_loss, final_stats, final_hard_triplets = self.final_loss(x['embedding'], positives_mask, negatives_mask)
        final_stats = {'final_{}'.format(e): final_stats[e] for e in final_stats}

        loss = 0.

        stats = final_stats
        if self.weights[0] > 0.:
            loss = self.weights[0] * final_loss + loss

        # Loss on the cloud-based descriptor
        if 'cloud_embedding' in x:
            cloud_loss, cloud_stats, _ = self.cloud_loss(x['cloud_embedding'], positives_mask, negatives_mask)
            cloud_stats = {'cloud_{}'.format(e): cloud_stats[e] for e in cloud_stats}
            stats.update(cloud_stats)
            if self.weights[1] > 0.:
                loss = self.weights[1] * cloud_loss + loss

        # Loss on the image-based descriptor
        if 'image_embedding' in x:
            image_loss, image_stats, _ = self.image_loss(x['image_embedding'], positives_mask, negatives_mask)
            image_stats = {'image_{}'.format(e): image_stats[e] for e in image_stats}
            stats.update(image_stats)
            if self.weights[2] > 0.:
                loss = self.weights[2] * image_loss + loss

        stats['loss'] = loss.item()
        return loss, stats, None

class MultiBatchHardTripletLossWithMasksAP:
    def __init__(self, margin, normalize_embeddings, weights):
        assert len(weights) == 3
        self.weights = weights
        
        tau1 = 0.01
        positives_per_query = 4
        similarity = 'cosine'

        self.final_loss = TruncatedSmoothAP(tau1=tau1, similarity=similarity,
                                    positives_per_query=positives_per_query)
        self.cloud_loss = TruncatedSmoothAP(tau1=tau1, similarity=similarity,
                                    positives_per_query=positives_per_query)
        self.image_loss = TruncatedSmoothAP(tau1=tau1, similarity=similarity,
                                    positives_per_query=positives_per_query)
        
        print('MultiBatchHardTripletLossWithMasks')
        print('Tau1: {}'.format(tau1))
        print('Similarity: {}'.format(similarity))
        print('Positives per query: {}'.format(positives_per_query))

    def __call__(self, x, positives_mask, negatives_mask):
        # Loss on the final global descriptor
        final_loss, final_stats, final_hard_triplets = self.final_loss(x['embedding'], positives_mask, negatives_mask)
        final_stats = {'final_{}'.format(e): final_stats[e] for e in final_stats}

        loss = 0.

        stats = final_stats
        if self.weights[0] > 0.:
            loss = self.weights[0] * final_loss + loss

        # Loss on the cloud-based descriptor
        if 'cloud_embedding' in x:
            cloud_loss, cloud_stats, _ = self.cloud_loss(x['cloud_embedding'], positives_mask, negatives_mask)
            cloud_stats = {'cloud_{}'.format(e): cloud_stats[e] for e in cloud_stats}
            stats.update(cloud_stats)
            if self.weights[1] > 0.:
                loss = self.weights[1] * cloud_loss + loss

        # Loss on the image-based descriptor
        if 'image_embedding' in x:
            image_loss, image_stats, _ = self.image_loss(x['image_embedding'], positives_mask, negatives_mask)
            image_stats = {'image_{}'.format(e): image_stats[e] for e in image_stats}
            stats.update(image_stats)
            if self.weights[2] > 0.:
                loss = self.weights[2] * image_loss + loss

        stats['loss'] = loss.item()
        return loss, stats, None


class MultiBatchHardTripletLossWithMasksAugmented(MultiBatchHardTripletLossWithMasks):
    def __init__(self, params):
        self.normalize_embeddings = params.normalize_embeddings
        self.margin = [params.margin] * 3
        self.mode = params.cfg.model.mode 

        if self.mode.lower() == "ransac":
            self.margin[1] = params.cfg.model.local_feat_margin[0]
            self.margin[2] = params.cfg.model.local_feat_margin[1]

        self.weights = params.weights
        print('MultiBatchHardTripletLossWithMasksAugmented, model mode: ', self.mode)
        super().__init__( self.margin, self.normalize_embeddings, self.weights)

        if self.mode == "superfeatures":
            print('Superfeature loss')

            self.local_weights = params.cfg.model.local_feat_weights
            local_feat_margin = params.cfg.model.local_feat_margin
            self.local_attn_weights = params.cfg.model.local_attn_weights

            self.criterion_superfeatures = SuperfeatureLoss(margin=local_feat_margin[1], weight=self.local_weights[1]).to("cuda")
            self.criterion_superfeatures_pc = SuperfeatureLoss(margin=local_feat_margin[0], weight=self.local_weights[0]).to("cuda")    
            
            self.criterion_attns = DecorrelationAttentionLoss(weight= self.local_attn_weights[1]).to("cuda")
            self.criterion_attns_3D = DecorrelationAttentionLoss(weight= self.local_attn_weights[0]).to("cuda")

    def __call__(self, x, positives_mask, negatives_mask):
        stats = {}
        loss = 0.
        
        embeddings = x['embedding'].float()
        final_loss, final_stats, final_hard_triplets = self.final_loss(embeddings, positives_mask, negatives_mask)
        final_stats = {'final_{}'.format(e): final_stats[e] for e in final_stats}
        device = x['embedding'][0].device

        loss = 0.
        stats = final_stats
        if self.weights[0] > 0.:
            loss = self.weights[0] * final_loss + loss

        if self.mode == "fusion":
            return loss, stats, None
        
        # Cloud - Loss on the cloud-based descriptor
        cloud_stats = {}
        image_stats = {}
        if self.mode == "ransac":
            cloud_loss, cloud_stats, pc_hard_triplets = self.cloud_loss(x['cloud_embedding'], positives_mask, negatives_mask)

            if self.weights[1] > 0.:
                loss = self.weights[1] * cloud_loss + loss
                
        if self.mode == "superfeatures":
            q_triplet = final_hard_triplets[0]
            p_triplet = final_hard_triplets[1]
            n_triplet = final_hard_triplets[2]

            a_pc_feat = x['pc_super_feat'][q_triplet]
            p_pc_feat = x['pc_super_feat'][p_triplet]
            feat_dim = a_pc_feat.shape[-1]
            n_pc_feat = x['pc_super_feat'][n_triplet.reshape(-1)] 
            n_pc_feat = n_pc_feat.view(*n_triplet.shape, -1, feat_dim) 
            
            a_img_feat = x['img_super_feat'][q_triplet]
            p_img_feat = x['img_super_feat'][p_triplet]
            feat_dim = a_img_feat.shape[-1]
            n_img_feat = x['img_super_feat'][n_triplet.reshape(-1)]
            n_img_feat = n_img_feat.view(*n_triplet.shape, -1, feat_dim)  

            # Initialize loss
            loss_local_3d = 0.0
            loss_local_img = 0.0

            loss_attn_3d = self.criterion_attns_3D(x['pc_attns'])
            loss_attn = self.criterion_attns(x['img_attns'])

            for i in range(len(q_triplet)):
                # Construct target for one positive and five negatives
                target = torch.tensor([-1, 1] + [0] * len(n_pc_feat[i])).to(device)
                
                # Concatenate anchor, positive, and negatives along the feature dimension
                pc_superfeatures = torch.cat([a_pc_feat[i].unsqueeze(0), p_pc_feat[i].unsqueeze(0), n_pc_feat[i]], dim=0)
                img_superfeatures = torch.cat([a_img_feat[i].unsqueeze(0), p_img_feat[i].unsqueeze(0), n_img_feat[i]], dim=0)

                loss_local_3d += self.criterion_superfeatures_pc(pc_superfeatures, target)
                loss_local_img += self.criterion_superfeatures(img_superfeatures, target)

            loss_local_3d /= len(q_triplet)
            loss_local_img /= len(q_triplet)
            loss_attn_3d /= len(q_triplet)
            loss_attn /= len(q_triplet)
            loss = loss + loss_attn_3d + loss_local_3d + loss_local_img + loss_attn
            cloud_stats["loss_attn_3d"] = loss_attn_3d.item()
            cloud_stats["loss_super_pc"] = loss_local_3d.item() 
            image_stats["loss_attn"] = loss_attn.item()
            image_stats["loss_super_im"] = loss_local_img.item()

        image_stats = {'image_{}'.format(e): image_stats[e] for e in image_stats}
        stats.update(image_stats)
            
        cloud_stats = {'cloud_{}'.format(e): cloud_stats[e] for e in cloud_stats}
        stats.update(cloud_stats)

        stats['loss'] = loss.item()
        return loss, stats, None
    
    
class BatchHardTripletLossWithMasks:
    def __init__(self, margin, normalize_embeddings):
        self.loss_fn = BatchHardTripletLossWithMasksHelper(margin, normalize_embeddings)

    def __call__(self, x, positives_mask, negatives_mask):
        embeddings = x['embedding']
        return self.loss_fn(embeddings, positives_mask, negatives_mask)


class BatchHardTripletLossWithMasksHelper:
    def __init__(self, margin, normalize_embeddings):
        self.margin = margin
        self.distance = LpDistance(normalize_embeddings=normalize_embeddings, collect_stats=True)
        # We use triplet loss with Euclidean distance
        self.miner_fn = HardTripletMinerWithMasks(distance=self.distance)
        reducer_fn = reducers.AvgNonZeroReducer(collect_stats=True)
        self.loss_fn = losses.TripletMarginLoss(margin=self.margin, swap=True, distance=self.distance,
                                                reducer=reducer_fn, collect_stats=True)

    def __call__(self, embeddings, positives_mask, negatives_mask):
        hard_triplets = self.miner_fn(embeddings, positives_mask, negatives_mask)
        # select only first negative
        hard_triplets_cp = (hard_triplets[0], hard_triplets[1], hard_triplets[2][:, 0])
        dummy_labels = torch.arange(embeddings.shape[0]).to(embeddings.device)
        loss = self.loss_fn(embeddings.float(), dummy_labels, hard_triplets_cp)
        stats = {'loss': loss.item(), 'avg_embedding_norm': self.loss_fn.distance.final_avg_query_norm,
                 'num_non_zero_triplets': self.loss_fn.reducer.triplets_past_filter,
                 'num_triplets': len(hard_triplets_cp[0]),
                 'mean_pos_pair_dist': self.miner_fn.mean_pos_pair_dist,
                 'mean_neg_pair_dist': self.miner_fn.mean_neg_pair_dist,
                 'max_pos_pair_dist': self.miner_fn.max_pos_pair_dist,
                 'max_neg_pair_dist': self.miner_fn.max_neg_pair_dist,
                 'min_pos_pair_dist': self.miner_fn.min_pos_pair_dist,
                 'min_neg_pair_dist': self.miner_fn.min_neg_pair_dist,
                 'normalized_loss': loss.item() * self.loss_fn.reducer.triplets_past_filter,
                 # total loss per batch
                 'total_loss': self.loss_fn.reducer.loss * self.loss_fn.reducer.triplets_past_filter
                 }


        return loss, stats, hard_triplets