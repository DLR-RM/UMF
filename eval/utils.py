from sklearn.neighbors import KDTree
import numpy as np
import pickle
import os
import torch
import tqdm
from how.layers import functional as HF
from models.UMF.utils.patch_matcher import PatchMatcher
from models.losses.super_feature_losses import match_super
import logging
import yaml

scores_cache_img = {}
scores_cache_pc = {}
NUM_NEIGHBORS = 25


def load_config(config_file):
    with open(config_file, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return cfg

def compute_and_log_stats(mode, modality, query_sets, database_sets, features_folder):
    count = 0
    recall = np.zeros(NUM_NEIGHBORS)
    one_percent_recall = []
    similarity = []
    
    for i in tqdm.tqdm(range(len(query_sets))):
        with open(os.path.join(features_folder, f'query_set_{i}.pkl'), 'rb') as f:
            query_outputs = pickle.load(f)
        for j in range(len(database_sets)):
            if i == j:
                continue
            with open(os.path.join(features_folder, f'db_set_{j}.pkl'), 'rb') as f:
                db_outputs = pickle.load(f)
            
            if mode == 'superfeatures':
                pair_recall, pair_similarity, pair_opr = get_recall_super(i, j, query_outputs, db_outputs,
                                                                    query_sets[i], mode=modality)
            else:
                pair_recall, pair_similarity, pair_opr = get_recall(i, j, query_outputs, db_outputs,
                                                                    query_sets[i], mode=modality)
                
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)
            similarity.extend(pair_similarity)
    
    ave_recall = recall / count
    average_similarity = np.mean(similarity)
    ave_one_percent_recall = np.mean(one_percent_recall)
    stats = {'ave_one_percent_recall': ave_one_percent_recall, 'ave_recall': ave_recall,
             'average_similarity': average_similarity}
    
    logging.info("-------------------------------------------------------------------")
    logging.info(f"Mode: {mode}")
    logging.info(f"ave_one_percent_recall: {ave_one_percent_recall}")
    logging.info(f"ave_recall: {ave_recall}")
    logging.info(f"average_similarity: {average_similarity}")
    logging.info("-------------------------------------------------------------------")
    
    return stats


def select_superfeatures(superfeatures, strengths, scales=[1], N=20):
    selected_superfeatures = torch.zeros(N, superfeatures.shape[-1])
    strengths = strengths.unsqueeze(0).unsqueeze(2)
    features, _, _, scales = HF.how_select_local(superfeatures, strengths, scales=scales, features_num=N)
    selected_superfeatures = features
    return selected_superfeatures


def get_recall_super(n, m, query_outputs, db_outputs, query_set, mode='base'):
    global scores_cache_img, scores_cache_pc
    database_output = db_outputs['embedding']
    queries_output = query_outputs['embedding']

    # log mode
    logging.info(f"-------------------------------------------------------------------")
    logging.info(f"Mode: {mode}")
    logging.info(f"-------------------------------------------------------------------")

    if mode == 'img' or mode == 'combined':
        db_feat_img = db_outputs['img_super_feat']
        q_feat_img = query_outputs['img_super_feat']
        db_strengths_img = db_outputs['img_strengths']
        q_strengths_img = query_outputs['img_strengths']
        N = 64
        LoweRatioTh = 0.9

    if mode == 'pc' or mode == 'combined':
        db_feat_pc = db_outputs['pc_super_feat']
        q_feat_pc = query_outputs['pc_super_feat']
        db_strengths_pc = db_outputs['pc_strengths']
        q_strengths_pc = query_outputs['pc_strengths']
        N = 128
        LoweRatioTh = 0.9
    database_nbrs = KDTree(database_output)
    recall = [0] * NUM_NEIGHBORS

    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output)/100.0)), 1)

    num_evaluated = 0
    for i in range(len(queries_output)):
        # i is query element ndx
        query_details = query_set[i]    # {'query': path, 'northing': , 'easting': }
        true_neighbors = query_details[m]
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1
        distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=NUM_NEIGHBORS)

        if len(indices) == 0:
            print("-----> Error")
            continue

        candidates = indices[0]
        final_candidates = candidates
        final_scores = distances[0]
        cache_key = (m, n, i)
        if mode != 'base':            

            if mode == 'img':
                q_img_superfeatures = select_superfeatures(q_feat_img[i], q_strengths_img[i], N=N)
                db_img_superfeatures = [select_superfeatures(db_feat_img[j], db_strengths_img[j], N=N) for j in candidates]
                scores = np.array([ -len(match_super(q_img_superfeatures, db_img_superfeatures[j], LoweRatioTh=LoweRatioTh)) / N for j in range(len(candidates))])
                scores_cache_img[cache_key] = scores

            elif mode == 'pc':
                q_pc_superfeatures = select_superfeatures(q_feat_pc[i], q_strengths_pc[i], N=N) 
                db_pc_superfeatures = [select_superfeatures(db_feat_pc[j], db_strengths_pc[j], N=N) for j in candidates]

                scores = np.array([ -len(match_super(q_pc_superfeatures, db_pc_superfeatures[j], LoweRatioTh=LoweRatioTh)) / N for  j in range(len(candidates))])
                scores_cache_pc[cache_key] = scores 
            elif mode == 'combined':
                scores_img = scores_cache_img[cache_key]
                scores_pc = scores_cache_pc[cache_key]

                scores = 1. * scores_img + 1 * scores_pc

            scores =  scores + distances[0]
            scores = [(score, candidate) for score, candidate in zip(scores, candidates)]

            final_candidates = [x for _, x in sorted(scores, key=lambda el: el[0])]
            final_scores =  [x[0] for x in sorted(scores)] 

        correct = []
        for j in range(len(final_candidates)):
            if final_candidates[j] in true_neighbors:
                correct.append( (final_candidates[j], final_scores[j]) )
                if j == 0:
                    similarity = np.dot(queries_output[i], database_output[final_candidates[j]])
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                break

        if len(list(set(final_candidates[0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1
    
        if mode != 'base':
            candidates = torch.tensor(candidates, dtype=torch.int32)
            final_candidates = torch.tensor(final_candidates, dtype=torch.int32)

            candidates_list = candidates.tolist()
            final_candidates_list = final_candidates.tolist()

            if len(correct) > 0:
                log_reranking_stats(candidates_list, final_candidates_list, correct, true_neighbors)

    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    recall = (np.cumsum(recall)/float(num_evaluated))*100
    return recall, top1_similarity_score, one_percent_recall


def get_recall(n, m, query_outputs, db_outputs, query_set, mode='base'):
    global scores_cache_img, scores_cache_pc
    database_output = db_outputs['embedding']
    queries_output = query_outputs['embedding']

    logging.info(f"-------------------------------------------------------------------")
    logging.info(f"Mode: {mode}")
    logging.info(f"-------------------------------------------------------------------")

    if mode == 'img' or mode == 'combined':
        db_feat_img = db_outputs['img_fine_feat']
        q_feat_img = query_outputs['img_fine_feat']
        q_attn_img = query_outputs['img_attns']
        db_attn_img = db_outputs['img_attns']

    if mode == 'pc' or mode == 'combined':
        db_feat_pc = db_outputs['pc_fine_feat']
        q_feat_pc = query_outputs['pc_fine_feat']
        q_attn_pc = query_outputs['pc_attns']
        db_attn_pc = db_outputs['pc_attns']


    database_nbrs = KDTree(database_output)

    recall = [0] * NUM_NEIGHBORS

    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output)/100.0)), 1)

    num_evaluated = 0
    for i in range(len(queries_output)):
        # i is query element ndx
        query_details = query_set[i]    # {'query': path, 'northing': , 'easting': }
        true_neighbors = query_details[m]
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1
        distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=NUM_NEIGHBORS)

        if len(indices) == 0:
            print("-----> Error")
            continue

        candidates = indices[0]
        final_candidates = candidates
        final_scores = distances[0]
        cache_key = (m, n, i)
        if mode != 'base':
            matcher = PatchMatcher(
                patch_sizes=[1], # [1, 5, 7]
                strides=[1],  # [1, 2, 3]
                patch_size3D=[1],  # [1, 3]
                stride3D=[1],  # [1, 2]
                th_img=0.6,
                th_pc=0.5,
            )

            if mode == 'img':
                scores = matcher.match(q_feat_img[i], db_feat_img[candidates], q_attn_img[i], db_attn_img[candidates])
                scores_cache_img[cache_key] = scores
            elif mode == 'pc':
                scores = matcher.match_pc(q_feat_pc[i], db_feat_pc[candidates], q_attn_pc[i], db_attn_pc[candidates])
                scores_cache_pc[cache_key] = scores

            elif mode == 'combined':
                if cache_key not in scores_cache_img:
                    scores_img = matcher.match(q_feat_img[i], db_feat_img[candidates], q_attn_img[i], db_attn_img[candidates])
                    scores_cache_img[cache_key] = scores_img
                else:
                    scores_img = scores_cache_img[cache_key]

                if cache_key not in scores_cache_pc:
                    scores_pc =  matcher.match_pc(q_feat_pc[i], db_feat_pc[candidates], q_attn_pc[i], db_attn_pc[candidates])
                    scores_cache_pc[cache_key] = scores_pc
                else:
                    scores_pc = scores_cache_pc[cache_key]

                scores = 1. * scores_img + 1 * scores_pc

            scores =   1 * scores + distances[0]
            scores = [(score, candidate) for score, candidate in zip(scores, candidates)]

            final_candidates = [x for _, x in sorted(scores, key=lambda el: el[0])]
            final_scores =  [x[0] for x in sorted(scores)] 

        correct = []
        for j in range(len(final_candidates)):
            if final_candidates[j] in true_neighbors:
                correct.append( (final_candidates[j], final_scores[j]) )
                if j == 0:
                    similarity = np.dot(queries_output[i], database_output[final_candidates[j]])
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                break

        if len(list(set(final_candidates[0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1
    
        if mode != 'base':
            candidates = torch.tensor(candidates, dtype=torch.int32)
            final_candidates = torch.tensor(final_candidates, dtype=torch.int32)

            candidates_list = candidates.tolist()
            final_candidates_list = final_candidates.tolist()

            if len(correct) > 0:
                log_reranking_stats(candidates_list, final_candidates_list, correct, true_neighbors)
                
    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    recall = (np.cumsum(recall)/float(num_evaluated))*100
    return recall, top1_similarity_score, one_percent_recall


def log_evaluation_stats(mode, ave_one_percent_recall, ave_recall, average_similarity):
    logging.info("-------------------------------------------------------------------")
    logging.info(f"Mode: {mode}")
    logging.info(f"ave_one_percent_recall: {ave_one_percent_recall}")
    logging.info(f"ave_recall: {ave_recall}")
    logging.info(f"average_similarity: {average_similarity}")
    logging.info("-------------------------------------------------------------------")

def log_reranking_stats(candidates, final_candidates, correct, true_neighbors):
    candidate, score = correct[0]

    # Identify the index of the correct match in the initial and final candidate lists
    initial_correct_index = next((candidates.index(candidate) for candidate in candidates if candidate in true_neighbors), None)
    final_correct_index = next((final_candidates.index(candidate) for candidate in final_candidates if candidate in true_neighbors), None)

    # Compute distance to top-1 for initial and final rankings
    initial_distance_to_top1 = initial_correct_index if initial_correct_index is not None else -1
    final_distance_to_top1 = final_correct_index if final_correct_index is not None else -1

    if final_distance_to_top1 < initial_distance_to_top1:
        best_method = "with reranking"
    elif final_distance_to_top1 == initial_distance_to_top1:
        best_method = "equal"
    else:
        best_method = "without reranking"
    
    log_message = (f"Candidate: {candidate}, Score: {score:.2f}, Initial top-1: {initial_distance_to_top1}, "
                f"Reranking top-1: {final_distance_to_top1}, Best: {best_method}.")


    logging.info(log_message)
