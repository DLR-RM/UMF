import numpy as np
import torch
import cv2
import torch.nn.functional as F
import open3d as o3d
import numpy as np
import torch
import cv2
from sklearn.preprocessing import RobustScaler


class PatchMatcher(object):
    """Patch matcher class, used to match keypoints and features from attention maps
       using strong geometric verification at different scales.
    """

    def __init__(self,  patch_sizes=[5], patch_size3D=[3], strides=[1], 
        stride3D=[1], th_img=0.5, th_pc=0.5):
        """Initialize PatchMatcher
        Args:
            patch_sizes: list of patch sizes for 2D attention maps
            patch_size3D: list of patch sizes for 3D attention maps
            strides: list of strides for 2D attention maps
            stride3D: list of strides for 3D attention maps
            th_img: threshold for 2D attention maps
            th_pc: threshold for 3D attention maps
        """
        assert len(patch_sizes) == len(strides)
        assert len(patch_size3D) == len(stride3D)
        assert th_img >= 0 and th_img <= 1
        assert th_pc >= 0 and th_pc <= 1
        self.patch_sizes = patch_sizes 
        self.patch_sizes3D = patch_size3D
        self.delta_img = th_img
        self.delta_pc = th_pc
        self.strides =  strides
        self.strides3D = stride3D


    def match(self, qfeats, dbfeats, qattn, dbattn):
        """Match query and database features using attention maps
        Args:
            qfeats: query features
            dbfeats: database features
            qattn: query attention maps
            dbattn: database attention maps
        Returns:
            scores: matching scores
        """
        keypoints_q_l, filtered_qfeats_l, outputs = [], [], []
        # Extract keypoints and features for query images at different scales
        for patch_size, stride in zip(self.patch_sizes, self.strides):
            # resize attn maps and features
            keypoints_q, filtered_qfeats = self.get_keypoints_from_attention_maps_2d(qattn, qfeats, patch_size, stride)
            keypoints_q_l.append(keypoints_q)
            filtered_qfeats_l.append(filtered_qfeats)
            # Similarly extract for database images
            out = [self.get_keypoints_from_attention_maps_2d(attn, feat, patch_size, stride) for attn, feat in zip(dbattn, dbfeats)]
            outputs.append(out)

        scores = np.zeros((len(outputs), len(outputs[0])))
        for scale_idx, output in enumerate(outputs):
            for img_idx, (keypoints_db, db_feats) in enumerate(output):
                scores[scale_idx, img_idx] = self.compare_two_ransac(filtered_qfeats_l[scale_idx], db_feats, keypoints_q_l[scale_idx], keypoints_db, self.patch_sizes[scale_idx], self.strides[scale_idx])

        final_scores = np.sum(scores, axis=0)
        return -final_scores

    def match_pc(self, qfeats, dbfeats, qattn, dbattn):
        """Match query and database 3D features using attention maps
        Args:
            qfeats: query features
            dbfeats: database features
            qattn: query attention maps
            dbattn: database attention maps
        Returns:
            scores: matching scores
        """
        keypoints_q_l, filtered_qfeats_l, outputs = [], [], []
        for patch_size, stride in zip(self.patch_sizes, self.strides):
            keypoints_q, filtered_qfeats = self.get_keypoints_from_attention_maps_3d(qattn, qfeats, patch_size, stride)
            keypoints_q_l.append(keypoints_q)
            filtered_qfeats_l.append(filtered_qfeats)
            out = [self.get_keypoints_from_attention_maps_3d(attn, feat, patch_size, stride) for attn, feat in zip(dbattn, dbfeats)]
            outputs.append(out)

        scores = np.zeros((len(outputs), len(outputs[0])))
        for i, output in enumerate(outputs):
            for j, (keypoints_db, db_feats) in enumerate(output):
                scores[i, j] = self.compare_two_ransac_pc(filtered_qfeats_l[i], db_feats, keypoints_q_l[i], keypoints_db, self.patch_sizes[i], self.strides[i])

        final_scores = np.sum(scores, axis=0)
        return -final_scores

        
    def get_keypoints_from_attention_maps_2d(self, attn, feat, patch_size, stride):
        attn = attn[0].unsqueeze(0).squeeze(-1)
        feat_i = feat[0].unsqueeze(0)

        attn = F.avg_pool2d(attn, kernel_size=patch_size, stride=stride, padding=0)
        feat_i = F.avg_pool2d(feat_i, kernel_size=patch_size, stride=stride, padding=0)
        attn = attn.squeeze(0).cpu().numpy()
        attn = cv2.GaussianBlur(attn, (5, 5), 0)
        feat_i = feat_i.squeeze(0)

        kp = filter_keypoint_attention(attn, th=self.delta_img)
        patches = feat_i[:, kp[:, 0], kp[:, 1]]
        return np.array(kp), np.array(patches)

    def get_keypoints_from_attention_maps_3d(self, attn, feat, patch_size, stride):
        attn = attn[0].cpu().squeeze(-1)

        attn = attn.unsqueeze(0).squeeze(-1)
        feat_i = feat[0].unsqueeze(0)
        attn = F.avg_pool3d(attn, kernel_size=patch_size, stride=stride, padding=0)
        feat_i = F.avg_pool3d(feat_i, kernel_size=patch_size, stride=stride, padding=0)

        attn = attn.squeeze(0).cpu().numpy()
        feat = feat_i.squeeze(0)

        kp = filter_keypoint_attention(attn, th=self.delta_pc)
        patches = feat[:, kp[:, 0], kp[:, 1], kp[:, 2]]
        return np.array(kp), np.array(patches)


    def compare_two_ransac_pc(self, qfeat, dbfeat, kpQ, kp2, patch_size, stride):
        if kpQ.shape[0] < 4 or kp2.shape[0] < 4:
            return 0

        if qfeat.shape[0] < 4 or dbfeat.shape[0] < 4:
            return 0
        
        MIN_MATCH_COUNT = 40
        threshold = 0.5

        qf = np.array(qfeat, dtype=np.float32).T 
        dbf = np.array(dbfeat, dtype=np.float32).T
        num_total_kps = len(kp2) 

        # find matches
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        good_matches = bf.match(qf, dbf)
        
        if len(good_matches) < MIN_MATCH_COUNT:
            return 0

        best_inliers = 0
        best_transformation = np.eye(4)
        all_src_pts = np.float32([kpQ[m.queryIdx] for m in good_matches])
        all_dst_pts = np.float32([kp2[m.trainIdx] for m in good_matches])
        for _ in range(120):  # Number of RANSAC iterations

            subset_indices = np.random.choice(len(good_matches), 3, replace=False)
            matches = [good_matches[i] for i in subset_indices]
            src_pts = np.float32([kpQ[m.queryIdx] for m in matches])
            dst_pts = np.float32([kp2[m.trainIdx] for m in matches])

            transformation = estimate_rigid_transform(src_pts, dst_pts)
            inliers = count_inliers(all_src_pts, all_dst_pts, transformation, threshold=threshold)

            if inliers > best_inliers:
                best_inliers = inliers
                best_transformation = transformation

        # Refining the transformation using ICP
        refined_transformation, fitness_score = refine_with_icp(all_src_pts, all_dst_pts, best_transformation, threshold=threshold)
        fitness_score =  fitness_score # / num_total_kps
        return fitness_score



    def compare_two_ransac(self, qfeat, dbfeat, kpQ, kp2, patch_size, stride):
        MIN_MATCH_COUNT = 10

        if kpQ.shape[0] < 4 or kp2.shape[0] < 4:
            return 0
        
        qf = np.array(qfeat, dtype=np.float32).T 
        dbf = np.array(dbfeat, dtype=np.float32).T

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        good_matches = bf.match(qf, dbf)

        if len(good_matches) > MIN_MATCH_COUNT:
            src_pts = np.float32([kpQ[m.queryIdx] for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx] for m in good_matches]).reshape(-1, 1, 2)

            _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, (patch_size)*1.5*4)

            inlier_index_keypoints = src_pts[mask.ravel() == 1]
            inlier_count =  inlier_index_keypoints.shape[0]

            total_keypoints = len(kpQ)
            normalized_score = inlier_count  / total_keypoints
            return normalized_score
        else:
            return 0


def estimate_rigid_transform(src, dst):
    """
    Estimate rigid transformation from src to dst.
    Args:
    src (np.array): Source points (Nx3).
    dst (np.array): Destination points (Nx3).

    Returns:
    np.array: 4x4 transformation matrix.
    """
    # Compute centroids
    centroid_src = np.mean(src, axis=0)
    centroid_dst = np.mean(dst, axis=0)

    # Center the points
    src_centered = src - centroid_src
    dst_centered = dst - centroid_dst

    # Compute the covariance matrix
    H = np.dot(src_centered.T, dst_centered)

    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # Special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    # Compute the translation
    t = centroid_dst.T - np.dot(R, centroid_src.T)

    transformation = np.identity(4)
    transformation[:3, :3] = R
    transformation[:3, 3] = t

    return transformation

def count_inliers(src, dst, transformation, threshold):
    """
    Count how many points in src are within 'threshold' distance to dst after applying transformation.
    Args:
    src (np.array): Source points (Nx3).
    dst (np.array): Destination points (Nx3).
    transformation (np.array): 4x4 transformation matrix.
    threshold (float): Distance threshold for counting inliers.

    Returns:
    int: Count of inliers.
    """
    src_homogeneous = np.hstack((src, np.ones((src.shape[0], 1))))
    transformed_src = np.dot(transformation, src_homogeneous.T).T[:, :3]

    distances = np.sqrt(np.sum((transformed_src - dst) ** 2, axis=1))
    inliers = np.sum(distances < threshold)

    return inliers

def refine_with_icp(source_points, target_points, initial_transformation, threshold=0.02):
    if isinstance(source_points, torch.Tensor):
        source_points = source_points.cpu().numpy()
    if isinstance(target_points, torch.Tensor):
        target_points = target_points.cpu().numpy()

    # Ensure the data type and shape are correct
    source_points = np.ascontiguousarray(source_points, dtype=np.float64)
    target_points = np.ascontiguousarray(target_points, dtype=np.float64)

    # Ensure the arrays are 2D
    if source_points.ndim != 2 or source_points.shape[1] != 3:
        raise ValueError("source_points must be a 2D array of shape [N, 3]")
    if target_points.ndim != 2 or target_points.shape[1] != 3:
        raise ValueError("target_points must be a 2D array of shape [N, 3]")

    # Convert numpy arrays to Open3D PointCloud objects
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_points)

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_points)

    # Set the ICP convergence criteria
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)

    # Perform ICP refinement
    result = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance=threshold, 
        init=initial_transformation, 
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=criteria
    )

    return result.transformation, result.fitness
    

def filter_keypoint_attention(attn, th=0.8):
    """Filter keypoints from attention map
    Args:
        attn: 2d or 2d attention map
        th: threshold
    Returns:
        keypoints: list of keypoints
    """
    if attn.ndim == 3:
        attn = attn / attn.max()
    else:
        attn = cv2.GaussianBlur(attn, (5, 5), 0)
        attn = RobustScaler().fit_transform(attn)
    keypoints = np.argwhere(attn > th)    
    return keypoints
