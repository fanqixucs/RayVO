#!/usr/bin/env python
# coding: utf-8

# In[14]:


import gzip
import json
import os.path as osp
import random
import socket
import time
import warnings

import ipdb  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageFile
from pytorch3d.renderer import FoVPerspectiveCameras, PerspectiveCameras
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
from tqdm.auto import tqdm
from accelerate import Accelerator

from pytorch3d.vis.plotly_vis import plot_scene
from ray_diffusion.inference.load_model import load_model
from ray_diffusion.inference.predict import predict_cameras_for_rec
import os
import pickle
from ray_diffusion.dataset.co3d_v2_kitti_test import Co3dDataset
from torch.utils.data import DataLoader
from liegroups import SE3





# In[10]:


def generate_transformation_matrix(R, t):
    """
    Generate a 4x4 transformation matrix from a rotation matrix and a translation vector.

    Args:
        R (numpy.ndarray): 3x3 rotation matrix.
        t (numpy.ndarray): 3x1 translation vector.

    Returns:
        numpy.ndarray: 4x4 transformation matrix.
    """
    # Initialize a 4x4 identity matrix
    transformation_matrix = np.eye(4)
    
    # Set the top-left 3x3 part to the rotation matrix R
    transformation_matrix[:3, :3] = R
    
    # Set the top-right 3x1 part to the translation vector t
    transformation_matrix[:3, 3] = t
    
    return transformation_matrix


# In[11]:


import pickle

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# In[12]:


from pyslam.metrics import TrajectoryMetrics
from liegroups import SE3
def compute_trajectory(vo_forward_estimated_poses, est_traj, gt_traj, method='odom', compute_seg_err=False):
    cum_dist = [0]
    for i in range(0,vo_forward_estimated_poses.shape[0]):
        # dT = SE3.from_matrix(vo_inverse_estimated_poses[i],normalize=True).inv() # the pose vec here is the inverse of the transformation
        dT = SE3.from_matrix(vo_forward_estimated_poses[i],normalize=True)
        cum_dist.append(cum_dist[i]+np.linalg.norm(dT.trans))
       
    gt_traj_se3 = [SE3.from_matrix(T,normalize=True) for T in gt_traj]
    est_traj_se3 = [SE3.from_matrix(T,normalize=True) for T in est_traj]
    
    tm_est = TrajectoryMetrics(gt_traj_se3, est_traj_se3, convention = 'Twv')
    est_mean_trans, est_mean_rot = tm_est.mean_err()
    est_mean_rot = ( est_mean_rot*180/np.pi ).round(3)
    est_mean_trans = est_mean_trans.round(3)
    
    print("{} mean trans. error: {} | mean rot. error: {}".format(method, est_mean_trans, est_mean_rot))

    if compute_seg_err==True:
        seg_lengths = list(range(100,801,100))
        _, seg_errs_est = tm_est.segment_errors(seg_lengths, rot_unit='rad')
        
        rot_seg_err = ( 100*np.mean(seg_errs_est[:,2])*180/np.pi).round(3)
        trans_seg_err = ( np.mean(seg_errs_est[:,1])*100).round(3)
    
        if np.isnan(trans_seg_err):
            max_dist = cum_dist[-1] - cum_dist[-1]%100 + 1 - 100
            print('max dist', max_dist)
            seg_lengths = list(range(100,int(max_dist),100))
            _, seg_errs_est = tm_est.segment_errors(seg_lengths, rot_unit='rad')

            rot_seg_err = ( 100*np.mean(seg_errs_est[:,2])*180/np.pi).round(3)
            trans_seg_err = ( np.mean(seg_errs_est[:,1])*100).round(3)        
    
    
        print("{} mean Segment Errors: {} (trans, %) | {} (rot, deg/100m)".format(method, trans_seg_err, rot_seg_err))
        
        errors = (est_mean_trans, est_mean_rot, trans_seg_err, rot_seg_err)
    else:
        errors = (est_mean_trans, est_mean_rot, 0, 0)            

    return np.array(est_traj), np.array(gt_traj), errors, np.array(cum_dist)


# In[4]:


def compute_trajectory_original(pose_vec, gt_traj, method='odom', compute_seg_err=False):
    est_traj = [gt_traj[0]]
    #print('gt_traj[0]',gt_traj[0])
    cum_dist = [0]
    for i in range(0,pose_vec.shape[0]):
        dT = SE3.exp(pose_vec[i])
        new_est = SE3.as_matrix((dT.dot(SE3.from_matrix(est_traj[i],normalize=True).inv())).inv())
        est_traj.append(new_est)
        cum_dist.append(cum_dist[i]+np.linalg.norm(dT.trans))
       
    gt_traj_se3 = [SE3.from_matrix(T,normalize=True) for T in gt_traj]
    est_traj_se3 = [SE3.from_matrix(T,normalize=True) for T in est_traj]
    #print('gt_traj_se3',gt_traj_se3)
    #print('tm_est',tm_est) ### object
    #print('est_traj_se3',est_traj_se3)
    
    tm_est = TrajectoryMetrics(gt_traj_se3, est_traj_se3, convention = 'Twv')
    est_mean_trans, est_mean_rot = tm_est.mean_err()
    est_mean_rot = ( est_mean_rot*180/np.pi ).round(3)
    est_mean_trans = est_mean_trans.round(3)
    
    #######################################################i think it is not correct 169.646 trans, 83.969 rot####################################
    print("{} mean trans. error: {} | mean rot. error: {}".format(method, est_mean_trans, est_mean_rot))

    if compute_seg_err==True:
        seg_lengths = list(range(100,801,100))
        #print('seg_lengths', seg_lengths)
        #####################################################################################################
        _, seg_errs_est = tm_est.segment_errors(seg_lengths, rot_unit='rad')
        #print('seg_errs_est', seg_errs_est)
        
        rot_seg_err = ( 100*np.mean(seg_errs_est[:,2])*180/np.pi).round(3)
        trans_seg_err = ( np.mean(seg_errs_est[:,1])*100).round(3)
    
        if np.isnan(trans_seg_err):
            max_dist = cum_dist[-1] - cum_dist[-1]%100 + 1 - 100
            print('max dist', max_dist)
            seg_lengths = list(range(100,int(max_dist),100))
            _, seg_errs_est = tm_est.segment_errors(seg_lengths, rot_unit='rad')

            rot_seg_err = ( 100*np.mean(seg_errs_est[:,2])*180/np.pi).round(3)
            trans_seg_err = ( np.mean(seg_errs_est[:,1])*100).round(3)        
    
    
        print("{} mean Segment Errors: {} (trans, %) | {} (rot, deg/100m)".format(method, trans_seg_err, rot_seg_err))
        
        errors = (est_mean_trans, est_mean_rot, trans_seg_err, rot_seg_err)
    else:
        errors = (est_mean_trans, est_mean_rot, 0, 0)            

    return np.array(est_traj), np.array(gt_traj), errors, np.array(cum_dist)


# In[23]:


def compute_trajectory_original_inverse(pose_vec, gt_traj, method='odom', compute_seg_err=False):
    est_traj = [gt_traj[0]]
    #print('gt_traj[0]',gt_traj[0])
    cum_dist = [0]
    for i in range(0,pose_vec.shape[0]):
        dT = SE3.exp(pose_vec[i]).inv()
        new_est = SE3.as_matrix((dT.dot(SE3.from_matrix(est_traj[i],normalize=True).inv())).inv())
        est_traj.append(new_est)
        cum_dist.append(cum_dist[i]+np.linalg.norm(dT.trans))
       
    gt_traj_se3 = [SE3.from_matrix(T,normalize=True) for T in gt_traj]
    est_traj_se3 = [SE3.from_matrix(T,normalize=True) for T in est_traj]
    #print('gt_traj_se3',gt_traj_se3)
    #print('tm_est',tm_est) ### object
    #print('est_traj_se3',est_traj_se3)
    
    tm_est = TrajectoryMetrics(gt_traj_se3, est_traj_se3, convention = 'Twv')
    est_mean_trans, est_mean_rot = tm_est.mean_err()
    est_mean_rot = ( est_mean_rot*180/np.pi ).round(3)
    est_mean_trans = est_mean_trans.round(3)
    
    #######################################################i think it is not correct 169.646 trans, 83.969 rot####################################
    print("{} mean trans. error: {} | mean rot. error: {}".format(method, est_mean_trans, est_mean_rot))

    if compute_seg_err==True:
        seg_lengths = list(range(100,801,100))
        #print('seg_lengths', seg_lengths)
        #####################################################################################################
        _, seg_errs_est = tm_est.segment_errors(seg_lengths, rot_unit='rad')
        #print('seg_errs_est', seg_errs_est)
        
        rot_seg_err = ( 100*np.mean(seg_errs_est[:,2])*180/np.pi).round(3)
        trans_seg_err = ( np.mean(seg_errs_est[:,1])*100).round(3)
    
        if np.isnan(trans_seg_err):
            max_dist = cum_dist[-1] - cum_dist[-1]%100 + 1 - 100
            print('max dist', max_dist)
            seg_lengths = list(range(100,int(max_dist),100))
            _, seg_errs_est = tm_est.segment_errors(seg_lengths, rot_unit='rad')

            rot_seg_err = ( 100*np.mean(seg_errs_est[:,2])*180/np.pi).round(3)
            trans_seg_err = ( np.mean(seg_errs_est[:,1])*100).round(3)        
    
    
        print("{} mean Segment Errors: {} (trans, %) | {} (rot, deg/100m)".format(method, trans_seg_err, rot_seg_err))
        
        errors = (est_mean_trans, est_mean_rot, trans_seg_err, rot_seg_err)
    else:
        errors = (est_mean_trans, est_mean_rot, 0, 0)            

    return np.array(est_traj), np.array(gt_traj), errors, np.array(cum_dist)


# In[5]:


import csv
class ResultsLogger():
    def __init__(self, filename):
        self.filename = filename
        csv_header1 = ['', '', '','m-ATE_inverse', '', 'Mean Segment Errors_inverse', '','m-ATE_forward', '', 'Mean Segment Errors_forward', '','m-ATE_inv_fwd', '', 'Mean Segment Errors_inv_fwd', '']
        csv_header2 = ['Sequence','Epoch', 'Name', 'Trans_inv. (m)', 'Rot_inv. (deg)', 'Trans_inv. (%)', 'Rot_inv. (deg/100m)', 'Trans_fwd. (m)', 'Rot_fwd. (deg)', 'Trans_fwd. (%)', 'Rot_fwd. (deg/100m)', 'Trans_inv_fwd. (m)', 'Rot_inv_fwd. (deg)', 'Trans_inv_fwd. (%)', 'Rot_inv_fwd. (deg/100m)']
        self.t_ate_list_inv = []
        self.r_ate_list_inv = []
        self.t_mse_list_inv = []
        self.r_mse_list_inv = []
        self.t_ate_list_fwd = []
        self.r_ate_list_fwd = []
        self.t_mse_list_fwd = []
        self.r_mse_list_fwd = []
        self.t_ate_list_inv_fwd = []
        self.r_ate_list_inv_fwd = []
        self.t_mse_list_inv_fwd = []
        self.r_mse_list_inv_fwd = []
        with open(filename, "w") as f:
            self.writer = csv.writer(f)
            self.writer.writerow(csv_header1)
            self.writer.writerow(csv_header2)
    
    def log(self, seq, epoch, name, t_ate_inv, r_ate_inv, t_mse_inv, r_mse_inv, t_ate_fwd, r_ate_fwd, t_mse_fwd, r_mse_fwd, t_ate_inv_fwd, r_ate_inv_fwd, t_mse_inv_fwd, r_mse_inv_fwd):
        stats_list = [seq, epoch, name, t_ate_inv, r_ate_inv, t_mse_inv, r_mse_inv, t_ate_fwd, r_ate_fwd, t_mse_fwd, r_mse_fwd, t_ate_inv_fwd, r_ate_inv_fwd, t_mse_inv_fwd, r_mse_inv_fwd]
        with open(self.filename, "a") as f:
            self.writer = csv.writer(f)
            self.writer.writerow(stats_list)
        
        self.t_ate_list_inv.append(t_ate_inv)
        self.r_ate_list_inv.append(r_ate_inv)
        self.t_mse_list_inv.append(t_mse_inv)
        self.r_mse_list_inv.append(r_mse_inv)
        self.t_ate_list_fwd.append(t_ate_fwd)
        self.r_ate_list_fwd.append(r_ate_fwd)
        self.t_mse_list_fwd.append(t_mse_fwd)
        self.r_mse_list_fwd.append(r_mse_fwd)
        self.t_ate_list_inv_fwd.append(t_ate_inv_fwd)
        self.r_ate_list_inv_fwd.append(r_ate_inv_fwd)
        self.t_mse_list_inv_fwd.append(t_mse_inv_fwd)
        self.r_mse_list_inv_fwd.append(r_mse_inv_fwd)

        self.results = {
            't_ate_list_inv': self.t_ate_list_inv,
            'r_ate_list_inv': self.r_ate_list_inv,
            't_mse_list_inv': self.t_mse_list_inv,
            'r_mse_list_inv': self.r_mse_list_inv,
            't_ate_list_fwd': self.t_ate_list_fwd,
            'r_ate_list_fwd': self.r_ate_list_fwd,
            't_mse_list_fwd': self.t_mse_list_fwd,
            'r_mse_list_fwd': self.r_mse_list_fwd,
            't_ate_list_inv_fwd': self.t_ate_list_inv_fwd,
            'r_ate_list_inv_fwd': self.r_ate_list_inv_fwd,
            't_mse_list_inv_fwd': self.t_mse_list_inv_fwd,
            'r_mse_list_inv_fwd': self.r_mse_list_inv_fwd,
        }
        

os.environ["CUDA_VISIBLE_DEVICES"] = str(3)
torch.cuda.device_count() 
accelerator = Accelerator(even_batches=False,device_placement=False,)
device = accelerator.device

# training parameters
batch_size = 8
# train on 00-10
# whole_train_size = 46402 
# train on 00-08
whole_train_size = 40818

# for dino config
# image_width = 640
# image_height = 192
# num_x = 40
# num_y = 12

# for dinov2
image_width = 630
image_height = 182
num_x = 45
num_y = 13

# # for dinov2
# image_width = 840
# image_height = 252
# num_x = 60
# num_y = 18

# image_width = 448
# image_height = 112
# num_x = 32
# num_y = 8

scale = image_width / image_height
# seq for inference
# seqs = ['09', '10']
seqs = ['09']
# seqs = ['10']
# Model directory
model_vo_diffusion_results_directory = "/hdd2/fanqi/ckps/checkpoints_for_rec/inverse_forward_regression_co3dpretrain_dinov2_dit4_perfect_unprojection_45_13_noise_step_1000_cos_beta_end_002_train08/output/0218_180208_100249" 
model_vo_diffusion_ckps_directory = osp.join(model_vo_diffusion_results_directory, "checkpoints")
# inference results directory
model_vo_diffusion_inference_results_directory = osp.join(model_vo_diffusion_results_directory, "results_large_epoch")
# model_vo_diffusion_inference_results_directory = osp.join(model_vo_diffusion_results_directory, "results_wm_scale_postprocess")
# model_vo_diffusion_inference_results_directory = osp.join(model_vo_diffusion_results_directory, "results_wom_scale_postprocess")
model_vo_diffusion_inference_traj_images = osp.join(model_vo_diffusion_inference_results_directory, "traj_images")
model_vo_diffusion_inference_prediced_poses = osp.join(model_vo_diffusion_inference_results_directory, "predicted_poses")
model_vo_diffusion_inference_prediced_rays = osp.join(model_vo_diffusion_inference_results_directory, "predicted_rays")
model_vo_diffusion_inference_ate_rpe_csv = osp.join(model_vo_diffusion_inference_results_directory, "ate_rpe_csv")

os.makedirs(model_vo_diffusion_inference_results_directory, exist_ok=True)
os.makedirs(model_vo_diffusion_inference_traj_images, exist_ok=True)
os.makedirs(model_vo_diffusion_inference_prediced_poses, exist_ok=True)
os.makedirs(model_vo_diffusion_inference_prediced_rays, exist_ok=True)
os.makedirs(model_vo_diffusion_inference_ate_rpe_csv, exist_ok=True)

# List all model files in the directory
model_files = sorted([f for f in os.listdir(model_vo_diffusion_ckps_directory) if f.endswith('.pth')])
# Create csv files for the results
# Create csv files for the results
if seqs == ['09']:
    logger_09_all = ResultsLogger('{}/09_all.csv'.format(model_vo_diffusion_inference_ate_rpe_csv))
elif seqs == ['10']:
    logger_10_all = ResultsLogger('{}/10_all.csv'.format(model_vo_diffusion_inference_ate_rpe_csv))


# In[26]:


for model in model_files:
    # Get the epoch number from the model name
    iteration_num = model.split(".")[0].split('_')[1]
    iteration_num = int(iteration_num)
    epoch_num = int(iteration_num * batch_size / whole_train_size)
    
    # print("model.append ndc", model_vo_diffusion.append_ndc) # false for rec
    if epoch_num >46:
        model_vo_diffusion, cfg_vo_diffusion = load_model(model_vo_diffusion_results_directory, model, device=device)
        for seq in seqs:
            vo_inverse_pred = []
            vo_forward_pred = []
            vo_inverse_lie_alg = []
            vo_forward_lie_alg = []
            rays_pred_inv = []
            rays_pred_fwd = []
            gt_vo_transformation_matrix =[]
            
            dataset = Co3dDataset(num_images=2, id_consecutive=True, category = [seq], image_width=image_width, image_height=image_height) 
            instances = np.arange(0, int(len(dataset)/2-1))
            
            with torch.no_grad():
                for counter, idx in enumerate(instances):
                    # if counter == 1:
                    #     break
                    
                    batch = dataset[idx]
                    seq_name = batch["model_id"]
                    filename = batch["filename"]
                    print(f"model of epoch {epoch_num} file {filename}")
                    images = batch["image"].to(device)
                    
                    # generate the gt poses forward pose
                    gt_R_tensor = batch["odometry_R"][1].cpu().numpy()
                    gt_t_tensor = batch["odometry_T"][1].cpu().numpy()
                    gt_transformation_matrix = generate_transformation_matrix(gt_R_tensor, gt_t_tensor)
                    gt_vo_transformation_matrix.append(gt_transformation_matrix)
                    
                    # create the perspective camera using the gt intrinsics
                    cameras_intrinsic = PerspectiveCameras(
                        focal_length=batch["focal_length"], # [-3.9929, -3.8221]
                        principal_point=batch["principal_point"], # [0.0628, 0.0102]
                        R=batch["odometry_R"],
                        T=batch["odometry_T"],
                    )
                    
                    # calculate the pose from known camera intrinsics
                    pred= predict_cameras_for_rec(
                                model_vo_diffusion,
                                images,
                                device,
                                pred_x0=True,
                                crop_parameters=None,
                                num_patches_x=num_x,
                                num_patches_y=num_y,
                                use_regression=True, 
                                max_num_images=2,
                                return_rays=True,
                                calculate_intrinsics=False, # use known intrinsics
                                scale = scale,
                                cameras = cameras_intrinsic,
                            )
                    # print("pred", pred)
                    # print("pred[0]", pred[0]) # PerspectiveCameras()
                    # print("pred[1]", pred[1]) # rays
                    # print("pred[2]", pred[2]) # []
                    # print("pred[3]", pred[3]) # []
                    
                    # processing rays 
                    pred_rays_inv = pred[1][0]
                    pred_rays_fwd = pred[1][1]
                    rays_pred_inv.append(pred_rays_inv)
                    rays_pred_fwd.append(pred_rays_fwd)
                    
                    # processing vo matrices
                    pred_cameras = pred[0]
                    # print("pred_cameras", pred_cameras.R)
                    R_pred_inv = pred_cameras.R[0].t().cpu().numpy()
                    t_pred_inv = pred_cameras.T[0].cpu().numpy()
                    R_pred_fwd = pred_cameras.R[1].t().cpu().numpy()
                    t_pred_fwd = pred_cameras.T[1].cpu().numpy()
                    vo_inverse_pred.append(generate_transformation_matrix(R_pred_inv, t_pred_inv))
                    vo_forward_pred.append(generate_transformation_matrix(R_pred_fwd, t_pred_fwd))
                    vo_inverse_lie_alg.append(SE3.from_matrix(generate_transformation_matrix(R_pred_inv, t_pred_inv),normalize=True).log())
                    vo_forward_lie_alg.append(SE3.from_matrix(generate_transformation_matrix(R_pred_fwd, t_pred_fwd),normalize=True).log())
                    # print("vo_inverse_lie",vo_inverse_lie_alg)
                    # print("vo_forward_lie",vo_forward_lie_alg)
            
            rays_pred_inv = np.array(rays_pred_inv)
            rays_pred_fwd = np.array(rays_pred_fwd)
            vo_forward_pred = np.array(vo_forward_pred)
            vo_inverse_pred = np.array(vo_inverse_pred)
            vo_forward_lie_alg = np.array(vo_forward_lie_alg)
            vo_inverse_lie_alg = np.array(vo_inverse_lie_alg)
            vo_inverse_forward_lie_alg = (vo_forward_lie_alg - vo_inverse_lie_alg)/2
            gt_vo_transformation_matrix = np.array(gt_vo_transformation_matrix)
            
            """ pre_traj_from_known_intrinsics = [np.eye(4)]
            for i in range(0, vo_forward_pred.shape[0]):
                dT = vo_forward_pred[i]
                new_est = np.linalg.inv(np.dot(dT,np.linalg.inv(pre_traj_from_known_intrinsics[i]))) # the estimation is inv
                pre_traj_from_known_intrinsics.append(new_est)
            pre_traj_from_known_intrinsics = np.array(pre_traj_from_known_intrinsics) """
            
            gt_traj = [np.eye(4)] # odometry is inverse pose from c to w/p
            for i in range(0, np.array(gt_vo_transformation_matrix).shape[0]):
                
                dT = gt_vo_transformation_matrix[i]
                new_est = np.linalg.inv(np.dot(dT,np.linalg.inv(gt_traj[i])))
                gt_traj.append(new_est)
            gt_traj = np.array(gt_traj)
            
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # save the rays
            rays = {"seq":seq, "rays_pred_inv":rays_pred_inv, "rays_pred_fwd":rays_pred_fwd}
            save_obj(rays, osp.join(model_vo_diffusion_inference_prediced_rays, f"seq-{seq_name}-epoch-{epoch_num}-ditmodel_regression_train_all_{timestamp}"))
            
            # save the predicted poses
            data = {"seq":seq,"vo_forward_estimated_poses_from_known_intrinsic":vo_forward_pred, "vo_inverse_estimated_poses_from_known_intrinsic":vo_inverse_pred}
            
            save_obj(data, osp.join(model_vo_diffusion_inference_prediced_poses, f"seq-{seq_name}-epoch-{epoch_num}-ditmodel_regression_train_all_{timestamp}"))
            
            """ est_traj_from_known_intrinsic, gt_traj, errors_known_intrinsics, cum_dist = compute_trajectory(vo_forward_pred, pre_traj_from_known_intrinsics, gt_traj, method='RT_from_known_intrinsic', compute_seg_err=True) """
            # error from forward lie algebra
            est_traj_inv, _, errors_inv_lie_alg, _ = compute_trajectory_original_inverse(vo_inverse_lie_alg,gt_traj, method='est_traj_inv', compute_seg_err=True)
            
            est_traj_fwd, _, errors_fwd_lie_alg, _ = compute_trajectory_original(vo_forward_lie_alg,gt_traj, method='est_traj_fwd', compute_seg_err=True)
            print(errors_fwd_lie_alg)
            
            est_traj_fwd_inv, _, errors_fwd_inv_lie_alg, _ = compute_trajectory_original(vo_inverse_forward_lie_alg,gt_traj, method='est_traj_fwd_inv', compute_seg_err=True)
            
            if seq == '09':
                logger_09_all.log(seq, epoch_num,'RT_from_known_intrinsic', errors_inv_lie_alg[0], errors_inv_lie_alg[1], errors_inv_lie_alg[2], errors_inv_lie_alg[3], errors_fwd_lie_alg[0], errors_fwd_lie_alg[1], errors_fwd_lie_alg[2], errors_fwd_lie_alg[3], errors_fwd_inv_lie_alg[0], errors_fwd_inv_lie_alg[1], errors_fwd_inv_lie_alg[2], errors_fwd_inv_lie_alg[3])
                
            elif seq == '10':
                logger_10_all.log(seq, epoch_num,'RT_from_known_intrinsic', errors_inv_lie_alg[0], errors_inv_lie_alg[1], errors_inv_lie_alg[2], errors_inv_lie_alg[3], errors_fwd_lie_alg[0], errors_fwd_lie_alg[1], errors_fwd_lie_alg[2], errors_fwd_lie_alg[3], errors_fwd_inv_lie_alg[0], errors_fwd_inv_lie_alg[1], errors_fwd_inv_lie_alg[2], errors_fwd_inv_lie_alg[3])
            ### new
            
            plt.figure()
            plt.grid()
            plt.plot(est_traj_fwd[:,0,3], est_traj_fwd[:,2,3], linewidth=1.5, color='green', label='fwd_est')
            plt.plot(est_traj_inv[:,0,3], est_traj_inv[:,2,3], linewidth=1.5, color='orange', label='inv_est')
            plt.plot(est_traj_fwd_inv[:,0,3], est_traj_fwd_inv[:,2,3], linewidth=1.5, color='blue', label='inv_fwd_est')        
            plt.plot(gt_traj[:,0,3], gt_traj[:,2,3], linewidth=1.5, color='red', label='gt')
            plt.legend()
            plt.title(f'Topdown (XY) Trajectory {seq_name}')
            # Save the plot with the timestamp in the filename
            output_path = osp.join(model_vo_diffusion_inference_traj_images, f'seq-{seq_name}-epoch-{epoch_num}-ditmodel_regression_train_all_{timestamp}.png')
            plt.savefig(output_path)
            # plt.show()
            plt.close()
