#!/usr/bin/env python
# coding: utf-8

# In[2]:

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


# In[3]:

os.environ["CUDA_VISIBLE_DEVICES"] = str(2)
torch.cuda.device_count()
accelerator = Accelerator(even_batches=False,device_placement=False,)
device = accelerator.device


# In[4]:


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


# In[5]:


import pickle

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# In[6]:


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


# In[7]:


import csv
class ResultsLogger():
    def __init__(self, filename):
        self.filename = filename
        csv_header1 = ['', '', '','m-ATE', '', 'Mean Segment Errors', '']
        csv_header2 = ['Sequence','Epoch', 'Name', 'Trans. (m)', 'Rot. (deg)', 'Trans. (%)', 'Rot. (deg/100m)']
        self.t_ate_list = []
        self.r_ate_list = []
        self.t_mse_list = []
        self.r_mse_list = []
        with open(filename, "w") as f:
            self.writer = csv.writer(f)
            self.writer.writerow(csv_header1)
            self.writer.writerow(csv_header2)
    
    def log(self, seq, epoch, name, t_ate, r_ate, t_mse, r_mse):
        stats_list = [seq, epoch, name, t_ate, r_ate, t_mse, r_mse]
        with open(self.filename, "a") as f:
            self.writer = csv.writer(f)
            self.writer.writerow(stats_list)
            
        self.t_ate_list.append(t_ate)
        self.r_ate_list.append(r_ate)
        self.t_mse_list.append(t_mse)
        self.r_mse_list.append(r_mse)
        
        self.results = {'t_ate_list': self.t_ate_list,
                        'r_ate_list': self.r_ate_list,
                        't_mse_list': self.t_mse_list,
                        'r_mse_list': self.r_mse_list,}


# In[10]:

# gpu
# os.umask(000)  # Default to 777 permissions
# for dinov2:
scale = 630 / 182
batch_size = 8
# seq for inference
# seqs = ['09']
seqs = ['10']
# seqs = ['09', '10']
# Model directory
model_vo_diffusion_results_directory = "/hdd2/fanqi/ckps/checkpoints_for_rec/identity_forward_regression_scratch_dinov2_dit1/output/0103_101944_425628"
model_vo_diffusion_ckps_directory = osp.join(model_vo_diffusion_results_directory, "checkpoints")
# inference results directory
model_vo_diffusion_inference_results_directory = osp.join(model_vo_diffusion_results_directory, "results")
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
logger_09_all = ResultsLogger('{}/09_all.csv'.format(model_vo_diffusion_inference_ate_rpe_csv))
logger_10_all = ResultsLogger('{}/10_all.csv'.format(model_vo_diffusion_inference_ate_rpe_csv))


# In[11]:


for model in model_files:
    model_vo_diffusion, cfg_vo_diffusion = load_model(model_vo_diffusion_results_directory, model, device=device)
    
    # Get the epoch number from the model name
    iteration_num = model.split(".")[0].split('_')[1]
    iteration_num = int(iteration_num)
    epoch_num = int(iteration_num * batch_size / 46402)
    
    # print("model.append ndc", model_vo_diffusion.append_ndc) # false for rec
    
    for seq in seqs:
        vo_forward_pred = []
        rays_pred = []
        gt_vo_transformation_matrix =[]
        
        dataset = Co3dDataset(num_images=2, id_consecutive=True, category = [seq]) 
        instances = np.arange(0, int(len(dataset)/2-1))
        
        with torch.no_grad():
            for counter, idx in enumerate(instances):
                
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
                            num_patches_x=45,
                            num_patches_y=13,
                            use_regression=True, 
                            max_num_images=2,
                            return_rays=True,
                            calculate_intrinsics=False, # use known intrinsics
                            scale = scale,
                            cameras = cameras_intrinsic,
                        )
                """ print("pred", pred)
                print("pred[0]", pred[0]) # PerspectiveCameras()
                print("pred[1]", pred[1]) # rays
                print("pred[2]", pred[2]) # []
                print("pred[3]", pred[3]) # [] """
                pred_rays = pred[1][1]
                rays_pred.append(pred_rays)
                
                pred_cameras = pred[0]
                R_pred = pred_cameras.R[1].cpu().numpy()
                t_pred = pred_cameras.T[1].cpu().numpy()
                vo_forward_pred.append(generate_transformation_matrix(R_pred, t_pred))
        
        vo_forward_pred = np.array(vo_forward_pred)
        rays_pred = np.array(rays_pred)
        gt_vo_transformation_matrix = np.array(gt_vo_transformation_matrix)
        
        pre_traj_from_known_intrinsics = [np.eye(4)]
        for i in range(0, vo_forward_pred.shape[0]):
            dT = vo_forward_pred[i]
            new_est = np.linalg.inv(np.dot(dT,np.linalg.inv(pre_traj_from_known_intrinsics[i]))) # the estimation is inv
            pre_traj_from_known_intrinsics.append(new_est)
        pre_traj_from_known_intrinsics = np.array(pre_traj_from_known_intrinsics)
        
        gt_traj = [np.eye(4)] # odometry is inverse pose from c to w/p
        for i in range(0, np.array(gt_vo_transformation_matrix).shape[0]):
            
            dT = gt_vo_transformation_matrix[i]
            new_est = np.linalg.inv(np.dot(dT,np.linalg.inv(gt_traj[i])))
            gt_traj.append(new_est)
        gt_traj = np.array(gt_traj)
        
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # save the rays
        rays = {"seq":seq, "rays_pred":rays_pred}
        save_obj(rays, osp.join(model_vo_diffusion_inference_prediced_rays, f"seq-{seq_name}-epoch-{epoch_num}-ditmodel_regression_train_all_{timestamp}"))
        
        # save the predicted poses
        data = {"seq":seq,"vo_forward_estimated_poses_from_known_intrinsic":vo_forward_pred,"est_traj_from_known_intrinsic": pre_traj_from_known_intrinsics, "gt_traj": gt_traj}
        
        save_obj(data, osp.join(model_vo_diffusion_inference_prediced_poses, f"seq-{seq_name}-epoch-{epoch_num}-ditmodel_regression_train_all_{timestamp}"))
        
        est_traj_from_known_intrinsic, gt_traj, errors_known_intrinsics, cum_dist = compute_trajectory(vo_forward_pred, pre_traj_from_known_intrinsics, gt_traj, method='RT_from_known_intrinsic', compute_seg_err=True)
        
        if seq == '09':
            logger_09_all.log(seq, epoch_num,'RT_from_known_intrinsic', errors_known_intrinsics[0], errors_known_intrinsics[1], errors_known_intrinsics[2], errors_known_intrinsics[3])
            
        elif seq == '10':
            logger_10_all.log(seq, epoch_num,'RT_from_known_intrinsic', errors_known_intrinsics[0], errors_known_intrinsics[1], errors_known_intrinsics[2], errors_known_intrinsics[3])
        ### new
        
        plt.figure()
        plt.grid()
        plt.plot(est_traj_from_known_intrinsic[:,0,3], est_traj_from_known_intrinsic[:,2,3], linewidth=1.5, color='blue', label='known_intrinsic')
        plt.plot(gt_traj[:,0,3], gt_traj[:,2,3], linewidth=1.5, color='red', label='gt')
        plt.legend()
        plt.title(f'Topdown (XY) Trajectory {seq_name}')
        # Save the plot with the timestamp in the filename
        output_path = osp.join(model_vo_diffusion_inference_traj_images, f'seq-{seq_name}-epoch-{epoch_num}-ditmodel_regression_train_all_{timestamp}.png')
        plt.savefig(output_path)
        # plt.show()
        plt.close()
    
    


