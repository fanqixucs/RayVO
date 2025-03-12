"""
Note that batch_size refers to the batch_size per GPU.

accelerate launch train.py training.batch_size=8 training.max_iterations=450000
"""

import datetime
import os
import random
import socket
import time
from glob import glob

import hydra
import ipdb  # noqa: F401
import numpy as np
import omegaconf
import torch
# torch.backends.cuda.preferred_linalg_library('cusolver')
import wandb
from accelerate import Accelerator
from pytorch3d.renderer import PerspectiveCameras

from ray_diffusion.dataset.co3d_v2_kitti_real_rec_inverse_forward import Co3dDataset
from ray_diffusion.model.diffuser import RayDiffuser
from ray_diffusion.model.scheduler import NoiseScheduler
from ray_diffusion.utils.normalize import normalize_cameras_batch
from ray_diffusion.utils.rays import cameras_to_rays_for_rec, rays_to_cameras_for_rec, Rays
from ray_diffusion.utils.visualization import (
    create_training_visualizations,
    create_plotly_cameras_visualization,
)
from liegroups import SE3

# os.umask(000)  # Default to 777 permissions
# os.environ["CUDA_VISIBLE_DEVICES"] = str(1)
# torch.cuda.device_count()  # print 1

class Trainer(object):
    def __init__(self, cfg):
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.cfg = cfg
        self.debug = cfg.debug
        self.resume = cfg.training.resume
        self.pretrain_path = cfg.training.pretrain_path

        self.batch_size = cfg.training.batch_size
        self.max_iterations = cfg.training.max_iterations
        self.mixed_precision = cfg.training.mixed_precision
        # self.interval_visualize = cfg.training.interval_visualize
        self.interval_save_checkpoint = cfg.training.interval_save_checkpoint
        # self.interval_delete_checkpoint = cfg.training.interval_delete_checkpoint
        # self.interval_evaluate = cfg.training.interval_evaluate
        # self.delete_all = cfg.training.delete_all_checkpoints_after_training
        self.freeze_encoder = cfg.training.freeze_encoder
        # self.translation_scale = cfg.training.translation_scale
        self.num_visualize = 2
        self.regression = cfg.training.regression
        # self.load_extra_cameras = cfg.training.load_extra_cameras
        # self.calculate_intrinsics = cfg.training.calculate_intrinsics
        # self.normalize_first_camera = cfg.training.normalize_first_camera

        self.model_type = cfg.model.model_type
        self.pred_x0 = cfg.model.pred_x0
        self.num_patches_x = cfg.model.num_patches_x
        self.num_patches_y = cfg.model.num_patches_y
        self.depth = cfg.model.depth
        self.num_images = cfg.model.num_images
        self.random_num_images = cfg.model.random_num_images
        self.feature_extractor = cfg.model.feature_extractor
        self.append_ndc = cfg.model.append_ndc
        self.image_dino_scale = cfg.model.image_dino_scale
        self.image_width = cfg.model.image_width
        self.image_height = cfg.model.image_height

        self.dataset_name = cfg.dataset.name
        self.category = cfg.dataset.category
        self.apply_augmentation = cfg.dataset.apply_augmentation

        if self.regression:
            assert self.pred_x0

        self.start_time = None
        self.iteration = 0
        self.epoch = 0
        self.wandb_id = None
        self.hostname = socket.gethostname()

        self.accelerator = Accelerator(
            even_batches=False,
            device_placement=False,
        )
        self.device = self.accelerator.device

        scheduler = NoiseScheduler(
            type=cfg.noise_scheduler.type,
            max_timesteps=cfg.noise_scheduler.max_timesteps,
            beta_start=cfg.noise_scheduler.beta_start,
            beta_end=cfg.noise_scheduler.beta_end,
        )

        self.model = RayDiffuser(
            depth=self.depth,
            height=self.num_patches_y,  
            width=self.num_patches_x,
            P=1,
            max_num_images=self.num_images,
            noise_scheduler=scheduler,
            freeze_encoder=self.freeze_encoder,
            feature_extractor=self.feature_extractor,
            append_ndc=self.append_ndc,
            img_width=self.image_width,
            img_height=self.image_height,
        ).to(self.device)
        self.lr = 1e-4
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.gradscaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        self.dataset = Co3dDataset(
            # category=self.category,
            split="train",
            num_images=self.num_images,
            image_width=self.image_width,
            image_height=self.image_height,
        )
        self.train_dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        self.model, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader
        )

        self.output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        if self.accelerator.is_main_process:
            """ name = os.path.basename(self.output_dir)
            name += f"_{self.category}"
            name += f"_{self.model_type}"
            name += f"_B{self.batch_size * self.accelerator.num_processes}"
            name += f"_N{self.num_images}"
            if self.random_num_images:
                name += "Rand"
            name += f"_D{self.depth}"
            name += f"_LR{self.lr}"
            name += f"_T{scheduler.max_timesteps}"
            if self.num_patches_x != 16 or self.num_patches_y != 16:
                name += f"_P{self.num_patches_x}x{self.num_patches_y}"
            if self.mixed_precision:
                name += "_AMP"
            if self.pred_x0:
                name += "_predX0"
                if self.regression:
                    name += "reg"
            if not self.freeze_encoder:
                name += "_FTEnc"
            if self.pretrain_path != "":
                name += "_Pretrained"
            else:
                if self.feature_extractor != "dino":
                    name += f"_{self.feature_extractor}"
            if self.normalize_first_camera:
                name += "_NormFirst" """
            name = cfg.training.name
            print("Output dir:", self.output_dir)
            with open(os.path.join(self.output_dir, name), "w"):
                # Create empty tag with name
                pass
            self.name = name

            conf_dict = omegaconf.OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            )
            conf_dict["output_dir"] = self.output_dir
            conf_dict["hostname"] = self.hostname

        if self.resume:
            checkpoint_files = sorted(glob(os.path.join(self.checkpoint_dir, "*.pth")))
            last_checkpoint = checkpoint_files[-1]
            print("Resuming from checkpoint:", last_checkpoint)
            self.load_model(last_checkpoint, load_metadata=True)
        elif self.pretrain_path != "":
            print("Loading pretrained model:", self.pretrain_path)
            self.load_model(self.pretrain_path, load_metadata=False)

        if self.accelerator.is_main_process:
            mode = "online" if cfg.debug.wandb else "disabled"
            if self.wandb_id is None:
                self.wandb_id = wandb.util.generate_id()
            self.wandb_run = wandb.init(
                mode=mode,
                name=name,
                project=cfg.debug.project_name,
                config=conf_dict,
                resume=self.resume,
                id=self.wandb_id,
            )
            wandb.define_metric("iteration")
            noise_schedule = self.get_module().noise_scheduler.plot_schedule(
                return_image=True
            )
            wandb.log(
                {"Schedule": wandb.Image(noise_schedule, caption="Noise Schedule")}
            )

    def get_module(self):
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            return self.model.module
        else:
            return self.model

    def train(self):
        while self.iteration < self.max_iterations:
            for batch in self.train_dataloader:
                t0 = time.time()
                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                    images = batch["image"].to(self.device)
                    focal_lengths = batch["focal_length"].to(self.device)
                    # crop_params = batch["crop_parameters"].to(self.device)
                    principal_points = batch["principal_point"].to(self.device)
                    # print("R", batch["odometry_R"])
                    R = batch["odometry_R"].permute(0,1,3,2).to(self.device)
                    # print("R", R)
                    T = batch["odometry_T"].to(self.device)

                    # print("focal_length", focal_lengths)
                    # print("principal_point", principal_points)
                    # print("images.shape", images.shape) # torch.Size([8, 2, 3, 182, 630])
                    # the dataset sets the focal length to negtive already
                    cameras_og = [
                        PerspectiveCameras(
                            focal_length=focal_lengths[b],
                            principal_point=principal_points[b],
                            R=R[b],
                            T=T[b],
                            device=self.device,
                        )
                        for b in range(self.batch_size)
                    ]
                    cameras = cameras_og
                    """ if self.num_images == 1:
                        cameras = cameras_og
                    else:
                        cameras, _ = normalize_cameras_batch(
                            cameras=cameras_og,
                            scale=self.translation_scale,
                            normalize_first_camera=self.normalize_first_camera,
                        ) """
                    """ # Now that cameras are normalized, fix shapes of camera parameters
                    if self.load_extra_cameras or self.random_num_images:
                        if self.random_num_images:
                            num_images = torch.randint(2, self.num_images + 1, (1,))
                        else:
                            num_images = self.num_images

                        # The correct number of images is already loaded.
                        # Only need to modify these camera parameters shapes.
                        focal_lengths = focal_lengths[:, :num_images]
                        crop_params = crop_params[:, :num_images]
                        R = R[:, :num_images]
                        T = T[:, :num_images]
                        images = images[:, :num_images]

                        cameras = [
                            PerspectiveCameras(
                                focal_length=cameras[b].focal_length[:num_images],
                                principal_point=cameras[b].principal_point[:num_images],
                                R=cameras[b].R[:num_images],
                                T=cameras[b].T[:num_images],
                                device=self.device,
                            )
                            for b in range(self.batch_size)
                        ] """

                    if self.regression:
                        low = self.get_module().noise_scheduler.max_timesteps - 1
                    else:
                        low = 0

                    t = torch.randint(
                        low=low,
                        high=self.get_module().noise_scheduler.max_timesteps,
                        size=(self.batch_size,),
                        device=self.device,
                    )

                    rays = []
                    scale = self.image_dino_scale
                    for camera in cameras:
                        r=  cameras_to_rays_for_rec(   
                            cameras=camera, # [0] means the first camera pair
                            num_patches_x=self.num_patches_x,
                            num_patches_y=self.num_patches_y,
                            scale = scale,
                            crop_parameters=None,
                        )
                        rays.append(
                            r.to_spatial(include_ndc_coordinates=self.append_ndc, num_patches_x=self.num_patches_x, num_patches_y=self.num_patches_y)
                        )
                        
                        # print("rays.shape", r.shape) # [2, 585, 6]

                    rays_tensor = torch.stack(rays, dim=0)

                    if self.append_ndc:
                        ndc_coordinates = rays_tensor[..., -2:, :, :]
                        rays_tensor = rays_tensor[..., :-2, :, :]  # (B, N, 6, H, W)
                    else:
                        ndc_coordinates = None


                    eps_pred, eps = self.model(
                        images=images,
                        rays=rays_tensor,
                        t=t,
                        ndc_coordinates=ndc_coordinates,
                    )


                    if self.pred_x0:
                        target = rays_tensor
                    else:
                        target = eps
                        
                    """ print("eps_pred.shape", eps_pred.shape) # torch.Size([8, 2, 6, 16, 16])
                    print("eps_pred", eps_pred)
                    print("target.shape", target.shape) # torch.Size([8, 2, 6, 16, 16])
                    print("target", target)
                    print("eps_pred - target  shape", (eps_pred - target).shape) # torch.Size([8, 2, 6, 16, 16])
                    print("eps_pred - target", eps_pred - target)
                    print("(eps_pred - target) ** 2 shape", ((eps_pred - target) ** 2).shape) # torch.Size([8, 2, 6, 16, 16])
                    print("(eps_pred - target) ** 2", (eps_pred - target) ** 2)
                    square = ((eps_pred - target) ** 2) # # torch.Size([8, 2, 6, 16, 16])
                    print("shape of the first pose's square error", square[:, 0, :, :, :].shape)
                    print("mean of the first pose's square error", torch.mean(square[:, 0, :, :, :]))
                    print("shape of the first pose's square error of direction 0", square[:, 0, 0:3, :, :].shape)
                    print("mean of the first pose's square error of direction 0", torch.mean(square[:, 0, 0:3, :, :]))
                    print("shape of the first pose's square error of moments", square[:, 0, 3:6, :, :].shape)
                    print("mean of the first pose's square error of moments", torch.mean(square[:, 0, 3:6, :, :]))
                    print("====================================================================================================")
                    print("shape of the second pose", square[:, 1, :, :, :].shape)
                    print("mean of the second pose", torch.mean(square[:, 1, :, :, :]))
                    print("mean of the second pose of direction 0", torch.mean(square[:, 1, 0:3, :, :]))
                    print("mean of the second pose of moments", torch.mean(square[:, 1, 3:6, :, :])) """
                    
                    # square = ((eps_pred - target) ** 2)
                    # loss = 0.3 * torch.mean(square[:, 1, 0:3, :, :]) + 0.7 * torch.mean(square[:, 1, 3:6, :, :])
                    # Ensure eps_pred and target require gradients
                    # loss = self.lie_loss(eps_pred, target, cameras).cuda().requires_grad_(True)
                    loss = 0.5 * torch.mean((eps_pred - target) ** 2) + 0.5 * self.lie_loss(eps_pred, target, cameras)
                    
                    # loss = torch.mean((eps_pred - target) ** 2)
                    # print("loss= torch.mean((eps_pred - target) ** 2)", torch.mean((eps_pred - target) ** 2))
                    

                if self.mixed_precision:
                    self.gradscaler.scale(loss).backward()
                    self.gradscaler.step(self.optimizer)
                    self.gradscaler.update()
                else:
                    self.accelerator.backward(loss)
                    self.optimizer.step()

                if self.accelerator.is_main_process:
                    if self.iteration % 10 == 0:
                        self.log_info(loss, t0)

                    # if self.iteration % self.interval_visualize == 0:
                    #     self.visualize(
                    #         images=images,
                    #         cameras_gt=cameras,
                    #         crop_parameters=None,
                    #     )

                if self.accelerator.is_main_process:
                    if self.iteration % self.interval_save_checkpoint == 0:
                        self.save_model()

                    """ if self.iteration % self.interval_delete_checkpoint == 0:
                        self.clear_old_checkpoints(self.checkpoint_dir) """

                    # if (
                    #     self.iteration % self.interval_evaluate == 0
                    #     and self.iteration > 0
                    # ):
                    #     self.evaluate_train_acc()

                    """ if self.iteration >= self.max_iterations + 1:
                        if self.delete_all:
                            self.clear_old_checkpoints(
                                self.checkpoint_dir, clear_all_old=True
                            )
                        return """
                self.iteration += 1
            self.epoch += 1

    def lie_loss(self, eps_pred, target, cameras):
        # print("eps_pred.shape", Rays.from_spatial(eps_pred).shape) # torch.Size([8, 2, 585, 6])
        # print(Rays.from_spatial(eps_pred)[0].shape) # torch.Size([2, 585, 6])
        # print(len(cameras)) #8
        all_lie_algebras_pred = []
        for cam_idx in range(len(cameras)):
            pred_cam = rays_to_cameras_for_rec(
                Rays.from_spatial(eps_pred)[cam_idx], # [0] is the rays, the rest are ndc, etc
                crop_parameters = None,
                num_patches_x=self.num_patches_x,
                num_patches_y=self.num_patches_y,
                scale = self.image_dino_scale,
                cameras = cameras[cam_idx],
            )
            
            transform_matrix = self.generate_transformation_matrix_batch(pred_cam.R.cpu().detach().numpy(), pred_cam.T.cpu().detach().numpy())
            # transform_matrix = self.generate_transformation_matrix_batch(pred_cam.R, pred_cam.T)
            # Initialize a list to store the Lie algebra representations
            lie_algebras = []

            # Loop through each transformation matrix and compute its Lie algebra representation
            for i in range(transform_matrix.shape[0]):
                lie_alg = self.generate_lie_algebra(transform_matrix[i])
                lie_algebras.append(lie_alg)

            # Convert the list of Lie algebra representations to a tensor
            # print("lie_algebras", lie_algebras)
            lie_algebras_tensor = torch.tensor(lie_algebras)
            all_lie_algebras_pred.append(lie_algebras_tensor)
            # print("lie_algebras_tensor.shape", lie_algebras_tensor.shape)  # (2, 6)
            # print("pred_cam.shape", pred_cam[0].R.shape) #torch.Size([1, 3, 3])
            # print("pred_cam.shape", pred_cam.R.shape) #torch.Size([2, 3, 3])
        # Stack all Lie algebra representations to form a tensor with shape [B, 2, 6]
        all_lie_algebras_pred_tensor = torch.stack(all_lie_algebras_pred)
        all_lie_algebras_pred_tensor = all_lie_algebras_pred_tensor.to(pred_cam.R.device)
        # print("all_lie_algebras_tensor.shape", all_lie_algebras_tensor.shape)  # (B, 2, 6)
        
        all_lie_algebras_target = []
        for cam_idx in range(len(cameras)):
            target_cam = rays_to_cameras_for_rec(
                Rays.from_spatial(target)[cam_idx], # [0] is the rays, the rest are ndc, etc
                crop_parameters = None,
                num_patches_x=self.num_patches_x,
                num_patches_y=self.num_patches_y,
                scale = self.image_dino_scale,
                cameras = cameras[cam_idx],
            )
            
            transform_matrix = self.generate_transformation_matrix_batch(target_cam.R.cpu().detach().numpy(), target_cam.T.cpu().detach().numpy())
            # Initialize a list to store the Lie algebra representations
            lie_algebras = []

            # Loop through each transformation matrix and compute its Lie algebra representation
            for i in range(transform_matrix.shape[0]):
                lie_alg = self.generate_lie_algebra(transform_matrix[i])
                lie_algebras.append(lie_alg)

            # Convert the list of Lie algebra representations to a tensor
            lie_algebras_tensor = torch.tensor(lie_algebras)
            all_lie_algebras_target.append(lie_algebras_tensor)
            # print("lie_algebras_tensor.shape", lie_algebras_tensor.shape)  # (2, 6)
            # print("target_cam.shape", target_cam[0].R.shape) #torch.Size([1, 3, 3])
            # print("target_cam.shape", target_cam.R.shape) #torch.Size([2, 3, 3])
        # Stack all Lie algebra representations to form a tensor with shape [B, 2, 6]
        all_lie_algebras_target_tensor = torch.stack(all_lie_algebras_target)
        all_lie_algebras_target_tensor = all_lie_algebras_target_tensor.to(target_cam.R.device)
        # print("all_lie_algebras_target_tensor.shape", all_lie_algebras_target_tensor.shape)  # (B, 2, 6)
        
        # Compute the angle loss (first three elements [0:3])
        angle_loss = torch.nn.functional.mse_loss(all_lie_algebras_pred_tensor[:, :, :3], all_lie_algebras_target_tensor[:, :, :3])

        # Compute the translation loss (last three elements [3:6])
        translation_loss = torch.nn.functional.mse_loss(all_lie_algebras_pred_tensor[:, :, 3:], all_lie_algebras_target_tensor[:, :, 3:])

        # Compute the total loss with different weights
        # lie_alg_loss = angle_loss + 100 * translation_loss
        lie_alg_loss = angle_loss + translation_loss

        # print("angle_loss:", angle_loss.item())
        # print("translation_loss:", translation_loss.item())
        # print("total_loss:", lie_alg_loss.item())
        
        return lie_alg_loss
        
    
    def generate_lie_algebra(self, matrix):
        
        lie_algebra = SE3.from_matrix(matrix,normalize=True).log()
        
        return lie_algebra
    
    
    
    
    def generate_transformation_matrix_batch(self, R, t):
        """
        Generate a batch of 4x4 transformation matrices from batches of rotation matrices and translation vectors.

        Args:
            R (numpy.ndarray): Batch of 3x3 rotation matrices with shape (N, 3, 3).
            t (numpy.ndarray): Batch of 3x1 translation vectors with shape (N, 3).

        Returns:
            numpy.ndarray: Batch of 4x4 transformation matrices with shape (N, 4, 4).
        """
        # Number of matrices in the batch
        N = R.shape[0]
        
        # Initialize a batch of 4x4 identity matrices
        transformation_matrices = np.eye(4).reshape(1, 4, 4).repeat(N, axis=0)
        
        # Set the top-left 3x3 part to the rotation matrices R
        transformation_matrices[:, :3, :3] = R
        
        # Set the top-right 3x1 part to the translation vectors t
        transformation_matrices[:, :3, 3] = t
    
        return transformation_matrices
    '''
    def generate_transformation_matrix_batch(self, R, t):
        """
        Generate a batch of 4x4 transformation matrices from batches of rotation matrices and translation vectors.

        Args:
            R (torch.Tensor): Batch of 3x3 rotation matrices with shape (N, 3, 3).
            t (torch.Tensor): Batch of 3x1 translation vectors with shape (N, 3).

        Returns:
            torch.Tensor: Batch of 4x4 transformation matrices with shape (N, 4, 4).
        """
        # Number of matrices in the batch
        N = R.shape[0]
        
        # Initialize a batch of 4x4 identity matrices on the same device as R
        transformation_matrices = torch.eye(4, device=R.device).unsqueeze(0).repeat(N, 1, 1)
        
        # Set the top-left 3x3 part to the rotation matrices R
        transformation_matrices[:, :3, :3] = R
        
        # Set the top-right 3x1 part to the translation vectors t
        transformation_matrices[:, :3, 3] = t
        
        return transformation_matrices
    '''
    

    def generate_transformation_matrix(self, R, t):
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
    
    
    def load_model(self, path, load_metadata=True):
        save_dict = torch.load(path, map_location=self.device)
        checkpoint_state_dict = save_dict['state_dict']
        
        # copying a param with shape torch.Size([1, 8, 1152]) from checkpoint, the shape in current model is torch.Size([1, 2, 1152])
        mismatched_param = checkpoint_state_dict['ray_predictor.x_pos_enc.image_pos_table']
        adjusted_param = mismatched_param[:, :self.num_images, :]
        checkpoint_state_dict['ray_predictor.x_pos_enc.image_pos_table'] = adjusted_param
        
        # copying a param with shape torch.Size([1152, 392, 1, 1]) from checkpoint, the shape in current model is torch.Size([1152, 390, 1, 1])
        mismatched_param_ndc = checkpoint_state_dict['ray_predictor.x_embedder.proj.weight']
        adjusted_param_ndc = mismatched_param_ndc[:, :390, :, :]
        checkpoint_state_dict['ray_predictor.x_embedder.proj.weight'] = adjusted_param_ndc
        
        # copying a param with shape torch.Size([1, 256, 1152]) from checkpoint, the shape in current model is torch.Size([1, 585, 1152]
        mismatched_param_patches = checkpoint_state_dict['ray_predictor.x_pos_enc.token_pos_table']
        import torch.nn.functional as F
        adjusted_param_patches = F.interpolate(mismatched_param_patches.unsqueeze(0), size = (self.num_patches_x * self.num_patches_y,1152), mode='nearest').squeeze(0)
        checkpoint_state_dict['ray_predictor.x_pos_enc.token_pos_table'] = adjusted_param_patches
        
        # Filter out the feature_extractor parameters
        if self.feature_extractor == "dino":
            print("Filtering out the feature extractor parameters from dinov2 -------------------------")
            filtered_state_dict = {k: v for k, v in save_dict["state_dict"].items() if not k.startswith('feature_extractor')}

            # Load the filtered state dictionary into the model
            missing, unexpected = self.get_module().load_state_dict(filtered_state_dict, strict=False)
        else:
            missing, unexpected = self.get_module().load_state_dict(
            save_dict["state_dict"],
            strict=False,
        )
        print(f"Missing keys: {missing}")
        print(f"Unexpected keys: {unexpected}")
        if load_metadata:
            self.iteration = save_dict["iteration"]
            self.epoch = save_dict["epoch"]
            time_elapsed = save_dict["elapsed"]
            self.start_time = time.time() - time_elapsed
            if "wandb_id" in save_dict:
                self.wandb_id = save_dict["wandb_id"]
            self.optimizer.load_state_dict(save_dict["optimizer"])
            self.gradscaler.load_state_dict(save_dict["gradscaler"])
        
        # print("Freezing all parameters except the VO model, the final layer==================================")
        # for name, param in self.model.named_parameters():
        #     if not name.startswith("ray_predictor.final_layer"):
        #         param.requires_grad = False
        # print("Freezing all parameters, the final layer==================================")
        # for name, param in self.model.named_parameters():
        #     param.requires_grad = False
        

    def save_model(self):
        path = os.path.join(self.checkpoint_dir, f"ckpt_{self.iteration:08d}.pth")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        elapsed = time.time() - self.start_time if self.start_time is not None else 0
        save_dict = {
            "epoch": self.epoch,
            "elapsed": elapsed,
            "gradscaler": self.gradscaler.state_dict(),
            "iteration": self.iteration,
            "state_dict": self.get_module().state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "wandb_id": self.wandb_id,
        }
        torch.save(save_dict, path)

    def clear_old_checkpoints(self, checkpoint_dir, clear_all_old=False):
        print("Clearing old checkpoints")
        checkpoint_files = sorted(glob(os.path.join(checkpoint_dir, "ckpt_*.pth")))
        if clear_all_old:
            for checkpoint_file in checkpoint_files[:-1]:
                os.remove(checkpoint_file)
        else:
            for checkpoint_file in checkpoint_files:
                checkpoint = os.path.basename(checkpoint_file)
                checkpoint_iteration = int("".join(filter(str.isdigit, checkpoint)))
                if checkpoint_iteration % self.interval_delete_checkpoint != 0:
                    os.remove(checkpoint_file)

    def log_info(self, loss, t0):
        if self.start_time is None:
            self.start_time = time.time()
        time_elapsed = round(time.time() - self.start_time)
        time_remaining = round(
            (time.time() - self.start_time)
            / (self.iteration + 1)
            * (self.max_iterations - self.iteration)
        )
        disp = [
            f"Iter: {self.iteration}/{self.max_iterations}",
            f"Epoch: {self.epoch}",
            f"Loss: {loss.item():.8f}",
            # f"Elap: {str(datetime.timedelta(seconds=time_elapsed))}",
            f"Rem: {str(datetime.timedelta(seconds=time_remaining))}",
            # self.hostname,
            self.name,
        ]
        print(", ".join(disp), flush=True)
        wandb.log(
            {
                "loss": loss.item(),
                "iter_time": time.time() - t0,
                "lr": self.lr,
                "iteration": self.iteration,
                "hours_remaining": time_remaining / 3600,
            }
        )

    def visualize(self, images, cameras_gt, crop_parameters=None):
        self.get_module().eval()
        for camera in cameras_gt:
            # AMP may not cast back to float
            camera.R = camera.R.float()
            camera.T = camera.T.float()

        vis_images, cameras_pred_batched = create_training_visualizations(
            model=self.get_module(),
            images=images[: self.num_visualize],
            device=self.device,
            cameras_gt=cameras_gt,
            pred_x0=self.pred_x0,
            num_images=images.shape[1],
            crop_parameters=crop_parameters,
            visualize_pred=self.regression,
            return_first=self.regression,
            calculate_intrinsics=self.calculate_intrinsics,
        )

        for i, cameras_pred in enumerate(cameras_pred_batched):
            fig = create_plotly_cameras_visualization(cameras_gt[i], cameras_pred, i)
            plot = wandb.Plotly(fig)
            wandb.log({f"Vis {i} plotly": plot})

        for i, vis_image in enumerate(vis_images):
            im = wandb.Image(
                vis_image, caption=f"iteration {self.iteration} example {i}"
            )
            wandb.log({f"Vis {i}": im})
        self.get_module().train()


@hydra.main(config_path="./conf", config_name="config", version_base="1.3")
def main(cfg):
    print(cfg)
    torch.autograd.set_detect_anomaly(cfg.debug.anomaly_detection)
    torch.set_float32_matmul_precision(cfg.training.matmul_precision)
    trainer = Trainer(cfg=cfg)
    trainer.train()


if __name__ == "__main__":
    main()
