import ipdb  # noqa: F401
import numpy as np
import torch
import torch.nn as nn

from ray_diffusion.model.dit import DiT
from ray_diffusion.model.feature_extractors import SpatialDino
from ray_diffusion.model.scheduler import NoiseScheduler


class RayDiffuser(nn.Module):
    def __init__(
        self,
        model_type="dit",
        depth=8,
        width=45,
        height=13,
        hidden_size=1152,
        P=1,
        max_num_images=1,
        noise_scheduler=None,
        freeze_encoder=True,
        feature_extractor="dinov2",
        append_ndc=True,
        use_unconditional=False,
        img_width = 448,
        img_height = 114,
    ):
        super().__init__()
        if noise_scheduler is None:
            self.noise_scheduler = NoiseScheduler()
        else:
            self.noise_scheduler = noise_scheduler

        self.ray_dim = 6

        self.append_ndc = append_ndc
        self.width = width
        self.height = height
        self.img_width = img_width
        self.img_height = img_height
        
        self.max_num_images = max_num_images
        self.model_type = model_type
        self.use_unconditional = use_unconditional

        if feature_extractor == "dinov2":
            self.feature_extractor = SpatialDino(
                freeze_weights=freeze_encoder, num_patches_x=width, num_patches_y=height, model_type="dinov2", img_width = img_width, img_height = img_height
            )
            self.feature_dim = self.feature_extractor.feature_dim
        elif feature_extractor == "dino":
            self.feature_extractor = SpatialDino(
                freeze_weights=freeze_encoder, num_patches_x=width, num_patches_y=height, model_type='dino',img_width = img_width, img_height = img_height
            )
            self.feature_dim = self.feature_extractor.feature_dim
        else:
            raise Exception(f"Unknown feature extractor {feature_extractor}")

        if self.use_unconditional:
            self.register_parameter(
                "null_token", nn.Parameter(torch.randn(self.feature_dim, 1, 1))
            )

        self.input_dim = self.ray_dim + self.feature_dim
        if self.append_ndc:
            self.input_dim += 2

        if model_type == "dit":
            self.ray_predictor = DiT(
                in_channels=self.input_dim,
                out_channels=self.ray_dim,
                width=width,
                height=height,
                depth=depth,
                hidden_size=hidden_size,
                max_num_images=max_num_images,
                P=P,
            )
        else:
            raise Exception(f"Unknown model type {model_type}")

    def forward_noise(self, x, t, epsilon=None, mask=None):
        """
        Applies forward diffusion (adds noise) to the input.

        If a mask is provided, the noise is only applied to the masked inputs.
        """
        t = t.reshape(-1, 1, 1, 1, 1)
        
        if epsilon is None:
            epsilon = torch.randn_like(x) # standard normal distribution    
            # print("epsilon shape", epsilon.shape)   
        else:
            epsilon = epsilon.reshape(x.shape)
        
        alpha_bar = self.noise_scheduler.alphas_cumprod[t]
        x_noise = torch.sqrt(alpha_bar) * x + torch.sqrt(1 - alpha_bar) * epsilon
        
        if mask is not None: # not done here
            # print("mask", mask)
            x_noise = x_noise * mask + x * (1 - mask)
        return x_noise, epsilon

    def forward(
        self,
        features=None,
        images=None,
        rays=None,
        rays_noisy=None,
        t=None,
        mask=None,
        ndc_coordinates=None,
        unconditional_mask=None,
        compute_x0=False,
    ):
        """
        Args:
            images: (B, N, 3, H, W).
            t: (B,).
            rays: (B, N, 6, H, W).
            rays_noisy: (B, N, 6, H, W).
            ndc_coordinates: (B, N, 2, H, W).
            unconditional_mask: (B, N) or (B,). Should be 1 for unconditional samples
                and 0 else.
        """

        if features is None:
            features = self.feature_extractor(images, autoresize=False)

        # print("features.shape", features.shape) # [8, 2, 384, 13, 45]
        # print("features", features)
        
        B = features.shape[0]
        
        # it is not done here
        if unconditional_mask is not None and self.use_unconditional:
            # print("unconditional_mask", unconditional_mask)
            null_token = self.null_token.reshape(1, 1, self.feature_dim, 1, 1)
            unconditional_mask = unconditional_mask.reshape(B, -1, 1, 1, 1)
            features = (
                features * (1 - unconditional_mask) + null_token * unconditional_mask
            )

        if isinstance(t, int) or isinstance(t, np.int64): # not done here
            # print("t is int")
            t = torch.ones(1, dtype=int).to(features.device) * t
        else:
            # print("t is not int")
            t = t.reshape(B) # when B is 3 then t is [99, 99, 99]

        # print("rays", rays)
        if rays_noisy is None:
            # print("rays_noisy is None")
            rays_noisy, epsilon = self.forward_noise(rays, t, mask=mask)
        else:
            epsilon = None
            
        # print("rays_noisy shape", rays_noisy.shape) # torch.Size([8, 2, 6, 13, 45])

        scene_features = torch.cat([features, rays_noisy], dim=2) # torch.Size([8, 2, 6, 13, 45])
        if self.append_ndc:
            # print("ndc_coordinates shape", ndc_coordinates.shape)
            scene_features = torch.cat([scene_features, ndc_coordinates], dim=2)
        
        # print("scene_features shape", scene_features.shape) # torch.Size([2, 2, 390, 13, 45])
        # print("t for the input of ray_predictor", t)
        epsilon_pred = self.ray_predictor(scene_features, t)

        if compute_x0: # nothing is done here
            # print("compute_x0")
            t = t.reshape(-1, 1, 1, 1, 1)
            a = self.noise_scheduler.alphas_cumprod[t]
            x0 = (rays_noisy - torch.sqrt(1 - a) * epsilon_pred) / torch.sqrt(a)
            return epsilon_pred, x0
        return epsilon_pred, epsilon
