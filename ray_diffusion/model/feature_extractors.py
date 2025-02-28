import ipdb  # noqa: F401
import torch
import torch.nn as nn


def resize(image, size=None, scale_factor=None):
    return nn.functional.interpolate(
        image,
        size=size,
        scale_factor=scale_factor,
        mode="bilinear",
        align_corners=False,
    )


class SpatialDino(nn.Module):
    def __init__(
        self,
        freeze_weights=True,
        model_type="dinov2",
        num_patches_x=16,
        num_patches_y=16,
        img_width = 630,
        img_height = 182,
    ):
        super().__init__()
        """ self.model = torch.hub.load("facebookresearch/dinov2", model_type)
        self.feature_dim = self.model.embed_dim
        self.num_patches_x = num_patches_x
        self.num_patches_y = num_patches_y """
        
        self.model_type = model_type    
        if self.model_type == "dinov2":
            print('loading dinov2=========================')
            self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
            self.patch_size = 14
        elif self.model_type == "dino":
            print('loading dino=========================')
            self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
            self.patch_size = 16
        else:
            raise Exception(f"Unknown model type for feature extractor")
        self.feature_dim = self.model.embed_dim
        self.width = img_width
        self.height = img_height
        # print("self.width", self.width) # 630
        # print("self.height", self.height)   #182
        # self.num_patches_x = self.width // self.patch_size
        # self.num_patches_y = self.height // self.patch_size
        self.num_patches_x = num_patches_x
        self.num_patches_y = num_patches_y
        
        if freeze_weights:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x, autoresize=False):
        """
        Spatial dimensions of output will be H // 14, W // 14. If autoresize is True,
        then the output will be resized to the correct dimensions.

        Args:
            x (torch.Tensor): Images (B, C, H, W). Should be ImageNet normalized.
            autoresize (bool): Whether to resize the input to match the num_patch
                dimensions.

        Returns:
            feature_map (torch.tensor): (B, C, h, w)
        """
        
        # print("input of dino shape",x.shape)
        # x = x[:,:,0:self.height,0:self.width]
        # x = x[:,:,0:self.width,0:self.height]
        """ from torchvision import transforms as pth_transforms
        transform = pth_transforms.Compose([
                pth_transforms.Resize(self.height, self.width),
            ]) """
        *B, c, h, w = x.shape

        x = x.reshape(-1, c, h, w)
        # print("reshaped input of dino shape",x.shape)

        # Output will be (B, H * W, C)
        # features = self.model.forward_features(x)["x_norm_patchtokens"]
        if self.model_type == "dinov2":
            features = self.model.forward_features(x)["x_norm_patchtokens"]
        elif self.model_type == "dino":
            features = self.model.get_intermediate_layers(x, n=12)
            features = features[-1][:,1:,:]
        # print("features.shape", features.shape)
        features = features.permute(0, 2, 1)
        # print("features.shape after permute", features.shape)
        features = features.reshape(  # (B, C, H, W)
            -1, self.feature_dim, h // self.patch_size, w // self.patch_size
        )
        # print("features.shape after reshape", features.shape)
        if autoresize:
            features = resize(features, size=(self.num_patches_y, self.num_patches_x))

        features = features.reshape(
            *B, self.feature_dim, self.num_patches_y, self.num_patches_x
        )
        
        # print("features.shape after reshape *B", features.shape)
        # print("features", features)
        return features
