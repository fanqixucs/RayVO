import ipdb  # noqa: F401
import torch

from ray_diffusion.inference.ddpm import inference_ddpm
from ray_diffusion.utils.rays import (
    Rays,
    compute_ndc_coordinates,
    rays_to_cameras_for_rec,
    rays_to_cameras_homography,
    rays_to_cameras_for_rec_moments_rescale,
    rays_to_cameras_for_rec_whole_moments_rescale,
)


def predict_cameras_with_homo_and_intrinsic(
    model,
    images,
    device,
    pred_x0,
    crop_parameters=None,
    num_patches_x=16,
    num_patches_y=16,
    additional_timesteps=(),
    calculate_intrinsics=True,
    use_beta_tilde=False,
    normalize_moments=True,
    rescale_noise="zero",
    use_regression=False,
    max_num_images=None,
    pbar=False,
    return_rays=False,
):
    """
    Args:
        images (torch.Tensor): (N, C, H, W)
        crop_parameters (torch.Tensor): (N, 4) or None
    """
    
    ray_to_cam_homo = rays_to_cameras_homography
    ray_to_cam_intrinsic = rays_to_cameras

    if use_regression:
        rays_final = inference_regression(
            model,
            images.unsqueeze(0),
            device,
            crop_parameters=crop_parameters.unsqueeze(0),
            num_patches_x=num_patches_x,
            num_patches_y=num_patches_y,
        )

    else:
        rays_final, rays_intermediate, pred_intermediate, _ = inference_ddpm(
            model,
            images.unsqueeze(0),
            device,
            visualize=True,
            pred_x0=pred_x0,
            crop_parameters=crop_parameters.unsqueeze(0),
            stop_iteration=-1,
            num_patches_x=num_patches_x,
            num_patches_y=num_patches_y,
            pbar=pbar,
            use_beta_tilde=use_beta_tilde,
            normalize_moments=normalize_moments,
            rescale_noise=rescale_noise,
            max_num_images=max_num_images,
        )

    # use regression false: diffusion model
    if pred_x0 and not use_regression:
        rays_final = rays_intermediate[-2]
    
    pred_cam = ray_to_cam_homo(
        Rays.from_spatial(rays_final)[0],
        crop_parameters,
        num_patches_x=num_patches_x,
        num_patches_y=num_patches_y,
    )

    additional_predictions_homo = []
    additional_predictions_intrinsic = []
    additional_predictions_rays = []
    for t in additional_timesteps:
        if pred_x0:
            ray = pred_intermediate[t]
        else:
            ray = rays_intermediate[t]
        ray = Rays.from_spatial(ray)[0]
        additional_predictions_homo.append(
            ray_to_cam_homo(
                ray,
                crop_parameters,
                num_patches_x=num_patches_x,
                num_patches_y=num_patches_y,
            )
        )
        additional_predictions_intrinsic.append(
            ray_to_cam_intrinsic(
                ray,
                crop_parameters,
                num_patches_x=num_patches_x,
                num_patches_y=num_patches_y,
            )
        )
        if return_rays:
            additional_predictions_rays.append(ray)
    
    if return_rays:
        return (
            pred_cam,
            Rays.from_spatial(rays_final)[0],
            additional_predictions_homo,
            additional_predictions_intrinsic,
            additional_predictions_rays,
        )
    else:
        return pred_cam, additional_predictions


def predict_cameras_for_rec_rescale_moment_postprocess(
    model,
    images,
    device,
    pred_x0,
    crop_parameters=None,
    num_patches_x=45,
    num_patches_y=13,
    additional_timesteps=False,
    calculate_intrinsics=False,
    use_beta_tilde=False,
    normalize_moments=True,
    rescale_noise="zero",
    use_regression=False,
    max_num_images=None,
    pbar=False,
    return_rays=False,
    scale = 3.31,
    cameras = None
):
    """
    Args:
        images (torch.Tensor): (N, C, H, W)
        crop_parameters (torch.Tensor): (N, 4) or None
    """
    if calculate_intrinsics:
        ray_to_cam = rays_to_cameras_homography
    else:
        ray_to_cam = rays_to_cameras_for_rec_moments_rescale

    if use_regression:
        rays_final = inference_regression(
            model,
            images.unsqueeze(0),
            device,
            crop_parameters=None,
            num_patches_x=num_patches_x,
            num_patches_y=num_patches_y,
        )

    else:
        rays_final, rays_intermediate, pred_intermediate, _ = inference_ddpm(
            model,
            images.unsqueeze(0),
            device,
            visualize=True,
            pred_x0=pred_x0,
            crop_parameters=None,
            stop_iteration=-1,
            num_patches_x=num_patches_x,
            num_patches_y=num_patches_y,
            pbar=pbar,
            use_beta_tilde=use_beta_tilde,
            normalize_moments=normalize_moments,
            rescale_noise=rescale_noise,
            max_num_images=max_num_images,
        )

    # use regression false: diffusion model
    if pred_x0 and not use_regression:
        rays_final = rays_intermediate[-1] # i xfq change this from -2 to -1
    
    pred_cam = ray_to_cam(
        Rays.from_spatial(rays_final)[0],
        crop_parameters = None,
        num_patches_x=num_patches_x,
        num_patches_y=num_patches_y,
        scale = scale,
        cameras = cameras,
    )
    additional_predictions = []
    additional_predictions_rays = []
    
    '''
    if additional_timesteps:
        additional_timesteps_t = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99] # 0 is x0 predicted from T=100
        additional_predictions = []
        additional_predictions_rays = []
        for t in additional_timesteps_t:
            if pred_x0:
                ray = pred_intermediate[t]
            else:
                ray = rays_intermediate[t]
            ray = Rays.from_spatial(ray)[0]
            additional_predictions.append(
                ray_to_cam(
                    ray,
                    crop_parameters,
                    num_patches_x=num_patches_x,
                    num_patches_y=num_patches_y,
                    scale = scale,
                    cameras = cameras,
                )
            )
            if return_rays:
                additional_predictions_rays.append(ray)
    '''
    
    
    if return_rays:
        return (
            pred_cam,
            Rays.from_spatial(rays_final)[0],
            additional_predictions,
            additional_predictions_rays,
        )
    else:
        return pred_cam, additional_predictions
    
def predict_cameras_for_rec_rescale_whole_moment_postprocess(
    model,
    images,
    device,
    pred_x0,
    crop_parameters=None,
    num_patches_x=45,
    num_patches_y=13,
    additional_timesteps=False,
    calculate_intrinsics=False,
    use_beta_tilde=False,
    normalize_moments=True,
    rescale_noise="zero",
    use_regression=False,
    max_num_images=None,
    pbar=False,
    return_rays=False,
    scale = 3.31,
    cameras = None
):
    """
    Args:
        images (torch.Tensor): (N, C, H, W)
        crop_parameters (torch.Tensor): (N, 4) or None
    """
    if calculate_intrinsics:
        ray_to_cam = rays_to_cameras_homography
    else:
        ray_to_cam = rays_to_cameras_for_rec_whole_moments_rescale

    if use_regression:
        rays_final = inference_regression(
            model,
            images.unsqueeze(0),
            device,
            crop_parameters=None,
            num_patches_x=num_patches_x,
            num_patches_y=num_patches_y,
        )

    else:
        rays_final, rays_intermediate, pred_intermediate, _ = inference_ddpm(
            model,
            images.unsqueeze(0),
            device,
            visualize=True,
            pred_x0=pred_x0,
            crop_parameters=None,
            stop_iteration=-1,
            num_patches_x=num_patches_x,
            num_patches_y=num_patches_y,
            pbar=pbar,
            use_beta_tilde=use_beta_tilde,
            normalize_moments=normalize_moments,
            rescale_noise=rescale_noise,
            max_num_images=max_num_images,
        )

    # use regression false: diffusion model
    if pred_x0 and not use_regression:
        rays_final = rays_intermediate[-1] # i xfq change this from -2 to -1
    
    pred_cam = ray_to_cam(
        Rays.from_spatial(rays_final)[0],
        crop_parameters = None,
        num_patches_x=num_patches_x,
        num_patches_y=num_patches_y,
        scale = scale,
        cameras = cameras,
    )
    additional_predictions = []
    additional_predictions_rays = []
    
    '''
    if additional_timesteps:
        additional_timesteps_t = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99] # 0 is x0 predicted from T=100
        additional_predictions = []
        additional_predictions_rays = []
        for t in additional_timesteps_t:
            if pred_x0:
                ray = pred_intermediate[t]
            else:
                ray = rays_intermediate[t]
            ray = Rays.from_spatial(ray)[0]
            additional_predictions.append(
                ray_to_cam(
                    ray,
                    crop_parameters,
                    num_patches_x=num_patches_x,
                    num_patches_y=num_patches_y,
                    scale = scale,
                    cameras = cameras,
                )
            )
            if return_rays:
                additional_predictions_rays.append(ray)
    '''
    
    
    if return_rays:
        return (
            pred_cam,
            Rays.from_spatial(rays_final)[0],
            additional_predictions,
            additional_predictions_rays,
        )
    else:
        return pred_cam, additional_predictions


def predict_cameras_for_rec(
    model,
    images,
    device,
    pred_x0,
    crop_parameters=None,
    num_patches_x=45,
    num_patches_y=13,
    additional_timesteps=False,
    calculate_intrinsics=False,
    use_beta_tilde=False,
    normalize_moments=True,
    rescale_noise="zero",
    use_regression=False,
    max_num_images=None,
    pbar=False,
    return_rays=False,
    scale = 3.31,
    cameras = None
):
    """
    Args:
        images (torch.Tensor): (N, C, H, W)
        crop_parameters (torch.Tensor): (N, 4) or None
    """
    if calculate_intrinsics:
        ray_to_cam = rays_to_cameras_homography
    else:
        ray_to_cam = rays_to_cameras_for_rec

    if use_regression:
        rays_final = inference_regression(
            model,
            images.unsqueeze(0),
            device,
            crop_parameters=None,
            num_patches_x=num_patches_x,
            num_patches_y=num_patches_y,
        )

    else:
        rays_final, rays_intermediate, pred_intermediate, _ = inference_ddpm(
            model,
            images.unsqueeze(0),
            device,
            visualize=True,
            pred_x0=pred_x0,
            crop_parameters=None,
            stop_iteration=-1,
            num_patches_x=num_patches_x,
            num_patches_y=num_patches_y,
            pbar=pbar,
            use_beta_tilde=use_beta_tilde,
            normalize_moments=normalize_moments,
            rescale_noise=rescale_noise,
            max_num_images=max_num_images,
        )

    # use regression false: diffusion model
    if pred_x0 and not use_regression:
        rays_final = rays_intermediate[-1] # i xfq change this from -2 to -1
    
    pred_cam = ray_to_cam(
        Rays.from_spatial(rays_final)[0],
        crop_parameters = None,
        num_patches_x=num_patches_x,
        num_patches_y=num_patches_y,
        scale = scale,
        cameras = cameras,
    )
    additional_predictions = []
    additional_predictions_rays = []
    
    '''
    if additional_timesteps:
        additional_timesteps_t = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99] # 0 is x0 predicted from T=100
        additional_predictions = []
        additional_predictions_rays = []
        for t in additional_timesteps_t:
            if pred_x0:
                ray = pred_intermediate[t]
            else:
                ray = rays_intermediate[t]
            ray = Rays.from_spatial(ray)[0]
            additional_predictions.append(
                ray_to_cam(
                    ray,
                    crop_parameters,
                    num_patches_x=num_patches_x,
                    num_patches_y=num_patches_y,
                    scale = scale,
                    cameras = cameras,
                )
            )
            if return_rays:
                additional_predictions_rays.append(ray)
    '''
    
    
    if return_rays:
        return (
            pred_cam,
            Rays.from_spatial(rays_final)[0],
            additional_predictions,
            additional_predictions_rays,
        )
    else:
        return pred_cam, additional_predictions


def predict_cameras(
    model,
    images,
    device,
    pred_x0,
    crop_parameters=None,
    num_patches_x=16,
    num_patches_y=16,
    additional_timesteps=(),
    calculate_intrinsics=True,
    use_beta_tilde=False,
    normalize_moments=True,
    rescale_noise="zero",
    use_regression=False,
    max_num_images=None,
    pbar=False,
    return_rays=False,
):
    """
    Args:
        images (torch.Tensor): (N, C, H, W)
        crop_parameters (torch.Tensor): (N, 4) or None
    """
    if calculate_intrinsics:
        ray_to_cam = rays_to_cameras_homography
    else:
        ray_to_cam = rays_to_cameras

    if use_regression:
        rays_final = inference_regression(
            model,
            images.unsqueeze(0),
            device,
            crop_parameters=crop_parameters.unsqueeze(0),
            num_patches_x=num_patches_x,
            num_patches_y=num_patches_y,
        )

    else:
        rays_final, rays_intermediate, pred_intermediate, _ = inference_ddpm(
            model,
            images.unsqueeze(0),
            device,
            visualize=True,
            pred_x0=pred_x0,
            crop_parameters=crop_parameters.unsqueeze(0),
            stop_iteration=-1,
            num_patches_x=num_patches_x,
            num_patches_y=num_patches_y,
            pbar=pbar,
            use_beta_tilde=use_beta_tilde,
            normalize_moments=normalize_moments,
            rescale_noise=rescale_noise,
            max_num_images=max_num_images,
        )

    # use regression false: diffusion model
    if pred_x0 and not use_regression:
        rays_final = rays_intermediate[-2]
    
    pred_cam = ray_to_cam(
        Rays.from_spatial(rays_final)[0],
        crop_parameters,
        num_patches_x=num_patches_x,
        num_patches_y=num_patches_y,
    )

    additional_predictions = []
    additional_predictions_rays = []
    for t in additional_timesteps:
        if pred_x0:
            ray = pred_intermediate[t]
        else:
            ray = rays_intermediate[t]
        ray = Rays.from_spatial(ray)[0]
        additional_predictions.append(
            ray_to_cam(
                ray,
                crop_parameters,
                num_patches_x=num_patches_x,
                num_patches_y=num_patches_y,
            )
        )
        if return_rays:
            additional_predictions_rays.append(ray)
    
    if return_rays:
        return (
            pred_cam,
            Rays.from_spatial(rays_final)[0],
            additional_predictions,
            additional_predictions_rays,
        )
    else:
        return pred_cam, additional_predictions


def inference_regression(
    model,
    images,
    device,
    crop_parameters=None,
    num_patches_x=16,
    num_patches_y=16,
):
    batch_size = images.shape[0]
    num_images = images.shape[1]
    images = images.to(device)

    t = model.noise_scheduler.max_timesteps

    with torch.no_grad():
        x_t = torch.randn(
            batch_size, num_images, 6, num_patches_y, num_patches_x, device=device
        )
        image_features = model.feature_extractor(images, autoresize=False)
        if model.append_ndc:
            # (B, N, H, W, 3)
            ndc_coordinates = compute_ndc_coordinates(
                crop_parameters=crop_parameters,
            )
            ndc_coordinates = ndc_coordinates.to(device)[..., :2]
            # (B, N, 2, H, W)
            ndc_coordinates = ndc_coordinates.permute(0, 1, 4, 2, 3)
        else:
            ndc_coordinates = None
        eps_pred, noise_sample = model(
            features=image_features,
            rays_noisy=x_t,
            t=t,
            ndc_coordinates=ndc_coordinates,
        )
    return eps_pred
