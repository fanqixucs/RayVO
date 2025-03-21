#### have been adapted to KITTI dataset

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
import pickle

CO3D_DIR = "/media/fanqixu/20XFQ/kitti/kitti_odometry_camerarays_preprocessed/med_res"
CO3D_ANNOTATION_DIR = "/home/fanqixu/experiments/datasets/kitti_odometry_camerarays_preprocessed"
#CO3D_ORDER_PATH = "data/co3d_v2_random_order_{sample_num}/{category}.json"


# TRAINING_CATEGORIES = ['00', '01', '02', '04', '05', '06', '07', '08']
# TEST_CATEGORIES = ['10']



#assert len(TRAINING_CATEGORIES) + len(TEST_CATEGORIES) == 51

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def square_bbox(bbox, padding=0.0, astype=None):
    """
    Computes a square bounding box, with optional padding parameters.
    Args:
        bbox: Bounding box in xyxy format (4,).
    Returns:
        square_bbox in xyxy format (4,).
    """
    if astype is None:
        astype = type(bbox[0])
    bbox = np.array(bbox)
    center = (bbox[:2] + bbox[2:]) / 2
    extents = (bbox[2:] - bbox[:2]) / 2
    s = max(extents) * (1 + padding)
    square_bbox = np.array(
        [center[0] - s, center[1] - s, center[0] + s, center[1] + s],
        dtype=astype,
    )
    return square_bbox


def unnormalize_image(image):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    image = image * std + mean
    return (image * 255.0).astype(np.uint8)


def _transform_intrinsic(image, bbox, principal_point, focal_length):
    # Rescale intrinsics to match bbox
    half_box = np.array([image.width, image.height]).astype(np.float32) / 2
    org_scale = min(half_box).astype(np.float32)

    # Pixel coordinates
    principal_point_px = half_box - (np.array(principal_point) * org_scale)
    focal_length_px = np.array(focal_length) * org_scale
    principal_point_px -= bbox[:2]
    new_bbox = (bbox[2:] - bbox[:2]) / 2
    new_scale = min(new_bbox)

    # NDC coordinates
    new_principal_ndc = (new_bbox - principal_point_px) / new_scale
    new_focal_ndc = focal_length_px / new_scale

    principal_point = torch.tensor(new_principal_ndc.astype(np.float32))
    focal_length = torch.tensor(new_focal_ndc.astype(np.float32))

    return principal_point, focal_length


def construct_camera_from_batch(batch, device):
    if isinstance(device, int):
        device = f"cuda:{device}"

    return PerspectiveCameras(
        R=batch["R"].reshape(-1, 3, 3),
        T=batch["T"].reshape(-1, 3),
        focal_length=batch["focal_lengths"].reshape(-1, 2),
        principal_point=batch["principal_points"].reshape(-1, 2),
        image_size=batch["image_sizes"].reshape(-1, 2),
        device=device,
    )


def save_batch_images(images, fname):
    cmap = plt.get_cmap("hsv")
    num_frames = len(images)
    num_rows = len(images)
    num_cols = 4
    figsize = (num_cols * 2, num_rows * 2)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
    axs = axs.flatten()
    for i in range(num_rows):
        for j in range(4):
            if i < num_frames:
                axs[i * 4 + j].imshow(unnormalize_image(images[i][j]))
                for s in ["bottom", "top", "left", "right"]:
                    axs[i * 4 + j].spines[s].set_color(cmap(i / (num_frames)))
                    axs[i * 4 + j].spines[s].set_linewidth(5)
                axs[i * 4 + j].set_xticks([])
                axs[i * 4 + j].set_yticks([])
            else:
                axs[i * 4 + j].axis("off")
    plt.tight_layout()
    plt.savefig(fname)


def jitter_bbox(square_bbox, jitter_scale=(1.1, 1.2), jitter_trans=(-0.07, 0.07)):
    square_bbox = np.array(square_bbox.astype(float))
    s = np.random.uniform(jitter_scale[0], jitter_scale[1])
    tx, ty = np.random.uniform(jitter_trans[0], jitter_trans[1], size=2)
    side_length = square_bbox[2] - square_bbox[0]
    center = (square_bbox[:2] + square_bbox[2:]) / 2 + np.array([tx, ty]) * side_length
    extent = side_length / 2 * s
    ul = center - extent
    lr = ul + 2 * extent
    return np.concatenate((ul, lr))

def is_valid_transformation_matrix(matrix):
    # print("start 8is_valid_transformation_matrix = ===============================")
    if matrix.shape != (4, 4):
        return False
    
    # Check if the upper-left 3x3 submatrix is a valid rotation matrix
    R = matrix[0:3, 0:3]
    if not np.allclose(np.dot(R, R.T), np.eye(3), atol=1e-6):
        return False
    if not np.isclose(np.linalg.det(R), 1.0, atol=1e-6):
        return False
    
    # Check if the last row is [0, 0, 0, 1]
    if not np.allclose(matrix[3, :], [0, 0, 0, 1]):
        return False
    
    return True


class Co3dDataset(Dataset):
    def __init__(
        self,
        split="train",
        transform=None,
        num_images=2,
        width=600,
        height=200,
        img_size=224,
        mask_images=False,
        crop_images=False,
        co3d_dir=None,
        co3d_annotation_dir=None,
        apply_augmentation=False,
        normalize_cameras=True,
        no_images=False,
        sample_num=None,
        seed=0,
        load_extra_cameras=False,
        seqs = ["10_02"],
    ):
        # print("aaaaaaaaaaaaaanew co3d_test start 1__init__ = ===============================")

        start_time = time.time()

        #self.category = category
        #print('self.category',self.category)
        self.split = split
        self.transform = transform
        self.num_images = num_images
        self.width = width
        self.height = height
        self.img_size = img_size
        self.mask_images = mask_images
        self.crop_images = crop_images
        self.apply_augmentation = apply_augmentation
        self.normalize_cameras = normalize_cameras
        self.no_images = no_images
        self.sample_num = sample_num
        self.load_extra_cameras = load_extra_cameras
        self.number = 0

        if self.apply_augmentation:
            self.jitter_scale = (1.1, 1.2)
            self.jitter_trans = (-0.07, 0.07)
        else:
            # Note if trained with apply_augmentation, we should still use
            # apply_augmentation at test time.
            self.jitter_scale = (1, 1)
            self.jitter_trans = (0.0, 0.0)

        if co3d_dir is not None:
            self.co3d_dir = co3d_dir
            self.co3d_annotation_dir = co3d_annotation_dir
        else:
            self.co3d_dir = CO3D_DIR
            self.co3d_annotation_dir = CO3D_ANNOTATION_DIR

        #if isinstance(self.category, str):
        #    self.category = [self.category]
        '''
        if "all_train" in self.category:
            self.category = TRAINING_CATEGORIES
            #print('self.category',self.category)
        if "all_test" in self.category:
            self.category = TEST_CATEGORIES
        if "full" in self.category:
            self.category = TRAINING_CATEGORIES + TEST_CATEGORIES
        self.category = sorted(self.category)
        #print('self.category sorted',self.category)
        self.is_single_category = len(self.category) == 1
        
        '''

        # Fixing seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.seq_info = {}
        self.number = 0

        for seq in seqs:
            print("processing seq=================",seq)
            # data_filename = osp.join(self.co3d_annotation_dir, seq)

            # with open(data_filename + '.pkl', 'rb') as f:
            #     annotation = pickle.load(f) # # print("annotation",annotation) # [{00000},{000001}]
            
            data_filename = osp.join(
                self.co3d_annotation_dir, f"{seq}.jgz"
            ) # train and test has different jgz files
            # print("data_filename",data_filename)

            with gzip.open(data_filename, "r") as fin:
                annotation = json.loads(fin.read())
            
            print("annotation",annotation) # loaded
            
            self.number += len(annotation)
            self.seq_info[seq] = annotation           
                
        # for whole training categories, whole training datasets
        self.sequence_list = list(self.seq_info.keys())
        # print("self.sequence_list",self.sequence_list)
        # print("self.seq_info",self.seq_info)   
        print(f"Data size in total for training: {len(self)}")

        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(self.img_size, antialias=True),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )


    def __len__(self):
        return self.number - 1

    def __getitem__(self, index): # gpu
        # print("index",index)
        
        num_to_load = self.num_images if not self.load_extra_cameras else 8
        if self.sample_num is not None:
            with open(
                CO3D_ORDER_PATH.format(
                    sample_num=self.sample_num, category=self.category[0]
                )
            ) as f:
                order = json.load(f)
            ids = order[sequence_name][:num_to_load]
        else:
            start_idx = index
            ids = list(range(start_idx, start_idx + num_to_load))

        # return index
        return self.get_data(index=index, ids=ids)

    def _get_scene_scale(self, sequence_name):
        print("start 4_get_scene_scale = ===============================")
        n = len(self.rotations[sequence_name])

        R = torch.zeros(n, 3, 3)
        T = torch.zeros(n, 3)

        for i, ann in enumerate(self.rotations[sequence_name]):
            R[i, ...] = torch.tensor(self.rotations[sequence_name][i]["R"])
            T[i, ...] = torch.tensor(self.rotations[sequence_name][i]["T"])

        cameras = PerspectiveCameras(R=R, T=T)
        cc = cameras.get_camera_center()
        centeroid = torch.mean(cc, dim=0)
        diff = cc - centeroid

        norm = torch.norm(diff, dim=1)
        scale = torch.max(norm).item()

        return scale

    def _crop_image(self, image, bbox):
        #print("start 5_crop_image = ===============================")
        image_crop = transforms.functional.crop(
            image,
            top=bbox[1],
            left=bbox[0],
            height=bbox[3] - bbox[1],
            width=bbox[2] - bbox[0],
        )
        return image_crop

    def _transform_intrinsic(self, image, bbox, principal_point, focal_length):
        print("start 6_transform_intrinsic = ===============================")
        half_box = np.array([image.width, image.height]).astype(np.float32) / 2
        org_scale = min(half_box).astype(np.float32)

        # Pixel coordinates
        principal_point_px = half_box - (np.array(principal_point) * org_scale)
        focal_length_px = np.array(focal_length) * org_scale
        principal_point_px -= bbox[:2]
        new_bbox = (bbox[2:] - bbox[:2]) / 2
        new_scale = min(new_bbox)

        # NDC coordinates
        new_principal_ndc = (new_bbox - principal_point_px) / new_scale
        new_focal_ndc = focal_length_px / new_scale

        return new_principal_ndc.astype(np.float32), new_focal_ndc.astype(np.float32)

    def get_data(self, index=None, sequence_name=None, ids=(0, 1), no_images=False):
        #print("start 7get_data = ===============================")
        if sequence_name is None:
            index = index % len(self.sequence_list)
            sequence_name = self.sequence_list[index]
            # print('sequence_name',sequence_name)
        
        metadata = self.seq_info[sequence_name]
        metadata_size = len(metadata)
        # print('metadata_size',metadata_size)
        
        # print('ids',ids)
        
        # for i in ids:
        #     print('i',i)
        #     annos = metadata[i] 
        #     # print('annos',annos)

        annos = [metadata[i] for i in ids]
        # print('annos',annos)
        # print("==================================================")

        images = []
        image_sizes = []
        PP = []
        FL = []
        crop_parameters = []
        filenames = []
        count = 0

        for anno in annos:
            # print("anno",anno)
            filepath = anno["filepath"]

            if not no_images:
                image = Image.open(osp.join(self.co3d_dir, filepath)).convert("RGB")
                # image.save(f'output_1kitti_orginal_image_{count}.png')
                image = image.resize((self.width, self.height))

                # Save the resized image if needed
                # image.save(f'output_1.5kitti_resized_image_{count}.png')

                # Optionally mask images with black background
                if self.mask_images:
                    black_image = Image.new("RGB", image.size, (0, 0, 0))
                    mask_name = osp.basename(filepath.replace(".jpg", ".png"))

                    mask_path = osp.join(
                        self.co3d_dir, category, sequence_name, "masks", mask_name
                    )
                    mask = Image.open(mask_path).convert("L")

                    if mask.size != image.size:
                        mask = mask.resize(image.size)
                    mask = Image.fromarray(np.array(mask) > 125)
                    image = Image.composite(image, black_image, mask)

                # Determine crop, Resnet wants square images
                bbox_init = ([100, 0, image.width - 150, image.height]) # [0, 0, 640, 192]
                # bbox = square_bbox(np.array(bbox_init))
                bbox = np.array(bbox_init)
                # print("bbox_init",bbox_init)
                # if self.apply_augmentation:
                bbox = jitter_bbox(
                    bbox,
                    jitter_scale=self.jitter_scale,
                    jitter_trans=self.jitter_trans,
                )
                bbox = np.around(bbox).astype(int)
                # print("bbox",bbox)

                # Crop parameters
                crop_center = (bbox[:2] + bbox[2:]) / 2
                # convert crop center to correspond to a "square" image
                width, height = image.size
                length = max(width, height)
                s = length / min(width, height)
                crop_center = crop_center + (length - np.array([width, height])) / 2
                # convert to NDC
                cc = s - 2 * s * crop_center / length
                crop_width = 2 * s * (bbox[2] - bbox[0]) / length
                crop_params = torch.tensor([-cc[0], -cc[1], crop_width, s])
                # print("crop_params",crop_params)

                principal_point = torch.tensor(anno["principal_point"])
                focal_length = torch.tensor(anno["focal_length"])

                # Crop and normalize image
                image = self._crop_image(image, bbox)
                # image.save(f'output_2kitti_crop_image_{count}.png')
                image = self.transform(image)
                # imagepil = transforms.ToPILImage()(image)
                # imagepil.save(f'output_3kitti_transform_image_{count}.png')
                # image = image[:, : self.img_size, : self.img_size]
                # imagepil = transforms.ToPILImage()(image)
                # imagepil.save(f'output_4kitti_final_image_{count}.png')

                images.append(image[:, : self.img_size, : self.img_size])
                crop_parameters.append(crop_params)
                # print("crop_params",crop_params)

            else:
                principal_point = torch.tensor(anno["principal_point"])
                focal_length = torch.tensor(anno["focal_length"])

            PP.append(principal_point)
            FL.append(focal_length)
            image_sizes.append(torch.tensor([self.width, self.height])) # img size is 224
            filenames.append(filepath)

        if not no_images:
            if self.load_extra_cameras:
                # Remove the extra loaded image, for saving space
                images = images[: self.num_images]

            images = torch.stack(images)
            crop_parameters = torch.stack(crop_parameters)
        else:
            images = None
            crop_parameters = None

        # Assemble batch info to send back
        R = torch.stack([torch.tensor(anno["R"]) for anno in annos])
        T = torch.stack([torch.tensor(anno["T"]) for anno in annos])
        # print("T from get_data",T)
        # print("anno odo",np.array(anno["odometry"])[:3, 3])
        R_inv = torch.stack([torch.tensor(anno["R_inv_vo"]) for anno in annos])
        T_inv = torch.stack([torch.tensor(anno["T_inv_vo"]) for anno in annos])
        R_forward = torch.stack([torch.tensor(anno["R_forward_vo"]) for anno in annos])
        T_forward = torch.stack([torch.tensor(anno["T_forward_vo"]) for anno in annos])
        focal_lengths = torch.stack(FL)
        principal_points = torch.stack(PP)
        
        
        odometry_R = torch.stack([torch.tensor(annos[1]["R_forward_vo"]),torch.tensor(annos[1]["R_inv_vo"])])
        odometry_T = torch.stack([torch.tensor(annos[1]["T_forward_vo"]),torch.tensor(annos[1]["T_inv_vo"])])
        image_sizes = torch.stack(image_sizes)

        #print("T from get_data",T)
        #print("filenames from get_data",filenames)

        batch = {
            "model_id": sequence_name,
            "n": len(metadata),
            "ind": torch.tensor(ids),
            "image": images,
            "R": R,
            "T": T,
            "odometry_R": odometry_R, # forwad and then inverse
            "odometry_T": odometry_T,
            "focal_length": focal_lengths,
            "principal_point": principal_points,
            "image_size_before_square224": image_sizes,
            "crop_parameters": crop_parameters,
            "filename": filenames,
        }

        print("image using",batch["filename"])
        # print("batch",batch)
        #print ("**************************************************")
        return batch


def main():
    dataset = Co3dDataset(
                num_images=2,
                apply_augmentation=False,
                seqs = ["10_02"],
    )

    dataset.__getitem__(0)
    # for i in range(120):
    #     dataset.__getitem__(i)

    
    train_dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=4,
                shuffle=True,
                num_workers=8,
                pin_memory=True,
                drop_last=True,
    )

    # Iterate through the DataLoader and print batch information
    # for batch_idx, batch in enumerate(train_dataloader):
    #     print(f"Batch {batch_idx + 1}")
    #     print(f"Batch size: {len(batch)}")
        
    #     # Assuming the batch is a tuple of (inputs, targets)
    #     inputs, targets = batch
    #     print(f"Inputs shape: {inputs.shape}")
    #     print(f"Targets shape: {targets.shape}")
        
    #     # Print a few samples from the batch
    #     print(f"Inputs: {inputs[:2]}")
    #     print(f"Targets: {targets[:2]}")
        
    #     # Break after the first batch for brevity
    #     break
        
    
    

if __name__ == "__main__":
    main()

