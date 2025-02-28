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

CO3D_DIR = "/hdd2/fanqi/datasets/kitti/kitti_odometry_camerarays_preprocessed/med_res"
CO3D_ANNOTATION_DIR = "/hdd2/fanqi/datasets/kitti/kitti_odometry_camerarays_preprocessed/whole_sequence"
#CO3D_ORDER_PATH = "data/co3d_v2_random_order_{sample_num}/{category}.json"


TRAINING_CATEGORIES = ['00', '01', '02','03', '04', '05', '06', '07', '08'] # ['00', '01', '02', '04', '05', '06', '07', '08']
TEST_CATEGORIES = ['10']


#assert len(TRAINING_CATEGORIES) + len(TEST_CATEGORIES) == 51

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True



class Co3dDataset(Dataset):
    def __init__(
        self,
        split="train",
        transform=None,
        num_images=2,
        image_width=600,
        image_height=200,
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
        id_consecutive=False,
        category = ["10"],
    ):
        # print("start 1__init__ = ===============================")

        start_time = time.time()

        #self.category = category
        #print('self.category',self.category)
        self.split = split
        self.transform = transform
        self.num_images = num_images
        self.width = image_width
        self.height = image_height
        self.img_size = img_size
        self.mask_images = mask_images
        self.crop_images = crop_images
        self.apply_augmentation = apply_augmentation
        self.normalize_cameras = normalize_cameras
        self.no_images = no_images
        self.sample_num = sample_num
        self.load_extra_cameras = load_extra_cameras
        self.number = 0
        self.id_consecutive = id_consecutive
        self.category = category    

        if co3d_dir is not None:
            self.co3d_dir = co3d_dir
            self.co3d_annotation_dir = co3d_annotation_dir
        else:
            self.co3d_dir = CO3D_DIR
            self.co3d_annotation_dir = CO3D_ANNOTATION_DIR

        # Fixing seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.low_quality_translations = []
        self.rotations = {}
        
        self.category_map = {}
        # print('self.category',self.category)

        all_valid_odometry = True

        for c in self.category:
            annotation_file = osp.join(
                self.co3d_annotation_dir, f"{c}.jgz"
            ) # train and test has different jgz files
            
            
            # annotation_file = osp.join(
            #     self.co3d_annotation_dir, "c*.jgz"
            # )
            
            # for inference result check in order to check whether the features are correct 
            # annotation_file = '/home/fanqixu/experiments/datasets/kitti_odometry_camerarays_preprocessed/10_02.jgz'

            with gzip.open(annotation_file, "r") as fin:
                annotation = json.loads(fin.read())

            # print('annotation_file',annotation)

            # counter = 0

            for seq_name, seq_data in annotation.items():
                # seq_name 08_02 08_03
                # size of seq_data 4071
                
                # counter += 1
                if len(seq_data) < self.num_images:
                    continue
                
                # print('seq_name',seq_name) # 08_02 08_03
                # print('seq_data',seq_data)
                # print('size of seq_data', len(seq_data))

                filtered_data = []
                self.category_map[seq_name] = c
                bad_seq = False
                
                for data in seq_data: # data: each image in the sequence

                    # Make sure translations are not ridiculous and rotations are valid
                    '''
                    det = np.linalg.det(data["R"])
                    if (np.abs(data["T"]) > 1e5).any() or det < 0.99 or det > 1.01:
                        bad_seq = True
                        self.low_quality_translations.append(seq_name)
                        break
                    '''
                    
                    matrix = np.array(data["odometry"])
                    
                    # Ignore all unnecessary information.
                    filtered_data.append(
                        {
                            "filepath": data["filepath"],
                            # "bbox": data["bbox"],
                            "R": data["R"],
                            "T": data["T"],
                            "odometry": data["odometry"],
                            "focal_length": data["focal_length"],
                            "principal_point": data["principal_point"],
                        },
                    )

                # print('filtered_data size',len(filtered_data)) # 4071
                self.number += len(filtered_data)
                

                if not bad_seq:
                    self.rotations[seq_name] = filtered_data
                
                #break

            # print(f"Loaded {counter} seqs/instances of the {c} sequence.")

        # for whole training categories, whole training datasets
        self.sequence_list = list(self.rotations.keys())
        # print('self.sequence_list',self.sequence_list)
        # ['00_02', '00_03', '01_02', '01_03', '02_02', '02_03', '04_02', '04_03', '05_02', '05_03', '06_02', '06_03', '07_02', '07_03', '08_02', '08_03']

        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
                    transforms.Resize((self.height, self.width), antialias=True), 
                ]
            )

        #print(
        #    f"Low quality translation sequences, not used: {self.low_quality_translations}"
        #)

        print(f"Data size in total for training: {len(self)}")
        # print("valid odometry", all_valid_odometry)
        #print(f"Data loading took {(time.time()-start_time)} seconds.")

    def __len__(self):
        #print("start 2__len__ = ===============================")
        # print('len(self.filterdata)',len(self.filtered_data))
        return self.number

    def __getitem__(self, index):
        #print("start 3__getting item__ = ===============================")
        num_to_load = self.num_images if not self.load_extra_cameras else 8
        #print('3__getting item__ num_to_load',num_to_load)

        #print("INDEX",index)
        #print("num_to_load",num_to_load)    

        sequence_name = self.sequence_list[index % len(self.sequence_list)]
        
        metadata = self.rotations[sequence_name]
        # print('metadata',len(metadata))

        if self.sample_num is not None:
            with open(
                CO3D_ORDER_PATH.format(
                    sample_num=self.sample_num, category=self.category[0]
                )
            ) as f:
                order = json.load(f)
            ids = order[sequence_name][:num_to_load]
        else:
            if self.id_consecutive:
                start_idx = index
                ids = list(range(start_idx, start_idx + num_to_load))
            else:
                start_idx = np.random.randint(0, len(metadata) - num_to_load + 1)
                    # ids = [0,1]
                ids = list(range(start_idx, start_idx + num_to_load))

            

            # print('index',index)
            # print('ids',ids)
            #ids = np.random.choice(len(metadata), num_to_load, replace=False)
            #print(ids) # [70 31 23 20 44 71 75 80]
            #print("length of metadata",len(metadata))

        return self.get_data(index=index, ids=ids)

    

    def get_data(self, index=None, sequence_name=None, ids=(0, 1), no_images=False):
        #print("start 7get_data = ===============================")
        if sequence_name is None:
            if self.id_consecutive:
                sequence_name = self.sequence_list[0]
            else:
                index = index % len(self.sequence_list)
                sequence_name = self.sequence_list[index]
            # print('sequence_name',sequence_name)
        
        metadata = self.rotations[sequence_name]
        #metadata_size = len(metadata)
        #print('metadata_size',metadata_size)

        category = self.category_map[sequence_name]

        # Read image & camera information from annotations
        annos = [metadata[i] for i in ids]
        

        images = []
        image_sizes = []
        PP = []
        FL = []
        crop_parameters = []
        filenames = []
        count = 0

        for anno in annos:
            filepath = anno["filepath"]

            if not no_images:
                image = Image.open(osp.join(self.co3d_dir, filepath)).convert("RGB")
                image = self.transform(image)
                images.append(image)
                
                # calculate intrisic in ndc 
                principal_point = torch.tensor(anno["principal_point"])
                focal_length = torch.tensor(anno["focal_length"])
                
                # first calculate the intrinsic after resizing for dino2
                H, W = image.shape[-2:]
                zoomx = W/640
                zoomy = H/192
                
                focal_length[0] = focal_length[0] * zoomx
                focal_length[1] = focal_length[1] * zoomy
                principal_point[0] = principal_point[0] * zoomx 
                principal_point[1] = principal_point[1] * zoomy
                
                # calculate the intrinsic in ndc
                # use minus focal length to match the ndc coordinate to meet kitti dataset
                s = min(W, H)
                focal_length[0] = -focal_length[0] * 2.0/s
                focal_length[1] = -focal_length[1] * 2.0/s

                principal_point[0] = - (principal_point[0]- W / 2.0) * 2.0 / s
                principal_point[1] = - (principal_point[1] - H / 2.0) * 2.0 / s
                                
                # crop_parameters.append(crop_params)

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
            # crop_parameters = torch.stack(crop_parameters)
        else:
            images = None
            crop_parameters = None

        # Assemble batch info to send back
        R = torch.stack([torch.tensor(anno["R"]) for anno in annos])
        T = torch.stack([torch.tensor(anno["T"]) for anno in annos])
        # print("T from get_data",T)
        # print("anno odo",np.array(anno["odometry"])[:3, 3])
        
        odometry_inverse_transformation = np.linalg.inv(annos[1]['odometry'])
        odometry_R = torch.stack([torch.tensor(np.array(annos[1]["odometry"])[:3, :3]), torch.tensor(odometry_inverse_transformation[:3, :3])], dim=0)
        odometry_T = torch.stack([torch.tensor(np.array(annos[1]["odometry"])[:3, 3]), torch.tensor(odometry_inverse_transformation[:3, 3])], dim=0)
        
        # odometry_R = torch.stack([torch.tensor(np.array(anno["odometry"])[:3, :3]) for anno in annos])
        # odometry_T = torch.stack([torch.tensor(np.array(anno["odometry"])[:3, 3]) for anno in annos]) 
        focal_lengths = torch.stack(FL)
        principal_points = torch.stack(PP)
        image_sizes = torch.stack(image_sizes)

        #print("T from get_data",T)
        #print("filenames from get_data",filenames)

        batch = {
            "model_id": sequence_name, # 09_02
            "category": category, # 09
            "n": len(metadata),
            "ind": torch.tensor(ids),
            "image": images,
            "R": R,
            "T": T,
            "odometry_R": odometry_R,
            "odometry_T": odometry_T,
            "focal_length": focal_lengths,
            "principal_point": principal_points,
            "image_size": image_sizes,
            "crop_parameters": crop_parameters,
            "filename": filenames,
        }

        # print("image using",batch["filename"])
        # print("batch",batch)
        #print ("**************************************************")
        return batch


def main():
    dataset = Co3dDataset(
                category=["10"],
                num_images=2,
                apply_augmentation=True,
                crop_images=True,
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

