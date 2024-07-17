import os
import numpy as np
import torch
import pandas as pd
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision.datasets import VisionDataset
from PIL import Image
from bigearthnet_common import constants as ben_constants
from kornia import augmentation as aug
from semilearn.datasets.augmentation import RandAugment
from semilearn.datasets.cv_datasets.datasetbase import BasicDataset


id_to_lbl43 = {
    0: 'Mixed forest',
    1: 'Coniferous forest',
    2: 'Non-irrigated arable land',
    3: 'Transitional woodland/shrub',
    4: 'Broad-leaved forest',
    5: 'Land principally occupied by agriculture, with significant areas of natural vegetation',
    6: 'Complex cultivation patterns',
    7: 'Pastures',
    8: 'Water bodies',
    9: 'Sea and ocean',
    10: 'Discontinuous urban fabric',
    11: 'Agro-forestry areas',
    12: 'Peatbogs',
    13: 'Permanently irrigated land',
    14: 'Industrial or commercial units',
    15: 'Natural grassland',
    16: 'Olive groves',
    17: 'Sclerophyllous vegetation',
    18: 'Continuous urban fabric',
    19: 'Water courses',
    20: 'Vineyards',
    21: 'Annual crops associated with permanent crops',
    22: 'Inland marshes',
    23: 'Moors and heathland',
    24: 'Sport and leisure facilities',
    25: 'Fruit trees and berry plantations',
    26: 'Mineral extraction sites',
    27: 'Rice fields',
    28: 'Road and rail networks and associated land',
    29: 'Bare rock',
    30: 'Green urban areas',
    31: 'Beaches, dunes, sands',
    32: 'Sparsely vegetated areas',
    33: 'Salt marshes',
    34: 'Coastal lagoons',
    35: 'Construction sites',
    36: 'Estuaries',
    37: 'Intertidal flats',
    38: 'Airports',
    39: 'Dump sites',
    40: 'Port areas',
    41: 'Salines',
    42: 'Burnt areas'
}

id_to_lbl19 = {
    0: 'Mixed forest',
    1: 'Coniferous forest',
    2: 'Arable land',
    3: 'Transitional woodland, shrub',
    4: 'Broad-leaved forest',
    5: 'Land principally occupied by agriculture, with significant areas of natural vegetation',
    6: 'Complex cultivation patterns',
    7: 'Pastures',
    8: 'Inland waters',
    9: 'Marine waters',
    10: 'Urban fabric',
    11: 'Agro-forestry areas',
    12: 'Inland wetlands',
    13: 'Industrial or commercial units',
    14: 'Natural grassland and sparsely vegetated areas',
    15: 'Permanent crops',
    16: 'Moors, heathland and sclerophyllous vegetation',
    17: 'Beaches, dunes, sands',
    18: 'Coastal wetlands'
}

lbl_to_id43 = {lbl: id for id, lbl in id_to_lbl43.items()}
lbl_to_id19 = {lbl: id for id, lbl in id_to_lbl19.items()}

lbl43_to_lbl19 = ben_constants.OLD2NEW_LABELS_DICT
id43_to_id19 = {id43: lbl_to_id19.get(
lbl43_to_lbl19[lbl43]) for id43, lbl43 in id_to_lbl43.items()}


def get_bigearthnet(args, alg, dataset, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):
    crop_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        aug.Resize(crop_size),
        aug.RandomCrop((crop_size, crop_size), padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        aug.RandomHorizontalFlip(),
        aug.RandomVerticalFlip(),
    ])
    transform_strong = transforms.Compose([
        aug.Resize(crop_size), 
        aug.RandomCrop((crop_size, crop_size), padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'), 
        aug.RandomHorizontalFlip(),
        aug.RandomVerticalFlip(),
        aug.auto.RandAugment(n=3, m=5),
    ])

    transform_val = transforms.Compose([
        aug.Resize(crop_size),
    ])

    # construct datasets for training and testing
    #if num_labels > 0:
    if 0 < num_labels < 1:
        train_labeled_dataset = BigEarthNet(
            alg, data_dir, 
            split=f"train{num_labels}_lb_{num_classes}", 
            num_classes=num_classes,
            transform=transform_weak, 
            transform_strong=transform_strong,
            pt_data_dir=args.pt_data_dir,
            no_loc=args.no_loc if hasattr(args, 'no_loc') else False,
            no_time=args.no_time if hasattr(args, 'no_time') else False
        )
        train_unlabeled_dataset = BigEarthNet(
            alg, data_dir, 
            split=f"train{num_labels}_ulb_{num_classes}", 
            num_classes=num_classes, 
            is_ulb=True,
            transform=transform_weak, 
            transform_strong=transform_strong,
            pt_data_dir=args.pt_data_dir,
            no_loc=args.no_loc if hasattr(args, 'no_loc') else False,
            no_time=args.no_time if hasattr(args, 'no_time') else False
        )
    elif num_labels == -1:
        train_labeled_dataset = BigEarthNet(
            alg, data_dir, 
            split=f"train", 
            num_classes=num_classes,
            transform=transform_weak, 
            transform_strong=transform_strong,
            pt_data_dir=args.pt_data_dir,
            no_loc=args.no_loc if hasattr(args, 'no_loc') else False,
            no_time=args.no_time if hasattr(args, 'no_time') else False
        )
        train_unlabeled_dataset = BigEarthNet(
            alg, data_dir, 
            split=f"train", 
            num_classes=num_classes, 
            is_ulb=True,
            transform=transform_weak, 
            transform_strong=transform_strong,
            pt_data_dir=args.pt_data_dir,
            no_loc=args.no_loc if hasattr(args, 'no_loc') else False,
            no_time=args.no_time if hasattr(args, 'no_time') else False
        )
    else:
        raise NotImplementedError
            
    val_dataset = BigEarthNet(
        alg, data_dir, 
        split="val", 
        num_classes=num_classes, 
        transform=transform_val, 
        pt_data_dir=args.pt_data_dir,
        no_loc=args.no_loc if hasattr(args, 'no_loc') else False,
        no_time=args.no_time if hasattr(args, 'no_time') else False
    )
    test_dataset = BigEarthNet(
        alg, data_dir, 
        split="test", 
        num_classes=num_classes, 
        transform=transform_val, 
        pt_data_dir=args.pt_data_dir,
        no_loc=args.no_loc if hasattr(args, 'no_loc') else False,
        no_time=args.no_time if hasattr(args, 'no_time') else False
    )


    print(f"#Labeled: {len(train_labeled_dataset)} #Unlabeled: {len(train_unlabeled_dataset)} "
          f"#Val: {len(val_dataset)}")

    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset


class BigEarthNet(VisionDataset, BasicDataset):
    def __init__(
        self, 
        alg, 
        root, 
        split, 
        num_classes=19, 
        bands=['B04', 'B03', 'B02'], 
        is_ulb=False, 
        transform=None, 
        target_transform=None, 
        transform_strong=None,
        pt_data_dir=None,
        no_loc=False,
        no_time=False
    ):
        """see comments at the beginning of the script"""
        super(BigEarthNet, self).__init__(
            root, transform=transform, target_transform=target_transform)

        self.is_ulb = is_ulb
        self.alg = alg
        self.root = root
        self.pt_data_dir = pt_data_dir
        self.split = split
        assert num_classes in [19, 43]
        self.num_classes = num_classes
        self.bands = bands
        self.strong_transform = transform_strong
        if self.strong_transform is None:
            if self.is_ulb:
                assert self.alg not in ['fullysupervised', 'supervised', 'pseudolabel', 'vat',
                                        'pimodel', 'meanteacher', 'mixmatch'], f"alg {self.alg} requires strong augmentation"
        split_file_path = os.path.join(root, f'splits/bigearthnet/{split}.pkl')
        self.df = pd.read_pickle(split_file_path)
        # rescale like here: https://github.com/ServiceNow/seasonal-contrast/blob/8285173ec205b64bc3e53b880344dd6c3f79fa7a/datasets/bigearthnet_dataset.py#L111
        self.min_value = np.array([ben_constants.BAND_STATS_S2['mean'][b] - 2 * ben_constants.BAND_STATS_S2['std'][b] for b in self.bands], dtype=np.float32)
        self.max_value = np.array([ben_constants.BAND_STATS_S2['mean'][b] + 2 * ben_constants.BAND_STATS_S2['std'][b] for b in self.bands], dtype=np.float32)
        self.mean = (np.array([ben_constants.BAND_STATS_S2['mean'][b] for b in self.bands], dtype=np.float32) - self.min_value) / \
                        (self.max_value - self.min_value).astype(np.float32)
        self.std = np.array([ben_constants.BAND_STATS_S2['std'][b] for b in self.bands], dtype=np.float32) / \
                        (self.max_value - self.min_value).astype(np.float32)
        
        self.no_loc = no_loc
        self.no_time = no_time
        if self.no_loc:
            self.df['lat'] = 0
            self.df['lng'] = 0
        if self.no_time:
            self.df['date'] = '2017-01-01 00:00:00'

        
    def _normalize(self, img):
        mean = self.mean.reshape(-1,1,1)
        std = self.std.reshape(-1,1,1)
        return (img - mean) / std

    def _load_img(self, patch):
        if self.pt_data_dir is None:
            paths = [os.path.join(self.root, f'bigearthnet/{patch}/{patch}_{band}.tif') for band in self.bands]
            bands = []
            for path in paths:
                band = Image.open(path)
                if band.size != (120, 120):
                    band = band.resize((120, 120))
                bands.append(np.array(band))
            img = np.stack(bands, axis=2)
            img = (img - self.min_value) / (self.max_value - self.min_value)
            img = img.clip(min=0, max=1)
            img = torch.from_numpy(img.transpose(2,0,1))
        else:
            pt_path = os.path.join(self.root, self.pt_data_dir, f'{patch}.pt')
            img = torch.load(pt_path).to(torch.float32)
        return img

    def __sample__(self, idx):
        sample = self.df.iloc[idx]
        img = self._load_img(sample.patch)
        target = sample.multi_hot_labels43 if self.num_classes == 43 else sample.multi_hot_labels19
        target = target.astype(float) # F.binary_cross_entropy_with_logits complains when target is long
        if self.transform is None:
            return  {'x_lb':  transforms.ToTensor()(img), 'y_lb': target}
        else:
            img_w = self.transform(img).squeeze(0)
            img_w = self._normalize(img_w)
            if not self.is_ulb:
                if 'defixmatch' in self.alg and self.strong_transform is not None:
                    # NOTE Strong augmentation on the labelled for DeFixMatch
                    img_s = self.strong_transform(img).squeeze(0)
                    img_s = self._normalize(img_s)
                    return {'idx_lb': idx, 'x_lb': img_w, 'x_lb_s': img_s, 'y_lb': target} 
                else:
                    return {'idx_lb': idx, 'x_lb': img_w, 'y_lb': target}
            else:
                if self.alg == 'fullysupervised' or self.alg == 'supervised':
                    return {'idx_ulb': idx}
                elif self.alg == 'pseudolabel' or self.alg == 'vat':
                    return {'idx_ulb': idx, 'x_ulb_w':img_w, 'y_ulb': target} 
                elif self.alg == 'pimodel' or self.alg == 'meanteacher' or self.alg == 'mixmatch':
                    # NOTE x_ulb_s here is weak augmentation
                    img_t = self.transform(img).squeeze(0)
                    img_t = self._normalize(img_t)
                    return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_s': img_t, 'y_ulb': target}
                elif self.alg == 'remixmatch':
                    rotate_v_list = [0, 90, 180, 270]
                    rotate_v1 = np.random.choice(rotate_v_list, 1).item()
                    img_s1 = self.strong_transform(img).squeeze(0)
                    img_s1 = self._normalize(img_s1)
                    img_s1_rot = transforms.functional.rotate(img_s1, rotate_v1)
                    img_s2 = self.strong_transform(img).squeeze(0)
                    img_s2 = self._normalize(img_s2)
                    return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_s_0': img_s1, 'x_ulb_s_1':img_s2, 'x_ulb_s_0_rot':img_s1_rot, 'rot_v':rotate_v_list.index(rotate_v1), 'y_ulb': target}
                elif self.alg == 'comatch':
                    img_s0 = self.strong_transform(img).squeeze(0)
                    img_s0 = self._normalize(img_s0)
                    img_s1 = self.strong_transform(img).squeeze(0)
                    img_s1 = self._normalize(img_s1)
                    return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_s_0': img_s0, 'x_ulb_s_1': img_s1, 'y_ulb': target} 
                else:
                    img_s = self.strong_transform(img).squeeze(0)
                    img_s = self._normalize(img_s)
                    return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_s': img_s, 'y_ulb': target} 

    def __getitem__(self, index): 
        sample_dict = self.__sample__(index)
        sample = self.df.iloc[index]  
        sample_dict[f"metainfo_{'ulb' if self.is_ulb else 'lb'}"] = {key:sample[key] \
            for key in ['patch', 'lat', 'lng', 'country', 'date', 'season', 'original_split']}
        return sample_dict

    def __len__(self):
        return len(self.df)


if __name__ == '__main__':
    save_dir = 'path/to/bigearthnet_rgb_pt'
    complete_dataset = BigEarthNet(
        alg=None,
        root='./data',
        split='all',
        num_classes=43,
        bands=['B04', 'B03', 'B02'],
        is_ulb=True,
        transform=None,
        target_transform=None,
        transform_strong=None
    )
    patches = complete_dataset.df.patch.values
    os.makedirs(save_dir, exist_ok=False)

    print(f'Start: Saving {len(patches)} imgs as pt-tensors in {save_dir}.')
    for patch in tqdm(patches, total=len(patches)):
        img = complete_dataset._load_img(patch)
        torch.save(img, os.path.join(save_dir, f'{patch}.pt'))
    print('Done.')

