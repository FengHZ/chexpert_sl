from torchvision import transforms
from torch.utils.data import Dataset
import torch
from collections import OrderedDict
import numpy as np
from PIL import Image
import re
from os import path
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler


class ChexPertDataset(Dataset):
    def __init__(self, dataset_base_path, data_name_list, label, weight, transform):
        """
        :param dataset_base_path: the base path store the CheXpert dataset
        :param data_name_list: item is the row name of csv for CheXpert dataset
        :param annotation_frame: the data_frame for CheXpert
        :param transform: the transform for data
        """
        self._dataset_base_path = dataset_base_path
        self._data_name_list = data_name_list
        self._transform = transform
        self._label = label
        self._weight = weight

    def __getitem__(self, index):
        data_path = self._data_name_list[index]
        label = self._label[index, :]
        weight = self._weight[index, :]
        # here we complete the data path
        data_path = path.join(self._dataset_base_path, data_path)
        # here we do resize and normalization
        image = Image.open(data_path)
        image = self._transform(image)
        image_name = re.match(r'(.*)/CheXpert/(.*)', data_path).group(2)
        # replace the / in imagename into |
        image_name = image_name.replace("/", "|")
        if image_name is None:
            raise ValueError("re can't find the image name")
        return image, index, image_name, weight, label

    def __len__(self):
        return len(self._data_name_list)

    @staticmethod
    def add_sp_noise(image, s_vs_p=0.5, amount=0.02):
        """
        :param s_vs_p : the ratio of salt vs pepper
        :param amount the number of pixels we need to add noise
        :return: image with added noise
        """
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        image[tuple(coords)] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        image[tuple(coords)] = 0
        return image


def chexpert_dataset(dataset_base_path, image_size=(320, 320), augment_prob=0.5, rotate_angle=10, train_flag=True,
                     generate_label=False):
    label_map = OrderedDict(
        {"Atelectasis": {"0": 0, "1": 1, "-1": 1}, "Cardiomegaly": {"0": 0, "1": 1, "-1": np.nan},
         "Consolidation": {"0": 0, "1": 1, "-1": 0}, "Edema": {"0": 0, "1": 1, "-1": np.nan},
         "Pleural Effusion": {"0": 0, "1": 1, "-1": np.nan}})
    if train_flag:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=augment_prob),
            transforms.RandomVerticalFlip(p=augment_prob),
            transforms.RandomRotation(rotate_angle),
            transforms.ToTensor()
        ])
        csv_path = path.join(dataset_base_path, "CheXpert-v1.0-small", "train.csv")
    else:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
        csv_path = path.join(dataset_base_path, "CheXpert-v1.0-small", "valid.csv")
    annotation_frame = pd.read_csv(csv_path)
    data_name_list = [img_name for img_name in annotation_frame.Path]
    if generate_label:
        label, weight = generate_label_weight(annotation_frame, data_name_list, label_map)
    else:
        if train_flag:
            label_path = path.join(dataset_base_path, "CheXpert-v1.0-small", "train.pth.tar")
        else:
            label_path = path.join(dataset_base_path, "CheXpert-v1.0-small", "valid.pth.tar")
        label_dict = torch.load(label_path)
        label = label_dict["label"]
        weight = label_dict["weight"]
    dataset = ChexPertDataset(dataset_base_path, data_name_list, label, weight, transform)
    return dataset


def get_chexpert_ssl_sampler(labels, weights, annotated_ratio):
    sampler_train_l = []
    for i in range(5):
        specified_labels = labels[:, i]
        specified_weights = weights[:, i]
        # sample label 0
        loc = torch.nonzero((specified_labels == 0) * (specified_weights == 1)).view(-1)
        annotated_num = round(loc.size(0) * annotated_ratio)
        loc = loc[torch.randperm(loc.size(0))]
        sampler_train_l.extend(loc[:annotated_num].tolist())
        # sample label 1
        loc = torch.nonzero((specified_labels == 1) * (specified_weights == 1)).view(-1)
        annotated_num = round(loc.size(0) * annotated_ratio)
        loc = loc[torch.randperm(loc.size(0))]
        sampler_train_l.extend(loc[:annotated_num].tolist())
    # drop the repeat label
    sampler_train_l = list(set(sampler_train_l))
    sampler_train_l = SubsetRandomSampler(sampler_train_l)
    return sampler_train_l


def generate_label_weight(annotation_frame, data_name_list, label_map):
    annotation_frame.set_index("Path", inplace=True)
    label = torch.ones(len(data_name_list), 5)
    weight = torch.ones(len(data_name_list), 5)
    for idx, data_path in enumerate(data_name_list):
        annotation = annotation_frame.loc[data_path]
        for jdx, (pathology, map_dict) in enumerate(label_map.items()):
            if np.isnan(annotation[pathology]):
                # nan in the csv means negative, which is zero
                label[idx][jdx] = 0
            else:
                # the annotation[pathology] is an numpy array with
                item_label = map_dict[str(int(annotation[pathology]))]
                if np.isnan(item_label):
                    # means uncertainty so we make weight into zero
                    weight[idx][jdx] = 0
                else:
                    label[idx][jdx] = item_label
    return label, weight


if __name__ == "__main__":
    from glob import glob
    from torch.utils.data import DataLoader
