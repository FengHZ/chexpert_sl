from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch
from collections import OrderedDict
import numpy as np
from PIL import Image
import re
from os import path
import pandas as pd


class ChexPertDataset(Dataset):
    def __init__(self, dataset_base_path, data_name_list, annotation_frame, transform):
        """
        :param dataset_base_path: the base path store the CheXpert dataset
        :param data_name_list: item is the row name of csv for CheXpert dataset
        :param annotation_frame: the data_frame for CheXpert
        :param transform: the transform for data
        """
        self._dataset_base_path = dataset_base_path
        self._annotation_frame = annotation_frame
        self._data_name_list = data_name_list
        # create label map for the 5 diseases
        self._label_map = OrderedDict(
            {"Atelectasis": {"0": 0, "1": 1, "-1": 1}, "Cardiomegaly": {"0": 0, "1": 1, "-1": np.nan},
             "Consolidation": {"0": 0, "1": 1, "-1": 0}, "Edema": {"0": 0, "1": 1, "-1": np.nan},
             "Pleural Effusion": {"0": 0, "1": 1, "-1": np.nan}})
        # set path as location
        self._annotation_frame.set_index("Path", inplace=True)
        self._transform = transform

    def __getitem__(self, index):
        data_path = self._data_name_list[index]
        annotation = self._annotation_frame.loc[data_path]
        label = torch.ones(5)
        weight = torch.ones(5)
        for idx, (pathology, map_dict) in enumerate(self._label_map.items()):
            if np.isnan(annotation[pathology]):
                # nan in the csv means negative, which is zero
                label[idx] = 0
            else:
                # the annotation[pathology] is an numpy array with
                item_label = map_dict[str(int(annotation[pathology]))]
                if np.isnan(item_label):
                    # means uncertainty so we make weight into zero
                    weight[idx] = 0
                else:
                    label[idx] = item_label
        # here we complete the data path
        data_path = path.join(self._dataset_base_path, data_path)
        # here we do resize and normalization
        image = Image.open(data_path)
        image = self._transform(image)
        image_name = re.match(r'(.*)/CheXpert(.*)', data_path).group(2)[1:]
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


def chexpert_dataset(dataset_base_path, image_size=(320, 384), augment_prob=0.5, rotate_angle=10, train_flag=True):
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
    data_name_list = [img_name for img_name in annotation_frame.Path if "frontal" in img_name]
    dataset = ChexPertDataset(dataset_base_path, data_name_list, annotation_frame, transform)
    return dataset


if __name__ == "__main__":
    from glob import glob
    from torch.utils.data import DataLoader
