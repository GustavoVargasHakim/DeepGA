# -*- coding: utf-8 -*-
#Dataset Class

from torch.utils.data import Dataset, DataLoader
import torch
#import cv2
from PIL import Image
import numpy as np
from torchvision import transforms

#Arrays of labels
c_labels = np.ones(918, dtype = np.int8) #Codiv
n_labels = np.zeros(918, dtype = np.int8) #Normal
p_labels = np.full((918,), 2) #Pneumonia

#Images path
c_root = '/home/proy_ext_adolfo.vargas/DeepGA/Images/COVID-19/covidp'
n_root = '/home/proy_ext_adolfo.vargas/DeepGA/Images/Normal/normalp'
p_root = '/home/proy_ext_adolfo.vargas/DeepGA/Images/Pneumonia/pneumoniap'

class CovidDataset(Dataset):
  def __init__(self, root, labels, transform = None):
    self.root = root #The folder path
    self.labels = labels #Labels array
    self.transform = transform #Transform composition
  
  def __len__(self):
    return len(self.labels)
  
  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    p_root = self.root[:] 
    img_name_p = p_root + str(idx+1) + '.png'
    #image_p = cv2.imread(img_name_p, 0)
    image_p = np.array(Image.open(img_name_p))
    [H, W] = image_p.shape
    image_p = image_p.reshape((H,W,1))
    p_label = self.labels[idx]
    sample = {'image': image_p, 'label': p_label}

    if self.transform:
      sample = self.transform(sample)

    return sample

#Class to transform image to tensor
class ToTensor(object):
  def __call__(self, sample):
    image, label = sample['image'], sample['label']

    #Swap dimmensions because:
    #       numpy image: H x W x C
    #       torch image: C x H x W
    #print(image.shape)
    image = image.transpose((2,0,1))
    #print(image.shape)
    return {'image':torch.from_numpy(image),
            'label':label}

def loading_data():
    #Loading Datasets
    covid_ds = CovidDataset(root = c_root, labels = c_labels, transform = transforms.Compose([ToTensor()]))
    normal_ds = CovidDataset(root = n_root, labels = n_labels, transform = transforms.Compose([ToTensor()]))
    pneumonia_ds = CovidDataset(root = p_root, labels = p_labels, transform = transforms.Compose([ToTensor()]))
    
    #Merging Covid, normal, and pneumonia Datasets
    dataset = torch.utils.data.ConcatDataset([covid_ds, normal_ds, pneumonia_ds])
    lengths = [int(len(dataset)*0.7), int(len(dataset)*0.3)+1]
    train_ds, test_ds = torch.utils.data.random_split(dataset = dataset, lengths = lengths)
    
    #i = 1836
    #Testing
    #print("Length of Training Dataset: {}".format(len(train_ds)))
    #print("Length of Test Dataset: {}".format(len(test_ds)))
    #print("Shape of images as tensors: {}".format(dataset[i]['image'].shape))
    #print("Label of image i: {}".format(dataset[i]['label']))
    
    #Creating Dataloaders
    train_dl = DataLoader(train_ds, batch_size = 24, shuffle = True)
    test_dl = DataLoader(test_ds, batch_size = 24, shuffle = True)
    
    return train_dl, test_dl

#train_dl, test_dl = loading_data()