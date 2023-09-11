from torch.utils.data import Dataset
from keras.preprocessing import image
import numpy as np
from utils import np2tensor
from utils import np2tensor_int
import random
from torchvision import transforms
from torch.utils.data import DataLoader
import torch


class FIW(Dataset):
    def __init__(self,
                 sample_path,
                 transform=None):

        self.sample_path=sample_path
        self.transform=transform
        self.sample_list=self.load_sample()
        self.bias=0
        self.race_dict={'AA':np.array([0], dtype=np.long),'A':np.array([1], dtype=np.long),'C':np.array([2], dtype=np.long),'I':np.array([3], dtype=np.long),\
            'AA&AA':np.array([4], dtype=np.long),'AA&A':np.array([5], dtype=np.long),'AA&C':np.array([6], dtype=np.long),'AA&I':np.array([7], dtype=np.long),\
            'A&AA':np.array([8], dtype=np.long),'A&A':np.array([9], dtype=np.long),'A&C':np.array([10], dtype=np.long),'A&I':np.array([11], dtype=np.long),\
            'C&AA':np.array([12], dtype=np.long),'C&A':np.array([13], dtype=np.long),'C&C':np.array([14], dtype=np.long),'C&I':np.array([15], dtype=np.long),\
            'I&AA':np.array([16], dtype=np.long),'I&A':np.array([17], dtype=np.long),'I&C':np.array([18], dtype=np.long),'I&I':np.array([19], dtype=np.long)}


    def load_sample(self):
        sample_list= []
        f = open(self.sample_path, "r+", encoding='utf-8')
        while True:
            line = f.readline().replace('\n', '')
            if not line:
                break
            else:
                tmp = line.split(' ')
                if tmp[4] == "AA":
                    race = 0
                elif tmp[4] == "A":
                    race = 1
                elif tmp[4] == "C":
                    race = 2
                elif tmp[4] == "I":
                    race = 3

                if tmp[2] == "fs":
                    kinship = 0
                elif tmp[2] == "fd":
                    kinship = 1
                elif tmp[2] == "ms":
                    kinship = 2
                elif tmp[2] == "md":
                    kinship = 3
                elif tmp[2] == "fms":
                    kinship = 4
                elif tmp[2] == "fmd":
                    kinship = 5
                elif tmp[2] == "fsd":
                    kinship = 6
                elif tmp[2] == "msd":
                    kinship = 7
                sample_list.append([tmp[0], tmp[1], kinship, tmp[3], tmp[4]])
        f.close()
        return sample_list

    def __len__(self):
        return len(self.sample_list)

    def read_image(self, path):
        img = image.load_img(path, target_size=(112, 112))
        return img

    def set_bias(self,bias):
        self.bias=bias%self.__len__()

    def preprocess(self, img):
        return np.transpose(img, (2, 0, 1))

    def __getitem__(self, item):
        sample = self.sample_list[(item+self.bias)%self.__len__()]
        img1,img2=self.read_image(sample[0]),self.read_image(sample[1])
        if self.transform is not None:
            img1,img2 = self.transform(img1),self.transform(img2)
        img1, img2 = np2tensor(self.preprocess(np.array(img1, dtype=float))), \
                     np2tensor(self.preprocess(np.array(img2, dtype=float)))
        
        label = np2tensor(np.array(sample[3], dtype=float))
        race = torch.squeeze(np2tensor(self.race_dict[sample[4]])).to(torch.long)
        kinship = np2tensor_int(np.array(sample[2], dtype=int))

        return img1, img2, kinship, label, race
