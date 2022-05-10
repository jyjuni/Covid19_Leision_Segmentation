import glob
import os
import cv2
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class BasicDataset(TensorDataset):
    def __init__(self, folder, n_sample=None, transforms=None):
        """
        takes folder name ('train', 'valid', 'test') as input and creates an instance of BasicDataset according to that folder.
        Also if you'd like to have less number of samples (for evaluation purposes), you may set the `n_sample` with an integer.
        """
        # loading
        self.folder = folder
        self.pairs_file = sorted(glob.glob(os.path.join(self.folder, '*.npy')))
                
        # sampling
        if not n_sample or n_sample > len(self.pairs_file):
            n_sample = len(self.pairs_file)
        self.n_sample = n_sample
        self.ids = list([i+1 for i in range(n_sample)])
          
        # transformations
        self.transforms = transforms
            
    def __len__(self):
        """return length of the dataset (AKA number of samples in that set)"""
        return self.n_sample
    
    def __getitem__(self, i):
        """
        takes: an index (i) which is between 0 to `len(BasicDataset)` (The return of the previous function)
        returns: grayscale image, mask (Binary), and the index of the file name (will use for visualization)
        The preprocessing step is also implemented in this function.
        """
        file_path = self.pairs_file[i]
        idx = self.ids[i]
        data = np.load(file_path, allow_pickle=True)

        img = data[0,:,:]
        mask = data[1,:,:]
        # show_pair(img, mask, idx)

        img = np.array(img * 255, dtype = np.uint8)
        # img = cv2.equalizeHist(img)
        
        # resize img
        img_size = 256
        img = cv2.resize(img, (img_size, img_size), interpolation = cv2.INTER_AREA).astype('uint8')
        mask = cv2.resize(mask, (img_size, img_size), interpolation = cv2.INTER_AREA).astype('uint8')

        # Scale img between 0 to 1
        img = np.array(img) / 255.0
        
        # convert mask to binary
        mask[mask <= 0.5] = 0
        mask[mask > 0.5] = 1

        # add channel axis
        img = np.expand_dims(img, axis=0)
        # mask = np.expand_dims(mask, axis=0)

        # # HWC to CHW
        # img = np.transpose(img, (2, 0, 1))

        # any customized transforms
        if self.transforms:
            img = self.transforms(image=img, mask=mask)
                    
        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.LongTensor),
            'img_id': idx
        }