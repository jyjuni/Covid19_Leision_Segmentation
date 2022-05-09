import nibabel as nib
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


# output_dir = os.path.join(data_dir, "train/")
# images = sorted(glob.glob(os.path.join(data_folder, "*_ct.nii.gz")))
# masks = sorted(glob.glob(os.path.join(data_folder, "*_seg.nii.gz")))

# # for i in range(len(images)):
# # for i in range(5):
# #     img = nib.load(images[i])
# #     img_mat = img.get_fdata()[:,:,:]
# #     for idx in range(img_mat.shape[-1]):
# #         filename = os.path.join(output_dir, "img{0:03d}_{1:03d}.png".format(i,idx))
# #         print(filename)
# #         cv2.imwrite(filename, img_mat[:,:,idx])

# img = nib.load(images[0])
# mask = nib.load(masks[0])

# data = img.get_fdata()[:,:,:]
# mask = mask.get_fdata()[:,:,:]
# for i in range(1, 5):
#     print(f"image {i}")
#     data = np.append(data, nib.load(images[i]).get_fdata(), axis=-1)
#     mask = np.append(mask, nib.load(masks[i]).get_fdata(), axis=-1)
# data.shape


def read_nii(filepath):
    img = nib.load(filepath)
    img_array   = img.get_fdata()
    img_array   = np.rot90(np.array(img_array))
    return(img_array)

def load_niis(data_folder, offset = 0, n=10):
    imgs = []
    segs = []
    img_size = 128
    img_filenames = sorted(glob.glob(os.path.join(data_folder, "*_ct.nii.gz")))
    seg_filenames = sorted(glob.glob(os.path.join(data_folder, "*_seg.nii.gz")))
    print(len(img_filenames))
    # for i in range(len(data)):
    for i in range(offset, offset+n):
        img = read_nii(img_filenames[i])
        print("image", i, img.shape)
        mask = read_nii(seg_filenames[i])
        
        for ii in range(img.shape[-1]):
            lung_img = cv2.resize(img[:, :, ii], dsize = (img_size, img_size),interpolation = cv2.INTER_AREA).astype('uint8')
            infec_img = cv2.resize(mask[:, :, ii],dsize=(img_size, img_size),interpolation = cv2.INTER_AREA).astype('uint8')
            imgs.append(lung_img[..., np.newaxis])
            segs.append(infec_img[..., np.newaxis])
    imgs = np.array(imgs)
    segs = np.array(segs)
    return imgs, segs

# data_folder = "/home/jinyijia/bme4460/data_COVID-19-20_v2/Train/"
# train_imgs, train_segs = load_niis(data_folder)




## Visualize pair
# i = 100
# show_pair(train_imgs[i], train_segs[i], i)
# pd.DataFrame(train_imgs[i].flatten()).describe()
# train_imgs.shape, train_segs.shape


class BasicDataset(TensorDataset):
    # This function takes folder name ('train', 'valid', 'test') as input and creates an instance of BasicDataset according to that folder.
    # Also if you'd like to have less number of samples (for evaluation purposes), you may set the `n_sample` with an integer.
    def __init__(self, folder, n_sample=None, transforms=None):
        if folder == "Validation":
            self.input_dir = os.path.join(data_dir, "Train")
            self.imgs, self.segs = load_niis(self.input_dir, offset = 20, n = 3)
        else:
            self.input_dir = os.path.join(data_dir, folder)
            self.imgs, self.segs = load_niis(self.input_dir)

        if not n_sample or n_sample > len(self.imgs):
            n_sample = len(self.imgs)
            print(n_sample)
        
        self.transforms = transforms
        
        # If n_sample is not None (It has been set by the user)
        if not n_sample or n_sample > len(self.pairs_file):
            n_sample = len(self.pairs_file)
        
        self.n_sample = n_sample
        self.ids = list([i+1 for i in range(n_sample)])

            
    # This function returns the lenght of the dataset (AKA number of samples in that set)
    def __len__(self):
        return self.n_sample
    
    
    # This function takes an index (i) which is between 0 to `len(BasicDataset)` (The return of the previous function), then returns RGB image, 
    # mask (Binary), and the index of the file name (Which we will use for visualization). The preprocessing step is also implemented in this function.
    def __getitem__(self, i):
        idx = self.ids[i]

        # img, mask = self.imgs[i], self.segs[i]
        
        file_name = glob.glob(os.path.join(self.pairs_dir, '*_pairs_{0:04d}.npy'.format(idx)))
        assert len(file_name) == 1
        # print(file_name[0])
        img, mask = np.load(file_name[0])

        # # Image histogram equalizaition
        img = np.array(img * 255, dtype = np.uint8)
        
        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)

        # Resize all images from 512 to 256 (H and W)
        img = cv2.resize(img, (128,128), interpolation = cv2.INTER_AREA).astype('uint8')
        mask = cv2.resize(mask, (128,128), interpolation = cv2.INTER_AREA).astype('uint8')

        
        # Scale between 0 to 1
        img = np.array(img) / 255.0
        
        # Make sure that the mask are binary (0 or 1)
        mask[mask <= 0.5] = 0
        mask[mask > 0.5] = 1

        # Add an axis to the image array so that it is in [channel, height, width] format.
        img = np.expand_dims(img, axis=0)
        
        # HWC to CHW
        # img = np.transpose(img, (2, 0, 1))
        # mask = np.transpose(mask, (2, 0, 1))
                    
        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.LongTensor),
            'img_id': idx
        }


## split data
valid_test_dataset = BasicDataset('PairedValidData', n_sample=400)

def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets['train'], datasets['val']

valid_dataloader, test_dataset = train_val_dataset(valid_test_dataset, val_split=0.5)

