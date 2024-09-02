from torch.utils.data.dataset import Dataset
import os
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import time
from torch.autograd import Variable
import scipy.io
from scipy.sparse import csc_matrix




class TrainDataset(Dataset):
    def __init__(self, train_img_path, transform=None):
        super(TrainDataset, self).__init__()
        self.img_hr = os.listdir(train_img_path)
        self.transform = transform
        self.hr_and_lr = []
        for i in range(len(self.img_hr)):
            self.hr_and_lr.append(
                    os.path.join(train_img_path, self.img_hr[i])
                )

    def __getitem__(self, item):
        hr_path = self.hr_and_lr[item]
        Data = scipy.io.loadmat(hr_path)  # 读取mat文件
        X = Data['data']
        # Y = Data['y']
        X = X / 1.0
        # Y = Y / 1.0

        return X

    def __len__(self):
        return len(self.img_hr)

if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor()])
    data = TrainDataset('data/TrainData', transform)
    print(len(data))
    data_loader = DataLoader(data, batch_size=1, shuffle=True)
    sample = next(iter(data_loader))
    print(sample[1][0].shape)