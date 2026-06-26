import gdown
import tarfile
import os
import torch 
from glob import glob
from torch.utils.data.dataset import Dataset
from PIL import Image
import gzip
import os
#os.chdir('./Aracnonet/')


class CIFAR10C(Dataset):


    
    def __init__(self, split, bias_amount, transform=None, root="./data",image_path_list=None):
        super().__init__()
        

        
        if bias_amount==95:
            percent='5pct'

        elif bias_amount==99:
            percent = '1pct'
        
        elif bias_amount==98:
            percent='2pct'
        elif bias_amount==99.5:
            percent = '0.5pct'
        else:
            percent='1pct'
        if not os.path.isdir(os.path.join(root, 'cifar10c')):
            self.download_dataset(root)
        

        print(percent)
       
        root = os.path.join(root, 'cifar10c', percent)

        self.transform = transform
        self.root = root
        self.image2pseudo = {}
        self.image_path_list = image_path_list
        if split == 'train':
            self.align = sorted(glob(os.path.join(root, 'align',"*","*")))
            self.conflict = sorted(glob(os.path.join(root, 'conflict',"*","*")))
            self.data = self.align + self.conflict
          

        elif split == 'val':
            self.data = sorted(glob(os.path.join(root,split,"*", "*")))

        elif split == 'test':
            self.data = sorted(glob(os.path.join(root, '../test',"*","*")))

    def download_dataset(self, path):
        url = "https://drive.google.com/file/d/1_eSQ33m2-okaMWfubO7b8hhvLMlYNJP-/view?usp=sharing"
        output = os.path.join(path, 'cifar10c.tar.gz')
        print(f'=> Downloading corrupted CIFAR10 dataset from {url}')
        gdown.download(url, output, quiet=False)

        print('=> Extracting dataset..')
        with tarfile.open(os.path.join(path, 'cifar10c.tar.gz'), 'r:gz') as tar:
            tar.extractall(path=path)
    
        os.remove(output)

    def __len__(self):
        return len(self.data)



    def perclass_populations(self, return_labels: bool = False):
        labels: torch.Tensor = torch.zeros(len(self))
        for i in range(len(self)):
            labels[i] = self[i][1][0]

        _, pop_counts = labels.unique(return_counts=True)

        if return_labels:
            return pop_counts.long(), labels.long()

        return pop_counts




    def __getitem__(self, index):
        label, bias = int(self.data[index].split('_')[-2]), int(self.data[index].split('_')[-1].split('.')[0])
        image = Image.open(self.data[index]).convert('RGB')
        # image = Image.open(self.data[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, (label, bias), index

if __name__ == '__main__':
    dataset = CIFAR10C(f'.\data\cifar', 'train', "0.5pct")
