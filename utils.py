from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10
import torch
import json
import numpy as np
import torchvision.transforms.functional as F
#from torchvision.transforms import ToPILImage
NUMINT = 18*3-2#12
from pdb import set_trace as st
from typing import Any, Callable, List, Optional, Sequence, Type, Union
import random
from PIL import ImageFilter
import os
#from dataset_3d import ShapeNetCore
'''from scipy.spatial.transform import Rotation
from pytorch3d.renderer.mesh.shader import HardFlatShader
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    TexturesVertex,
    look_at_view_transform,
)'''

class GaussianBlur:
    def __init__(self, sigma: Sequence[float] = None):
        """Gaussian blur as a callable object.

        Args:
            sigma (Sequence[float]): range to sample the radius of the gaussian blur filter.
                Defaults to [0.1, 2.0].
        """

        if sigma is None:
            sigma = [0.1, 2.0]

        self.sigma = sigma

    def __call__(self, img: Image) -> Image:
        """Applies gaussian blur to an input image.

        Args:
            img (Image): an image in the PIL.Image format.

        Returns:
            Image: blurred image.
        """

        sigma = random.uniform(self.sigma[0], self.sigma[1])
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img


train_transform_sp32 = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply(
        [transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)],
        p=0.8,
    ),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur()], p=0.5),
    #transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    #transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])


test_transform_sp32 = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

train_transform_mnist_full = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.ToTensor(),
    transforms.Normalize([0.4914], [0.2023])])

test_transform_mnist_full = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize([0.4914], [0.2023])])

train_transform_sp_full = transforms.Compose([
    transforms.CenterCrop(80),
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply(
        [transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)],
        p=0.8,
    ),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur()], p=0.5),
    #transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    #transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform_sp_full = transforms.Compose([
    transforms.CenterCrop(75),
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

SIZE = 32#48


train_transform_sp = transforms.Compose([
    transforms.RandomResizedCrop(SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply(
        [transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)],
        p=0.8,
    ),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur()], p=0.5),
    #transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    #transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
'''train_transform_sp = transforms.Compose([
    #transforms.CenterCrop(80),
    transforms.Resize(32),#transforms.RandomResizedCrop(32),
    #transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])'''

test_transform_sp = transforms.Compose([
    transforms.Resize(SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])


def pil_loader(path, to_grayscale=False):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    img = Image.open(path)
    if to_grayscale:
        img = img.convert('L')
    return img

def get_adj_pose_id(pose_id, max_id=23):
    in_id = int(pose_id.split('/')[-1].split('.')[0])

    if not in_id:
        l_id = max_id
        r_id = in_id + 1
    elif in_id == max_id:
        l_id = max_id - 1
        r_id = 0
    else:
        l_id = in_id - 1
        r_id = in_id + 1

    return pose_id.replace(str(in_id)+".png", str(l_id)+".png"), pose_id.replace(str(in_id)+".png", str(r_id)+".png")

def get_adj_pose_id_only(pose_id, max_id=17):
    in_id = int(pose_id)

    if not in_id:
        l_id = max_id
        r_id = in_id + 1
    elif in_id == max_id:
        l_id = max_id - 1
        r_id = 0
    else:
        l_id = in_id - 1
        r_id = in_id + 1
    return l_id, r_id
        
class Dataset_pose(torch.utils.data.Dataset):
    def __init__(self, train_F, master_dir = "/ssd/peterwg/s3dis/area_1/data/rgb", transform=None):
        'Initialization'
        with open(train_F, 'r') as openfile: 
            data = json.load(openfile) 
        data = np.array(data)
        #self.labels = data[:,1].astype(int)
        self.data = data[:,0]
        self.targets = data[:,1].astype(int)
        self.classes = np.unique(self.targets)
        self.transform = transform
        #self.master_dir = master_dir

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        ids = img.split('/')[-1]
        meta_path = img.replace('/ShapeNetRendering_scale/', '/ShapeNetRendering/').replace(ids, "rendering/rendering_metadata.txt")
        meta_name = np.loadtxt(meta_path)
        id_2 = np.random.choice(24,1, replace=False)[0]
        
        img2 = pil_loader(img.replace(ids, str(id_2)+".png"))
        img = pil_loader(img)#Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img) #self.transform(img)
            pos_2 = self.transform(img2)
        angle = ((meta_name[int(ids.split(".")[0]), 0] - meta_name[id_2, 0])/360.0 ).astype(np.float32)
        if angle <0:angle+=1
        return pos_1, pos_2, np.floor(angle*10).astype(int)
    
class Dataset_3d_single(torch.utils.data.Dataset):
    def __init__(self, train_F, master_dir = "/home/peterwg/dataset/ShapeNetRendering3d", transform=None, cat_only=None):
        'Initialization'
        with open(train_F, 'r') as openfile: 
            data = json.load(openfile) 
        data = np.array(data)
        self.data = data[:,0]
        #self.targets = data[:,1].astype(int)
        #self.classes = np.unique(self.targets)
        self.transform = transform
        
        self.datapath = master_dir 
        
        if cat_only is not None:
            self.data = [i for i in self.data if "/"+cat_only+'/' in i]
        all_data = []
        for i in self.data:
            for num in range(18):
                all_data.append(os.path.join(i, "x_"+str(num)+".png"))
                all_data.append(os.path.join(i, "y_"+str(num)+".png"))
                all_data.append(os.path.join(i, "z_"+str(num)+".png"))
        self.data = all_data
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)
    
    def __getitem__(self, index):
        img = self.data[index]
        cat_id, sid, img_id = img.split("/")[-3], img.split("/")[-2], img.split("/")[-1][:-4]
        axis_id = img_id.split("_")[0] #np.random.choice(['x', 'y', 'z'], 1, replace=False)[0]
        c_id = int(img_id.split("_")[1])#np.random.randint(0, 17)
        if axis_id == "y" and c_id:
            c_id += 17
        elif axis_id == "z" and c_id:
            c_id += 17*2

        img = pil_loader(img)#Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img) #self.transform(img)
        return pos_1, c_id
    
    def a__getitem__(self, index):
        img = self.data[index]
        cat_id, sid = img.split("/")[-2], img.split("/")[-1]
        axis_id = "x" #np.random.choice(['x', 'y', 'z'], 1, replace=False)[0]
        c_id = np.random.randint(0, 17)
        curnum = c_id
        
        c_id = os.path.join(self.datapath, cat_id, sid, axis_id+"_"+str(c_id)+".png")

        img = pil_loader(c_id)#Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img) #self.transform(img)

        return pos_1, curnum//2

class Dataset_pose_single(torch.utils.data.Dataset):
    def __init__(self, train_F, master_dir = "/ssd/peterwg/s3dis/area_1/data/rgb", transform=None, cat_only=None):
        'Initialization'
        with open(train_F, 'r') as openfile: 
            data = json.load(openfile) 
        data = np.array(data)
        self.data = data[:,0]

        if cat_only is not None:
            self.data = [i for i in self.data if "/"+cat_only+'/' in i]
            
        #self.labels = data[:,1].astype(int)
        #self.targets = data[:,1].astype(int)
        self.classes = np.arange(12) #np.arange(10) 
        self.transform = transform
        #self.master_dir = master_dir

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        img= self.data[index]
        ids = img.split('/')[-1]
        #meta_path = img.replace('/ShapeNetRendering_scale/', '/ShapeNetRendering/').replace(ids, "rendering/rendering_metadata.txt")
        #meta_name = np.loadtxt(meta_path)
        #ids = np.random_choice(24,2, replace=False)

        img = pil_loader(img)#Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img) #self.transform(img)
        angle = int(ids.split('.')[0]) # (meta_name[int(ids.split(".")[0]), 0] -1e-10).astype(np.float32)
        
        return pos_1, angle//2


class Dataset_3d_train(torch.utils.data.Dataset):
    def __init__(self, train_F, master_dir = "/ssd/peterwg/s3dis/area_1/data/rgb", transform=None, test_transform=test_transform_sp, cat_only=None, return_rot=False):
        'Initialization'
        with open(train_F, 'r') as openfile: 
            data = json.load(openfile) 
        if cat_only is not None:
            data = [i for i in data if cat_only+'/' in i[0]]
            
        data = np.array(data)
        #self.labels = data[:,1].astype(int)
        self.data = data[:,0]
        self.data2 = data[:,1]
        self.targets = data[:,2].astype(int)
        self.classes = np.unique(self.targets)
        self.transform = transform
        self.test_transform = test_transform
        self.datapath = "/ssd/peterwg/ShapeNetSO3_train/"#"/home/peterwg/dataset/ShapeNetRendering3d"
        if cat_only is not None:
            self.data = [i for i in self.data if cat_only+'/' in i]
        self.return_rot = return_rot

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    
    def __getitem__(self, index):
        img, img2, target = self.data[index], self.data2[index], self.targets[index]
        rot_target = int(img2.split('_')[0])
        #img = img.replace("/home/peterwg/dataset/ShapeNetRendering3d/", "/ssd/peterwg/ShapeNetSO3/")

        #cat_id, sid, img_id = img.split("/")[-3], img.split("/")[-2], img.split("/")[-1][:-4]
        #pose_id = img_id.split("_")[0] #np.random.choice(['x', 'y', 'z'], 1, replace=False)[0]
        #axis_id = img_id.split("_")[1] #np.random.choice(['x', 'y', 'z'], 1, replace=False)[0]
        #c_id = int(img_id.split("_")[2])
        
        #cat_id, sid = img.split("/")[-2], img.split("/")[-1]
        #axis_id = np.random.choice(['x', 'y', 'z'], 1, replace=False)[0]
        #c_id = np.random.randint(0, 17)
        #l_id, r_id = get_adj_pose_id_only(c_id)
        
        c_id = os.path.join(self.datapath, img, img2.replace("_0.png", "_1.png"))#, pose_id+"_"+axis_id+"_"+str(1)+".png")
        l_id = os.path.join(self.datapath, img, img2) #, pose_id+"_"+axis_id+"_"+str(0)+".png")
        r_id = os.path.join(self.datapath, img, img2.replace("_0.png", "_2.png") )#, pose_id+"_"+axis_id+"_"+str(2)+".png")

        try:
            img = pil_loader(c_id)#Image.fromarray(img)
        except:
            print(c_id)
            return self.__getitem__(0)
        img_l = pil_loader(l_id)
        img_r = pil_loader(r_id)        

        if self.transform is not None:
            pos_1 = self.transform(img) #self.transform(img)
            pos_2 = self.transform(img)
            if self.return_rot:
                data = (pos_1, pos_2)
                data = (data, target, rot_target)
                return (index, *data) 
            pos_c = self.test_transform(img)
            #pos_a = test_transform_sp(img)
            pos_l = self.test_transform(img_l)
            pos_r = self.test_transform(img_r)
            

        data = (pos_1, pos_2, pos_c, pos_l, pos_r)
        if self.return_rot:
            data = (data, target, rot_target)
        else:
            data = (data, target)
        ##cid
        '''axis_id = img_id.split("_")[0] #np.random.choice(['x', 'y', 'z'], 1, replace=False)[0]
        c_id = int(img_id.split("_")[1])#np.random.randint(0, 17)
        if axis_id == "y" and c_id:
            c_id += 17
        elif axis_id == "z" and c_id:
            c_id += 17*2
        data = (pos_1, pos_2, pos_c, pos_l, pos_r)
        data = (data, c_id)'''
        ##
        return (index, *data) 
    
class Dataset_3d_val(torch.utils.data.Dataset):
    def __init__(self, train_F, master_dir = None, transform=None, cat_only=None, trainset=False, give_pose=False, OOD=False, return_rot=False):
        'Initialization'
        with open(train_F, 'r') as openfile: 
            data = json.load(openfile) 
        if cat_only is not None:
            data = [i for i in data if cat_only+'/' in i[0]]
    
        data = np.array(data)
        #self.labels = data[:,1].astype(int)
        self.data = data[:,0]
        self.data2 = data[:,1]
        self.targets = data[:,2].astype(int)
        self.classes = np.unique(self.targets)
        self.transform = transform
        
        if trainset:
            self.datapath = "/ssd/peterwg/ShapeNetSO3_train"
            self.data2 = np.char.replace(self.data2, "_0.png", "_1.png" )
        elif OOD:
            self.datapath = "/ssd/peterwg/ShapeNetSO3_OOD" 
        else:
            self.datapath = "/ssd/peterwg/ShapeNetSO3" #"/home/peterwg/dataset/ShapeNetRendering3d"
        self.give_pose = give_pose
        self.trainset = trainset
        if master_dir:
            self.datapath = master_dir
        self.return_rot = return_rot
            
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        img, img2, target = self.data[index],self.data2[index], self.targets[index]
        rot_target = int(img2.split('.')[0])

        #img = img.replace("/home/peterwg/dataset/ShapeNetRendering3d/", "/ssd/peterwg/ShapeNetSO3/")
        #cat_id, sid, img_id = img.split("/")[-3], img.split("/")[-2], img.split("/")[-1][:-4]
        #axis_id = img_id.split("_")[0] #np.random.choice(['x', 'y', 'z'], 1, replace=False)[0]
        #c_id = int(img_id.split("_")[1])
        #cat_id, sid = img.split("/")[-2], img.split("/")[-1]
        #axis_id = np.random.choice(['x', 'y', 'z'], 1, replace=False)[0]
        #c_id = np.random.randint(0, 17)
        
        #c_id = os.path.join(self.datapath, cat_id, sid, axis_id+"_"+str(c_id)+".png")
        
        c_id = os.path.join(self.datapath, img, img2)
        try:
            img = pil_loader(c_id)#Image.fromarray(img)
        except:
            #print(c_id)
            return self.__getitem__(0)
            
        if self.transform is not None:
            pos_1 = self.transform(img) #self.transform(img)

        ##cid
        '''axis_id = img_id.split("_")[0] #np.random.choice(['x', 'y', 'z'], 1, replace=False)[0]
        c_id = int(img_id.split("_")[1])#np.random.randint(0, 17)
        if axis_id == "y" and c_id:
            c_id += 17
        elif axis_id == "z" and c_id:
            c_id += 17*2
        return pos_1, c_id'''
        ##
        if self.give_pose:
            if self.trainset:
                return pos_1, int(img2.split("_")[0])
            else:
                return pos_1, int(img2.replace('.png', ''))
        if self.return_rot:
            return pos_1, target, rot_target
        else:
            return pos_1, target
    
class Dataset_3d_val_rel(torch.utils.data.Dataset):
    def __init__(self, train_F, master_dir = None, transform=None, cat_only=None, train_transform=None,TEST_ODD_CAT=False, TEST_ODD_ROT=False, return_sem=False):
        'Initialization'
        with open(train_F, 'r') as openfile: 
            data = json.load(openfile) 
        if cat_only is not None:
            data = [i for i in data if cat_only+'/' in i[0]]
    
        data = np.array(data)
        #self.labels = data[:,1].astype(int)
        self.data = data[:,0]
        self.data2 = data[:,1]
        self.targets = data[:,2].astype(int)
        self.classes = np.unique(self.targets)
        self.transform = transform
        self.map_index = np.load("map_index.npy")
        
        self.datapath = "/ssd/peterwg/ShapeNetSO3" #"/home/peterwg/dataset/ShapeNetRendering3d"
        self.train_transform = train_transform
        if TEST_ODD_CAT:
            self.datapath = '/ssd/peterwg/ShapeNetSO3_OOD'
        elif TEST_ODD_ROT:
            self.datapath = '/ssd/peterwg/ShapeNetSO3_100'      
            self.map_index = np.load("map_index_100.npy")
        self.TEST_ODD_ROT = TEST_ODD_ROT
        #self.give_pose = give_pose
        if master_dir:
            self.datapath = master_dir
        self.return_sem = return_sem
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        img,ind1,  target = self.data[index], self.data2[index], self.targets[index]
        
        ind1 = int(ind1.replace('.png', ''))
        if self.TEST_ODD_ROT:
            ind2 = np.random.choice(100,1, replace=False)[0]
        else:
            ind2 = np.random.choice(50,1, replace=False)[0]
        c_id = os.path.join(self.datapath, img, str(ind1)+".png")
        r_id = os.path.join(self.datapath, img, str(ind2)+".png")
        try:
            img = pil_loader(c_id)#Image.fromarray(img)
            img2 = pil_loader(r_id)
        except:
            print(c_id)
            return self.__getitem__(0)
            
        if self.transform is not None:
            pos_1 = self.transform(img) #self.transform(img)
            pos_2 = self.transform(img2)
            if self.train_transform:
                pos_1p = self.train_transform(img)
                pos_2p = self.train_transform(img)
        rel_pose = self.map_index[ind1, ind2]
        if self.train_transform:
            data = (torch.cat([pos_1, pos_2], 0), torch.cat([pos_1p, pos_2p], 0))
            data = (data, target, rel_pose)
            #return (pos_1, pos_2, pos_1p), rel_pose, target
            return (index, *data) 
        elif self.return_sem:
            return torch.cat([pos_1, pos_2], 0), target, rel_pose
        else: # do not change!!
            return (pos_1, pos_2), rel_pose
    
class Dataset_3d_train2(torch.utils.data.Dataset):
    def __init__(self, train_F, master_dir = "/ssd/peterwg/s3dis/area_1/data/rgb", transform=None, cat_only=None):
        'Initialization'
        with open(train_F, 'r') as openfile: 
            data = json.load(openfile) 
        data = np.array(data)
        #self.labels = data[:,1].astype(int)
        self.data = data[:,0]
        self.targets = data[:,1].astype(int)
        self.classes = np.unique(self.targets)
        self.transform = transform
        
        self.datapath = "/home/peterwg/dataset/ShapeNetRendering3d"
    
        if cat_only is not None:
            self.data = [i for i in self.data if "/"+cat_only+'/' in i]
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        cat_id, sid, img_id = img.split("/")[-3], img.split("/")[-2], img.split("/")[-1][:-4]
        axis_id = img_id.split("_")[0] #np.random.choice(['x', 'y', 'z'], 1, replace=False)[0]
        c_id = int(img_id.split("_")[1])
        
        #cat_id, sid = img.split("/")[-2], img.split("/")[-1]
        #axis_id = np.random.choice(['x', 'y', 'z'], 1, replace=False)[0]
        #c_id = np.random.randint(0, 17)
        l_id, r_id = get_adj_pose_id_only(c_id)
        
        c_id = os.path.join(self.datapath, cat_id, sid, axis_id+"_"+str(c_id)+".png")
        l_id = os.path.join(self.datapath, cat_id, sid, axis_id+"_"+str(l_id)+".png")
        r_id = os.path.join(self.datapath, cat_id, sid, axis_id+"_"+str(r_id)+".png")

        img = pil_loader(c_id)#Image.fromarray(img)
        img_l = pil_loader(l_id)
        img_r = pil_loader(r_id)        

        if self.transform is not None:
            pos_1 = self.transform(img) #self.transform(img)
            pos_2 = self.transform(img)
            pos_c = test_transform_sp(img)
            #pos_a = test_transform_sp(img)
            pos_l = test_transform_sp(img_l)
            pos_r = test_transform_sp(img_r)
            

        data = (pos_1, pos_2, pos_c, pos_l, pos_r)
        
        ##cid
        axis_id = img_id.split("_")[0] #np.random.choice(['x', 'y', 'z'], 1, replace=False)[0]
        c_id = int(img_id.split("_")[1])#np.random.randint(0, 17)
        if axis_id == "y" and c_id:
            c_id += 17
        elif axis_id == "z" and c_id:
            c_id += 17*2
        data = (pos_1, pos_2, pos_c, pos_l, pos_r)
        data = (data, c_id)
        ##
        return (index, *data) 
    
class Dataset_3d_val2(torch.utils.data.Dataset):
    def __init__(self, train_F, master_dir = "/ssd/peterwg/s3dis/area_1/data/rgb", transform=None, cat_only=None):
        'Initialization'
        with open(train_F, 'r') as openfile: 
            data = json.load(openfile) 
        data = np.array(data)
        #self.labels = data[:,1].astype(int)
        self.data = data[:,0]
        self.targets = data[:,1].astype(int)
        self.classes = np.unique(self.targets)
        self.transform = transform
        
        self.datapath = "/home/peterwg/dataset/ShapeNetRendering3d"
        if cat_only is not None:
            self.data = [i for i in self.data if "/"+cat_only+'/' in i]

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        cat_id, sid, img_id = img.split("/")[-3], img.split("/")[-2], img.split("/")[-1][:-4]
        axis_id = img_id.split("_")[0] #np.random.choice(['x', 'y', 'z'], 1, replace=False)[0]
        c_id = int(img_id.split("_")[1])
        #cat_id, sid = img.split("/")[-2], img.split("/")[-1]
        #axis_id = np.random.choice(['x', 'y', 'z'], 1, replace=False)[0]
        #c_id = np.random.randint(0, 17)
        
        c_id = os.path.join(self.datapath, cat_id, sid, axis_id+"_"+str(c_id)+".png")

        img = pil_loader(c_id)#Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img) #self.transform(img)

        c_id = int(img_id.split("_")[1])#np.random.randint(0, 17)
        if axis_id == "y" and c_id:
            c_id += 17
        elif axis_id == "z" and c_id:
            c_id += 17*2
        return pos_1, c_id
        ##
        
        return pos_1, target
    
class Dataset_sp_val(torch.utils.data.Dataset):
    def __init__(self, train_F, master_dir = "/ssd/peterwg/s3dis/area_1/data/rgb", transform=None):
        'Initialization'
        with open(train_F, 'r') as openfile: 
            data = json.load(openfile) 
        data = np.array(data)
        #self.labels = data[:,1].astype(int)
        self.data = data[:,0]
        self.targets = data[:,1].astype(int)
        
        '''ind = np.random.choice(len(self.data), int(0.1*len(self.data)), replace=False)
        self.data = self.data[ind]
        self.targets = self.targets[ind]'''
        
        self.classes = np.unique(self.targets)
        self.transform = transform
        #self.master_dir = master_dir

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = pil_loader(img)#Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img) #self.transform(img)


        return pos_1, target
    
    
class Dataset_sp_train(torch.utils.data.Dataset):
    def __init__(self, train_F, master_dir = "/ssd/peterwg/s3dis/area_1/data/rgb", transform=None):
        'Initialization'
        with open(train_F, 'r') as openfile: 
            data = json.load(openfile) 
        data = np.array(data)
        #self.labels = data[:,1].astype(int)
        self.data = data[:,0]
        self.targets = data[:,1].astype(int)
        self.classes = np.unique(self.targets)
        self.transform = transform
        
        self.stride = 3
        #self.master_dir = master_dir

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        l_id, r_id = get_adj_pose_id(img)

        img = pil_loader(img)#Image.fromarray(img)
        img_l = pil_loader(l_id)
        img_r = pil_loader(r_id)
        
        patchlen = np.random.randint(12, 26)
        loc = np.random.randint(31+self.stride, 106-self.stride-patchlen, 2)

        img_b = Image.fromarray(np.array(img)[loc[0]:loc[0]+patchlen, loc[1]:loc[1]+patchlen, :])
        if np.random.uniform() >= 0.5:
            img_a = Image.fromarray(np.array(img)[loc[0]-self.stride:loc[0]-self.stride+patchlen, loc[1]:loc[1]+patchlen, :])
            img_c = Image.fromarray(np.array(img)[loc[0]+self.stride:loc[0]+self.stride+patchlen, loc[1]:loc[1]+patchlen, :])
        else:
            img_a = Image.fromarray(np.array(img)[loc[0]:loc[0]+patchlen, loc[1]-self.stride:loc[1]-self.stride+patchlen, :])
            img_c = Image.fromarray(np.array(img)[loc[0]:loc[0]+patchlen, loc[1]+self.stride:loc[1]+self.stride+patchlen, :])            

        if self.transform is not None:
            pos_1 = self.transform(img) #self.transform(img)
            pos_2 = self.transform(img)
            pos_c = test_transform_sp_full(img)
            #pos_a = test_transform_sp(img)
            pos_l = test_transform_sp_full(img_l)
            pos_r = test_transform_sp_full(img_r)
            
            pos_a = test_transform_sp(img_a)
            pos_b = test_transform_sp(img_b)
            pos_c = test_transform_sp(img_c)

        data = (pos_1, pos_2, pos_c, pos_l, pos_r, pos_a, pos_b, pos_c)
        data = (data, target)
        return (index, *data) #pos_1, pos_2, pos_c, pos_l, pos_r, target
    
class Dataset_pose_single_patch(torch.utils.data.Dataset):
    def __init__(self, train_F, master_dir = "/ssd/peterwg/s3dis/area_1/data/rgb", transform=None, num_patch=5):
        'Initialization'
        with open(train_F, 'r') as openfile: 
            data = json.load(openfile) 
        data = np.array(data)
        #self.labels = data[:,1].astype(int)
        self.data = data[:,0]
        #self.targets = data[:,1].astype(int)
        self.classes = np.arange(8*2) 
        self.transform = transform
        self.num_patch = num_patch
        #self.master_dir = master_dir

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        img= self.data[index]
        
        ids = img.split('/')[-1]
        #meta_path = img.replace('/ShapeNetRendering_scale/', '/ShapeNetRendering/').replace(ids, "rendering/rendering_metadata.txt")
        #meta_name = np.loadtxt(meta_path)
        #ids = np.random_choice(24,2, replace=False)

        img = pil_loader(img)#Image.fromarray(img)

        img = np.asarray(img)
        # center crop
        img = img[34:104, 34:104, :]
        patchsize = img.shape[0]//self.num_patch
        out_patch = [] 
        for i in range(self.num_patch):
            for j in range(self.num_patch):
                cur_patch = img[i*patchsize:(i+1)*patchsize, j*patchsize:(j+1)*patchsize, :]
                cur_patch = Image.fromarray(cur_patch)#(np.uint8(cur_patch))
                if self.transform is not None:
                    cur_patch = self.transform(cur_patch) #self.transform(img)
                out_patch.append(cur_patch)
                
        angle = int(ids.split('.')[0]) # (meta_name[int(ids.split(".")[0]), 0] -1e-10).astype(np.float32)
        
        return out_patch, angle//2
    
class Dataset_sp(torch.utils.data.Dataset):
    def __init__(self, train_F, master_dir = "/ssd/peterwg/s3dis/area_1/data/rgb", transform=None):
        'Initialization'
        with open(train_F, 'r') as openfile: 
            data = json.load(openfile) 
        data = np.array(data)
        #self.labels = data[:,1].astype(int)
        self.data = data[:,0]
        self.targets = data[:,1].astype(int)
        self.classes = np.unique(self.targets)
        self.transform = transform
        #self.master_dir = master_dir

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        l_id, r_id = get_adj_pose_id(img)

        img = pil_loader(img)#Image.fromarray(img)
        img_l = pil_loader(l_id)
        img_r = pil_loader(r_id)

        if self.transform is not None:
            pos_1 = self.transform(img) #self.transform(img)
            pos_2 = self.transform(img)
            pos_c = test_transform_sp_full(img)
            #pos_a = test_transform_sp(img)
            pos_l = test_transform_sp_full(img_l)
            pos_r = test_transform_sp_full(img_r)
            
        return pos_1, pos_2, pos_c, pos_l, pos_r, target



'''def Shapenet3d(parent_dataset):
    class paired_dataset(parent_dataset):
        def __init__(self, data_dir, cat_only=3, numint=NUMINT, *args, **kwargs):
            super(paired_dataset, self).__init__(data_dir, *args, **kwargs)
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            else:
                self.device = torch.device("cpu")

            self.raster_settings = RasterizationSettings(image_size=192, cull_backfaces=True, perspective_correct=True, faces_per_pixel=8)
            self.lights = PointLights( device=self.device, location=[[0.0, 5.0, -10.0]], diffuse_color=((0, 0, 0),), specular_color=((0, 0, 0),), )
            
        def __getitem__(self, index):
            R, T = look_at_view_transform(1.0, elev=0, azim=0)
            R = torch.Tensor(Rotation.from_euler('yxz', [[ 0,0,72*4]], degrees=True).as_dcm())
            cameras = OpenGLPerspectiveCameras(R=R, T=T, device=self.device)
            
            images_by_idxs = self.render(
                idxs=[index],
                device=self.device,
                cameras=cameras,
                raster_settings=self.raster_settings,
                lights=self.lights,
                shader=HardFlatShader(),
            )
            
            target = self.to_label_dict[self._get_item_ids(index)['synset_id']]

            return images_by_idxs, target
    return paired_dataset'''
def tensor2array(tens):
    return Image.fromarray( (tens.cpu().numpy()*255).astype(np.uint8) )

class Shapenet3d_val(torch.utils.data.Dataset):
    def __init__(self, basedataset, transform=None):
        'Initialization'
        self.basedataset = basedataset
        if False:#torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.raster_settings = RasterizationSettings(image_size=64, cull_backfaces=True)
        self.lights = PointLights( device=self.device, location=[[0.0, 5.0, -10.0]], diffuse_color=((0, 0, 0),), specular_color=((0, 0, 0),), )
        self.transform = transform
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.basedataset)

    def __getitem__(self, index):
        target = self.basedataset.to_label_dict[self.basedataset[index]['synset_id']]
        #render
        R, T = look_at_view_transform(1.0, elev=0, azim=0)
        splits = 24
        angle_int = 360.0/splits
        centerp = np.random.randint(0, splits)
        axis_ind = np.random.randint(0, 3)
        if  axis_ind==0:
            R = torch.Tensor(Rotation.from_euler('yxz', [[ 0,centerp*angle_int,0]], degrees=True).as_dcm())
        elif  axis_ind==1:
            R = torch.Tensor(Rotation.from_euler('yxz', [[ centerp*angle_int,0,0]], degrees=True).as_dcm())
        else:
            R = torch.Tensor(Rotation.from_euler('yxz', [[ 0,0,centerp*angle_int]], degrees=True).as_dcm())
        cameras = OpenGLPerspectiveCameras(R=R, T=T, device=self.device)
            
        images_by_idxs = self.basedataset.render(
            idxs=[index],
            device=self.device,
            cameras=cameras,
            raster_settings=self.raster_settings,
            lights=self.lights,
            shader=HardFlatShader(),
        )
        images_by_idxs = images_by_idxs[...,:3]

        pos_1 = self.transform(tensor2array(images_by_idxs.squeeze()))
        
        return pos_1, target
    
class Shapenet3d(torch.utils.data.Dataset):
    def __init__(self, basedataset, transform=None):
        'Initialization'
        self.basedataset = basedataset
        if False: #torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.raster_settings = RasterizationSettings(image_size=64, cull_backfaces=True, perspective_correct=True, faces_per_pixel=8)
        self.lights = PointLights( device=self.device, location=[[0.0, 5.0, -10.0]], diffuse_color=((0, 0, 0),), specular_color=((0, 0, 0),), )
        self.transform = transform
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.basedataset)

    def __getitem__(self, index):
        #render
        R, T = look_at_view_transform(1.0, elev=0, azim=0)
        T = torch.cat((T,T, T))
        splits = 24
        angle_int = 360.0/splits
        centerp = np.random.randint(0, splits)
        axis_ind = np.random.randint(0, 3)
        if  axis_ind==0:
            R = torch.cat([torch.Tensor(Rotation.from_euler('yxz', [[ 0,(centerp-1)*angle_int,0]], degrees=True).as_dcm()),torch.Tensor(Rotation.from_euler('yxz', [[ 0,centerp*angle_int,0]], degrees=True).as_dcm()),torch.Tensor(Rotation.from_euler('yxz', [[ 0,(centerp+1)*angle_int,0]], degrees=True).as_dcm()) ] )
        elif  axis_ind==1:
            R = torch.cat([torch.Tensor(Rotation.from_euler('yxz', [[ (centerp-1)*angle_int,0,0]], degrees=True).as_dcm()),torch.Tensor(Rotation.from_euler('yxz', [[ centerp*angle_int,0,0]], degrees=True).as_dcm()),torch.Tensor(Rotation.from_euler('yxz', [[ (centerp+1)*angle_int,0,0]], degrees=True).as_dcm()) ] )
        else:
            R = torch.cat([torch.Tensor(Rotation.from_euler('yxz', [[ 0,0,(centerp-1)*angle_int]], degrees=True).as_dcm()),torch.Tensor(Rotation.from_euler('yxz', [[ 0,0,centerp*angle_int]], degrees=True).as_dcm()),torch.Tensor(Rotation.from_euler('yxz', [[ 0,0,(centerp+1)*angle_int]], degrees=True).as_dcm()) ] )
        cameras = OpenGLPerspectiveCameras(R=R, T=T, device=self.device)
            
        images_by_idxs = self.basedataset.render(
            idxs=[index],
            device=self.device,
            cameras=cameras,
            raster_settings=self.raster_settings,
            lights=self.lights,
            shader=HardFlatShader(),
        )
        images_by_idxs = images_by_idxs[...,:3]
        #topil = ToPILImage()
        pos_1 = self.transform(tensor2array(images_by_idxs[1]))
        pos_2 = self.transform(tensor2array(images_by_idxs[1]))
        pos_l = test_transform_sp(tensor2array(images_by_idxs[0]))
        pos_c = test_transform_sp(tensor2array(images_by_idxs[1]))
        pos_r = test_transform_sp(tensor2array(images_by_idxs[2]))
        
        data = (pos_1, pos_2, pos_c, pos_l, pos_r)
        target = self.basedataset.to_label_dict[self.basedataset[index]['synset_id']]

        data = (data, target)

        return (index, *data)
            
def MNIST_sp(parent_dataset):
    class paired_dataset(parent_dataset):

        def __getitem__(self, index):
            img, target = self.data[index], self.targets[index]
            img = F.rotate(img.unsqueeze(0), np.random.uniform(0, 359))
            
            ang1 = np.random.uniform(8, 20)
            ang2 = np.random.uniform(-20, -8)
            img_l = F.rotate(img, ang1)
            img_r = F.rotate(img, ang2)
            img = Image.fromarray(img.squeeze().numpy())
            img_l = Image.fromarray(img_l.squeeze().numpy())
            img_r = Image.fromarray(img_r.squeeze().numpy())

            if self.transform is not None:
                pos_1 = self.transform(img)
                pos_2 = self.transform(img)
                pos_c = test_transform_mnist_full(img)
                pos_l = test_transform_mnist_full(img_l)
                pos_r = test_transform_mnist_full(img_r)
            
            if self.target_transform is not None:
                target = self.target_transform(target)

            return pos_1, pos_2, pos_c, pos_l, pos_r, target
    return paired_dataset

def MNIST_rot(parent_dataset):
    class paired_dataset(parent_dataset):
        def __init__(self, cat_only=3, numint=NUMINT, *args, **kwargs):
            super(paired_dataset, self).__init__(*args, **kwargs)
            
            if cat_only is not None:
                ind = (self.targets == cat_only)
                self.data = self.data[ind]
                self.targets = self.targets[ind]
            
            self.interval = int(360/numint)
            self.all_angs = np.arange(numint)
            
        def __getitem__(self, index):
            
            img, target = self.data[index], self.targets[index]
            target = np.random.choice(self.all_angs, 1)

            try:
                img = F.rotate(img.unsqueeze(0), int(target[0]*self.interval))
            except:
                print(target[0]*self.interval)
                st()
            img = Image.fromarray(img.squeeze().numpy())

            if self.transform is not None:
                pos_1 = self.transform(img)
            
            if self.target_transform is not None:
                target = self.target_transform(target)

            return pos_1, target
    return paired_dataset

def MNIST_rotcls(parent_dataset):
    class paired_dataset(parent_dataset):
        def __init__(self, cat_only=6, *args, **kwargs):
            super(paired_dataset, self).__init__(*args, **kwargs)
            if cat_only is not None:
                ind = (self.targets == 6) | (self.targets == 9)
                
                self.data = self.data[ind]
                self.targets = self.targets[ind]
                
                self.targets[self.targets==6] = 0
                self.targets[self.targets==9] = 1
                
        def __getitem__(self, index):
            img, target = self.data[index], self.targets[index]
            
            #img = F.rotate(img.unsqueeze(0), np.random.uniform(0, 359))
            img = F.rotate(img.unsqueeze(0), np.random.uniform(-70, 70))
            #img = F.rotate(img.unsqueeze(0), 180)
            img = Image.fromarray(img.squeeze().numpy())

            if self.transform is not None:
                pos_1 = self.transform(img)
            
            if self.target_transform is not None:
                target = self.target_transform(target)

            return pos_1, target
    return paired_dataset

def MNIST_rotclstest(parent_dataset):
    class paired_dataset(parent_dataset):
        def __init__(self, cat_only=6, *args, **kwargs):
            super(paired_dataset, self).__init__(*args, **kwargs)
            if cat_only is not None:
                ind = (self.targets == 6) | (self.targets == 9)
                
                self.data = self.data[ind]
                self.targets = self.targets[ind]
                
                self.targets[self.targets==6] = 0
                self.targets[self.targets==9] = 1
                
        def __getitem__(self, index):
            img, target = self.data[index], self.targets[index]
            
            #img = F.rotate(img.unsqueeze(0), np.random.uniform(0, 359))
            #img = F.rotate(img.unsqueeze(0), np.random.uniform(-70, 70))
            img = F.rotate(img.unsqueeze(0), np.random.uniform(40, 80))
            img = Image.fromarray(img.squeeze().numpy())

            if self.transform is not None:
                pos_1 = self.transform(img)
            
            if self.target_transform is not None:
                target = self.target_transform(target)

            return pos_1, target
    return paired_dataset

class Dataset_sp_base(torch.utils.data.Dataset):
    def __init__(self, train_F, master_dir = "/ssd/peterwg/s3dis/area_1/data/rgb", transform=None):
        'Initialization'
        with open(train_F, 'r') as openfile: 
            data = json.load(openfile) 
        data = np.array(data)
        #self.labels = data[:,1].astype(int)
        self.data = data[:,0]
        self.targets = data[:,1].astype(int)
        self.classes = np.unique(self.targets)
        self.transform = transform
        #self.master_dir = master_dir

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = pil_loader(img)#Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        return pos_1, pos_2, target
        """'Generates one sample of data'
        
        impath = os.path.join(self.master_dir, self.list_IDs[index].replace("depth", "rgb"))
        X = pil_loader(impath)
        target = pil_loader(impath.replace("rgb", "depthz"))
        #target = pil_loader_log(impath.replace("rgb", "depthz2"))
        inter = pil_loader(impath.replace("rgb", "normal"))
        mask = pil_loader(impath.replace("rgb", "depthmask"))
        
        if self.transform is not None:
            seed = np.random.randint(2147483647) # make a seed with numpy generator 
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            X = self.transform(X)
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            target = self.transform(target)
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            inter = self.transform(inter)
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            mask = self.transform(mask)
        
        return X, target, inter, 127.5*(mask+1), impath.split("/")[-1]"""
    
def create_pair_dataset(parent_dataset):
    class paired_dataset(parent_dataset):

        def __getitem__(self, index):
            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(img)

            if self.transform is not None:
                pos_1 = self.transform(img)
                pos_2 = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return pos_1, pos_2, target
    return paired_dataset
