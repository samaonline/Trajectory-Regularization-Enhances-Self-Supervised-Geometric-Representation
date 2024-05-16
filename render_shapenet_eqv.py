from dataset_3d import ShapeNetCore
from pdb import set_trace as st
import torch
import pytorch3d
from dataset_3d import SHAPENET_PATH
#from scipy.spatial.transform import Rotation
from pytorch3d.renderer.mesh.shader import HardFlatShader
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    TexturesVertex,
    look_at_view_transform,
)
import numpy as np
import imageio
from pathlib import Path
import os
from pdb import set_trace as st
from pytorch3d.transforms import quaternion_to_matrix
import cv2

from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

dataset_class = ShapeNetCore(SHAPENET_PATH, version=2, synsets=['02691156', '02958343', '03636649', '04256520', '04530566', '02828884', '03001627', '03691459', '04379243', '02933112', '03211117', '04090263', '04401088'], phase="train")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

raster_settings = RasterizationSettings(image_size=128, cull_backfaces=True, perspective_correct=True, faces_per_pixel=20)
lights = PointLights( device=device, location=[[0.0, 5.0, -10.0]], diffuse_color=((0, 0, 0),), specular_color=((0, 0, 0),), )
SAVE_PATH = '/ssd/peterwg/ShapeNetSO3_150'#'/home/peterwg/dataset/ShapeNetRendering3d'
NUM_SPLITS = 150

def generate_superfibonacci(n=1, device="cpu"):
    """
    Samples n rotations equivolumetrically using a Super-Fibonacci Spiral.

    Reference: Marc Alexa, Super-Fibonacci Spirals. CVPR 22.

    Args:
        n (int): Number of rotations to sample.
        device (str): CUDA Device. Defaults to CPU.

    Returns:
        (tensor): Rotations (n, 3, 3).
    """
    phi = np.sqrt(2.0)
    psi = 1.533751168755204288118041
    ind = torch.arange(n, device=device)
    s = ind + 0.5
    r = torch.sqrt(s / n)
    R = torch.sqrt(1.0 - s / n)
    alpha = 2 * np.pi * s / phi
    beta = 2.0 * np.pi * s / psi
    Q = torch.stack(
        [
            r * torch.sin(alpha),
            r * torch.cos(alpha),
            R * torch.sin(beta),
            R * torch.cos(beta),
        ],
        1,
    )
    return quaternion_to_matrix(Q).float()

def rgba2rgb( rgba, background=(0,0,0) ):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' )# / 255.0
    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray( rgb )

#for index, model in enumerate(dataset_class):
#sum_ =0 

for index in range(len(dataset_class)):
    '''if sum_>1000:
        continue'''
    try:
        model = dataset_class[index]
    except:
        continue
    if os.path.isdir( os.path.join(SAVE_PATH, model["synset_id"], model["model_id"]) ):
        continue
    Path(os.path.join(SAVE_PATH, model["synset_id"], model["model_id"])).mkdir(parents=True, exist_ok=True)

    R, T = look_at_view_transform(1.0, elev=0, azim=0)
    splits = 18
    angle_int = 360.0/splits
    
    '''R = []
    for i in range(splits):
        R.append(torch.Tensor(Rotation.from_euler('yxz', [[ 0,i*angle_int,0]], degrees=True).as_dcm()))
    for i in range(splits):
        R.append(torch.Tensor(Rotation.from_euler('yxz', [[ i*angle_int,0,0]], degrees=True).as_dcm()))    
    for i in range(splits):
        R.append(torch.Tensor(Rotation.from_euler('yxz', [[ 0,0,i*angle_int]], degrees=True).as_dcm()))'''
    #R = torch.cat(R)
    #st()
    R = generate_superfibonacci(NUM_SPLITS)
    T = torch.cat([T]*NUM_SPLITS)
    
    '''centerp = np.random.randint(0, splits)
    axis_ind = np.random.randint(0, 3)
    if  axis_ind==0:
        R = torch.Tensor(Rotation.from_euler('yxz', [[ 0,centerp*angle_int,0]], degrees=True).as_dcm())
    elif  axis_ind==1:
        R = torch.Tensor(Rotation.from_euler('yxz', [[ centerp*angle_int,0,0]], degrees=True).as_dcm())
    else:
        R = torch.Tensor(Rotation.from_euler('yxz', [[ 0,0,centerp*angle_int]], degrees=True).as_dcm())'''
    cameras = OpenGLPerspectiveCameras(R=R, T=T, device=device)
    try:
        images_by_idxs = dataset_class.render(
                idxs=[index],
                device=device,
                cameras=cameras,
                raster_settings=raster_settings,
                lights=lights,
                shader=HardFlatShader(),
            )
    except:
        continue
    for i, img in enumerate(images_by_idxs.cpu().numpy()):
        imageio.imwrite( os.path.join(SAVE_PATH, model["synset_id"], model["model_id"], str(i)+".png"), rgba2rgb(img*255).astype(np.uint8))
    '''for i, img in enumerate(images_by_idxs[:splits].cpu().numpy()):
        imageio.imwrite( os.path.join(SAVE_PATH, model["synset_id"], model["model_id"], "x_"+str(i)+".png"), rgba2rgb(img))
    for i, img in enumerate(images_by_idxs[splits:2*splits].cpu().numpy()):
        imageio.imwrite( os.path.join(SAVE_PATH, model["synset_id"], model["model_id"], "y_"+str(i)+".png"), rgba2rgb(img))
    for i, img in enumerate(images_by_idxs[2*splits:].cpu().numpy()):
        imageio.imwrite( os.path.join(SAVE_PATH, model["synset_id"], model["model_id"], "z_"+str(i)+".png"), rgba2rgb(img))'''