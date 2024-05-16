# Copyright 2022 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
from pprint import pprint
os.environ['WANDB_SILENT']="true"
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from solo.args.setup import parse_args_pretrain
from solo.data.classification_dataloader import prepare_data as prepare_data_classification
'''from solo.data.pretrain_dataloader import (
    prepare_dataloader,
    prepare_datasets,
    prepare_n_crop_transform,
    prepare_transform,
)'''
from solo.methods import METHODS
from solo.utils.auto_resumer import AutoResumer
from solo.utils.checkpointer import Checkpointer
from solo.utils.misc import make_contiguous
from pytorch_lightning.loggers import CSVLogger
import torchvision.transforms.functional as F
import torch.nn.functional as F2
from PIL import Image
from tqdm import tqdm
from solo.utils.knn import WeightedKNNClassifier
import utils

try:
    from solo.data.dali_dataloader import PretrainDALIDataModule
except ImportError:
    _dali_avaliable = False
else:
    _dali_avaliable = True

try:
    from solo.utils.auto_umap import AutoUMAP
except ImportError:
    _umap_available = False
else:
    _umap_available = True
from torch import nn
import torch
from pdb import set_trace as st
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, dataset
import numpy as np
from torchvision import transforms
from solo.data.classification_dataloader import (
    prepare_dataloaders,
    prepare_datasets,
    prepare_transforms,
)
from typing import Tuple
from utils import MNIST_rotcls

#NUMR = 100#50
BATCHSIZE = 200
GEN_VIS = False#True
PATCH_NORM = True
TEST_OOD = False #True

test_transform_mnist_full = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize([0.4914], [0.2023])])

@torch.no_grad()
def run_knn(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    test_features: torch.Tensor,
    test_targets: torch.Tensor,
    k: int,
    T: float,
    distance_fx: str,
) -> Tuple[float]:
    """Runs offline knn on a train and a test dataset.

    Args:
        train_features (torch.Tensor, optional): train features.
        train_targets (torch.Tensor, optional): train targets.
        test_features (torch.Tensor, optional): test features.
        test_targets (torch.Tensor, optional): test targets.
        k (int): number of neighbors.
        T (float): temperature for the exponential. Only used with cosine
            distance.
        distance_fx (str): distance function.

    Returns:
        Tuple[float]: tuple containing the the knn acc@1 and acc@5 for the model.
    """

    # build knn
    knn = WeightedKNNClassifier(k=k, T=T, distance_fx=distance_fx,)

    # add features
    knn(
        train_features=train_features,
        train_targets=train_targets,
        test_features=test_features,
        test_targets=test_targets,
    )

    # compute
    acc1, acc5 = knn.compute() #knn.compute_angle()

    # free up memory
    del knn

    return acc1, acc5

def extract_features(loader: DataLoader, model: nn.Module) -> Tuple[torch.Tensor]:
    """Extract features from a data loader using a model.

    Args:
        loader (DataLoader): dataloader for a dataset.
        model (nn.Module): torch module used to extract features.

    Returns:
        Tuple(torch.Tensor): tuple containing the backbone features, projector features and labels.
    """

    model.eval()
    backbone_features, proj_features, labels = [], [], []
    with torch.no_grad():
        for im, lab in tqdm(loader):
            im = im.cuda(non_blocking=True)
            lab = lab.cuda(non_blocking=True)
            temp_model = model.backbone
            outs = model(im)
            
            #outs = outs['feats']
            outs = outs['conv4']
            if not PATCH_NORM:
                outs = outs.reshape(len(im), -1)
                outs = F2.normalize(outs, dim=-1)
            else:
                outs = F2.normalize(outs.reshape(outs.shape[0], outs.shape[1], -1), dim=-1)
                outs = outs.reshape(len(im), -1)
            '''#outs = F2.normalize(outs.reshape(outs.shape[0], outs.shape[1], -1), dim=-1)
            outs = outs.reshape(len(im), -1)
            #outs = temp_model(im)
            
            outs = F2.normalize(outs, dim=-1)'''
            backbone_features.append(outs.detach().cpu())
            #proj_features.append(outs["z"].cpu())
            proj_features.append(outs.detach().cpu())
            labels.append(lab.cpu())
    model.train()
    backbone_features = torch.cat(backbone_features)
    proj_features = torch.cat(proj_features)
    labels = torch.cat(labels)
    return backbone_features, proj_features, labels


def main():
    seed_everything(5)

    args = parse_args_pretrain()

    assert args.method in METHODS, f"Choose from {METHODS.keys()}"

    if args.num_large_crops != 2:
        assert args.method in ["wmse", "mae"]

    model = METHODS[args.method](**args.__dict__)
    
    if args.dataset == "mnist":
        model.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1,bias=False)
        model.backbone.maxpool = nn.Identity()
        model.classifier = nn.Linear(512, 13)
        #model.backbone.avgpool = nn.Identity()
        #model.backbone.layer4 = nn.Identity()
    elif args.dataset == "shapenet" or  args.dataset == "shapenet3d":
        model.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,bias=False)
        model.backbone.maxpool = nn.Identity()
        model.classifier = nn.Linear(512, 13)#13) #(512, 13)  
        #model.backbone.layer4 = nn.Identity()
        #model.backbone.avgpool = nn.Identity()

    make_contiguous(model)
    #model.load_state_dict(torch.load( "traj/p0.01/vicreg/0/vicreg-mnist-0-ep=999.ckpt")[ 'state_dict'])
    #model.load_state_dict(torch.load( "traj/p0.2/vicreg/0/vicreg-mnist-0-ep=40.ckpt")[ 'state_dict'])
    #model.load_state_dict(torch.load( "trained_models/vicreg/1/vicreg-mnist-1-ep=999.ckpt", map_location="cpu")[ 'state_dict'])
    model.load_state_dict(torch.load(args.project, map_location="cpu")[ 'state_dict'], strict=False)
    model.eval()
    model.cuda()
    
    
    '''_, T = prepare_transforms(args.dataset)
    train_dataset, val_dataset = prepare_datasets(
        args.dataset,
        T_train=T,
        T_val=T,
        #data_dir=args.train_dir,#args.data_dir,
        #train_dir=args.train_dir,
        #val_dir=args.val_dir,
        download=True,
    )'''
    #for cur_class in range(10):
    if TEST_OOD:
        cur_class = '02691156'#'02871439' #'03790512'
        train_dataset = utils.Dataset_3d_val("/home/peterwg/dataset/ShapeNetRendering/ShapeNetRenderingso3_100_train.json", cat_only=cur_class,
                                             transform=utils.test_transform_sp, give_pose=True, OOD=True, master_dir='/ssd/peterwg/ShapeNetSO3_100')
        val_dataset =  utils.Dataset_3d_val("/home/peterwg/dataset/ShapeNetRendering/ShapeNetRenderingso3_100_test.json",cat_only=cur_class,
                                            transform=utils.test_transform_sp, give_pose=True, OOD=True, master_dir='/ssd/peterwg/ShapeNetSO3_100')
    else:
        for cur_class in ['02691156']:#["02691156"]: #, "02828884", "02933112", "02958343", "03001627", "03211117", "03636649", "03691459", "04090263", "04256520", "04379243", "04401088", "04530566"]:
            print("EVAL "+str(cur_class))
            if args.dataset == "mnist":
                from torchvision.datasets import MNIST
                dataset_class = utils.MNIST_rot(MNIST)
                train_dataset = dataset_class(root='data', train=True, transform=utils.test_transform_mnist_full, download=True, cat_only=cur_class)
                val_dataset = dataset_class(root='data', train=False, transform=utils.test_transform_mnist_full, download=True, cat_only=cur_class)
            elif args.dataset == "shapenet":
                '''train_dataset = utils.Dataset_sp_val("/home/peterwg/dataset/ShapeNetRendering_scale/ShapeNetRendering_train.json", transform=utils.test_transform_sp_full)
                val_dataset =  utils.Dataset_sp_val("/home/peterwg/dataset/ShapeNetRendering_scale/ShapeNetRendering_test.json", transform=utils.test_transform_sp_full)'''
                train_dataset = utils.Dataset_pose_single("/home/peterwg/dataset/ShapeNetRendering_scale/ShapeNetRendering_train.json", cat_only=cur_class, 
                                                          transform=utils.test_transform_sp)
                val_dataset =  utils.Dataset_pose_single("/home/peterwg/dataset/ShapeNetRendering_scale/ShapeNetRendering_test.json",cat_only=cur_class, 
                                                         transform=utils.test_transform_sp) # "02958343" for car
            elif args.dataset == "shapenet3d":
                '''train_dataset = utils.Dataset_3d_val("/home/peterwg/dataset/ShapeNetRendering/ShapeNetRendering3d_train.json", transform=utils.test_transform_sp)
                val_dataset =  utils.Dataset_3d_val("/home/peterwg/dataset/ShapeNetRendering/ShapeNetRendering3d_test.json", transform=utils.test_transform_sp)'''        
                train_dataset = utils.Dataset_3d_val("/home/peterwg/dataset/ShapeNetRendering/ShapeNetRenderingso3_100_train.json", cat_only=cur_class, 
                                                          transform=utils.test_transform_sp, trainset=True, master_dir="/ssd/peterwg/ShapeNetSO3_100_train", give_pose=True)
                val_dataset =  utils.Dataset_3d_val("/home/peterwg/dataset/ShapeNetRendering/ShapeNetRenderingso3_100_test.json",cat_only=cur_class, 
                                                         transform=utils.test_transform_sp, give_pose=True)
            
    train_loader, val_loader = prepare_dataloaders(
        train_dataset, val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
    )


    # extract train features
    train_features_bb, train_features_proj, train_targets = extract_features(train_loader, model)
    train_features = {"backbone": train_features_bb, "projector": train_features_proj}

    # extract test features
    test_features_bb, test_features_proj, test_targets = extract_features(val_loader, model)
    test_features = {"backbone": test_features_bb, "projector": test_features_proj}

    # run k-nn for all possible combinations of parameters
    distance_function = ["euclidean", "cosine"]
    c_temps = [0.01,  0.05,0.1,0.5, 1]#[0.01, 0.02, 0.05, 0.07, 0.1, 0.2, 0.5, 1]
    for feat_type in ['backbone']:
        print(f"\n### {feat_type.upper()} ###")
        for k in [200, 100, 50]:#[1, 2, 5, 10, 20, 50, 100, 200]:
            for distance_fx in distance_function:
                temperatures = c_temps if distance_fx == "cosine" else [None] 
                for T in temperatures:
                    print("---")
                    print(f"Running k-NN with params: distance_fx={distance_fx}, k={k}, T={T}...")
                    acc1, acc5 = run_knn(
                        train_features=train_features[feat_type],
                        train_targets=train_targets,
                        test_features=test_features[feat_type],
                        test_targets=test_targets,
                        k=k,
                        T=T,
                        distance_fx=distance_fx,
                    )
                    print(f"Result: acc@1={acc1}, acc@5={acc5}")
    return#st()
    # validation dataloader for when it is available
    if args.dataset == "custom" and (args.no_labels or args.val_data_path is None):
        val_loader = None
    elif args.dataset in ["imagenet100", "imagenet"] and (args.val_data_path is None):
        val_loader = None
    else:
        if args.data_format == "dali":
            val_data_format = "image_folder"
        else:
            val_data_format = args.data_format

        _, val_loader = prepare_data_classification(
            args.dataset,
            train_data_path=args.train_data_path,
            val_data_path=args.val_data_path,
            data_format=val_data_format,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    # pretrain dataloader
    if args.data_format == "dali":
        assert (
            _dali_avaliable
        ), "Dali is not currently avaiable, please install it first with pip3 install .[dali]."

        dali_datamodule = PretrainDALIDataModule(
            dataset=args.dataset,
            train_data_path=args.train_data_path,
            unique_augs=args.unique_augs,
            transform_kwargs=args.transform_kwargs,
            num_crops_per_aug=args.num_crops_per_aug,
            num_large_crops=args.num_large_crops,
            num_small_crops=args.num_small_crops,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            no_labels=args.no_labels,
            data_fraction=args.data_fraction,
            dali_device=args.dali_device,
            encode_indexes_into_labels=args.encode_indexes_into_labels,
        )
        dali_datamodule.val_dataloader = lambda: val_loader
    else:
        transform_kwargs = (
            args.transform_kwargs if args.unique_augs > 1 else [args.transform_kwargs]
        )
        transform = prepare_n_crop_transform(
            [prepare_transform(args.dataset, **kwargs) for kwargs in transform_kwargs],
            num_crops_per_aug=args.num_crops_per_aug,
        )

        if args.debug_augmentations:
            print("Transforms:")
            pprint(transform)

        train_dataset = prepare_datasets(
            args.dataset,
            transform,
            train_data_path=args.train_data_path,
            data_format=args.data_format,
            no_labels=args.no_labels,
            data_fraction=args.data_fraction,
        )
        train_loader = prepare_dataloader(
            train_dataset, batch_size=args.batch_size, num_workers=args.num_workers
        )

    # 1.7 will deprecate resume_from_checkpoint, but for the moment
    # the argument is the same, but we need to pass it as ckpt_path to trainer.fit
    ckpt_path, wandb_run_id = None, None
    if args.auto_resume and args.resume_from_checkpoint is None:
        auto_resumer = AutoResumer(
            checkpoint_dir=os.path.join(args.checkpoint_dir, args.method),
            max_hours=args.auto_resumer_max_hours,
        )
        resume_from_checkpoint, wandb_run_id = auto_resumer.find_checkpoint(args)
        if resume_from_checkpoint is not None:
            print(
                "Resuming from previous checkpoint that matches specifications:",
                f"'{resume_from_checkpoint}'",
            )
            ckpt_path = resume_from_checkpoint
    elif args.resume_from_checkpoint is not None:
        ckpt_path = args.resume_from_checkpoint
        del args.resume_from_checkpoint

    callbacks = []

    if args.save_checkpoint:
        # save checkpoint on last epoch only
        ckpt = Checkpointer(
            args,
            logdir=os.path.join(args.checkpoint_dir, args.method),
            frequency=args.checkpoint_frequency,
        )
        callbacks.append(ckpt)

    if args.auto_umap:
        assert (
            _umap_available
        ), "UMAP is not currently avaiable, please install it first with [umap]."
        auto_umap = AutoUMAP(
            args,
            logdir=os.path.join(args.auto_umap_dir, args.method),
            frequency=args.auto_umap_frequency,
        )
        callbacks.append(auto_umap)

    # wandb logging
    if args.wandb:
        wandb_logger = WandbLogger(
            name=args.name,
            project=args.project,
            entity=args.entity,
            offline=args.offline,
            resume="allow" if wandb_run_id else None,
            id=wandb_run_id,
        )
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(args)

        # lr logging
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)
        
    csvlog = CSVLogger(os.path.join(args.checkpoint_dir, args.method))
    trainer = Trainer.from_argparse_args(
        args,
        logger=wandb_logger if args.wandb else csvlog,
        callbacks=callbacks,
        enable_checkpointing=False,
        strategy=DDPStrategy(find_unused_parameters=False)
        if args.strategy == "ddp"
        else args.strategy,
    )

    # fix for incompatibility with nvidia-dali and pytorch lightning
    # with dali 1.15 (this will be fixed on 1.16)
    # https://github.com/Lightning-AI/lightning/issues/12956
    try:
        from pytorch_lightning.loops import FitLoop

        class WorkaroundFitLoop(FitLoop):
            @property
            def prefetch_batches(self) -> int:
                return 1

        trainer.fit_loop = WorkaroundFitLoop(
            trainer.fit_loop.min_epochs, trainer.fit_loop.max_epochs
        )
    except:
        pass

    if args.data_format == "dali":
        trainer.fit(model, ckpt_path=ckpt_path, datamodule=dali_datamodule)
    else:
        trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
