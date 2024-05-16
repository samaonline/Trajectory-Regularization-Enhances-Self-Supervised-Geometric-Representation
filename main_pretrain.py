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
from solo.data.pretrain_dataloader import (
    prepare_dataloader,
    prepare_datasets,
    prepare_n_crop_transform,
    prepare_transform,
)
from solo.methods import METHODS
from solo.utils.auto_resumer import AutoResumer
from solo.utils.checkpointer import Checkpointer
from solo.utils.misc import make_contiguous
from pytorch_lightning.loggers import CSVLogger
#from dataset_3d import ShapeNetCore, SHAPENET_PATH

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
import utils
from pdb import set_trace as st
import torch

from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass
SUPERVISED = True

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
        model.classifier = nn.Linear(512, 10)
    elif args.dataset == "shapenet" or args.dataset == "shapenet3d":
        #model.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,bias=False)
        model.backbone.maxpool = nn.Identity()
        model.classifier = nn.Linear(512, 1000)        

    make_contiguous(model)
    try:
        model.backbone.load_state_dict(torch.load("resnet18-f37072fd.pth", map_location="cpu"), strict=False)
    except:
        st()
        model.backbone.load_state_dict(torch.load("resnet50-19c8e357.pth", map_location="cpu"), strict=False)        
    model.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,bias=False)
    model.classifier = nn.Linear(512, 13) #13, 52        
    model.rot_classifier = nn.Linear(512*2, 50)
    #model.classifier = nn.Linear(2048, 13)        

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
            SUPERVISED=SUPERVISED,
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
    elif args.dataset == "shapenet":
        train_dataset = utils.Dataset_sp_train("/home/peterwg/dataset/ShapeNetRendering_scale/ShapeNetRendering_train.json", transform=utils.train_transform_sp_full, return_rot=True)
        train_loader = prepare_dataloader(
            train_dataset, batch_size=args.batch_size, num_workers=args.num_workers
        )
    elif args.dataset == "shapenet3d":
        train_dataset = utils.Dataset_3d_train("/home/peterwg/dataset/ShapeNetRendering/ShapeNetRenderingso3_train.json", transform=utils.train_transform_sp, test_transform=utils.test_transform_sp, return_rot=SUPERVISED)
        '''dataset_class = ShapeNetCore(SHAPENET_PATH, version=2, synsets=['02691156', '02958343', '03636649', '04256520', '04530566', '02828884', '03001627', '03691459', '04379243', '02933112', '03211117', '04090263', '04401088'], phase="train")
        train_dataset = utils.Shapenet3d(dataset_class, transform=utils.train_transform_sp)'''
        train_loader = prepare_dataloader(
            train_dataset, batch_size=args.batch_size, num_workers=args.num_workers
        )
    elif args.dataset == "shapenet3drel":
        train_dataset = utils.Dataset_3d_val_rel("/home/peterwg/dataset/ShapeNetRendering/ShapeNetRenderingso3_linear_train.json", train_transform=utils.train_transform_sp, transform=utils.test_transform_sp, return_sem=True)
        '''dataset_class = ShapeNetCore(SHAPENET_PATH, version=2, synsets=['02691156', '02958343', '03636649', '04256520', '04530566', '02828884', '03001627', '03691459', '04379243', '02933112', '03211117', '04090263', '04401088'], phase="train")
        train_dataset = utils.Shapenet3d(dataset_class, transform=utils.train_transform_sp)'''
        train_loader = prepare_dataloader(
            train_dataset, batch_size=args.batch_size, num_workers=args.num_workers
        )
    elif args.dataset == "shapenet3ds":
        train_dataset = utils.Dataset_3d_train2("/home/peterwg/dataset/ShapeNetRendering/ShapeNetRendering3d_train.json", transform=utils.train_transform_sp)
        '''dataset_class = ShapeNetCore(SHAPENET_PATH, version=2, synsets=['02691156', '02958343', '03636649', '04256520', '04530566', '02828884', '03001627', '03691459', '04379243', '02933112', '03211117', '04090263', '04401088'], phase="train")
        train_dataset = utils.Shapenet3d(dataset_class, transform=utils.train_transform_sp)'''
        train_loader = prepare_dataloader(
            train_dataset, batch_size=args.batch_size, num_workers=args.num_workers
        )
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
            name=str(args.checkpoint_dir).split("/")[-1],#args.name,
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
