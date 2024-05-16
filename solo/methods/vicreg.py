import sys

sys.path.append('/home/peterwg/repos/geo-ssl')
import argparse
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
from solo.losses.vicreg import vicreg_loss_func
from solo.losses.dcl import PoseLoss, DCL, TotalCodingRate, Z_loss
from solo.methods.base import BaseMethod
import torch.nn.functional as F
import numpy as np
from pdb import set_trace as st
USE_LINEAR_TRAJ = False

def patch_normalize(f_c):
    f_c = F.normalize(f_c.reshape(f_c.shape[0], f_c.shape[1], -1), dim=-1)
    f_c = f_c.reshape(len(f_c), -1)   
    return f_c
    
class VICReg(BaseMethod):
    def __init__(
        self,
        proj_output_dim: int,
        proj_hidden_dim: int,
        sim_loss_weight: float,
        var_loss_weight: float,
        cov_loss_weight: float,
        alpha=1.0,
        **kwargs
    ):
        """Implements VICReg (https://arxiv.org/abs/2105.04906)

        Args:
            proj_output_dim (int): number of dimensions of the projected features.
            proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
            sim_loss_weight (float): weight of the invariance term.
            var_loss_weight (float): weight of the variance term.
            cov_loss_weight (float): weight of the covariance term.
        """

        super().__init__(**kwargs)

        self.sim_loss_weight = sim_loss_weight
        self.var_loss_weight = var_loss_weight
        self.cov_loss_weight = cov_loss_weight
        self.alpha = alpha

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(VICReg, VICReg).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("vicreg")

        # projector
        parser.add_argument("--proj_output_dim", type=int, default=2048)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # parameters
        parser.add_argument("--sim_loss_weight", default=25, type=float)
        parser.add_argument("--var_loss_weight", default=25, type=float)
        parser.add_argument("--cov_loss_weight", default=1.0, type=float)
        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Performs the forward pass of the backbone and the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        out = super().forward(X)
        z = self.projector(out["feats"])
        out.update({"z": z})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for VICReg reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of VICReg loss and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        
        class_loss = out["loss"]
        rot_loss = out["rot_loss"]
        l_pose = 0.0

        z1, z2 = out["z"]
        if USE_LINEAR_TRAJ:
            #f1, f2, f_c, f_l, f_r,  = out["feats"]
            f1, f2, f_c, f_l, f_r,  = out["conv4"]
        
            f_c = f_c.reshape(len(z1), -1)
            f_l = f_l.reshape(len(z1), -1)
            f_r = f_r.reshape(len(z1), -1)
            #_, _, f_c, f_l, f_r = out["x3"]
        
            # all feats
            '''vicreg_loss2 = vicreg_loss_func(
                f1,
                f2,
                sim_loss_weight=self.sim_loss_weight,
                var_loss_weight=self.var_loss_weight,
                cov_loss_weight=self.cov_loss_weight,
            )'''
        
            # add normalize
            #f_c = patch_normalize(f_c)
            #f_l = patch_normalize(f_l)
            #f_r = patch_normalize(f_r)
        
            # project
            #f_c = F.normalize(z_c, dim=-1)
            #f_l = F.normalize(z_l, dim=-1)
            #f_r = F.normalize(z_r, dim=-1)
        
            # old
            """f1 = F.normalize(f_c, dim=-1)
            f2 = F.normalize(f_l, dim=-1)"""
        
            f_c = F.normalize(f_c, dim=-1)
            f_l = F.normalize(f_l, dim=-1)
            f_r = F.normalize(f_r, dim=-1)# commented on 8/14/23

            #f_a = F.normalize(f_a, dim=-1)
            #f_b = F.normalize(f_b, dim=-1)
            #f_c = F.normalize(f_c, dim=-1)

            # NMCE loss
            '''criterion = TotalCodingRate(eps=0.5)
            loss_coding = criterion(z1.T)+criterion(z2.T)

            criterion_z = Z_loss()
            z_loss = criterion_z(torch.cat((z1, z2)).T)'''
            l2 = PoseLoss()
            l_pose =  l2(f_c, f_l, f_r)  #l2(f_b, f_a, f_c) #
            
        # ------- vicreg loss -------
        vicreg_loss = vicreg_loss_func(
            z1,
            z2,
            sim_loss_weight=self.sim_loss_weight,
            var_loss_weight=self.var_loss_weight,
            cov_loss_weight=self.cov_loss_weight,
        )
        
        #dcl = DCL(temperature=0.5)
        #dcl_loss = dcl(f1, f2) 

        if torch.isnan(vicreg_loss):
            st()
        



        self.log("train_vicreg_loss", vicreg_loss, on_epoch=True, sync_dist=True)
        if USE_LINEAR_TRAJ:
            self.log("l_pose", l_pose, on_epoch=True, sync_dist=True)
        
        if USE_LINEAR_TRAJ:
            return vicreg_loss + class_loss + rot_loss + self.alpha*l_pose #+ 1*loss_coding
        else:
            return vicreg_loss + class_loss + self.alpha*rot_loss #+ l_pose 