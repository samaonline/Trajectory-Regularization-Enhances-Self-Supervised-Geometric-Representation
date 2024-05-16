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

import logging
from argparse import ArgumentParser
from functools import partial
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union
import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from solo.backbones import (
    convnext_base,
    convnext_large,
    convnext_small,
    convnext_tiny,
    poolformer_m36,
    poolformer_m48,
    poolformer_s12,
    poolformer_s24,
    poolformer_s36,
    resnet18,
    resnet50,
    swin_base,
    swin_large,
    swin_small,
    swin_tiny,
    vit_base,
    vit_large,
    vit_small,
    vit_tiny,
    wide_resnet28w2,
    wide_resnet28w8,
)
from solo.utils.knn import WeightedKNNClassifier
from solo.utils.lars import LARS
from solo.utils.metrics import accuracy_at_k, weighted_mean
from solo.utils.misc import remove_bias_and_norm_from_weight_decay
from solo.utils.momentum import MomentumUpdater, initialize_momentum_params
from torch.optim.lr_scheduler import MultiStepLR
from pdb import set_trace as st
USE_CONV3_FEATS = True
sem_SUPERVISED = False
rot_SUPERVISED = True #False

def static_lr(
    get_lr: Callable, param_group_indexes: Sequence[int], lrs_to_replace: Sequence[float]
):
    lrs = get_lr()
    for idx, lr in zip(param_group_indexes, lrs_to_replace):
        lrs[idx] = lr
    return lrs

class batch_mul_gaussian():
    def __init__(self,D=512,H=4,W=4,seed=0):
        torch.manual_seed(seed)
        self.sign = torch.bernoulli(torch.empty(D*H*W).uniform_(0, 1)).cuda()*2 - 1

        perm = []
        for i in range(H*W):
            torch.manual_seed(i)
            perm.append(torch.randperm(D))
        self.perm = torch.cat(perm)
        
    def forward(self, input_):
        B, D, H, W = input_.shape
        input_ = input_.reshape(B, D, -1).permute(0,2,1).reshape(B, -1)
        input_ = input_[:, self.perm]*self.sign
        input_ = input_.reshape((B, H*W,D)).permute(0,2,1).reshape((B, D, H, W))
        return input_

'''def mul_gaussian(input_,seed=0):
    B, L = input_.shape
    #input_ = input_.flatten(start_dim=1)
    
    torch.manual_seed(seed)
    sign = torch.bernoulli(torch.empty(L).uniform_(0, 1)).cuda()*2 - 1

    torch.manual_seed(seed)
    perm = torch.randperm(L).cuda()
    
    input_ = input_[input_[:, perm]*sign
    #input_ = input_.reshape(toshape)
    return input_'''
    
### transformer
CONFIG = {"hidden_size": 512, "num_heads": 4, "attention_dropout_rate": 0.0,"mlp_dim": 512,"dropout_rate": 0.1}
class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config["num_heads"]
        self.attention_head_size = int(config["hidden_size"] / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config["hidden_size"], self.all_head_size)
        self.key = nn.Linear(config["hidden_size"], self.all_head_size)
        self.value = nn.Linear(config["hidden_size"], self.all_head_size)

        self.out = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.attn_dropout = nn.Dropout(config["attention_dropout_rate"])
        self.proj_dropout = nn.Dropout(config["attention_dropout_rate"])

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(config["hidden_size"], config["mlp_dim"])
        self.fc2 = nn.Linear(config["mlp_dim"], config["hidden_size"])
        ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu} #, "swish": swish}
        self.act_fn = ACT2FN["gelu"]
        self.dropout = nn.Dropout(config["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        hidden_size = 512
        
        self.hidden_size = hidden_size
        self.attention_norm = nn.LayerNorm( hidden_size, eps=1e-6).cuda()
        self.ffn_norm = nn.LayerNorm( hidden_size, eps=1e-6).cuda()
        self.ffn = Mlp(CONFIG).cuda()
        self.attn = Attention(CONFIG, vis=False).cuda()

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights
    
###

class BaseMethod(pl.LightningModule):
    _BACKBONES = {
        "resnet18": resnet18,
        "resnet50": resnet50,
        "vit_tiny": vit_tiny,
        "vit_small": vit_small,
        "vit_base": vit_base,
        "vit_large": vit_large,
        "swin_tiny": swin_tiny,
        "swin_small": swin_small,
        "swin_base": swin_base,
        "swin_large": swin_large,
        "poolformer_s12": poolformer_s12,
        "poolformer_s24": poolformer_s24,
        "poolformer_s36": poolformer_s36,
        "poolformer_m36": poolformer_m36,
        "poolformer_m48": poolformer_m48,
        "convnext_tiny": convnext_tiny,
        "convnext_small": convnext_small,
        "convnext_base": convnext_base,
        "convnext_large": convnext_large,
        "wide_resnet28w2": wide_resnet28w2,
        "wide_resnet28w8": wide_resnet28w8,
    }
    _OPTIMIZERS = {
        "sgd": torch.optim.SGD,
        "lars": LARS,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }
    _SCHEDULERS = [
        "reduce",
        "warmup_cosine",
        "step",
        "exponential",
        "none",
    ]

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        backbone_args: dict,
        max_epochs: int,
        batch_size: int,
        optimizer: str,
        lr: float,
        weight_decay: float,
        classifier_lr: float,
        accumulate_grad_batches: Union[int, None],
        extra_optimizer_args: Dict,
        scheduler: str,
        num_large_crops: int,
        num_small_crops: int,
        min_lr: float = 0.0,
        warmup_start_lr: float = 0.00003,
        warmup_epochs: float = 10,
        scheduler_interval: str = "step",
        lr_decay_steps: Sequence = None,
        knn_eval: bool = False,
        knn_k: int = 20,
        no_channel_last: bool = False,
        **kwargs,
    ):
        """Base model that implements all basic operations for all self-supervised methods.
        It adds shared arguments, extract basic learnable parameters, creates optimizers
        and schedulers, implements basic training_step for any number of crops,
        trains the online classifier and implements validation_step.

        Args:
            backbone (str): architecture of the base backbone.
            num_classes (int): number of classes.
            backbone_params (dict): dict containing extra backbone args, namely:
                #! optional, if it's not present, it is considered as False
                cifar (bool): flag indicating if cifar is being used.
                #! only for resnet
                zero_init_residual (bool): change the initialization of the resnet backbone.
                #! only for vit
                patch_size (int): size of the patches for ViT.
            max_epochs (int): number of training epochs.
            batch_size (int): number of samples in the batch.
            optimizer (str): name of the optimizer.
            lr (float): learning rate.
            weight_decay (float): weight decay for optimizer.
            classifier_lr (float): learning rate for the online linear classifier.
            accumulate_grad_batches (Union[int, None]): number of batches for gradient accumulation.
            extra_optimizer_args (Dict): extra named arguments for the optimizer.
            scheduler (str): name of the scheduler.
            num_large_crops (int): number of big crops.
            num_small_crops (int): number of small crops .
            min_lr (float): minimum learning rate for warmup scheduler. Defaults to 0.0.
            warmup_start_lr (float): initial learning rate for warmup scheduler.
                Defaults to 0.00003.
            warmup_epochs (float): number of warmup epochs. Defaults to 10.
            scheduler_interval (str): interval to update the lr scheduler. Defaults to 'step'.
            lr_decay_steps (Sequence, optional): steps to decay the learning rate if scheduler is
                step. Defaults to None.
            knn_eval (bool): enables online knn evaluation while training.
            knn_k (int): the number of neighbors to use for knn.
            no_channel_last (bool). Disables channel last conversion operation which
                speeds up training considerably. Defaults to False.
                https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html#converting-existing-models

        .. note::
            When using distributed data parallel, the batch size and the number of workers are
            specified on a per process basis. Therefore, the total batch size (number of workers)
            is calculated as the product of the number of GPUs with the batch size (number of
            workers).

        .. note::
            The learning rate (base, min and warmup) is automatically scaled linearly based on the
            batch size and gradient accumulation.

        .. note::
            For CIFAR10/100, the first convolutional and maxpooling layers of the ResNet backbone
            are slightly adjusted to handle lower resolution images (32x32 instead of 224x224).

        """

        super().__init__()

        # resnet backbone related
        self.backbone_args = backbone_args

        # training related
        self.num_classes = num_classes
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.classifier_lr = classifier_lr
        self.accumulate_grad_batches = accumulate_grad_batches
        self.extra_optimizer_args = extra_optimizer_args
        self.scheduler = scheduler
        self.lr_decay_steps = lr_decay_steps
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.warmup_epochs = warmup_epochs
        assert scheduler_interval in ["step", "epoch"]
        self.scheduler_interval = scheduler_interval
        self.num_large_crops = num_large_crops
        self.num_small_crops = num_small_crops
        self.knn_eval = knn_eval
        self.knn_k = knn_k
        self.no_channel_last = no_channel_last

        # multicrop
        self.num_crops = self.num_large_crops + self.num_small_crops

        # all the other parameters
        self.extra_args = kwargs

        # turn on multicrop if there are small crops
        self.multicrop = self.num_small_crops != 0

        # if accumulating gradient then scale lr
        if self.accumulate_grad_batches:
            self.lr = self.lr * self.accumulate_grad_batches
            self.classifier_lr = self.classifier_lr * self.accumulate_grad_batches
            self.min_lr = self.min_lr * self.accumulate_grad_batches
            self.warmup_start_lr = self.warmup_start_lr * self.accumulate_grad_batches

        assert backbone in BaseMethod._BACKBONES
        self.base_model = self._BACKBONES[backbone]

        self.backbone_name = backbone

        # initialize backbone
        kwargs = self.backbone_args.copy()
        cifar = kwargs.pop("cifar", False)
        # swin specific
        if "swin" in self.backbone_name and cifar:
            kwargs["window_size"] = 4

        method = self.extra_args.get("method", None)
        self.backbone = self.base_model(method,pretrained=True,  **kwargs) ###
        
        if self.backbone_name.startswith("resnet"):
            self.features_dim = self.backbone.inplanes
            # remove fc layer
            self.backbone.fc = nn.Identity()
            if cifar:
                self.backbone.conv1 = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=2, bias=False
                )
                self.backbone.maxpool = nn.Identity()
        else:
            self.features_dim = self.backbone.num_features

        self.classifier = nn.Linear(self.features_dim, num_classes)
        self.rot_classifier = nn.Linear(self.features_dim, num_classes)

        if self.knn_eval:
            self.knn = WeightedKNNClassifier(k=self.knn_k, distance_fx="euclidean")

        if scheduler_interval == "step":
            logging.warn(
                f"Using scheduler_interval={scheduler_interval} might generate "
                "issues when resuming a checkpoint."
            )

        # can provide up to ~20% speed up
        if not no_channel_last:
            self = self.to(memory_format=torch.channels_last)
        # bmg and transformer
        self.bmg = batch_mul_gaussian(seed=100)
        
        hidden_size = 512
        n_patches = 16
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.dropout = nn.Dropout(0.1)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, hidden_size))
        
        self.conv1x1 = nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=False)


    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds shared basic arguments that are shared for all methods.

        Args:
            parent_parser (ArgumentParser): argument parser that is used to create a
                argument group.

        Returns:
            ArgumentParser: same as the argument, used to avoid errors.
        """

        parser = parent_parser.add_argument_group("base")

        # backbone args
        BACKBONES = BaseMethod._BACKBONES

        parser.add_argument("--backbone", choices=BACKBONES, type=str)
        # extra args for resnet
        parser.add_argument("--zero_init_residual", action="store_true")
        # extra args for ViT
        parser.add_argument("--patch_size", type=int, default=16)

        # general train
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--lr", type=float, default=0.3)
        parser.add_argument("--classifier_lr", type=float, default=0.3)
        parser.add_argument("--weight_decay", type=float, default=0.0001)
        parser.add_argument("--num_workers", type=int, default=4)

        # wandb
        parser.add_argument("--name")
        parser.add_argument("--project")
        parser.add_argument("--entity", default=None, type=str)
        parser.add_argument("--wandb", action="store_true")
        parser.add_argument("--offline", action="store_true")

        parser.add_argument(
            "--optimizer", choices=BaseMethod._OPTIMIZERS.keys(), type=str, required=True
        )
        parser.add_argument("--exclude_bias_n_norm_wd", action="store_true")
        # lars args
        parser.add_argument("--grad_clip_lars", action="store_true")
        parser.add_argument("--eta_lars", default=1e-3, type=float)
        parser.add_argument("--exclude_bias_n_norm_lars", action="store_true")
        # adamw args
        parser.add_argument("--adamw_beta1", default=0.9, type=float)
        parser.add_argument("--adamw_beta2", default=0.999, type=float)

        parser.add_argument(
            "--scheduler", choices=BaseMethod._SCHEDULERS, type=str, default="reduce"
        )
        parser.add_argument("--lr_decay_steps", default=None, type=int, nargs="+")
        parser.add_argument("--min_lr", default=0.0, type=float)
        parser.add_argument("--warmup_start_lr", default=0.00003, type=float)
        parser.add_argument("--warmup_epochs", default=10, type=int)
        parser.add_argument(
            "--scheduler_interval", choices=["step", "epoch"], default="step", type=str
        )

        # online knn eval
        parser.add_argument("--knn_eval", action="store_true")
        parser.add_argument("--knn_k", default=20, type=int)

        # disables channel last optimization
        parser.add_argument("--no_channel_last", action="store_true")

        return parent_parser

    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Defines learnable parameters for the base class.

        Returns:
            List[Dict[str, Any]]:
                list of dicts containing learnable parameters and possible settings.
        """

        return [
            {"name": "backbone", "params": self.backbone.parameters()},
            {
                "name": "classifier",
                "params": self.classifier.parameters(),
                "lr": self.classifier_lr,
                "weight_decay": 0,
            },
        ]

    def configure_optimizers(self) -> Tuple[List, List]:
        """Collects learnable parameters and configures the optimizer and learning rate scheduler.

        Returns:
            Tuple[List, List]: two lists containing the optimizer and the scheduler.
        """

        learnable_params = self.learnable_params

        # exclude bias and norm from weight decay
        if self.extra_args.get("exclude_bias_n_norm_wd", False):
            learnable_params = remove_bias_and_norm_from_weight_decay(learnable_params)

        # indexes of parameters without lr scheduler
        idxs_no_scheduler = [i for i, m in enumerate(learnable_params) if m.pop("static_lr", False)]

        assert self.optimizer in self._OPTIMIZERS
        optimizer = self._OPTIMIZERS[self.optimizer]

        # create optimizer
        optimizer = optimizer(
            learnable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.extra_optimizer_args,
        )

        if self.scheduler.lower() == "none":
            return optimizer

        if self.scheduler == "warmup_cosine":
            max_warmup_steps = (
                self.warmup_epochs * (self.trainer.estimated_stepping_batches / self.max_epochs)
                if self.scheduler_interval == "step"
                else self.warmup_epochs
            )
            max_scheduler_steps = (
                self.trainer.estimated_stepping_batches
                if self.scheduler_interval == "step"
                else self.max_epochs
            )
            scheduler = {
                "scheduler": LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=max_warmup_steps,
                    max_epochs=max_scheduler_steps,
                    warmup_start_lr=self.warmup_start_lr if self.warmup_epochs > 0 else self.lr,
                    eta_min=self.min_lr,
                ),
                "interval": self.scheduler_interval,
                "frequency": 1,
            }
        elif self.scheduler == "step":
            scheduler = MultiStepLR(optimizer, self.lr_decay_steps)
        else:
            raise ValueError(f"{self.scheduler} not in (warmup_cosine, cosine, step)")

        if idxs_no_scheduler:
            partial_fn = partial(
                static_lr,
                get_lr=scheduler["scheduler"].get_lr
                if isinstance(scheduler, dict)
                else scheduler.get_lr,
                param_group_indexes=idxs_no_scheduler,
                lrs_to_replace=[self.lr] * len(idxs_no_scheduler),
            )
            if isinstance(scheduler, dict):
                scheduler["scheduler"].get_lr = partial_fn
            else:
                scheduler.get_lr = partial_fn

        return [optimizer], [scheduler]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        """
        This improves performance marginally. It should be fine
        since we are not affected by any of the downsides descrited in
        https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html#torch.optim.Optimizer.zero_grad

        Implemented as in here
        https://pytorch-lightning.readthedocs.io/en/1.5.10/guides/speed.html#set-grads-to-none
        """
        try:
            optimizer.zero_grad(set_to_none=True)
        except:
            optimizer.zero_grad()
            
    def transformer(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        
        layer = Block()
        x = layer(x)[0]
        
        encoder_norm = nn.LayerNorm(CONFIG["hidden_size"], eps=1e-6).cuda()
        x = encoder_norm(x)
        return x.mean(1)
        
    def forward(self, X) -> Dict:
        """Basic forward method. Children methods should call this function,
        modify the ouputs (without deleting anything) and return it.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict: dict of logits and features.
        """

        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)
        if rot_SUPERVISED:
            x_1, x_2 = X[:, :3,...],  X[:, 3:,...]
            feats_1 = self.backbone(x_1)
            feats_2 = self.backbone(x_2)
            
            if sem_SUPERVISED:
                logits = self.classifier(feats_1)
            else:
                logits = self.classifier(feats_1.detach()) #change!
            rot_logits = self.rot_classifier( torch.cat([feats_1,feats_2],axis=1) )
            return {"logits": logits, "rot_logits": rot_logits,"feats": feats_1} #, "x3": x3}
        elif USE_CONV3_FEATS:
            x = self.backbone.conv1(X)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)

            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            ## different!!
            conv3 = x
            #conv3 = self.conv1x1(conv3)
            ##
            x = self.backbone.layer4(x)
            conv4 = x
            #x3 = torch.flatten(x, 1)
            #x3 = F.normalize(x3, dim=-1)
            
            ### binding
            #x = self.bmg.forward(x)
            
            ###
            ### transformer
            #x = self.transformer(x)
            ###
            
            x = self.backbone.avgpool(x)
            feats = torch.flatten(x, 1)
            if sem_SUPERVISED:
                logits = self.classifier(feats)
            else:
                logits = self.classifier(feats.detach()) #change!
                
            if rot_SUPERVISED: # for specific pose only, depreciated 
                rot_logits = self.rot_classifier(feats)
            else:
                rot_logits = self.rot_classifier(feats.detach())
            #x = torch.flatten(x, 1)
            #x = self.fc(x)
            return {"logits": logits, "rot_logits": rot_logits,"feats": feats, "conv4": conv4} #, "x3": x3}
        else:
            feats = self.backbone(X)
            if SUPERVISED:
                logits = self.classifier(feats) #change!
            else:
                logits = self.classifier(feats.detach()) 
            return {"logits": logits, "feats": feats}

    def multicrop_forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Basic multicrop forward method that performs the forward pass
        for the multicrop views. Children classes can override this method to
        add new outputs but should still call this function. Make sure
        that this method and its overrides always return a dict.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict: dict of features.
        """

        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)
        feats = self.backbone(X)
        return {"feats": feats}

    def _base_shared_step(self, X: torch.Tensor, targets: torch.Tensor, rot_targets) -> Dict:
        """Forwards a batch of images X and computes the classification loss, the logits, the
        features, acc@1 and acc@5.

        Args:
            X (torch.Tensor): batch of images in tensor format.
            targets (torch.Tensor): batch of labels for X.

        Returns:
            Dict: dict containing the classification loss, logits, features, acc@1 and acc@5.
        """

        out = self(X)
        logits = out["logits"]
        rot_logits = out["rot_logits"]

        loss = F.cross_entropy(logits, targets, ignore_index=-1)
        rot_loss = F.cross_entropy(rot_logits, rot_targets, ignore_index=-1)

        # handle when the number of classes is smaller than 5
        top_k_max = min(5, logits.size(1))
        acc1, acc5 = accuracy_at_k(logits, targets, top_k=(1, top_k_max))
        acc1_rot, acc5_rot = accuracy_at_k(rot_logits, rot_targets, top_k=(1, top_k_max))

        out.update({"loss": loss, "rot_loss": rot_loss, "acc1": acc1, "acc5": acc5,  "acc1_rot": acc1_rot, "acc5_rot": acc5_rot})
        return out

    def base_training_step(self, X: torch.Tensor, targets: torch.Tensor, rot_targets) -> Dict:
        """Allows user to re-write how the forward step behaves for the training_step.
        Should always return a dict containing, at least, "loss", "acc1" and "acc5".
        Defaults to _base_shared_step

        Args:
            X (torch.Tensor): batch of images in tensor format.
            targets (torch.Tensor): batch of labels for X.

        Returns:
            Dict: dict containing the classification loss, logits, features, acc@1 and acc@5.
        """

        return self._base_shared_step(X, targets, rot_targets)

    def training_step(self, batch: List[Any], batch_idx: int) -> Dict[str, Any]:
        """Training step for pytorch lightning. It does all the shared operations, such as
        forwarding the crops, computing logits and computing statistics.

        Args:
            batch (List[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            Dict[str, Any]: dict with the classification loss, features and logits.
        """
        _, X, targets, rot_targets = batch

        X = [X] if isinstance(X, torch.Tensor) else X

        # check that we received the desired number of crops
        #assert len(X) == self.num_crops

        #outs = [self.base_training_step(x, targets) for x in X[: self.num_large_crops]]
        outs = [self.base_training_step(x, targets, rot_targets) for x in X]
        outs = {k: [out[k] for out in outs] for k in outs[0].keys()}

        if self.multicrop:
            multicrop_outs = [self.multicrop_forward(x) for x in X[self.num_large_crops :]]
            for k in multicrop_outs[0].keys():
                outs[k] = outs.get(k, []) + [out[k] for out in multicrop_outs]

        # loss and stats
        outs["loss"] = sum(outs["loss"]) / self.num_large_crops
        outs["acc1"] = sum(outs["acc1"]) / self.num_large_crops
        outs["acc5"] = sum(outs["acc5"]) / self.num_large_crops
        outs["rot_loss"] = sum(outs["rot_loss"]) / self.num_large_crops
        outs["acc1_rot"] = sum(outs["acc1_rot"]) / self.num_large_crops
        outs["acc5_rot"] = sum(outs["acc5_rot"]) / self.num_large_crops
        
        metrics = {
            "train_class_loss": outs["loss"],
            "train_acc1": outs["acc1"],
            "train_acc5": outs["acc5"],
            "train_rot_loss": outs["rot_loss"],
            "train_rot_acc1": outs["acc1_rot"],
            "train_rot_acc5": outs["acc5_rot"],
        }

        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        if self.knn_eval:
            targets = targets.repeat(self.num_large_crops)
            mask = targets != -1
            self.knn(
                train_features=torch.cat(outs["feats"][: self.num_large_crops])[mask].detach(),
                train_targets=targets[mask],
            )

        return outs

    def base_validation_step(self, X: torch.Tensor, targets: torch.Tensor, rot_targets) -> Dict:
        """Allows user to re-write how the forward step behaves for the validation_step.
        Should always return a dict containing, at least, "loss", "acc1" and "acc5".
        Defaults to _base_shared_step

        Args:
            X (torch.Tensor): batch of images in tensor format.
            targets (torch.Tensor): batch of labels for X.

        Returns:
            Dict: dict containing the classification loss, logits, features, acc@1 and acc@5.
        """

        return self._base_shared_step(X, targets, rot_targets)

    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int, dataloader_idx: int = None
    ) -> Dict[str, Any]:
        """Validation step for pytorch lightning. It does all the shared operations, such as
        forwarding a batch of images, computing logits and computing metrics.

        Args:
            batch (List[torch.Tensor]):a batch of data in the format of [img_indexes, X, Y].
            batch_idx (int): index of the batch.

        Returns:
            Dict[str, Any]: dict with the batch_size (used for averaging), the classification loss
                and accuracies.
        """
        X, targets, rot_targets = batch #X shape: [300, 6, 32, 32]
        batch_size = targets.size(0)
        
        out = self.base_validation_step(X, targets, rot_targets)

        if self.knn_eval and not self.trainer.sanity_checking:
            self.knn(test_features=out.pop("feats").detach(), test_targets=targets.detach())

        metrics = {
            "batch_size": batch_size,
            "val_loss": out["loss"],
            "val_acc1": out["acc1"],
            "val_acc5": out["acc5"],
            "val_rot_loss": out["rot_loss"],
            "val_rot_acc1": out["acc1_rot"],
            "val_rot_acc5": out["acc5_rot"],
        }
        return metrics

    def validation_epoch_end(self, outs: List[Dict[str, Any]]):
        """Averages the losses and accuracies of all the validation batches.
        This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.

        Args:
            outs (List[Dict[str, Any]]): list of outputs of the validation step.
        """

        val_loss = weighted_mean(outs, "val_loss", "batch_size")
        val_acc1 = weighted_mean(outs, "val_acc1", "batch_size")
        val_acc5 = weighted_mean(outs, "val_acc5", "batch_size")
        val_rot_acc1 = weighted_mean(outs, "val_rot_acc1", "batch_size")
        val_rot_acc5 = weighted_mean(outs, "val_rot_acc5", "batch_size")
        log = {"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5, "val_rot_acc1": val_rot_acc1, "val_rot_acc5": val_rot_acc5}

        if self.knn_eval and not self.trainer.sanity_checking:
            val_knn_acc1, val_knn_acc5 = self.knn.compute()
            log.update({"val_knn_acc1": val_knn_acc1, "val_knn_acc5": val_knn_acc5})

        self.log_dict(log, sync_dist=True)


class BaseMomentumMethod(BaseMethod):
    def __init__(
        self,
        base_tau_momentum: float,
        final_tau_momentum: float,
        momentum_classifier: bool,
        **kwargs,
    ):
        """Base momentum model that implements all basic operations for all self-supervised methods
        that use a momentum backbone. It adds shared momentum arguments, adds basic learnable
        parameters, implements basic training and validation steps for the momentum backbone and
        classifier. Also implements momentum update using exponential moving average and cosine
        annealing of the weighting decrease coefficient.

        Args:
            base_tau_momentum (float): base value of the weighting decrease coefficient (should be
                in [0,1]).
            final_tau_momentum (float): final value of the weighting decrease coefficient (should be
                in [0,1]).
            momentum_classifier (bool): whether or not to train a classifier on top of the momentum
                backbone.
        """

        super().__init__(**kwargs)

        # momentum backbone
        kwargs = self.backbone_args.copy()
        cifar = kwargs.pop("cifar", False)
        # swin specific
        if "swin" in self.backbone_name and cifar:
            kwargs["window_size"] = 4

        method = self.extra_args.get("method", None)
        self.momentum_backbone = self.base_model(method, **kwargs)
        if self.backbone_name.startswith("resnet"):
            # remove fc layer
            self.momentum_backbone.fc = nn.Identity()
            if cifar:
                self.momentum_backbone.conv1 = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=2, bias=False
                )
                self.momentum_backbone.maxpool = nn.Identity()
        else:
            self.features_dim = self.momentum_backbone.num_features

        initialize_momentum_params(self.backbone, self.momentum_backbone)

        # momentum classifier
        if momentum_classifier:
            self.momentum_classifier: Any = nn.Linear(self.features_dim, self.num_classes)
        else:
            self.momentum_classifier = None

        # momentum updater
        self.momentum_updater = MomentumUpdater(base_tau_momentum, final_tau_momentum)

    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Adds momentum classifier parameters to the parameters of the base class.

        Returns:
            List[Dict[str, Any]]:
                list of dicts containing learnable parameters and possible settings.
        """

        momentum_learnable_parameters = []
        if self.momentum_classifier is not None:
            momentum_learnable_parameters.append(
                {
                    "name": "momentum_classifier",
                    "params": self.momentum_classifier.parameters(),
                    "lr": self.classifier_lr,
                    "weight_decay": 0,
                }
            )
        return super().learnable_params + momentum_learnable_parameters

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Defines base momentum pairs that will be updated using exponential moving average.

        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs (two element tuples).
        """

        return [(self.backbone, self.momentum_backbone)]

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds basic momentum arguments that are shared for all methods.

        Args:
            parent_parser (ArgumentParser): argument parser that is used to create a
                argument group.

        Returns:
            ArgumentParser: same as the argument, used to avoid errors.
        """

        parent_parser = super(BaseMomentumMethod, BaseMomentumMethod).add_model_specific_args(
            parent_parser
        )
        parser = parent_parser.add_argument_group("base")

        # momentum settings
        parser.add_argument("--base_tau_momentum", default=0.99, type=float)
        parser.add_argument("--final_tau_momentum", default=1.0, type=float)
        parser.add_argument("--momentum_classifier", action="store_true")

        return parent_parser

    def on_train_start(self):
        """Resets the step counter at the beginning of training."""
        self.last_step = 0

    @torch.no_grad()
    def momentum_forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Momentum forward method. Children methods should call this function,
        modify the ouputs (without deleting anything) and return it.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict: dict of logits and features.
        """

        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)
        feats = self.momentum_backbone(X)
        return {"feats": feats}

    def _shared_step_momentum(self, X: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
        """Forwards a batch of images X in the momentum backbone and optionally computes the
        classification loss, the logits, the features, acc@1 and acc@5 for of momentum classifier.

        Args:
            X (torch.Tensor): batch of images in tensor format.
            targets (torch.Tensor): batch of labels for X.

        Returns:
            Dict[str, Any]:
                a dict containing the classification loss, logits, features, acc@1 and
                acc@5 of the momentum backbone / classifier.
        """

        out = self.momentum_forward(X)

        if self.momentum_classifier is not None:
            feats = out["feats"]
            logits = self.momentum_classifier(feats)

            loss = F.cross_entropy(logits, targets, ignore_index=-1)
            acc1, acc5 = accuracy_at_k(logits, targets, top_k=(1, 5))
            out.update({"logits": logits, "loss": loss, "acc1": acc1, "acc5": acc5})

        return out

    def training_step(self, batch: List[Any], batch_idx: int) -> Dict[str, Any]:
        """Training step for pytorch lightning. It performs all the shared operations for the
        momentum backbone and classifier, such as forwarding the crops in the momentum backbone
        and classifier, and computing statistics.
        Args:
            batch (List[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            Dict[str, Any]: a dict with the features of the momentum backbone and the classification
                loss and logits of the momentum classifier.
        """

        outs = super().training_step(batch, batch_idx)

        _, X, targets = batch
        X = [X] if isinstance(X, torch.Tensor) else X

        # remove small crops
        X = X[: self.num_large_crops]

        momentum_outs = [self._shared_step_momentum(x, targets) for x in X]
        momentum_outs = {
            "momentum_" + k: [out[k] for out in momentum_outs] for k in momentum_outs[0].keys()
        }

        if self.momentum_classifier is not None:
            # momentum loss and stats
            momentum_outs["momentum_loss"] = (
                sum(momentum_outs["momentum_loss"]) / self.num_large_crops
            )
            momentum_outs["momentum_acc1"] = (
                sum(momentum_outs["momentum_acc1"]) / self.num_large_crops
            )
            momentum_outs["momentum_acc5"] = (
                sum(momentum_outs["momentum_acc5"]) / self.num_large_crops
            )

            metrics = {
                "train_momentum_class_loss": momentum_outs["momentum_loss"],
                "train_momentum_acc1": momentum_outs["momentum_acc1"],
                "train_momentum_acc5": momentum_outs["momentum_acc5"],
            }
            self.log_dict(metrics, on_epoch=True, sync_dist=True)

            # adds the momentum classifier loss together with the general loss
            outs["loss"] += momentum_outs["momentum_loss"]

        outs.update(momentum_outs)
        return outs

    def on_train_batch_end(self, outputs: Dict[str, Any], batch: Sequence[Any], batch_idx: int):
        """Performs the momentum update of momentum pairs using exponential moving average at the
        end of the current training step if an optimizer step was performed.

        Args:
            outputs (Dict[str, Any]): the outputs of the training step.
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.num_crops containing batches of images.
            batch_idx (int): index of the batch.
        """

        if self.trainer.global_step > self.last_step:
            # update momentum backbone and projector
            momentum_pairs = self.momentum_pairs
            for mp in momentum_pairs:
                self.momentum_updater.update(*mp)
            # log tau momentum
            self.log("tau", self.momentum_updater.cur_tau)
            # update tau
            self.momentum_updater.update_tau(
                cur_step=self.trainer.global_step,
                max_steps=self.trainer.estimated_stepping_batches,
            )
        self.last_step = self.trainer.global_step

    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int, dataloader_idx: int = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Validation step for pytorch lightning. It performs all the shared operations for the
        momentum backbone and classifier, such as forwarding a batch of images in the momentum
        backbone and classifier and computing statistics.
        Args:
            batch (List[torch.Tensor]): a batch of data in the format of [X, Y].
            batch_idx (int): index of the batch.
        Returns:
            Tuple(Dict[str, Any], Dict[str, Any]): tuple of dicts containing the batch_size (used
                for averaging), the classification loss and accuracies for both the online and the
                momentum classifiers.
        """

        parent_metrics = super().validation_step(batch, batch_idx)

        X, targets = batch
        batch_size = targets.size(0)

        out = self._shared_step_momentum(X, targets)

        metrics = None
        if self.momentum_classifier is not None:
            metrics = {
                "batch_size": batch_size,
                "momentum_val_loss": out["loss"],
                "momentum_val_acc1": out["acc1"],
                "momentum_val_acc5": out["acc5"],
            }

        return parent_metrics, metrics

    def validation_epoch_end(self, outs: Tuple[List[Dict[str, Any]]]):
        """Averages the losses and accuracies of the momentum backbone / classifier for all the
        validation batches. This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.
        Args:
            outs (Tuple[List[Dict[str, Any]]]):): list of outputs of the validation step for self
                and the parent.
        """

        parent_outs = [out[0] for out in outs]
        super().validation_epoch_end(parent_outs)

        if self.momentum_classifier is not None:
            momentum_outs = [out[1] for out in outs]

            val_loss = weighted_mean(momentum_outs, "momentum_val_loss", "batch_size")
            val_acc1 = weighted_mean(momentum_outs, "momentum_val_acc1", "batch_size")
            val_acc5 = weighted_mean(momentum_outs, "momentum_val_acc5", "batch_size")

            log = {
                "momentum_val_loss": val_loss,
                "momentum_val_acc1": val_acc1,
                "momentum_val_acc5": val_acc5,
            }
            self.log_dict(log, sync_dist=True)
