# Trajectory Regularization Enhances Self-Supervised Geometric Representation
[[Paper]](https://arxiv.org/pdf/2403.14973) 

## Overview
This is the author's re-implementation of the method described in:  
"[Trajectory Regularization Enhances Self-Supervised Geometric Representation](https://arxiv.org/abs/2403.14973)"   
[Jiayun Wang](http://pwang.pw/),&nbsp; [Yubei Chen](https://redwood.berkeley.edu/people/yubei-chen/),&nbsp;
[Stella X. Yu](https://www1.icsi.berkeley.edu/~stellayu/); (UC Berkeley & UC Davis & U Michigan)&nbsp;
in ArXiv (2024).

## Dataset Generation
For the ShapeNet dataset rendering, please refer to the [data_rendering folder](./data_rendering). Detailed instructions will be added soon.

## Training
To train the model on the rendered ShapeNet dataset, run the following code:
```
CUDA_VISIBLE_DEVICES=0 sh scripts/pretrain/cifar/vicreg_sn.sh shapenet3d SAVED_WANDB_LOG_NAME
```

## Evaluation
After training the model, you can use linear probe to evaluate the performance
```
CUDA_VISIBLE_DEVICES=0 sh scripts/pretrain/shapenet/vicreg_linear.sh shapenet3d PATH_TO_PRETRAINED_MODEL
```
Or 
```
CUDA_VISIBLE_DEVICES=0 sh scripts/pretrain/shapenet/vicreg.sh shapenet3d PATH_TO_PRETRAINED_MODEL
```

## Acknowledgement
This repo is heavily built on [solo-learn](https://github.com/vturrisi/solo-learn).

## License and Citation
The use of this software is released under [GNU GENERAL PUBLIC LICENSE](./LICENSE).
```
@article{wang2024trajectory,
  title={Trajectory Regularization Enhances Self-Supervised Geometric Representation},
  author={Wang, Jiayun and Yu, Stella X and Chen, Yubei},
  journal={arXiv preprint arXiv:2403.14973},
  year={2024}
}
```