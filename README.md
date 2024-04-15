# Semantic-Aware Implicit Template Learning via Part Deformation Consistency



Official implementation of ICCV 2023 [paper](https://arxiv.org/pdf/2308.11916.pdf), "Semantic-Aware Implicit Template Learning via Part Deformation Consistency" by Sihyeon Kim, Minseok Joo, Jaewon Lee, Juyeon Ko, Juhan Cha, and Hyunwoo J. Kim

![poster](sources/poster.png)


## Dense correspondence
We evaluate unsupervised dense correspondence between shapes with various surrogate tasks.

### ● Keypoint transfer
<p align="center"> 
  <img src="https://raw.githubusercontent.com/mlvlab/PDC/main/sources/git_feature_kp.gif" width="80%" />
</p>

### ● Part label transfer
<p align="center"> 
  <img src="https://raw.githubusercontent.com/mlvlab/PDC/main/sources/git_feature_plt.gif" width="80%" />
</p>

### ● Texture transfer
<p align="center"> 
  <img src="https://raw.githubusercontent.com/mlvlab/PDC/main/sources/git_feature_text.gif" width="80%" />
</p>

## Settings
* Clone repository

```
git clone https://github.com/mlvlab/PDC.git
cd PDC
```

* Setup conda environment
```
conda env create --file env.yaml
conda activate PDC
```

* Install packages that require manual setup
```
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install setuptools==59.5.0
```

* Install [torchmeta](https://github.com/tristandeleu/pytorch-meta/tree/794bf82348fbdc2b68b04f5de89c38017d54ba59)

Additionally, the torchmeta library is necessary. Please install it by following the instructions(Install from Source) in its GitHub page.


## Data preprocessing
For data preprocessing, we follow [DIF](https://github.com/microsoft/DIF-Net/tree/be2d8be0c5190c5db6da98ef4cf8c1d401e60000) to generate SDF values/surface points/normal vectors for original meshes based on [mesh_to_sdf](https://github.com/marian42/mesh_to_sdf).
Go DIF repo for some example SDF datasets for [ShapeNetV2](https://shapenet.org/).



## Training
```
python train.py --config=configs/train/<category>.yml
```


## UPDATES
04.10.23. Initial code release

10.11.23. Main code release

15.04.22. Refine some codes & add utils

## Acknowledgement
This repo is based on [SIREN](https://github.com/vsitzmann/siren) and [DIF](https://github.com/microsoft/DIF-Net/tree/be2d8be0c5190c5db6da98ef4cf8c1d401e60000).

We especially thank Yu Deng, the author of DIF, for helping us with data preprocessing!

## Citation
```
@inproceedings{kim2023semantic,
  title={Semantic-Aware Implicit Template Learning via Part Deformation Consistency},
  author={Kim, Sihyeon and Joo, Minseok and Lee, Jaewon and Ko, Juyeon and Cha, Juhan and Kim, Hyunwoo J},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
}
```

## License
Code is released under [MIT License].

> Copyright (c) 2023-present Korea University Research and Business Foundation & MLV Lab  
