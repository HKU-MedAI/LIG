# Large Images are Gaussians: High-Quality Large Image Representation with Levels of 2D Gaussian Splatting

This is the official code for https://arxiv.org/abs/2502.09039.

## Installation

Clone this repository and install packages:
```
git clone git@github.com:HKU-MedAI/LIG.git
conda create -n lig python=3.10
pip install -r requirements.txt
cd gsplat2d/gsplat2d/cuda/csrc
mkdir third_party
cd third_party
git clone https://github.com/g-truc/glm.git
cd ../../../..
python setup.py build
python setup.py install
cd ..
conda activate lig
```

## Dataset

Download the STimage [here](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/ltzhu99_connect_hku_hk/ETT5fZwxKUNPuvevfgdMXkcBKft_yCVnY1mZ7qS_LEMRxg?e=3ahAVm) and DIV-HR data (Validation Data (HR images)) [here](https://data.vision.ee.ethz.ch/cvl/DIV2K/).

The dataset folder is organized as follows.

```bash
├── dataset
|   | STimage
|     ├── Human_Heart_0.png
|     ├── Human_Heart_1.png
│     ├── ...
│   | DIV2K_valid_HR
│     ├── 0801.png
│     ├── 0802.png
│     ├── ...
```

## Training

Run `python train.py` to start training. The dataset and the parameters can be editted in the script. The metrics, representations, and the images will be saved.

## Acknowledgement

The codebase is developed based on [GaussianImage](https://github.com/Xinjie-Q/GaussianImage) and [gsplat](https://github.com/nerfstudio-project/gsplat).

## Citation

If you find our work useful, please kindly cite as:
```
@article{zhu2025large,
  title={Large Images are Gaussians: High-Quality Large Image Representation with Levels of 2D Gaussian Splatting},
  author={Zhu, Lingting and Lin, Guying and Chen, Jinnan and Zhang, Xinjie and Jin, Zhenchao and Wang, Zhao and Yu, Lequan},
  journal={arXiv preprint arXiv:2502.09039},
  year={2025}
}
```
