## High-Quality Infrared Image Synthesis via a Three-Stage Cascaded Generative Adversarial Network-guide

## Highlights

![](E:\红外成像仿真\2025论文\pytorch-CycleGAN-and-pix2pix-master (1)\pytorch-CycleGAN-and-pix2pix-master\总体概述图.jpg)

**<p align="justify"> Abstract:** This paper proposes a novel three-stage cascaded generative adversarial network (GAN) for generating high-quality infrared images from visible light images. The proposed framework comprises three key stages: a U-Net network for initial image segmentation and structural information extraction, an improved DoubleU-Net network for detail enhancement and edge contour strengthening, and a noise reduction network for denoising and final image refinement. By incorporating residual connections, atrous spatial pyramid pooling (ASPP), and squeeze-and-excitation (SE) blocks, the network effectively captures multi-scale features and refines image details. Experimental results, evaluated both qualitatively and quantitatively using metrics such as Peak Signal-to-Noise Ratio (PSNR), Structure Similarity Index Measure (SSIM), and Root Mean Square Error (RMSE), demonstrate that the proposed model outperforms existing methods in generating infrared images with richer texture details and higher fidelity. 

## Comparisons of objective indicators

Three objective metrics of PSNR, SSIM, and RMSE are employed to compare the proposed algorithm with other typical image conversion methods. Please refer to our paper for more details.

|     algorithm      |  PSNR  |  SSIM   |   RMSE   |
| :----------------: | :----: | :-----: | :------: |
|      pix2pix       | 17.334 | 0.65313 | 36.10325 |
| DoubleUnet_pix2pix | 17.412 | 0.64075 | 35.37405 |
|      proposed      | 24.083 | 0.84835 | 17.73998 |

## Visualization of Three-Stage  Cascaded Generative Adversarial Network

Comparison of the effect of simulating infrared images with self-built datasets

![](E:\红外成像仿真\2025论文\pytorch-CycleGAN-and-pix2pix-master (1)\pytorch-CycleGAN-and-pix2pix-master\可视化图1.jpg)

Comparison of infrared simulation effects on Kaist lab dataset

![](E:\红外成像仿真\2025论文\pytorch-CycleGAN-and-pix2pix-master (1)\pytorch-CycleGAN-and-pix2pix-master\可视化图2.jpg)

## Environment:

Install [PyTorch](http://pytorch.org) and 0.4+ and other dependencies (e.g., torchvision, [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)).

Install the required packages:

```bash
pip install -r requirements.txt
```

## Datasets Download:

Put the datasets in `./dataset/aligenedkaist` folder.

## How to Run

- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.

- To log training progress and test images to W&B dashboard, set the `--use_wandb` flag with train and test script

- Run a model:

  ```bash
  python train.py --dataroot ./datasets/aligenedkaist --name aligenedkaist_pix2pix --model pix2pix --direction BtoATrain a model
  ```