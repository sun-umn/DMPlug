# DMPlug: A Plug-in Method for Solving Inverse Problems with Diffusion Models

This is the official implementation of "DMPlug: A Plug-in Method for Solving Inverse Problems with Diffusion Models". You can find our paper via [arXiv](https://arxiv.org/abs/2405.16749).


## Getting started 

### 1) Clone the repository

```
git clone https://github.com/sun-umn/DMPlug.git

cd DMPlug
```


### 2) Download pretrained checkpoint

From the [link](https://onedrive.live.com/?authkey=%21AOIJGI8FUQXvFf8&id=72419B431C262344%21103807&cid=72419B431C262344), download the checkpoint "celebahq_p2.pt" and paste it to ./models/;

From the [link](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh), download the checkpoint "ffhq_10m.pt" and paste it to ./models/;

From the [link](https://github.com/openai/guided-diffusion), download the checkpoint "lsun_bedroom.pt" and paste it to ./models/.
```
mkdir models
mv {DOWNLOAD_DIR}/celebahq_p2.pt ./models/
mv {DOWNLOAD_DIR}/ffqh_10m.pt ./models/
mv {DOWNLOAD_DIR}/lsun_bedroom.pt ./models/
```
{DOWNLOAD_DIR} is the directory that you downloaded checkpoint to.


### 3) Set environment

We use the external codes for motion-blurring and non-linear deblurring.

```
git clone https://github.com/VinAIResearch/blur-kernel-space-exploring bkse

git clone https://github.com/LeviBorodenko/motionblur motionblur
```

From the [link](https://drive.google.com/file/d/1vRoDpIsrTRYZKsOMPNbPcMtFDpCT6Foy/view), download the checkpoint "GOPRO_wVAE.pth" and paste it to ./experiments/pretrained/.
```
mv {DOWNLOAD_DIR}/GOPRO_wVAE.pt ./experiments/pretrained/
```
{DOWNLOAD_DIR} is the directory that you downloaded checkpoint to.

Install dependencies

```
conda env create -f environment.yml
conda activate dmplug
```

## Inference

```
# Superresolution, inpainting, nonlinear deblurring
python sr_inp_nonlinear.py --task 'super_resolution'
python sr_inp_nonlinear.py --task 'inpainting'
python sr_inp_nonlinear.py --task 'nonlinear_deblur'

# BID
python bid.py --kernel 'motion'
python bid.py --kernel 'gaussian'

# BID with turbulence
python turbulence.py
```

## References
This repo is developed based on [DPS](https://github.com/DPS2022/diffusion-posterior-sampling) and [BlindDPS](https://github.com/BlindDPS/blind-dps), especially for forward operations. Please also consider citing them if you use this repo.


## Contact

- Hengkang Wang, wang9881@umn.edu, [Homepage](https://scholar.google.com/citations?hl=en&user=APqDZvUAAAAJ)
- Xu Zhang, spongezhang@gmail.com, [Homepage](https://xu-zhang-1987.github.io)
- Taihui Li, lixx5027@umn.edu, [Homepage](https://taihui.github.io/)
- Tiancong Chen, chen6271@umn.edu, [Homepage](https://sites.google.com/view/tiancong-chen)
- Yuxiang Wan, wan01530@umn.edu, [Homepage](https://www.linkedin.com/in/yuxiang-wan-31518921a/)
- Ju Sun, jusun@umn.edu, [Homepage](https://sunju.org/)
