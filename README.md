# DMPlug: A Plug-in Method for Solving Inverse Problems with Diffusion Models (NeurIPS 2024)

This is the official implementation of "DMPlug: A Plug-in Method for Solving Inverse Problems with Diffusion Models". This paper has been accpeted by [NeurIPS 2024](https://neurips.cc/). You can find our paper via [arXiv](https://arxiv.org/abs/2405.16749).

## Abstract
In this paper, we advocate viewing the reverse process in DMs as a function and propose a novel plug-in method for solving 
IPs using pretrained DMs, dubbed DMPlug. DMPlug addresses the issues of manifold feasibility and measurement feasibility 
in a principled manner, and also shows great potential for being **robust to unknown types and levels of noise**. Through 
extensive experiments across various IP tasks, including two linear and three nonlinear IPs, we demonstrate that DMPlug 
consistently **outperforms state-of-the-art methods**, often by large margins especially for nonlinear IPs (typically **3-6dB in terms of PSNR**).

![title](images/main.png)

## Getting started 

### 1) Clone the repository

```
git clone https://github.com/sun-umn/DMPlug.git

cd DMPlug
```


### 2) Download pretrained checkpoint

From the [link](https://drive.google.com/drive/u/0/folders/15sp0fzSvITsu77ZETEaJf6y9G3HJAjai), download the checkpoint "celebahq_p2.pt" and paste it to ./models/;

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

## Citation

If you find our work interesting, please consider citing
```
@misc{wang_dmplug_2024,
	title = {{DMPlug}: {A} {Plug}-in {Method} for {Solving} {Inverse} {Problems} with {Diffusion} {Models}},
	shorttitle = {{DMPlug}},
	url = {http://arxiv.org/abs/2405.16749},
	doi = {10.48550/arXiv.2405.16749},
	urldate = {2024-06-03},
	publisher = {arXiv},
	author = {Wang, Hengkang and Zhang, Xu and Li, Taihui and Wan, Yuxiang and Chen, Tiancong and Sun, Ju},
	month = may,
	year = {2024},
	note = {arXiv:2405.16749 [cs]},
	keywords = {Computer Science - Computer Vision and Pattern Recognition, Computer Science - Machine Learning},
}

```

## Contact

- Hengkang Wang, wang9881@umn.edu, [Homepage](https://scholar.google.com/citations?hl=en&user=APqDZvUAAAAJ)
- Xu Zhang, spongezhang@gmail.com, [Homepage](https://xu-zhang-1987.github.io)
- Taihui Li, lixx5027@umn.edu, [Homepage](https://taihui.github.io/)
- Tiancong Chen, chen6271@umn.edu, [Homepage](https://sites.google.com/view/tiancong-chen)
- Yuxiang Wan, wan01530@umn.edu, [Homepage](https://www.linkedin.com/in/yuxiang-wan-31518921a/)
- Ju Sun, jusun@umn.edu, [Homepage](https://sunju.org/)
