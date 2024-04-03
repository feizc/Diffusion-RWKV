## Diffusion-RWKV: Scaling RWKV-Like Architectures for Diffusion Models <br><sub>Official PyTorch Implementation</sub>

This repo contains PyTorch model definitions, pre-trained weights and training/sampling code for our paper scalable diffusion models with RWKV-like architectures (Diffusion-RWKV). 
It builds a series of architectures adapted from the RWKV model used in the NLP, with requisite modifications tailored for diffusion model applied to image generation tasks. 

![Diffusion-RWKV framework](visuals/framework.jpg) 


### 1. Environments

- Python 3.10
  - `conda create -n your_env_name python=3.10`

- Requirements file
  - `pip install -r requirements.txt`

- Install ``mmcv-full`` and ``mmcls``
  - `pip install -U openmim`
  - `mim install mmcv-full==1.7.0`
  - `pip install mmcls==0.25.0`


### 2. Training

We provide a training script for Diffusion-RWKV in [`train.py`](train.py). This script can be used to train unconditional, class-conditional Diffusion-RWKV models, it can be easily modified to support other types of conditioning. 

To launch DRWKV-H/2 (256x256) in the latent space training with `N` GPUs on one node:

```bash
torchrun --nnodes=1 --nproc_per_node=N train.py \
--model DRWKV-H/2 \
--dataset-type imagenet \
--data-path /path/to/imagenet/train \
--image-size 256 \
--latent_space True \
--task-type class-cond \
--vae_path /path/to/vae \
--num-classes 1000 
```

To launch DRWKV-B/2 (32x32) in the pixel space training with `N` GPUs on one node:
```bash
torchrun --nnodes=1 --nproc_per_node=N train.py \
--model DRWKV-B/2 \
--dataset-type celeba \
--data-path /path/to/imagenet/train \
--image-size 32 \
--task-type uncond 
```



There are several additional options; see [`train.py`](train.py) for details. 
All experiments in our work of training script can be found in file direction `script`. 


For convenience, the pre-trained DiS models can be downloaded in  
[huggingface](https://huggingface.co/feizhengcong/Diffusion-RWKV).

### 3. Evaluation

We include a [`sample.py`](sample.py) script which samples images from a Diffusion-RWKV model. Besides, we support other metrics evaluation, e.g., FLOPS and model parameters, in [`test.py`](test.py) script. 

```bash
python sample.py \
--model DRWKV-H/2 \
--ckpt /path/to/model \
--image-size 256 \
--num-classes 1000 \
--cfg-scale 1.5 \
--latent_space True
```

### 4. BibTeX

```bibtex
@article{FeiDRWKV2024,
  title={Diffusion-RWKV: Scaling RWKV-Like Architectures for Diffusion Models},
  author={Zhengcong Fei, Mingyuan Fan, Changqian Yu, Debang Li, Jusnshi Huang},
  year={2024},
  journal={arXiv preprint},
}
```
### 5. Acknowledgments

The codebase is based on the awesome [DiT](https://github.com/facebookresearch/DiT), [RWKV](https://github.com/BlinkDL/RWKV-LM), [DiS](https://github.com/feizc/DiS), and [Vision-RWKV](https://github.com/OpenGVLab/Vision-RWKV) repos. 


