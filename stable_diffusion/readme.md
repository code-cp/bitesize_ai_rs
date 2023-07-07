## about 

- this repo contains a simple implementation of stable diffusion 
- tested on Mac M1 chip 
- cargo run --release --features tch-gpu

## todo 

- [x] try https://github.com/tcapelle/Diffusion-Models-pytorch/blob/e9bab9a1ae1f1745493031f4163427fe884e12fb/modules.py#L161
- [ ] debug why train loss does not decrease 

## references 

- 2023 - 李宏毅 - Diffusion Model 原理剖析
- https://github.com/SingleZombie/DL-Demos/tree/master/dldemos/ddpm
- https://wandb.ai/capecape/train_sd/reports/How-to-Train-a-Conditional-Diffusion-Model-from-Scratch--VmlldzoyNzIzNTQ1
- https://jalammar.github.io/illustrated-stable-diffusion/
- https://amaarora.github.io/posts/2020-09-13-unet.html
- https://hackmd.io/@gianghoangcotai/HJSaDvKuI#Residual-block
- https://github.com/LaurentMazare/diffusers-rs/tree/main