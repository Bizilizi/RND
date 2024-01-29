**Description**

This page contains the code I produced during my research at Oxford University.  
It contains a framework that syncs training loops of Pytorch Lightning and Avalanche.

**Models**  
I've developed 4 different architectures to solve the task of the "Continous Representation Learning" in Class Incremental learning. 
Each model has its own repository that is independent from others.

**RND, VAE FT, VQ VAE**  
Those are intermediate and failed experiments that eventually led me to the final architecture -- VQ-MAE (transformer_vq_vae folder)

**VQ-MAE**  
Vector-Quantized Masked Auto Encoder is an architecture that successfully bootstraps images from its own latent space and reuses them for further training.
Architecture, experiments and results (SOTA in Continous Representation Learning) are described in my masters thesis:
[Deep Learning Methods for Continual
Representation Learning](https://drive.google.com/file/d/1gz64N2WN5eoSBkObWaYj1AYzmuAAKg_8/view?usp=sharing)
