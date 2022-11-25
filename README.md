<h1> Variational Auto-Encoders </h1>

This is my implementation of Variational Auto-Encoders for [CSE-291B - Deep Generative Models](https://sites.google.com/view/cse291). My implementation builds on top of a starter code, and includes the following - 
1. Reparameterization trick for computing gradients.
1. Negative ELBO bound for computing the training loss.
1. Vanilla VAE - using a unit-mean isotropic gaussian prior. The variational posterior is a gaussian.
1. GMVAE - VAE using a gaussian mixture prior. The variational posterior is a gaussian.
1. Using a gaussian mixture variational posterior.

The models can take a while to run on CPU, so please prepare accordingly. On a
2018 Macbook Pro, it takes ~5 minutes each to run `vae.py` and `gmvae.py`.

1. `sample_gaussian` in `utils.py`
1. `negative_elbo_bound` in `vae.py`
1. `log_normal` in `utils.py`
1. `log_normal_mixture` in `utils.py`
1. `negative_elbo_bound` in `gmvae.py`

### Dependencies

This code was built and tested using the following libraries

```
tqdm==4.47.0
numpy==1.18.5
torchvision==0.8.1
torch==1.7.0
```
