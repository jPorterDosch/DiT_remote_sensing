### Overview
This project adapts diffusion transformer (DiT) models for downstream remote sensing tasks by adding a latent reconstruction objective. 

The original code which we fork off of is from the paper "Unleashing Diffusion Transformers for Visual Correspondence by Modulating Massive Activations
", published in NeurIPS 2025. Their Git repository is located at the following link: https://github.com/ganchaofan0000/DiTF/tree/main.

We take their findings on the specific properties required to extract features from diffusion models, and extend this to fine-tune DiTs in order to extract efficient representations of remote sensing images in a label-scarce environment.

### Project Setup
#### Download conda (if not already installed)
```
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
bash miniconda.sh
source ~/.bashrc
```
#### Navigate back to copied directory
```
conda env create -f environment.yml
conda activate DiTF
pip install -e ".[all]"
```
