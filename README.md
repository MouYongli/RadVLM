# RadVLM: Vision Language Models for Radiology Report Generation - A Reasoning and Knowledge Graph Retrieval Augmented Generation Approach 

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-green)](https://developer.nvidia.com/cuda-downloads)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.0-red)](https://pytorch.org/get-started/locally/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[![Forks](https://img.shields.io/github/forks/MouYongli/RadVLM?style=social)](https://github.com/MouYongli/RadVLM/network/members)
[![Stars](https://img.shields.io/github/stars/MouYongli/RadVLM?style=social)](https://github.com/MouYongli/RadVLM/stargazers)
[![Issues](https://img.shields.io/github/issues/MouYongli/RadVLM)](https://github.com/MouYongli/RadVLM/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr/MouYongli/RadVLM)](https://github.com/MouYongli/RadVLM/pulls)
[![Contributors](https://img.shields.io/github/contributors/MouYongli/RadVLM)](https://github.com/MouYongli/RadVLM/graphs/contributors)
[![Last Commit](https://img.shields.io/github/last-commit/MouYongli/RadVLM)](https://github.com/MouYongli/RadVLM/commits/main)
<!-- [![Build Status](https://img.shields.io/github/actions/workflow/status/MouYongli/RadVLM/ci.yml)](https://github.com/MouYongli/RadVLM/actions)
[![Code Quality](https://img.shields.io/lgtm/grade/python/g/MouYongli/RadVLM.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/MouYongli/RadVLM/context:python) -->

[![Docker](https://img.shields.io/badge/Docker-Supported-blue)](https://hub.docker.com/r/YOUR_DOCKER_IMAGE)
[![Colab](https://img.shields.io/badge/Open%20in-Colab-yellow)](https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME/blob/main/notebooks/demo.ipynb)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)


[![WeChat](https://img.shields.io/badge/WeChat-公众号名称-green)](https://your-wechat-link.com)
[![Weibo](https://img.shields.io/badge/Weibo-关注-red)](https://weibo.com/YOUR_WEIBO_LINK)
<!-- [![Discord](https://img.shields.io/discord/YOUR_DISCORD_SERVER_ID?label=Discord&logo=discord&color=5865F2)](https://discord.gg/YOUR_INVITE_LINK) -->
<!-- [![Twitter](https://img.shields.io/twitter/follow/YOUR_TWITTER_HANDLE?style=social)](https://twitter.com/YOUR_TWITTER_HANDLE) -->



This is official repo for "RadVLM: Vision Language Models for Radiology Report Generation" by DBIS group at RWTH Aachen University
([Yongli Mou*](mou@dbis.rwth-aachen.de), Antonia Gustke and Stefan Decker)

## Overview

**RadVLM** is a research project focused on enhancing radiology report generation using Vision-Language Models (VLMs). It integrates reasoning and knowledge graph retrieval to improve accuracy and contextual understanding. The repository provides tools for dataset preprocessing, model training, and evaluation, along with pre-trained models and benchmarks.

## Installation

#### Anaconda
1. create conda environment
```bash
conda create --name dugle python=3.11
conda activate dugle
```

2. Install Jupyter lab and kernel
```bash
conda install -c conda-forge jupyterlab
conda install ipykernel
```

3. Install dependencies
```bash          
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 # torch==2.6.0+cu126, torchvision==0.21.0+cu126, torchaudio==2.6.0+cu126
pip install torch_geometric # torch_geometric==2.6.1torch_spline_conv-1.2.2+pt25cu124
pip install -e .
```
#### Docker

## Datasets

### Download datasets


## Usage

Here’s an example of how to use the model:

```python
from radvlm.models.modeling_radvlm import RadVLM
model = RadVLM.load_pretrained("base")
```

## Project Structure
```
📦 Dugle
├── 📁 data         # Sample datasets and preprocessing scripts
├── 📁 models           # Pre-trained models and checkpoints
├── 📁 notebooks        # Jupyter notebooks with tutorials
├── 📁 docs             # Documentation and API references
├── 📁 experiments      # Experimental configurations, logs and results
├── 📁 src              # Core implementation of foundation models
└── README.md           # Project description
```

## Benchmark Results

| Model        | accuracy |
|--------------|-------:|
| Baseline | xx     |  
| Ours | xx     | 
More benchmarks are available in the [research paper](https://your-project-website.com/paper).


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


## Citation

If you use this project in your research, please cite:
```bibtex
@article{mou2025radvlm,
  author  = {Yongli Mou, Antonia Gustke and Stefan Decker},
  title   = {XXX},
  journal = {XXX},
  year    = {202X}
}
```

---
<!-- ---Developed by **Your Name** | [LinkedIn](https://linkedin.com/in/YOURNAME) | [Twitter](https://twitter.com/YOURHANDLE) -->
