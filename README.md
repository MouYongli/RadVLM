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

### Anaconda

#### DeepSeek-VL2
1. Clone DeepSeek-VL2 repository
```bash
git clone https://github.com/deepseek-ai/DeepSeek-VL2.git
```

2. Update requirements.txt in DeepSeek-VL2 folder
```text
transformers==4.38.2
xformers==0.0.29.post3
timm==1.0.15
accelerate==1.4.0
sentencepiece==0.2.0
attrdict==2.0.1
einops==0.8.1

# for gradio demo
#gradio==3.48.0
#gradio-client==0.6.1
#mdtex2html==1.3.0
#pypinyin==0.50.0
#tiktoken==0.5.2
#tqdm==4.64.0
#colorama==0.4.5
#Pygments==2.12.0
#markdown==3.4.1
#SentencePiece==0.1.96
```

3. Update pyproject.toml in DeepSeek-VL2 folder
```TOML
[build-system]
requires = ["setuptools>=40.6.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "deepseek_vl2"
version = "1.0.0"
description = "DeepSeek-VL2"
authors = [{name = "DeepSeek-AI"}]
license = {file = "LICENSE-CODE"}
urls = {homepage = "https://github.com/deepseek-ai/DeepSeek-VL2"}
readme = "README.md"
requires-python = ">=3.8"
dependencies = [
    "transformers==4.38.2",
    "xformers== 0.0.29.post3",
    "timm==1.0.15",
    "accelerate==1.4.0",
    "sentencepiece==0.2.0",
    "attrdict==2.0.1",
    "einops==0.8.1",
]

[tool.setuptools]
packages = {find = {exclude = ["images"]}}
```

4. Create conda environment
```bash
conda create --name deepseekenv python=3.10
conda activate deepseekenv
cd DeepSeek-VL2
```

5. Install dependencies
```bash          
pip install torch torchvision torchaudio
pip install -r requirements.txt
pip install -e .
```

#### Qwen2.5-VL

1. Create conda environment
```bash
conda create --name qwenenv python
conda activate qwenenv
```

5. Install dependencies
```bash          
pip install transformers==4.51.3 accelerate
pip install qwen-vl-utils[decord]
```

### Docker

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
📦 RadVLM
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
