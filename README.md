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

## 1. Overview

**RadVLM** is a research project focused on enhancing radiology report generation using Vision-Language Models (VLMs). It integrates reasoning and knowledge graph retrieval to improve accuracy and contextual understanding. The repository provides tools for dataset preprocessing, model training, and evaluation, along with pre-trained models and benchmarks.

## 2. Installation

```bash
git clone https://github.com/MouYongli/RadVLM.git
cd RadVLM
```

### Anaconda

Because the DeepSeek-VL2 and Qwen2.5-VL require different versions of dependencies, we need to install them in separate conda environments.

```bash
export PROJECT_ROOT=$(pwd)
```

1. DeepSeek-VL2
```bash
cd $PROJECT_ROOT/baselines
mkdir deepseek
# Clone the DeepSeek-VL2 repository
git clone https://github.com/deepseek-ai/DeepSeek-VL2.git
mv DeepSeek-VL2/* deepseek
rm -rf DeepSeek-VL2
# Update requirements.txt in DeepSeek-VL2 folder
cp requirements.deepseek.txt deepseek/requirements.txt
#  Update pyproject.toml in DeepSeek-VL2 folder
cp pyproject.deepseek.toml deepseek/pyproject.toml
# Install dependencies and install the deepseek-vl2 package
cd deepseek
# Create a new conda environment for DeepSeek-VL2
conda create --name deepseekenv python=3.10
conda activate deepseekenv
pip install -r requirements.txt
pip install -e .
# Install PyTorch with CUDA 12.6
# For other version, please refer to https://pytorch.org/get-started/locally, for example:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install torch torchvision torchaudio 
```

2. Qwen2.5-VL

```bash
cd $PROJECT_ROOT/baselines
mkdir qwen
conda create --name qwenenv python=3.10
conda activate qwenenv
cp requirements.qwen.txt qwen/requirements.txt
cd qwen
pip install -r requirements.txt
```

3. Our project and dependencies
```bash
cd $PROJECT_ROOT
conda activate deepseekenv
pip install -e .
conda activate qwenenv
pip install -e .
```

## Datasets

### Download datasets

- MIMIC-CXR: https://physionet.org/content/mimic-cxr/2.0.0/
- ChestExpert







## Usage

Here's an example of how to use the model:

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
