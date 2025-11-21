<p align="center">
  <h1 align="center">Monet: Reasoning in Latent Visual Space Beyond Images and Language</h1>
  <p align="center">
  </p>
  <p align="center">
    <a href="https://novaglow646.github.io/">Qixun Wang</a>,
    <a href="https://frankyang-17.github.io/">Yang Shi</a>,
    <a href="https://yifeiwang77.com/">Yifei Wang</a>,
    <a href="https://scholar.google.com/citations?user=COdftTMAAAAJ&hl=en">Yuanxing Zhang</a>,
    <a href="https://magicwpf.github.io/">Pengfei Wan</a>,
    <a href="https://scholar.google.com/citations?user=PXO4ygEAAAAJ&hl=zh-CN">Kun Gai</a>,
    <a href="https://scholar.google.com/citations?user=27o9L1wAAAAJ&hl=en">Xianghua Ying</a>,
    <a href="https://yisenwang.github.io/">Yisen Wang</a>,
  </p>
  <p align="center">
    <a href="https://www.arxiv.org/abs/2506.17218">
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red' alt='Paper PDF'>
    </a>
    <a href="https://huggingface.co/NOVAglow646/Monet-7B" target="_blank" rel="noopener noreferrer">
      <img src="https://img.shields.io/badge/Model-HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=orange" alt="HF Model">
    </a>
    <a href="https://huggingface.co/datasets/NOVAglow646/Monet-SFT-125K" target="_blank" rel="noopener noreferrer">
      <img src="https://img.shields.io/badge/Dataset-HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black" alt="HF Dataset">
    </a>
  </p>
</p>


[![arXiv](https://img.shields.io/badge/arXiv-PyVision-b31b1b)](...)
[![Model](https://img.shields.io/badge/Demo-Monet7B-yellow)](https://huggingface.co/NOVAglow646/Monet-7B)


<p align="center">
    <img src="images/overview.png" alt="Logo" width="190%">
</p>

<br>

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Tabel of Contents</summary>
  <ol>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#data-preparation">Data Preparation</a>
    </li>
    <li>
      <a href="#training">Training</a>
    </li>
    <!-- <li>
      <a href="#inference">Inference</a>
    </li> -->
    <li>
      <a href="#citation">Citation</a>
    </li>
    <li>
      <a href="#acknowledgement">Acknowledgement</a>
    </li>
  </ol>
</details>

## Installation
Create a conda environment and install the required packages:
```bash
conda create -n monet python=3.10
conda activate monet

git clone https://github.com/NOVAglow646/Monet.git
cd Monet
pip install -r requirements.txt
```

## Data Preparation
We provide a sample dataset of 100 examples for the VSP spatial reasoning task. Please format your data file as follows:

```json
{
    "text_input": "Question",
    "text_output": "Answer",
    "image_input": ["input1.jpg"],
    "image_output": "helper_image.jpg"
}
```

## Training
We train our model in two stages:
- Stage 1 jointly supervises text and latent visual tokens, grounding the latter in the visual subspace.
- Stage 2 drops the latent supervision, anchoring the grounded latent tokens for subsequent text generation.

<p align="center">
    <img src="asset/pipeline.png" alt="Logo" width="190%">
</p>

Run the following commands to reproduce the training. Make sure to configure the `data_path` and `model_path` as needed.

**Training Stage 1**
```bash
python src/main.py \
    --model Qwen/Qwen2.5-VL-7B-Instruct --epochs 10 \
    --task vsp-spatial-reasoning \
    --latent_size 4 \
    --stage stage1 \
    --data_path ./data/sample.jsonl \
    --log_file ./log.txt \
    --save_model_path ./checkpoints/model_stage1 
```

**Training Stage 2**
```bash
python src/main.py \
    --model Qwen/Qwen2.5-VL-7B-Instruct --epochs 10 \
    --task vsp-spatial-reasoning \
    --latent_size 4 \
    --stage stage2 \
    --data_path ./data/sample.jsonl \
    --log_file ./log.txt \
    --load_model_path ./checkpoints/model_stage1
    --save_model_path ./checkpoints/model_stage2 
```

<!-- ## Inference -->


## Citation
If you find our work useful, please consider citing: 

```bibtex
@article{yang2025machine,
  title={Machine Mental Imagery: Empower Multimodal Reasoning with Latent Visual Tokens}, 
  author={Zeyuan Yang and Xueyang Yu and Delin Chen and Maohao Shen and Chuang Gan},
  year={2025},
  eprint={2506.17218},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2506.17218}, 
}
```

## Acknowledgement
We would like to thank the following works for their code and models:
- Training: [Coconut](https://arxiv.org/abs/2412.06769), [Qwen](https://huggingface.co/Qwen) and [MVoT](https://arxiv.org/pdf/2501.07542)
- Datasets: [VSP](https://arxiv.org/abs/2407.01863), [Blink](https://zeyofu.github.io/blink/), [COMT](https://arxiv.org/abs/2412.12932) and [SAT](https://arxiv.org/abs/2412.07755)

We are extremely grateful to Haoyu Zhen, Bairu Hou, Guangtao Zeng, Yuncong Yang,
Jiaben Chen, Ziwei Liu, Zonghan Yang, Sunli Chen, Lixing Fang, and many other friends in our [Embodied AGI Lab](https://embodied-agi.cs.umass.edu/)
for their helpful feedback and insightful discussions.
