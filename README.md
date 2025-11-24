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
      <img alt="HF Model: ViGaL" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Model-Monet7B-ffc107?color=ffc107&logoColor=white" height="20" />
    </a>
    <a href="https://huggingface.co/datasets/NOVAglow646/Monet-SFT-125K" target="_blank">
    <img alt="HF Model: ViGaL" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Model-MonetSFT125K-ffc107?color=ffc107&logoColor=white" height="20" />
    </a>

  </p>
</p>


[![arXiv](https://img.shields.io/badge/arXiv-PyVision-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org)
[![Model](https://img.shields.io/badge/Demo-Monet7B-yellow?style=flat&logo=https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/huggingface_logo.svg)](https://huggingface.co/NOVAglow646/Monet-7B)


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



## Training
The training requires a modification of the official code of Qwen2.5-VL-7B, which is implemented in `new/avt_qwen_model/modeling_qwen2_5_vl_avt.py`. The main implementation of the forward process with latent embeddings is in `Qwen2_5_VLModel:forward` and `Qwen2_5_VLForConditionalGeneration:forward`.


## Inference
The inference requires replacing the official code of vLLM.


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

