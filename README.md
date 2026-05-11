# RADAR
RADAR: Redundancy-Aware Diffusion for Multi-Agent Communication Structure Generation (ICML 2026)

![](https://github.com/cszhangzhen/RADAR/blob/main/fig/model.png)

This is a PyTorch implementation of the RADAR algorithm, which is a redundancy-aware and query-adaptive generative framework that actively reduce communication overhead. Motivated by recent progress in conditional discrete graph diffusion models, we formulate communication topology design as a step-by-step generation process, guided by the effective size of the graph. Comprehensive experiments on six benchmarks demonstrate that RADAR consistently outperforms recent baselines, achieving higher accuracy, lower token consumption, and greater robustness across diverse scenarios.

### Requirements
* python3.10
* pytorch==2.10.0
* torch-geometric==2.7.0
* numpy==2.2.6
* scipy==1.15.3
* openai==2.30.0

### Project Structure

```
RADAR
├── data
│   └── AQuA
├── mas
│   ├── agents
│   ├── datasets
│   ├── domain
│   ├── gnn
│   ├── graph
│   ├── __init__.py
│   ├── llm
│   ├── prompt
│   ├── tools
│   └── utils
├── model
│   ├── denoising.py
│   ├── gd.py
│   ├── ordering.py
│   └── utils.py
├── accuracy.py
├── process_datasets.py
├── run_aqua.py
├── run_gsm8k.py
├── run_humaneval.py
├── run_mmlu.py
├── run_multiarith.py
├── run_svamp.py
├── template.env
└── utils.py
```

### Add API keys in template.env and change its name to .env
```
BASE_URL = "" # your base url
API_KEY = "" # your api key
```

### Download Datasets
Download MMLU, HumanEval and GSM8K ect. And put them in different folders.

### Run on MMLU dataset
```
python run_mmlu.py

python run_gsm8k.py
```
### Citing
If you find RADAR useful for your research, please consider citing the following paper:
```
@inproceedings{zhang2026radar,
title={{RADAR}: Redundancy-Aware Diffusion for Multi-Agent Communication Structure Generation},
author={Anonymous},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=GtpiqFaJtZ}
}
```

### Acknowledgments
This code refers to GPTSwarm, GDesigner, GraphARM, etc.