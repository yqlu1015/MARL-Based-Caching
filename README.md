# Caching for Edge Inference at Scale: A Mean Field Multi-Agent Reinforcement Learning Approach

Code for our IEEE GLOBECOM 2023 paper ["Caching for Edge Inference at Scale: A Mean Field Multi-Agent Reinforcement Learning Approach"](https://ieeexplore.ieee.org/abstract/document/10436965?casa_token=H7SPXO1A35YAAAAA:PRYn6KzxJwGwLCPm9DI3gl__thF0I0A7Fu2P4Uk-cGLRCN_nN6GMOyxyNSh9-rAyTihe4CzAojE).
In this work, we simulated an edge intelligence system with mean field MARL based caching.

## Preparation

### Clone

```bash
git clone https://github.com/yqlu1015/MARL-Based-Caching.git
```

### Create a conda environment [Optional]:

```bash
conda create -n simulation_env
conda activate simulation_env
pip install -r requiresments.txt
```

## Usage

### Training and Evaluation:

```bash
python run_comp_2.py
```

### Plot figures:
```bash
python run_comparison.py
```

## Citation

If this work is helpful, please cite as:

```bibtex
@inproceedings{lu2023caching,
  title={Caching for Edge Inference at Scale: A Mean Field Multi-Agent Reinforcement Learning Approach},
  author={Lu, Yanqing and Zhang, Meng and Tang, Ming},
  booktitle={GLOBECOM 2023-2023 IEEE Global Communications Conference},
  pages={332--337},
  year={2023},
  organization={IEEE}
}
```
