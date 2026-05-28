<div align="center">

# SHARP: Short-Window Streaming for Accurate and Robust Prediction in Motion Forecasting

[![arXiv](https://img.shields.io/badge/arXiv-2603.16550-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2603.28091)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg?style=flat-square&logo=googlechrome&logoColor=white)](https://a-pru.github.io/sharp/)
[![Poster](https://img.shields.io/badge/CVPR%202026-Poster-brightgreen.svg?style=flat-square&logo=adobeacrobatreader&logoColor=white)](docs/static/pdfs/prutsch_sharp_cvpr2026_poster.pdf)

</div>

> [**SHARP: Short-Window Streaming for Accurate and Robust Prediction in Motion Forecasting**](https://arxiv.org/abs/2603.28091)  
> Alexander Prutsch, Christian Fruhwirth-Reisinger, David Schinagl, Horst Possegger  
> **Graz University of Technology**   
> **CVPR 2026**

**This repository provides full data preprocessing, training and inference support for the nuScenes, Argoverse 1 (AV1), and Argoverse 2 (AV2) datasets.**  
**It also includes pretrained checkpoints for AV2 single- and multi-agent settings, and visualization tools for AV2.**

## Getting Started

### Create and Activate Virtual Environment
```
conda create -n sharp python=3.11
conda activate sharp
```

### Install PyTorch
We tested our implementation with torch 2.8.0 and CUDA 12.8

Install PyTorch e.g.
```
pip install torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu128
```

### Install Dependencies
```
pip install -r ./requirements.txt
```

## Dataset Setup
### Download the [Argoverse 2 Motion Forecasting Dataset](https://argoverse.github.io/user-guide/datasets/motion_forecasting.html#download)
The expected structure of the AV2 data should be:
```
data_root
    ├── train
    │   ├── 0000b0f9-99f9-4a1f-a231-5be9e4c523f7
    │   ├── 0000b6ab-e100-4f6b-aee8-b520b57c0530
    │   ├── ...
    ├── val
    │   ├── 00010486-9a07-48ae-b493-cf4545855937
    │   ├── 00062a32-8d6d-4449-9948-6fedac67bfcd
    │   ├── ...
    ├── test
    │   ├── 0000b329-f890-4c2b-93f2-7e2413d4ca5b
    │   ├── 0008c251-e9b0-4708-b762-b15cb6effc27
    │   ├── ...
```

### Download the [Argoverse 1 Motion Forecasting Dataset](https://www.argoverse.org/av1.html#forecasting-link)

### nuScenes
Please follow the [official guidelines.](https://www.nuscenes.org/nuscenes#download)

### Data Preprocessing
Preprocess the **Argoverse 2** dataset by executing
```
python preprocess_av.py --data_root=/path/to/av2_data_root -p
```

Preprocess the **Argoverse 1** dataset by executing
```
python preprocess_av.py --data_root=/path/to/av1_data_root -p --av1
```

For nuScenes please run the standalone [`nus_extractor.py`](https://github.com/a-pru/sharp/blob/main/src/datamodules/nus_extractor.py) script in the `src/datamodules` folder.

## Training on Single-Agent Benchmarks
Train SHARP model on single-agent data using
```
python train.py datamodule.pl_module.data_root=/path/to/data_root/sharp_processed/
```
Select the data root for the corresponding dataset.  

## Evaluation on Single-Agent Benchmarks
Evaluate SHARP model using
```
python eval.py datamodule.pl_module.data_root=/path/to/data_root/sharp_processed/ checkpoint=/path/to/checkpoint.ckpt
```
Select the data root for the corresponding dataset. 

**AV2 Single-Agent Checkpoint provided in the repository.**
```
python eval.py datamodule.pl_module.data_root=/path/to/av2_data_root/sharp_processed/ checkpoint=exps/av2_single_agent/checkpoints/av2_sa.ckpt
```

Expected results:
| MR | minADE1 | minADE6 | minFDE1 | minFDE6 | b-minFDE6 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0.140 | 1.569 | 0.639 | 3.85 | 1.197 | 1.822 |

## AV2 Single-Agent Visualization
Visualize the prediction results using
```
python visualize_av2_sa.py
```

Please update the data_root, chkpt_dir, and av2_raw_data_dir variable in the script.

## Training and Evaluation on AV2 Multi-Agent Benchmark
1. Update config_name to "config_ma" in [`train.py`](https://github.com/a-pru/sharp/blob/main/train.py)
2. Initialize single-agent model with your checkpoint (see TODO in [`sharp.py`](https://github.com/a-pru/sharp/blob/main/train.py))
3. Train SHARP model with multi-agent consistency module using
```
python train.py datamodule.pl_module.data_root=/path/to/av2_data_root/sharp_processed/
```

## Evaluation on AV2 Multi-Agent Benchmark
1. Update config_name to "config_ma" in [`eval.py`](https://github.com/a-pru/sharp/blob/main/eval.py)
2. Evaluate SHARP model using
```
python eval.py datamodule.pl_module.data_root=/path/to/av2_data_root/sharp_processed/ checkpoint=/path/to/checkpoint.ckpt
```
**AV2 Multi-Agent Checkpoint provided in the repository.**
```
python eval.py datamodule.pl_module.data_root=/path/to/av2_data_root/sharp_processed/ checkpoint=exps/av2_multi_agent/checkpoints/av2_ma.ckpt
```
Expected results:
| AvgMinADE | AvgMinFDE | AvgBrierMinFDE |
| :--- | :--- | :--- |
| 0.55 | 1.14 | 1.78 |

This checkpoint differs from our final challenge submission which is trained on both train and validation set.


## Bibtex
```bibtex
@inproceedings{prutsch2026sharp,
    title={{SHARP: Short-Window Streaming for Accurate and Robust Prediction in Motion Forecasting}},
    author={Prutsch, Alexander and Fruhwirth-Reisinger, Christian and Schinagl, David and Possegger, Horst},
    booktitle={In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2026}
}
```

## Acknowledgements
This repository is based on [RealMotion](https://github.com/fudan-zvg/RealMotion/) and integrates code from [Forecast-MAE](https://github.com/jchengai/forecast-mae), [DeMo](https://github.com/fudan-zvg/DeMo), and [EMP](https://github.com/a-pru/emp). We thank them for their work!
