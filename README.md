# SHARP: Short-Window Streaming for Accurate and Robust Prediction in Motion Forecasting

### Paper
> **SHARP: Short-Window Streaming for Accurate and Robust Prediction in Motion Forecasting**
> Alexander Prutsch, Christian Fruhwirth-Reisinger, David Schinagl, Horst Possegger
> **Graz University of Technology**  
> **CVPR 2026**

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
### Download Exampe Data [Argoverse 2 Motion Forecasting Dataset](https://argoverse.github.io/user-guide/datasets/motion_forecasting.html#download)
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

### Other Datasets
For AV1 and nuScenes please follow the official guidelines.

### Data Preprocessing
Preprocess the Argoverse 2 dataset by executing
```
python preprocess.py --data_root=/path/to/data_root -p
```

For AV1, please switch the extractor import in `preprocess.py`. For nuScenes we provide a standalone extraction script in the `src/datamodules` folder.

## Code and model weights coming soon!

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
