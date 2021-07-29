# Multi-level Landmark-Guided Deep Network for Face Super-resolution

## Dependencies
- Python 3 (Recommend to use Anaconda)
- PyTorch >= 1.1.0
- NVIDIA GPU + CUDA
- Python packages: pip install numpy opencv-python tqdm imageio pandas matplotlib tensorboardX
## Dataset Preparation
- Download datasets,CelebA dataset can be downloaded here. Please download and unzip the img_celeba.7z file.
- Helen dataset can be downloaded here. Please download and unzip the 5 parts of All images.
- The test files can be downloaded here.
## Training

```
cd code
python train.py -opt options/train/train_(MLG|MLGGAN)_(CelebA|Helen).json
```
- The json file will be processed by options/options.py.
- Before running this code, please modify option files to your own configurations.
## Testing
```
cd code
python test.py -opt options/test/train_(MLG|MLGGAN)_(CelebA|Helen).json
```

## Acknowledgements
- Thank Cheng Ma. Our code structure is derived from his repository [DIC/DICGAN](https://github.com/Maclory/Deep-Iterative-Collaboration).
- Thank Zhen Li.  [SRFBN](https://github.com/Paper99/SRFBN_CVPR19).He provide many useful codes which facilitate our work.
