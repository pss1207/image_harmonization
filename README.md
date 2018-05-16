# Image Harmonization

PyTorch implementation for image harmonization. 

## Prerequisites
- Pytorch


## Getting Started
### Installation
- Install PyTorch and dependencies from http://pytorch.org
- Install Torch vision from the source.
```bash
git clone https://github.com/pytorch/vision
cd vision
python setup.py install
```
- Install python libraries [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate).
```bash
pip install visdom
pip install dominate
```


### Train
- Train a model:
```bash
python train.py --dataroot ./datasets 
```
- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097. To see more intermediate results, check out `./checkpoints/web/index.html`

### Test
- Test the model:
```bash
#!./scripts/test_cyclegan.sh
python test.py --dataroot ./datasets
```
The test results will be saved to a html file here: `./results/test_latest/index.html`.


