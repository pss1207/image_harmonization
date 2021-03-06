# Image Harmonization

PyTorch implementation for image harmonization. 
![alt text](https://github.com/pss1207/image_harmonization/blob/master/test_results.png)

## Prerequisites
- Pytorch
- [MS COCO Dataset](http://cocodataset.org/#home)

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

### Harmonization Dataset
- To train the model, generate a harmonization dataset with MS COCO
- The harmonization dataset will be randomly generated by converting brightness, contrast, and gamma value of masked region.
```bash
cd data_gen
python data_gen.py --image_dir coco_dataset_path --save_dir directory_for_saving_images 
```

### Train
- Train a model:
```bash
python train.py --dataroot ./datasets 
```
- To view training results, run `python -m visdom.server` and click the URL http://localhost:8097. To see more intermediate results, check out `./checkpoints/web/index.html`

### Test
- Test the model:
```bash
python test.py --dataroot ./datasets
```
The test results will be saved to a html file here: `./results/test_latest/index.html`.



