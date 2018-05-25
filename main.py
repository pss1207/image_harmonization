import os
from options.test_options import TestOptions
from models import create_model
from PIL import Image
import numpy as np
import torch
from util.util import tensor2im, save_image
from scipy.misc import imresize

def save_images(img, aspect_ratio=1.0):
    im = tensor2im(img)
    image_name = 'output.png'
    save_path = os.path.join('./', image_name)
    h, w, _ = im.shape
    if aspect_ratio > 1.0:
        im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
    if aspect_ratio < 1.0:
        im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')
    save_image(im, save_path)


def image_harmonization_eval(input_file_path, mask_file_path):
    opt = TestOptions().parse()
    opt.nThreads = 1  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.display_id = -1  # no visdom display

    with open(input_file_path, 'rb') as f:
        with Image.open(f) as input_img:
            input_img = input_img.convert('RGB')

    with open(mask_file_path, 'rb') as f:
        with Image.open(f) as mask_img:
            mask_img = mask_img.convert('L')

    input_image = np.array(input_img, np.float32) / 255.0
    input_image = input_image.transpose((2, 0, 1))

    input_mask = np.expand_dims(np.array(mask_img, np.float32) / 255.0, axis=0)

    input_image = torch.from_numpy(input_image)
    input_mask = torch.from_numpy(input_mask)

    mean = torch.tensor(0.5).view(-1, 1, 1)
    std = torch.tensor(0.5).view(-1, 1, 1)

    input_image = (input_image - mean) / std

    input_image = torch.unsqueeze(input_image,0)
    input_mask = torch.unsqueeze(input_mask, 0)

    model = create_model(opt)
    model.set_input_eval([input_image, input_mask])
    pred = model.eval()
    save_images(pred, aspect_ratio=1.0)

    return pred


if __name__ == '__main__':
    input_file_path = '/media/hdd/data/harmonization/canvas_arr.jpg'
    mask_file_path = '/media/hdd/data/harmonization/fore_alpha.png'
    image_harmonization_eval(input_file_path, mask_file_path)



