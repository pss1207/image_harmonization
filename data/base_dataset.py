import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        input_image = sample[0]
        input_mask = sample[1]
        target_image = sample[2]

        h, w = input_image.size
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        input_image_re = input_image.resize((new_h, new_w), Image.BICUBIC)
        input_mask_re = input_mask.resize((new_h, new_w), Image.NEAREST)
        target_image_re = target_image.resize((new_h, new_w), Image.BICUBIC)

        return [input_image_re, input_mask_re, target_image_re]


class RandomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        input_image = sample[0]
        input_mask = sample[1]
        target_image = sample[2]

        h, w = input_image.size
        new_h, new_w = self.output_size

        if h == new_h:
            top = 0
        else:
            top = np.random.randint(0, h - new_h)
        if w == new_w:
            left = 0
        else:
            left = np.random.randint(0, w - new_w)

        input_image = input_image.crop((top, left, top + new_h, left + new_w))
        input_mask = input_mask.crop((top, left, top + new_h, left + new_w))
        target_image = target_image.crop((top, left, top + new_h, left + new_w))

        return [input_image, input_mask, target_image]


class ToTensor(object):

    def __call__(self, sample):
        input_image = np.array(sample[0], np.float32) / 255.0
        input_mask = np.expand_dims(np.array(sample[1], np.float32) / 255.0, axis=0)
        target_image = np.array(sample[2], np.float32) / 255.0
        #image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        input_image = input_image.transpose((2, 0, 1))
        #input_mask = input_mask.transpose((2, 0, 1))
        target_image = target_image.transpose((2, 0, 1))
        return [torch.from_numpy(input_image), torch.from_numpy(input_mask), torch.from_numpy(target_image)]

class Normalization(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, sample):
        input_image = sample[0]
        input_mask = sample[1]
        target_image = sample[2]

        input_image = (input_image - self.mean) / self.std
        input_mask = input_mask
        target_image = (target_image - self.mean) / self.std

        return [input_image, input_mask, target_image]

def get_transform(opt):
    rescale = Rescale(opt.loadSize)
    crop = RandomCrop(opt.fineSize)
    to_tensor = ToTensor()
    norm = Normalization(0.5, 0.5)

    return transforms.Compose([rescale, crop, to_tensor, norm])

