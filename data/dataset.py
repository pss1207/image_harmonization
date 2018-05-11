from data.base_dataset import BaseDataset, get_transform
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import os.path

INPUT_IMAGE_EXTENSIONS = ['_i.jpg']
MASK_EXTENSIONS = ['_m.jpg']
TARGET_IMAGE_EXTENSIONS = ['_t.jpg']
SEMANTIC_EXTENSIONS = ['_s.jpg']


def is_input_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in INPUT_IMAGE_EXTENSIONS)

def is_mask_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in MASK_EXTENSIONS)

def is_target_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in TARGET_IMAGE_EXTENSIONS)

def is_semantic_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in SEMANTIC_EXTENSIONS)

def make_dataset(dir):

    input_image = []
    input_mask = []
    target_image = []


    dir = os.path.expanduser(dir)
    input_dir_dir = dir + '/input'
    for root, _, fnames in sorted(os.walk(input_dir_dir)):
        count = 0
        for fname in sorted(fnames):
            if is_input_image_file(fname):
                path = os.path.join(root, fname)
                input_image.append(path)

        for fname in sorted(fnames):
            if is_mask_file(fname):
                path = os.path.join(root, fname)
                input_mask.append(path)

    target_dir_dir = dir + '/target'
    for root, _, fnames in sorted(os.walk(target_dir_dir)):
        for fname in sorted(fnames):
            if is_target_image_file(fname):
                path = os.path.join(root, fname)
                target_image.append(path)


    return input_image, input_mask, target_image

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
class ImageData(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_data = os.path.join(opt.dataroot, opt.phase)
        input_image, input_mask, target_image = make_dataset(self.dir_data)

        self.input_image = input_image
        self.input_mask = input_mask
        self.target_image = target_image
        self.transform = get_transform(opt)
        self.loader = default_loader

    def __getitem__(self, index):
        input_image_path = self.input_image[index]
        input_mask_path = self.input_mask[index]
        target_image_path = self.target_image[index]

        input_image = self.loader(input_image_path)
        input_mask = self.loader(input_mask_path)
        input_mask = input_mask.convert('L')
        target_image = self.loader(target_image_path)


        input_image, input_mask, target_image = self.transform([input_image, input_mask, target_image])

        return input_image, input_mask, target_image

    def __len__(self):
        return len(self.input_image)

    def name(self):
        return 'Dataset'