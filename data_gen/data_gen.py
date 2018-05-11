import argparse
import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import utils
import scipy.misc


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

class CocoDataset(utils.Dataset):
    def load_coco(self, dataset_dir, subset, class_ids=None,
                  class_map=None, return_coco=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        """
        coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, '2017'))

        image_dir = "{}/{}/{}{}".format(dataset_dir, 'images', subset, '2017')

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco


    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_mask(image_id)

    def load_semantic(self, image_id, semantic_dir):
        file_name = os.path.splitext(os.path.basename(self.image_info[image_id]['path']))[0] + '.png'
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        semantic = Image.open(semantic_dir + '/' +file_name, 'r')

        return np.array(semantic)



    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(CocoDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

def data_gen(image, mask_array, semantic, display_on):
    # Convert Image
    # Brightness
    MIN_ALPHA = 0.9
    MAX_ALPHA = 1.1
    MIN_BETA = -5
    MAX_BETA = 5
    MIN_GAMMA = 0.4
    MAX_GAMMA = 2.5

    MIN_ALPHA = np.int(MIN_ALPHA * 100)
    MAX_ALPHA = np.int(MAX_ALPHA * 100)

    MIN_GAMMA = np.int(MIN_GAMMA * 100)
    MAX_GAMMA = np.int(MAX_GAMMA * 100)

    img_linear = np.zeros_like(image)

    # Linear Transformation - Contrast and Brightness
    alpha_r = np.random.randint(MIN_ALPHA, MAX_ALPHA + 1) * 0.01
    alpha_g = np.random.randint(MIN_ALPHA, MAX_ALPHA + 1) * 0.01
    alpha_b = np.random.randint(MIN_ALPHA, MAX_ALPHA + 1) * 0.01
    beta_r = np.random.randint(MIN_BETA, MAX_BETA + 1)
    beta_g = np.random.randint(MIN_BETA, MAX_BETA + 1)
    beta_b = np.random.randint(MIN_BETA, MAX_BETA + 1)

    img_linear[:, :, 0] = cv2.convertScaleAbs(image[:, :, 0], alpha=alpha_b, beta=beta_b)
    img_linear[:, :, 1] = cv2.convertScaleAbs(image[:, :, 1], alpha=alpha_g, beta=beta_g)
    img_linear[:, :, 2] = cv2.convertScaleAbs(image[:, :, 2], alpha=alpha_r, beta=beta_r)

    # Gamma Correction
    gamma = np.random.randint(MIN_GAMMA, MAX_GAMMA + 1) * 0.01
    img_gamma = adjust_gamma(img_linear, gamma=gamma)

    # Image Mask Generation
    class_num = mask_array.shape[2]
    mask_index = 0
    max_mask_sum = 0
    for index in range(class_num):
        mask_sum = np.sum(mask_array[:, :, index])
        if mask_sum > max_mask_sum:
            mask_index = index
            max_mask_sum = mask_sum

    #mask_index = np.random.randint(0, class_num, 1)

    mask = np.squeeze(mask_array[:,:,mask_index])*255

    mask_not = cv2.bitwise_not(mask)

    img_bg = cv2.bitwise_and(image, image, mask=mask_not)
    img_fg = cv2.bitwise_and(img_gamma, img_gamma, mask=mask)
    input_image = cv2.add(img_bg, img_fg)

    mask_shape = (img_gamma.shape[0], img_gamma.shape[1])
    mask_image = cv2.bitwise_and(np.ones(mask_shape, img_gamma.dtype) * 255,
                                np.ones(mask_shape, img_gamma.dtype) * 255, mask=mask)



    if display_on == 1:
        plt.subplot(141)
        plt.title('1. Input Image')
        plt.imshow(input_image)
        plt.subplot(142)
        plt.title('2. Mask')
        plt.imshow(mask_image, cmap='gray')
        plt.subplot(143)
        plt.title('3. Target')
        plt.imshow(image)
        plt.subplot(144)
        plt.title('4. Semantic')
        plt.imshow(semantic)


        plt.show()


    return np.array(input_image), np.array(mask_image)


def resize_image(image, size):
    """Resize an image to the given size."""
    return image.resize(size, Image.ANTIALIAS)

def pre_proc(CocoDataset, semantic_dir, output_dir):
    """Resize the images in 'image_dir' and save into 'output_dir'."""

    count = 0
    for image_id in CocoDataset.image_ids:
        image = CocoDataset.load_image(image_id)
        mask, class_ids = CocoDataset.load_mask(image_id)
        semantic = CocoDataset.load_semantic(image_id, semantic_dir)



        input_image, mask_image = data_gen(image, mask, semantic, 0)

        scipy.misc.toimage(input_image, cmin=0.0, cmax=255.0).save(
            output_dir + '/input/' + str(image_id) + '_i.jpg')

        scipy.misc.toimage(mask_image, cmin=0.0, cmax=255.0).save(
            output_dir + '/input/' + str(image_id) + '_m.jpg')

        scipy.misc.toimage(image, cmin=0.0, cmax=255.0).save(
            output_dir + '/target/' + str(image_id) + '_t.jpg')

        count = count + 1
        if count % 500 == 0:
            print (str(count) + '/' + str(len(CocoDataset.image_ids)))



def main(args):
    image_dir = args.image_dir
    save_dir = args.save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(save_dir + '/train')
        os.makedirs(save_dir + '/val')
        os.makedirs(save_dir + '/train/input')
        os.makedirs(save_dir + '/train/target')
        os.makedirs(save_dir + '/test/input')
        os.makedirs(save_dir + '/test/target')

    dataset = CocoDataset()
    dataset.load_coco(image_dir, "train")
    dataset.prepare()

    print("Image Count: {}".format(len(dataset.image_ids)))
    print("Class Count: {}".format(dataset.num_classes))
    for i, info in enumerate(dataset.class_info):
        print("{:3}. {:50}".format(i, info['name']))

    pre_proc(dataset, image_dir+'/annotations/train2017', save_dir+'/train')


    dataset = CocoDataset()
    dataset.load_coco(image_dir, "val")
    dataset.prepare()

    print("Image Count: {}".format(len(dataset.image_ids)))
    print("Class Count: {}".format(dataset.num_classes))
    for i, info in enumerate(dataset.class_info):
        print("{:3}. {:50}".format(i, info['name']))

    pre_proc(dataset, image_dir+'/annotations/val2017', save_dir + '/test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='/media/hdd/data/coco',
                        help='directory for dataset images')

    parser.add_argument('--save_dir', type=str, default='/media/hdd/data/harmonization',
                        help='directory for saving images')

    args = parser.parse_args()
    main(args)