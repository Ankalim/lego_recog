import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(r"/home/nik1/lego/lego_recog/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


class GenerateDataset_custom(utils.Dataset):

    def load_details(self, count, height, width):
        self.ROOT_PATH = os.path.abspath(r'/home/nik1/lego/lego_recog/images')
        self.DIC_DETAILS_IMG_DIR_PATH = {'brick': '3003 Brick 2x2',
                                         'plate': '3022 Plate 2x2'}
        self.DIC_DETAILS_IMG_NAMES = {}
        for ind in self.DIC_DETAILS_IMG_DIR_PATH:
            self.DIC_DETAILS_IMG_NAMES.update({ind: os.listdir(os.path.join(self.ROOT_PATH,
                                                                            self.DIC_DETAILS_IMG_DIR_PATH[ind]))})
        self.add_class("details", 1, "brick")
        self.add_class("details", 2, "plate")

        for i in range(count):
            #             bg_color = np.array([random.randint(0, 255) for _ in range(3)])
            bg_color, details = self.random_image(height, width)
            self.add_image("details", image_id=i, path=None,
                           width=width, height=height,
                           bg_color=bg_color, details=details)

        # загрузка изображений и масок
        self.__dicImg = {}
        self.__dicMasks = {}
        for inf in self.image_info:
            result_img, result_masks = self.__prepare_data(inf)
            self.__dicImg.update({inf['id']: result_img})
            self.__dicMasks.update({inf['id']: result_masks})

    def random_detail(self, height, width):
        detail = random.choice(["brick", "plate"])
        # Color
        color = tuple([random.randint(0, 255) for _ in range(3)])
        # Center x, y
        buffer = 20
        s = random.randint(buffer, height // 4)
        y = random.randint(buffer, height - buffer - 1 - s)
        x = random.randint(buffer, width - buffer - 1 - s)
        # Size
        path_to_img = os.path.join(self.ROOT_PATH,
                                   self.DIC_DETAILS_IMG_DIR_PATH[detail],
                                   random.choice(self.DIC_DETAILS_IMG_NAMES[detail]))

        return detail, color, (x, y, s), path_to_img

    def random_image(self, height, width):
        # Pick random background color
        bg_color = np.array([random.randint(0, 255) for _ in range(3)])
        # Generate a few random shapes and record their
        # bounding boxes
        details = []
        boxes = []
        N = random.randint(1, 15)
        for _ in range(N):
            detail, color, dims, path_to_img = self.random_detail(height, width)
            details.append((detail, color, dims, path_to_img))
            x, y, s = dims
            boxes.append([y - s, x - s, y + s, x + s])
        # Apply non-max suppression wit 0.3 threshold to avoid
        # shapes covering each other
        keep_ixs = utils.non_max_suppression(np.array(boxes), np.arange(N), 0.3)
        details = [s for i, s in enumerate(details) if i in keep_ixs]
        return bg_color, details

    def load_image(self, image_id):
        return self.__dicImg[image_id]

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        details = info['details']
        class_ids = np.array([self.class_names.index(s[0]) for s in details])
        return self.__dicMasks[image_id].astype(np.bool), class_ids.astype(np.int32)

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "details":
            return info["details"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def __change_color(self, img, clr):
        img_norm = img.copy()
        img_norm = img_norm / 255.
        clr = np.array(clr)
        clr = clr / 255.
        img_norm *= clr
        img_norm = (img_norm * 255.).astype(np.uint8)
        #     plt.imshow(img_norm)
        return img_norm

    def __image_resize(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)

        # return the resized image
        return resized

    def __prepare_data(self, img_info):
        bg_color = np.array(img_info['bg_color']).reshape([1, 1, 3])
        img1 = np.ones([img_info['height'], img_info['width'], 3], dtype=np.uint8)
        img1 = img1 * bg_color.astype(np.uint8)
        mask_full = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        masks = []
        for detail_type, color, (x, y, s), path in img_info['details']:
            img2 = cv2.imread(path)

            img2 = self.__image_resize(img2, height=s)
            img2 = self.__change_color(img2, color)
            # I want to put logo on top-left corner, So I create a ROI
            rows, cols, channels = img2.shape
            mask = np.zeros((rows, cols), dtype=np.uint8)
            roi = img1[y:rows + y, x:cols + x]
            #         print(x)
            #         print(roi.shape)
            #         print(img2.shape)
            # Now create a mask of logo and create its inverse mask also
            img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            #         print(mask_inv.shape)
            # Now black-out the area of logo in ROI
            img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            # Take only region of logo from logo image.
            img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
            # Put logo in ROI and modify the main image
            dst = cv2.add(img1_bg, img2_fg)
            img1[y:rows + y, x:cols + x] = dst
            tmp_mask_full = mask_full.copy()
            tmp_mask_full[y:rows + y, x:cols + x] = mask
            tmp_mask_full = tmp_mask_full.reshape((img_info['height'], img_info['width'], 1))

            masks.append(tmp_mask_full)
        masks = np.concatenate(masks, axis=2)
        return img1, masks

class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    #     NAME = "shapes"
    NAME = "details"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    #     NUM_CLASSES = 1 + 3  # background + 3 shapes
    NUM_CLASSES = 1 + 2  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 640
    IMAGE_MAX_DIM = 640

    #     IMAGE_MIN_DIM = 128
    #     IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

config = ShapesConfig()
config.display()

# Training dataset
# dataset_train = ShapesDataset()
dataset_train = GenerateDataset_custom()
dataset_train.load_details(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()

# Validation dataset
dataset_val = GenerateDataset_custom()
dataset_val.load_details(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()

model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=1,
            layers='heads')

class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)