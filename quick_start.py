# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import time
import json

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

from io import BytesIO
from PIL import Image
import base64

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))
    plt.savefig()

def read_image(fname):
    image = cv2.imread(fname)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    

if __name__ == "__main__":
    sam_checkpoint="ckpts/sam_vit_h_4b8939.pth"
    model_type="default"
    device="cuda:0"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)    

    image = read_image("assets/notebook1.png")
    masks = mask_generator.generate(image)

    show_anns(masks)