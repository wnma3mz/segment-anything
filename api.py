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

from flask import Flask, Response, jsonify, make_response, render_template, request
from io import BytesIO
from PIL import Image
import base64

app = Flask(__name__)


class M:
    def __init__(self, sam_checkpoint, model_type, device):

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        self.mask_generator = SamAutomaticMaskGenerator(sam)

    def generate(self, image, **params):
        for key in [
            "points_per_side",
            "crop_n_layers",
            "min_mask_region_area",
            "crop_n_points_downscale_factor",
        ]:
            v = params.get(key, -1)
            if v == -1:
                continue
            if not isinstance(v, int):
                print(key, "Error")
                return -1
            setattr(self.mask_generator, key, v)

        for key in [
            "pred_iou_thresh",
            "stability_score_thresh",
            "stability_score_offset",
            "box_nms_thresh",
            "crop_nms_thresh",
            "crop_overlap_ratio",
        ]:
            v = params.get(key, -1.0)
            if v == -1.0:
                continue
            if not isinstance(v, float):
                print(key, "Error")
                return -1
            setattr(self.mask_generator, key, v)
        output_mode = params.get("output_mode", "binary_mask")
        assert output_mode in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ], f"Unknown output_mode {output_mode}."
        setattr(self.mask_generator, "output_mode", output_mode)

        return self.mask_generator.generate(image)

    def show_anns(self, image, anns, fname):
        if len(anns) == 0:
            return

        plt.xticks([])
        plt.yticks([])
        plt.imshow(image)
        sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
        ax = plt.gca()
        for ann in sorted_anns:
            m = ann["segmentation"]
            img = np.ones((m.shape[0], m.shape[1], 3))
            color_mask = np.random.random((1, 3)).tolist()[0]
            for i in range(3):
                img[:, :, i] = color_mask[i]
            ax.imshow(np.dstack((img, m * 0.35)))
        plt.savefig(fname, bbox_inches="tight", pad_inches=-0.1)


m = M(
    sam_checkpoint="ckpts/sam_vit_h_4b8939.pth", model_type="default", device="cuda:0"
)


@app.route("/api", methods=["GET", "POST"])
def run_func():
    if request.method == "POST":
        data = request.json
        if data.get("image"):
            try:
                image_b64 = data.get("image")

                img_array = np.frombuffer(base64.b64decode(image_b64), dtype=np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                fname = f"{int(time.time())}.png"
                mask_output = m.generate(image, **data.get("params", {}))
                if mask_output == -1:
                    return jsonify({"error": "params error"})
                m.show_anns(image, mask_output, fname)

                with open(fname, "rb") as f:
                    data = f.read()
                return jsonify({"image": base64.b64encode(data)})
            except Exception as e:
                print(e)
                return jsonify({"error": "image error"})
        return jsonify({"error": "No image Provided."})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7006, debug=False)