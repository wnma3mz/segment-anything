import gradio as gr
import requests
import numpy as np
import base64
import cv2
import time

max_q = 3

seg_url = "http://10.26.34.12:7006/api"


def sepia(
    input_img,
    # output_mode,
    points_per_side,
    crop_n_layers,
    min_mask_region_area,
    crop_n_points_downscale_factor,
    pred_iou_thresh,
    stability_score_thresh,
    stability_score_offset,
    box_nms_thresh,
    crop_nms_thresh,
    crop_overlap_ratio,
):
    # print(input_img)
    fname = f"{int(time.time())}.png"
    cv2.imwrite(fname, input_img)
    with open(fname, "rb") as f:
        data = f.read()
    params = {
        k: v
        for k, v in zip(
            [
                # "output_mode",
                "points_per_side",
                "crop_n_layers",
                "min_mask_region_area",
                "crop_n_points_downscale_factor",
                "pred_iou_thresh",
                "stability_score_thresh",
                "stability_score_offset",
                "box_nms_thresh",
                "crop_nms_thresh",
                "crop_overlap_ratio",
            ],
            [
                # output_mode,
                points_per_side,
                crop_n_layers,
                min_mask_region_area,
                crop_n_points_downscale_factor,
                pred_iou_thresh,
                stability_score_thresh,
                stability_score_offset,
                box_nms_thresh,
                crop_nms_thresh,
                crop_overlap_ratio,
            ],
        )
    }
    for key in [
        "pred_iou_thresh",
        "stability_score_thresh",
        "stability_score_offset",
        "box_nms_thresh",
        "crop_nms_thresh",
        "crop_overlap_ratio",
    ]:
        params[key] = float(params[key])
    print(params)
    s = requests.post(seg_url, json={"image": base64.b64encode(data), "params": params})

    image_b64 = s.json()["image"]
    img_array = np.frombuffer(base64.b64decode(image_b64), dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


label_points_per_side = gr.Slider(1, 100, step=1, label="points_per_side", value=32)
label_crop_n_layers = gr.Slider(0, 100, step=1, label="crop_n_layers", value=0)
label_min_mask_region_area = gr.Slider(
    0, 100, step=1, label="min_mask_region_area", value=0
)
label_crop_n_points_downscale_factor = gr.Slider(
    1, 100, step=1, label="crop_n_points_downscale_factor", value=1
)

label_pred_iou_thresh = gr.Slider(0, 1, label="pred_iou_thresh", value=0.88)
label_stability_score_thresh = gr.Slider(
    0, 1, label="stability_score_thresh", value=0.95
)
label_stability_score_offset = gr.Slider(
    0, 10, label="stability_score_offset", value=1.0
)
label_box_nms_thresh = gr.Slider(0, 10, label="box_nms_thresh", value=0.7)
label_crop_nms_thresh = gr.Slider(0, 10, label="crop_nms_thresh", value=0.7)
label_crop_overlap_ratio = gr.Slider(
    0, 10, label="crop_overlap_ratio", value=512 / 1500
)


demo = gr.Interface(
    sepia,
    [
        gr.Image(),
        # gr.Radio(["binary_mask", "uncompressed_rle", "coco_rle"], value="binary_mask"),
        label_points_per_side,
        label_crop_n_layers,
        label_min_mask_region_area,
        label_crop_n_points_downscale_factor,
        label_pred_iou_thresh,
        label_stability_score_thresh,
        label_stability_score_offset,
        label_box_nms_thresh,
        label_crop_nms_thresh,
        label_crop_overlap_ratio,
    ],
    gr.Image(shape=(200, 200)),
)

if __name__ == "__main__":
    # network_test()
    gr.close_all()
    # demo.queue(concurrency_count=3, max_size=20)
    demo.launch(
        debug=True,
        share=False,
        server_name="0.0.0.0",
        server_port=7861,
        # ssl_keyfile="./localhost+2-key.pem",
        # ssl_certfile="./localhost+2.pem",
    )
