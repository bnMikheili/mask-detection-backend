# -*- coding:utf-8 -*-
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from mask.utils import load_tflite_model
from config import config_import as conf
from mask import utils


MODEL_PATH = conf.get_config_data_by_key("mask_detection")["MODEL_PATH"]

interpreter, input_details, output_details = load_tflite_model(MODEL_PATH)

# Configure anchors
feature_map_sizes = utils.get_feature_map_sizes("tf")
anchor_sizes = utils.get_anchor_sizes()
anchor_ratios = utils.get_anchor_ratios()

# Generate anchors
anchors = utils.generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

# For inference , the batch size is 1, the model output shape is [1, N, 4],
# so we expand dim for anchors to [1, anchor_num, 4]
anchors_exp = np.expand_dims(anchors, axis=0)
id2class = {0: "Mask", 1: "NoMask"}


# def inference(image):
#     pred = face_mask_detection(image)
#     return pred


def inference(
    image, conf_thresh=0.5, iou_thresh=0.4, target_shape=(260, 260), show_result=False
):
    """
    Driver function for face mask detection inference

    Args:
        image (np.array): 3D numpy array of input image
        conf_thresh (float): Min threshold of classification probability
        iou_thresh (float): IOU threshold of NMS
        target_shape (tuple): Model input shape
        show_result (bool): Image display flag

    Returns:
        (array): Array of bounding boxes
    """
    image_resized = cv2.resize(image, target_shape)
    image_np = image_resized / 255.0
    image_exp = np.expand_dims(image_np, axis=0)
    image_exp = tf.cast(image_exp, dtype=tf.float32)

    # Set img_in as tensor in the model's input_details
    interpreter.set_tensor(input_details[0]["index"], image_exp)
    interpreter.invoke()

    # Get the output_details tensors (based on the given input above)
    y_bboxes_output = interpreter.get_tensor(output_details[0]["index"])
    y_cls_output = interpreter.get_tensor(output_details[1]["index"])

    # remove the batch dimension, for batch is always 1 for inference
    y_bboxes = utils.decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]

    # to speed up, do single class NMS, not multiple classes NMS
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)

    # keep_idx is the alive bounding box after nms
    keep_idxs = utils.single_class_non_max_suppression(
        y_bboxes, bbox_max_scores, conf_thresh=conf_thresh, iou_thresh=iou_thresh
    )

    boxes = utils.draw_results(
        keep_idxs, image, bbox_max_scores, bbox_max_score_classes, y_bboxes
    )

    if show_result:
        Image.fromarray(image).show()

    return boxes
