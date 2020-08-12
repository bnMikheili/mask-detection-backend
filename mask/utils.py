import numpy as np
import os
import cv2
import tensorflow as tf


def load_tflite_model(tf_model_path):
    """
    Loads tflite model from a given path

    Args:
        tf_model_path (str): Path to the model

    Returns:
        (interpreter, input_details, output_details):
        tf.lite.Interpreter, interpreter's input and output details
    """
    # Load the tfline model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=tf_model_path)
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


def decode_bbox(anchors, raw_outputs, variances=[0.1, 0.1, 0.2, 0.2]):
    """
    Decode the actual bbox according to the anchors.
    the anchor value order is:[xmin,ymin, xmax, ymax]

    Args:
        anchors (np.array): numpy array of anchors with
            shape [batch, num_anchors, 4]
        raw_outputs (np.array): numpy array with the same
            shape with anchors
        variances (list): list of float as scaling variances,
            default=[0.1, 0.1, 0.2, 0.2]

    Returns:
        (np.array): numpy array of prediction boxes
    """
    anchor_centers_x = (anchors[:, :, 0:1] + anchors[:, :, 2:3]) / 2
    anchor_centers_y = (anchors[:, :, 1:2] + anchors[:, :, 3:]) / 2
    anchors_w = anchors[:, :, 2:3] - anchors[:, :, 0:1]
    anchors_h = anchors[:, :, 3:] - anchors[:, :, 1:2]

    raw_outputs_rescale = raw_outputs * np.array(variances)

    predict_center_x = raw_outputs_rescale[
                                       :,
                                       :, 0:1] * anchors_w + anchor_centers_x
    predict_center_y = raw_outputs_rescale[
                                       :,
                                       :, 1:2] * anchors_h + anchor_centers_y
    predict_w = np.exp(raw_outputs_rescale[:, :, 2:3]) * anchors_w
    predict_h = np.exp(raw_outputs_rescale[:, :, 3:]) * anchors_h
    predict_xmin = predict_center_x - predict_w / 2
    predict_ymin = predict_center_y - predict_h / 2
    predict_xmax = predict_center_x + predict_w / 2
    predict_ymax = predict_center_y + predict_h / 2
    predict_bbox = np.concatenate(
        [predict_xmin, predict_ymin, predict_xmax, predict_ymax], axis=-1)

    return predict_bbox


def generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios):
    """
    Generate anchors.

    Args:
        feature_map_sizes (list): feature map list of lists,
            for example: [[40,40], [20,20]]
        anchor_sizes (list): anchor sizes list of lists,
            for example: [[0.05, 0.075], [0.1, 0.15]]
        anchor_ratios (list): anchors ratios list of lists,
            for example: [[1, 0.5], [1, 0.5]]

    Returns:
        (np.array): numpy array of anchor bounding boxes
    """
    anchor_bboxes = []

    # generates anchors for each feature_map_size, generating centers of each
    # anchor and the X,Y coordinates in relation with the center
    for idx, feature_size in enumerate(feature_map_sizes):
        cx = (np.linspace(0, feature_size[0] - 1,
                          feature_size[0]) + 0.5) / feature_size[0]
        cy = (np.linspace(0, feature_size[1] - 1,
                          feature_size[1]) + 0.5) / feature_size[1]

        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid_expend = np.expand_dims(cx_grid, axis=-1)
        cy_grid_expend = np.expand_dims(cy_grid, axis=-1)

        center = np.concatenate((cx_grid_expend, cy_grid_expend), axis=-1)

        num_anchors = len(anchor_sizes[idx]) + len(anchor_ratios[idx]) - 1
        center_tile = np.tile(center, (1, 1, 2 * num_anchors))
        anchor_width_heights = []

        # generates anchor centers by evenly
        #   distributing the center coordinates
        # on each X,Y axis. Afterwards, calculates
        #   box coordinates for each anchor
        # in relation with the previously generated anchor center
        for scale in anchor_sizes[idx]:
            ratio = anchor_ratios[idx][0]  # select the first ratio
            width = scale * np.sqrt(ratio)
            height = scale / np.sqrt(ratio)
            anchor_width_heights.extend(
                [-width / 2.0, -height / 2.0, width / 2.0, height / 2.0])

        #  run on the first scale, with different aspect ratios
        #  (except for the first one)
        for ratio in anchor_ratios[idx][1:]:
            scale_1 = anchor_sizes[idx][0]  # select the first scale
            width = scale_1 * np.sqrt(ratio)
            height = scale_1 / np.sqrt(ratio)
            anchor_width_heights.extend(
                [-width / 2.0, -height / 2.0, width / 2.0, height / 2.0])

        bbox_coords = center_tile + np.array(anchor_width_heights)
        bbox_coords_reshape = bbox_coords.reshape((-1, 4))
        anchor_bboxes.append(bbox_coords_reshape)

    anchor_bboxes = np.concatenate(anchor_bboxes, axis=0)
    return anchor_bboxes


def single_class_non_max_suppression(
   bboxes,
   confidences,
   conf_thresh=0.2,
   iou_thresh=0.5,
   keep_top_k=-1):
    """
    Do Non Max Suppresssion on single class.
    For the specific class, given the bbox and its confidence,

    1) sort the bbox according to the confidence from top to down, we call this
    a set

    2) select the bbox with the highest confidence, remove it from set, and do
    IOU calculate with the rest bbox

    3) remove the bbox whose IOU(Intersection of Union) is higher than
        the iou_thresh from the set,

    4) loop step 2 and 3, until the set is empty.

    Leaves only the bounding box with the highest confidence.

    Args:
        bboxes (np.array): bboxes numpy array of 2D, [num_bboxes, 4]
        confidences (np.array): Confidences numpy array of 1D. [num_bboxes]
        conf_thresh (float): Confidence threshold (0.2 by default)
        iou_thresh (float): Intersection of Union threshold (0.5 by default)
        keep_top_k (int): How many bboxes to keep, in case of -1 keeps all
            of them (-1 by default)

    Returns:
        (list): List of indexes which should be kept
    """
    if len(bboxes) == 0:
        return []

    conf_keep_idx = np.where(confidences > conf_thresh)[0]

    bboxes = bboxes[conf_keep_idx]
    confidences = confidences[conf_keep_idx]

    # pick bboxes with highest confidence
    pick_highest_conf = []
    xmin = bboxes[:, 0]
    ymin = bboxes[:, 1]
    xmax = bboxes[:, 2]
    ymax = bboxes[:, 3]

    area = (xmax - xmin + 1e-3) * (ymax - ymin + 1e-3)
    idxs = np.argsort(confidences)

    # pick bboxes with highest confidence,
    # remove bboxes with IOU higher than threshold,
    # so we are picking bbox with highest confidence
    # for each
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick_highest_conf.append(i)

        # keep top k
        if keep_top_k != -1:
            if len(pick_highest_conf) >= keep_top_k:
                break

        overlap_xmin = np.maximum(xmin[i], xmin[idxs[:last]])
        overlap_ymin = np.maximum(ymin[i], ymin[idxs[:last]])
        overlap_xmax = np.minimum(xmax[i], xmax[idxs[:last]])
        overlap_ymax = np.minimum(ymax[i], ymax[idxs[:last]])
        overlap_w = np.maximum(0, overlap_xmax - overlap_xmin)
        overlap_h = np.maximum(0, overlap_ymax - overlap_ymin)
        overlap_area = overlap_w * overlap_h

        overlap_ratio = overlap_area / \
            (area[idxs[:last]] + area[i] - overlap_area)

        need_to_be_deleted_idx = np.concatenate(
            ([last], np.where(overlap_ratio > iou_thresh)[0]))

        idxs = np.delete(idxs, need_to_be_deleted_idx)

    # if the number of final bboxes is less than keep_top_k, we need to pad it.
    # TODO
    return conf_keep_idx[pick_highest_conf]


def get_feature_map_sizes(type):
    """
    Defines and returns the sizes of the feature map

    Args:
        type (string): ?describe this arg

    Returns:
        (list): feature_map_sizes
    """
    if type == 'tf':
        feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
    elif type == 'pt':
        feature_map_sizes = [[45, 45], [23, 23], [12, 12], [6, 6], [4, 4]]
    else:
        print('Incorrect type')

    return feature_map_sizes


def get_anchor_sizes():
    """
    Defines and returns the anchor sizes

    Returns:
        (list): anchor_sizes
    """
    anchor_sizes = [[0.04, 0.056], [0.08, 0.11],
                    [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
    return anchor_sizes


def get_anchor_ratios():
    """
    Defines and returns the anchor ratios

    Returns:
        (list): anchor_ratios
    """
    anchor_ratios = [[1, 0.62, 0.42]] * 5
    return anchor_ratios


def write_output_video(vid, output_video_name):
    """
    Creates a video writer

    Args:
        vid (cv2.VideoCapture object): video capture
        output_video_name (str): name of the output video file

    Returns:
        (cv2.VideoWriter): writer ?describe
    """
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(output_video_name, codec, fps, (width, height))
    return writer


def check_directory(directory_name):
    """
    Checks if the given directory exists, and if not
        it creates a new directory.

    Args:
        directory_name (str)

    Returns:
        None
    """
    if not os.path.exists(directory_name):
        print("""The directory you are looking for it doesn't exist,
            creating a new one..""")
        os.mkdir(directory_name)
    else:
        print("The directory already exists, moving on..")


def draw_results(
   idxs,
   image,
   bbox_max_scores,
   bbox_max_score_classes,
   y_bboxes,
   blur = True
   ):
    """
    Draw object bounding boxes, object classes, and confidence scores for
    indices needed for NMS.

    Args:
        idxs (list): List with bounding box indices needed for NMS
        image (np.array): 3D numpy array of input image
        bbox_max_scores (list): List of max class scores per bbox
        bbox_max_score_classes (list): List with classes per bbox
        y_bboxes (np.array): Bbox coordinates
        blur (bool): Flag for blurring the face

    Returns:
        (list): List of tuples ([xmin, ymin, xmax, ymax], class_id)
    """
    id2class = {0: 'Mask', 1: 'NoMask'}
    height, width, _ = image.shape
    boxes = []

    # for each list of bbox indices needed for NMS,
    # we compute a confidence interval,
    # and draw object bboxes
    for idx in idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]

        # clip the coordinate, avoid the value exceed the image boundary.
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)
        box = ([xmin, ymin, xmax, ymax], class_id)
        boxes.append(box)

        # when wearing mask
        if class_id == 0:
            color = (0, 255, 0)

        else:
            color = (255, 0, 0)

        if blur:
            image[ymin:ymax, xmin:xmax] = cv2.blur(
                image[ymin:ymax, xmin:xmax], (40, 40))

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(image, "%s: %.2f" % (id2class[class_id], conf),
                    (xmin + 2, ymin - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)

    return boxes
