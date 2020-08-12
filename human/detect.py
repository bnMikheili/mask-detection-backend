'''
Version2
Tracking + reidentification.
Tracking is done using last k=15 frames.
'''
import cv2
import tensorflow as tf

from mask.utils import load_tflite_model
from config import config_import as conf

HUMAN_DETECT_CONF = conf.get_config_data_by_key('human_detection')
COCO_INP_SIZE = HUMAN_DETECT_CONF['COCO_INP_SIZE']
MOBILENET_SSD_PATH = HUMAN_DETECT_CONF['MOBILENET_SSD_PATH']

class cocoDetection:
    """
    Class for object detection with Coco ssd mobilenet
    """
    def __init__(self):
        self.interpreter, self.input_details, self.output_details = load_tflite_model(MOBILENET_SSD_PATH)

    def inference(self, img, draw=False):
        """
        People detection with ssd mobilenet coco

        Args:
            img (np.array): Input image
            draw (bool): Whether to draw bounding boxes on image or not

        Returns:
            (list): list of human bounding boxes
        """
        # Resize the image to size [1, 300, 300, 3]
        img_in = tf.expand_dims(img, 0)
        img_in = tf.image.resize(img_in, (COCO_INP_SIZE, COCO_INP_SIZE))
        img_in = tf.cast(img_in, dtype=tf.uint8)

        # Set img_in as tensor in the model's input_details
        self.interpreter.set_tensor(self.input_details[0]['index'], img_in)
        self.interpreter.invoke()

        # Get the output_details tensors (based on the given input above)
        bbox = self.interpreter.get_tensor(self.output_details[0]['index'])
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])
        nums = self.interpreter.get_tensor(self.output_details[3]['index'])

        boxes, objectness, classes, nums = bbox[0], scores[0], classes[0], nums[0]

        # Getting the images width and height
        wh = img.shape[0:2]

        # Person class id = 0
        person_id = 0
        ppl_boxes = []

        # Go through the prediction results
        for i in range(int(nums)):
            # Iterate over the predicted bounding boxes and filter
            #   the boxes with class "person"
            if classes[i] == person_id and scores[0][i] > 0.60:
                x1 = int(boxes[i][1] * wh[1])
                y1 = int(boxes[i][0] * wh[0])
                x2 = int(boxes[i][3] * wh[1])
                y2 = int(boxes[i][2] * wh[0])

                display_str = 'Person ' + str(round(objectness[i], 2))
                text_w, text_h = cv2.getTextSize(
                    display_str,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6,
                    thickness=1
                )[0]

                ppl_boxes.append([x1, y1, x2, y2])
                if draw:
                    cv2.rectangle(
                        img,
                        (x1, y1),
                        (x2, y2),
                        (60, 179, 113),
                        4
                    )
                    cv2.rectangle(
                        img,
                        (x1 + 2, y1 - 2),
                        (x1 + text_w + 2, y1 - text_h - 2),
                        (60, 179, 113),
                        cv2.FILLED
                    )
                    cv2.putText(
                        img,
                        display_str,
                        (x1 + 2, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.6,
                        thickness=1,
                        color=(255, 255, 255)
                    )

        return ppl_boxes
