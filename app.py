from PIL import Image
import numpy as np
import cv2

from human.detect import cocoDetection as human_detector
from mask.detect import inference as detect_masks

def show_people(img_path):
    detector = human_detector()
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes = detector.inference(img, draw=True)
    Image.fromarray(img).show()

def show_masks(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detect_masks(img, show_result=True)

show_people('')
show_masks('')

