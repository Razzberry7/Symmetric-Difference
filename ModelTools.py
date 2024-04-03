from typing import List, Tuple
import yolov5
from PIL import Image
import os
import math
import torch
import csv
import cv2
import numpy as np

class Model:
    """Creates a ModelTools object

    :param model_path: The path to the model used for detections
    :param conf: Confidence level to mark a prediction as valid
    :param iou: IoU level to mark a prediction as valid
    :param agnostic: Disallow multiple classes per annotation
    """
    def __init__(self, model_path: str, conf: float = 0.25, iou: float = 0.45, agnostic: bool = False):
        self.model = yolov5.load(model_path)
        self.model.conf = conf
        self.model.iou = iou
        self.model.agnostic = agnostic
    
    """Predicts annotations for a single image

    :param image_path: The path to the image
    :param image: The actual image (only need image_path or image, not both)
    :returns: A Tensor containing annotation outputs [x1, y1, x2, y2, conf, label]
    """
    def prediction(self, image_path: str = None, image: Image = None) -> List[List[int]]:
        if image_path:
            image = Image.open(image_path)
        if image:
            res = []
            model_res = self.model(image).xyxy[0]
            for a in model_res:
                a = a.tolist()
                a[0], a[2] = a[0] / image.width, a[2] / image.width
                a[1], a[3] = a[1] / image.height, a[3] / image.height
                res.append(a)
            return res
        return None
    
    """Predicts annotations for a batch of images

    :param image_directory: The path to the image directory
    :returns: A list containing annotation outputs for each image
    (accessible by calling .xyxy[index] on the output) [x1, y1, x2, y2, conf, label]
    """
    def batch_prediction(self, image_directory: str)-> torch.Tensor:
        files = os.listdir(image_directory)
        imgs = [image_directory + f for f in files]
        return self.model(imgs)

    """Predicts annotations for a single image with cropped tiles
    
    :param crop_dim: The size of each cropped image
    :param border_size: The amount of pixels to cut off each side of the cropped image
    :param image_path: The path to the image
    :param image: The actual image (only need image_path or image, not both)
    :returns: A list containing annotation outputs [x1, y1, x2, y2, conf, label]
    """
    def tile_prediction(self, crop_dim, border_size, image_path: str = None, image: Image = None) -> List[List[int]]:
        if image_path:
            image = Image.open(image_path)
            
        images = []
        slide_increment = crop_dim - (border_size*2)
        for x in range(0, image.width, slide_increment):
            for y in range(0, image.height, slide_increment):
                images.append(image.crop((x-border_size, y-border_size, x+crop_dim-border_size, y+crop_dim-border_size)))
        
        x_dim = math.ceil(image.width/slide_increment)
        y_dim = math.ceil(image.height/slide_increment)
                
        results = self.model(images)
        output = []
        
        for xi in range(x_dim):
            for yi in range(y_dim): 
                boxes = results.xyxy[xi*y_dim + yi].cpu().numpy()
                
                x_offset = xi*slide_increment
                y_offset = yi*slide_increment
                
                for box in boxes:
                    x_m = (box[0] + box[2]) / 2
                    y_m = (box[1] + box[3]) / 2
                    if x_m < border_size or x_m > crop_dim - border_size:
                        continue
                    if y_m < border_size or y_m > crop_dim - border_size:
                        continue
                    box_new = []
                    box_new.append((box[0] + x_offset - border_size) / image.width)
                    box_new.append((box[1] + y_offset - border_size) / image.height)
                    box_new.append((box[2] + x_offset - border_size) / image.width)
                    box_new.append((box[3] + y_offset - border_size) / image.height)
                    box_new.append(box[4])
                    box_new.append(box[5])
                    output.append(box_new)
                    
        return output
    
    """Compares ground truth to prediction and finds false negatives and positives

    :param ground_truth_path: The path to the ground truth annotation file
    :param image_path: The path to the image
    :param image: The actual image (only need image_path or image, not both)
    :param threshold: IoU required for two annotations to be considered the same
    :returns: Two lists (false_negatives, false_positives) containing annotation outputs [x1, y1, x2, y2, conf, label]
    """
    def find_nonoverlapping(self, ground_truth_path: str, image_path: str = None, image: Image = None, threshold: float = 0.7, tiled=False) -> Tuple[List[List[float]]]:
        ground_truth = yolov5_to_xyxy(ground_truth_path)
        if not tiled:
            prediction = self.prediction(image_path=image_path, image=image)
        else:
            prediction = self.tile_prediction(640, 60, image_path=image_path, image=image)
        false_negatives = []
        false_positives = []
        for box1 in ground_truth:
            valid_overlap = False
            for box2 in prediction:
                if do_annotations_overlap(box1, box2) and iou(box1, box2) > threshold:
                    valid_overlap = True
                    break
            if not valid_overlap:
                box1.append(box1[4])
                box1[4] = 1.0
                false_negatives.append(box1)
        for box1 in prediction:
            valid_overlap = False
            for box2 in ground_truth:
                if do_annotations_overlap(box1, box2) and iou(box1, box2) > threshold:
                    valid_overlap = True
                    break
            if not valid_overlap:
                false_positives.append(box1)
        return (false_negatives, false_positives, len(ground_truth), len(prediction))



"""Returns IoU value for two annotations

:param box1: First annotation [x1, y1, x2, y2]
:param box2: Second annotation [x1, y1, x2, y2]
:returns: IoU value
"""
def iou(box1, box2) -> float:
    x_min_intercept = max(box1[0], box2[0])
    x_max_intercept = min(box1[2], box2[2])

    y_min_intercept = max(box1[1], box2[1])
    y_max_intercept = min(box1[3], box2[3])

    intercept_area = max(0, x_max_intercept - x_min_intercept) * max(0, y_max_intercept - y_min_intercept)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return intercept_area / float(box1_area + box2_area - intercept_area)

"""Returns a label file converted to XYXY format

:param label_path: The path to the label file
:returns: XYXY formatted annotation data [x1, y1, x2, y2, label]
"""
def yolov5_to_xyxy(label_path) -> List[List[float]]:
    # Open annotation file
    with open(label_path, newline='') as csvfile:
        data = list(csv.reader(csvfile, delimiter=' ', quotechar='|'))
        for box in data:
            x_center = float(box[1])
            y_center = float(box[2])
            half_width = float(box[3]) / 2
            half_height = float(box[4]) / 2
            box[4] = int(box[0])
            box[0] = x_center - half_width
            box[1] = y_center - half_height
            box[2] = x_center + half_width
            box[3] = y_center + half_height
        return data

"""Checks if two annotations have any overlap

:param box1: First annotation [x1, y1, x2, y2]
:param box2: Second annotation [x1, y1, x2, y2]
:returns: Whether the annotations have overlap
"""
def do_annotations_overlap(box1, box2) -> bool:
    if (box1[0]>=box2[2]) or (box1[2]<=box2[0]) or (box1[3]<=box2[1]) or (box1[1]>=box2[3]):
        return False
    return True

"""Exports an image with labeled annotations

:param annotations: List of annotations [x1, y1, x2, y2, conf, label]
:param output_directory: The path for outputting the image
:param file_name: The name of the output file
:param image_path: The path to the image
:param image: The actual image (only need image_path or image, not both)
:param conf_text: Display confidence level text above each annotation
:param text_size: Size of the confidence level text
:param label_thickness: Thickness of each label box in pixels
:param label_color: List of custom label colors in the format (B, G, R), 0 <= B,G,R <= 255
:returns: True if successful output, else False
"""
def export_annotated_image(annotations, output_directory, file_name, image_path=None, image=None, conf_text=True, text_scale=1, label_thickness=3, label_colors=None) -> bool:
    if not image_path and not image:
        return False
    if image_path:
        image = Image.open(image_path)
    w, h = image.width, image.height
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    if not label_colors:
        label_colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255)]
    for box in annotations:
        cv2.rectangle(image, (int(box[0]*w), int(box[1]*h)), (int(box[2]*w), int(box[3]*h)), label_colors[int(box[5])], label_thickness)
        if conf_text:
            cv2.putText(image,f"{box[4]:.3f}", (int(box[0]), int(box[1]) - 20), cv2.FONT_HERSHEY_DUPLEX, text_scale, (255,255,255), 2, 2)
    cv2.imwrite(output_directory + file_name, image)

"""Exports an image as cropped tiles

:param output_path: The path to output the cropped images to
:param image_path: The path to the image
:param image: The actual image (only need image_path or image, not both)
:param tile_dim: The dimensions of the tiles
"""
def export_image_crops(output_path, image_path=None, image=None, tile_dim=640):
    if not image_path and not image:
        return False
    if image_path:
        image = Image.open(image_path)
    x, y = image.size
    
    for i in range(tile_dim, x+tile_dim, tile_dim):
        for j in range(tile_dim, y+tile_dim, tile_dim):
            im = image.crop((i - tile_dim, j - tile_dim, i, j))
            im.save(output_path + str(i) + "-" + str(j) + os.path.basename(image_path))