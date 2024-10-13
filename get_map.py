import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map
from yolo import YOLO

import os

if __name__ == "__main__":
    #------------------------------------------------------------------------------------------------------------------#
    # map_mode is used to specify the content of the file's runtime calculation.
    # map_mode is 0 to represent the entire map calculation process, including obtaining the prediction results, obtaining the ground truth boxes, and calculating the VOC_map.
    # map_mode is 1 to represent obtaining only the prediction results.
    # map_mode is 2 to represent obtaining only the ground truth boxes.
    # map_mode is 3 to represent calculating only the VOC_map.
    # map_mode is 4 to represent calculating the 0.50:0.95 map for the current dataset using the COCO toolkit. You need to obtain the prediction results, obtain the ground truth boxes, and install pycocotools to do this.
    #-------------------------------------------------------------------------------------------------------------------#
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    map_mode        = 0
    #--------------------------------------------------------------------------------------#
    #   The classes_path here is used to specify the categories for which the VOC_map needs to be measured.
    #   In most cases, the classes_path used for training and prediction should be the same.
    #--------------------------------------------------------------------------------------#
    classes_path    = 'model_data/insulator_classes.txt'
    #--------------------------------------------------------------------------------------#
    # MINOVERLAP is used to specify the desired mAP@0.x
    #--------------------------------------------------------------------------------------#
    MINOVERLAP      = 0.5
    confidence      = 0.001
    #--------------------------------------------------------------------------------------#
    #  non-maximum suppression value
    #--------------------------------------------------------------------------------------#
    nms_iou         = 0.5
    #---------------------------------------------------------------------------------------------------------------#
    #   By default, the Recall and Precision values calculated by this code represent the Recall and Precision values when the threshold value is 0.5 (which is defined as score_threhold here).
    #---------------------------------------------------------------------------------------------------------------#
    score_threhold  = 0.5
    #-------------------------------------------------------#
    #   map_vis is used to specify whether VOC_map computation visualization should be enabled.
    #-------------------------------------------------------#
    map_vis         = False
    #-------------------------------------------------------#
    # Point to the folder where the VOC dataset is located
    # Defaults to the VOC dataset located in the root directory.
    #-------------------------------------------------------#
    VOCdevkit_path  = 'VOCdevkit'
    #-------------------------------------------------------#
    #   The output folder, by default, is named "map_out".
    #-------------------------------------------------------#
    map_out_path    = 'map_out'

    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/test.txt")).read().strip().split()

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names, _ = get_classes(classes_path)

    if map_mode == 0 or map_mode == 1:
        print("Load model.")
        yolo = YOLO(confidence = confidence, nms_iou = nms_iou)
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
            image       = Image.open(image_path)
            if map_vis:
                image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".jpg"))
            yolo.get_map_txt(image_id, image, class_names, map_out_path)
        print("Get predict result done.")
        
    if map_mode == 0 or map_mode == 2:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            with open(os.path.join(map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
                root = ET.parse(os.path.join(VOCdevkit_path, "VOC2007/Annotations/"+image_id+".xml")).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult')!=None:
                        difficult = obj.find('difficult').text
                        if int(difficult)==1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue
                    bndbox  = obj.find('bndbox')
                    left    = bndbox.find('xmin').text
                    top     = bndbox.find('ymin').text
                    right   = bndbox.find('xmax').text
                    bottom  = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        get_map(MINOVERLAP, True, score_threhold = score_threhold, path = map_out_path)
        print("Get map done.")

    if map_mode == 4:
        print("Get map.")
        get_coco_map(class_names = class_names, path = map_out_path)
        print("Get map done.")
