import cv2
import time
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt

try:
    os.mkdir("Output_images")
    print("A new file has been created.")
except:
    print("A file with the specified name already exists.")

input_image_path = "/home/kamber/Desktop/KamberAgcan/Python/GitHub/Yolov3Training/images"
output_image_path = "/home/kamber/Desktop/KamberAgcan/Python/GitHub/Yolov3Training/Output_images"
yolo_cfg = "/home/kamber/Desktop/KamberAgcan/Python/Yolov3/darknet-master/darknet-master/cfg/yolov3.cfg"
coco_names = "/home/kamber/Desktop/KamberAgcan/Python/Yolov3/darknet-master/darknet-master/data/coco.names"
yolo_weights = "/home/kamber/Desktop/KamberAgcan/Python/Yolov3/darknet-master/darknet-master/yolov3.weights"
confidence_threshold = 0.5
nms_threshold = 0.5

def load_input_image(image_path):
    test_img = cv2.imread(image_path)
    h, w, _ = test_img.shape
    return test_img, h, w

def yolov3(yolo_weights, yolo_cfg, coco_names):
    net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
    classes = open(coco_names).read().strip().split("\n")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

def perform_detection(net, img, output_layers, w, h, confidence_threshold):
    blob = cv2.dnn.blobFromImage(img, 1 / 255., (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)
    boxes = []
    confidences = []
    class_ids = []
    
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Object is deemed to be detected
            if confidence > confidence_threshold:            
                center_x, center_y, width, height = list(map(int, detection[0:4] * [w, h, w, h]))
                top_left_x = int(center_x - (width / 2))
                top_left_y = int(center_y - (height / 2))
                boxes.append([top_left_x, top_left_y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return boxes, confidences, class_ids

def draw_boxes(boxes, confidences, class_ids, classes, img, colors, confidence_threshold, NMS_threshold):
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, NMS_threshold)
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]          
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(classes[class_ids[i]], confidences[i])
            #cv2.putText(img, text, (x, y - 5), FONT, 0.5, color, 2)
            k = cv2.putText(img, text, (x, y - 5), FONT, 0.5, color, 2)
    return img
    print(k)

def detection_image_file(image_path, yolo_weights, yolo_cfg, coco_names, confidence_threshold, nms_threshold):
    img, h, w = load_input_image(image_path)
    net, classes, output_layers = yolov3(yolo_weights, yolo_cfg, coco_names)
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    boxes, confidences, class_ids = perform_detection(net, img, output_layers, w, h, confidence_threshold)
    img = draw_boxes(boxes, confidences, class_ids, classes, img, colors, confidence_threshold, nms_threshold)
    cv2.imwrite(output_image_path + "/" + image_path.split("/")[-1],img)
    
def saving_detected_images(input_image_path, yolo_weights, yolo_cfg, coco_names, confidence_threshold, nms_threshold):  
    input_images = os.listdir(input_image_path)
    image_path = []
    image_path2 = [] 
    
    for dirname, _, filenames in os.walk(input_image_path):
        for filename in filenames:
            image_path2.append(os.path.join(dirname, filename)) 
    image_path2 = np.asarray(image_path2) 

    for i in range(len(input_images)): 
        input_images[i]
        image_path.append(image_path2[i])  
        detection_image_file(image_path[i], yolo_weights, yolo_cfg, coco_names, confidence_threshold, nms_threshold)  
    print("Detected ""{}"" images has been saved in Output_image file.".format(len(input_images)))


saving_detected_images(input_image_path, yolo_weights, yolo_cfg, coco_names, confidence_threshold, nms_threshold)