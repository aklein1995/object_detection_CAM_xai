import torch
import numpy as np
import cv2
from PIL import Image
import torchvision
# add the following to avoid ssl issues from the server
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

COCO_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
'hair drier', 'toothbrush']

class BaseObjectDetection:
    def __init__(self, device='cpu'):
        self.COLORS = np.random.uniform(0, 255, size=(80, 3))
        self.device = device
        # Placeholder for the model, to be defined by subclasses
        self.model = None

    def preprocess_input(self, input):
        """Preprocess input image(s) before passing them to the model."""
        raise NotImplementedError("Subclasses should implement this method.")

    def infer(self, preprocessed_input):
        """Run inference on the preprocessed input."""
        raise NotImplementedError("Subclasses should implement this method.")

    def postprocess_output(self, model_output, confidence_threshold):
        """Postprocess the raw output of the model."""
        raise NotImplementedError("Subclasses should implement this method.")

    def detect_objects(self, input, confidence_threshold=0.2):
        """Detect objects in the input image(s)."""
        preprocessed_input = self.preprocess_input(input)
        model_output = self.infer(preprocessed_input)
        detections = self.postprocess_output(model_output, confidence_threshold)
        return detections
    

class YOLOv5(BaseObjectDetection):
    def __init__(self, device='cpu', model_name='yolov5x'):
        super(YOLOv5, self).__init__(device)
        # from ultralytics import YOLO
        # self.model =  YOLO('models/yolov8n-cls.pt') 
        
        # https://pytorch.org/hub/ultralytics_yolov5/
        # TUTORIAL API: https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading/#simple-example
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        self.model.eval().to(self.device)
        
    def preprocess_input(self, input):
        # Implement specific preprocessing for YOLOv5 if necessary
        return input.to(self.device)

    def infer(self, preprocessed_input):
        return self.model(preprocessed_input)

    def postprocess_output(self, model_output, confidence_threshold):
        detections = model_output.pandas().xyxy[0]  # Extract detection results
        boxes, colors, names, labels = [], [], [], []

        for i in range(len(detections)):
            confidence = detections.iloc[i]['confidence']
            if confidence < confidence_threshold:
                continue
            xmin, ymin, xmax, ymax = map(int, detections.iloc[i][['xmin', 'ymin', 'xmax', 'ymax']])
            name = detections.iloc[i]['name']
            category = int(detections.iloc[i]['class'])
            color = self.COLORS[category]

            boxes.append((xmin, ymin, xmax, ymax))
            colors.append(color)
            names.append(name)
            labels.append(None)
            
        return boxes, colors, names, labels
        
class FasterRCNN(BaseObjectDetection):
    def __init__(self, device='cpu'):
        super(FasterRCNN, self).__init__(device)
        # https://pytorch.org/vision/main/models.html --> look at object detection model
        # https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html#torchvision.models.detection.fasterrcnn_resnet50_fpn --> check the inference List Dict format to postprocess
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval().to(self.device)

    def preprocess_input(self, input):
        # Implement specific preprocessing for YOLOv5 if necessary
        return input.to(self.device)

    def infer(self, preprocessed_input):
        return self.model(preprocessed_input)

    def postprocess_output(self, model_output, confidence_threshold):
       
        pred_classes = [COCO_NAMES[i] for i in model_output[0]['labels'].cpu().numpy()]
        pred_labels = model_output[0]['labels'].cpu().numpy()
        pred_scores = model_output[0]['scores'].detach().cpu().numpy()
        pred_bboxes = model_output[0]['boxes'].detach().cpu().numpy()
        
        # boxes, classes, labels, indices = [], [], [], []
        boxes, colors, names, labels = [],[],[],[]
        for i in range(len(pred_scores)):
            
            if pred_scores[i] < confidence_threshold:
                continue
            
            boxes.append(pred_bboxes[i].astype(np.int32))
            colors.append(self.COLORS[pred_labels[i]]) #labels.append(pred_labels[i])
            names.append(pred_classes[i]) #classes.append(pred_classes[i])
            labels.append(pred_labels[i])
        boxes = np.int32(boxes)
        
        return boxes, colors, names, labels
        # return boxes, classes, labels, indices