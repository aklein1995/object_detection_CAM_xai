# credits: https://jacobgil.github.io/pytorch-gradcam-book/EigenCAM%20for%20YOLO5.html

# Import necessary libraries
import warnings
warnings.filterwarnings('ignore')  # Ignore warnings to clean up output
warnings.simplefilter('ignore')
import torch  # PyTorch, a machine learning library
import numpy as np  # NumPy for numerical operations

from pytorch_grad_cam import EigenCAM, AblationCAM, ScoreCAM,GradCAM  # Grad-CAM implementation
from pytorch_grad_cam.utils.image import show_cam_on_image

# import my own utils --> name it different (for example with subscript _, if not it can report errors)
from xai.common._utils import process_image, save_image, draw_detections, renormalize_cam_in_bounding_boxes
from xai.common.object_detection_models import YOLOv5,FasterRCNN
#############################################################################################

################################
# set device
################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
################################
# Get image & process them
################################
path = "data/"
image_name = "stop.png"
tensor, rgb_img, img_norm_np = process_image(path+image_name)

################################
# Model where XAI is to be applied
################################
name = 'faster'
if name == 'faster':
    object_detection_model = FasterRCNN(device)
    input_to_model = object_detection_model.preprocess_input(tensor)
else:
    object_detection_model = YOLOv5(device)
    input_to_model = rgb_img
    
results = object_detection_model.infer(input_to_model)
boxes, colors, names, labels = object_detection_model.postprocess_output(model_output=results, confidence_threshold=0.2)
detections = draw_detections(boxes=boxes,colors=colors,names=names,img=rgb_img)
save_image(img=detections,name='detection.jpg')

################################
# Create and apply the CAM model
################################
if name == 'faster':
    target_layers = [
        object_detection_model.model.backbone, 
    ] 
else:
    target_layers = [
        object_detection_model.model.model.model.model[-1].m[0], 
        object_detection_model.model.model.model.model[-1].m[1],
        object_detection_model.model.model.model.model[-1].m[2],    
    ]

import torchvision
class FasterRCNNBoxScoreTarget:
    def __init__(self, labels, bounding_boxes, iou_threshold=0.5):
        self.labels = labels
        self.bounding_boxes = bounding_boxes
        self.iou_threshold = iou_threshold

    def __call__(self, model_outputs):
        output = torch.Tensor([0])
        if torch.cuda.is_available():
            output = output.cuda()

        if len(model_outputs["boxes"]) == 0:
            return output

        for box, label in zip(self.bounding_boxes, self.labels):
            box = torch.Tensor(box[None, :])
            if torch.cuda.is_available():
                box = box.cuda()

            ious = torchvision.ops.box_iou(box, model_outputs["boxes"])
            index = ious.argmax()
            if ious[0, index] > self.iou_threshold and model_outputs["labels"][index] == label:
                score = ious[0, index] + model_outputs["scores"][index]
                output = output + score
        return output
     
# specific targets
targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=boxes)]
def fasterrcnn_reshape_transform(x):
    target_size = x['pool'].size()[-2 : ]
    activations = []
    for key, value in x.items():
        activations.append(torch.nn.functional.interpolate(torch.abs(value), target_size, mode='bilinear'))
    activations = torch.cat(activations, axis=1)
    return activations

_i = 0
cam = EigenCAM(model=object_detection_model.model, 
                target_layers=target_layers,
                reshape_transform=fasterrcnn_reshape_transform,
                )

grayscale_cam = cam(tensor, 
                    targets=targets
                )
grayscale_cam = grayscale_cam[0, :] 
cam_image = show_cam_on_image(img_norm_np, grayscale_cam, use_rgb=True)
save_image(cam_image,f'detections_w_cam_s{_i}.jpg')

################################
# REMOVE THE HEATMAP OUT OF THE BOUNDING BOXES
################################
# Apply the renormalized CAM on the bounding boxes
renormalized_cam_image = renormalize_cam_in_bounding_boxes(boxes, colors, names, img_norm_np, grayscale_cam)
save_image(renormalized_cam_image,f'detections_w_cam_normalized_s{_i}.jpg')


# pipeline for CAM based methods:
# FORWARD --> COMPUTE CAM PER LAYER --> get_cam_image -- get_cam_weights