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
from xai.common.object_detection_models import YOLOv5,FasterRCNN
from xai.common._utils import process_image, save_image, draw_detections, renormalize_cam_in_bounding_boxes
#############################################################################################

################################
# set device
################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

################################
# Get image & process them
################################
path = "data/toy_examples/"
image_name = "stop.png"
tensor, rgb_img, img_norm_np = process_image(path+image_name)

################################
# Model where XAI is to be applied
################################
object_detection_model = YOLOv5(device)
input_to_model = rgb_img
    
results = object_detection_model.infer(input_to_model)
boxes, colors, names, labels = object_detection_model.postprocess_output(model_output=results, confidence_threshold=0.2)
detections = draw_detections(boxes=boxes,colors=colors,names=names,img=rgb_img)
save_image(img=detections,name='detection.jpg')

################################
# Create and apply the CAM model
################################
# target_layers = [
#     object_detection_model.model.model.model.model[-1].m[0], 
#     object_detection_model.model.model.model.model[-1].m[1],
#     object_detection_model.model.model.model.model[-1].m[2],    
# ]
        # target_layers = [
#                 model.model.model.model[-8], # menor resolucion 
#                 model.model.model.model[-5],  # media
#                 model.model.model.model[-2],  #max
# ]  
target_layers = [
    object_detection_model.model.model.model.model[-2]
]


for _i,t in enumerate(target_layers):
    cam = EigenCAM(model=object_detection_model.model, 
                   target_layers=[t],
                   reshape_transform=None,
                   )
    # Generate CAM for the image
    grayscale_cam = cam(tensor)
    grayscale_cam = grayscale_cam[0, :, :]  
    cam_image = show_cam_on_image(img_norm_np, grayscale_cam, use_rgb=True)
    save_image(cam_image,f'detections_w_cam_s{_i}.jpg')

    ################################
    # REMOVE THE HEATMAP OUT OF THE BOUNDING BOXES
    ################################
    # Apply the renormalized CAM on the bounding boxes
    renormalized_cam_image = renormalize_cam_in_bounding_boxes(boxes, colors, names, img_norm_np, grayscale_cam)
    save_image(renormalized_cam_image,f'detections_w_cam_normalized_s{_i}.jpg')

    # Concatenate the original image, CAM image, and renormalized CAM image for comparison
    comparison_image = np.hstack((rgb_img, cam_image, renormalized_cam_image))
    save_image(comparison_image,f'detections_s{_i}_all.jpg')




"""
# ***SCORE-CAM***
for _i,t in enumerate(target_layers):
    cam = ScoreCAM(model=object_detection_model.model, 
                   target_layers=t,
                   reshape_transform=None,
                   )
    # Generate CAM for the image
    grayscale_cam = cam(tensor)
    grayscale_cam = grayscale_cam[0, :, :]  
    cam_image = show_cam_on_image(img_norm_np, grayscale_cam, use_rgb=True)
    save_image(cam_image,f'detections_w_cam_s{_i}.jpg')

    ################################
    # REMOVE THE HEATMAP OUT OF THE BOUNDING BOXES
    ################################
    # Apply the renormalized CAM on the bounding boxes
    renormalized_cam_image = renormalize_cam_in_bounding_boxes(boxes, colors, names, img_norm_np, grayscale_cam)
    save_image(renormalized_cam_image,f'detections_w_cam_normalized_s{_i}.jpg')

    # Concatenate the original image, CAM image, and renormalized CAM image for comparison
    comparison_image = np.hstack((rgb_img, cam_image, renormalized_cam_image))
    save_image(comparison_image,f'detections_s{_i}_all.jpg')
"""
# pipeline for CAM based methods:
# FORWARD --> COMPUTE CAM PER LAYER --> get_cam_image -- get_cam_weights