import requests
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
import torchvision.transforms as transforms  # Torchvision for image transformations
from scipy.ndimage.filters import gaussian_filter
import torch

def process_image(path):
    if 'https' in path:
        # Download/Load image
        image_url = path #"https://upload.wikimedia.org/wikipedia/commons/f/f1/Puppies_%284984818141%29.jpg"
        img = download_and_process_image(image_url)
    else:
        img = np.array(Image.open(path)) 

    # process image (assuume it comes a numpy)
    img = cv2.resize(img, (640, 640))  # Resize image
    rgb_img = img.copy() #store it for later!
    # Normalize image
    img = np.float32(img) / 255  
    # Convert image to PyTorch tensor and add batch dim on idx=0
    transform = transforms.ToTensor()
    tensor = transform(img).unsqueeze(0)  
    return tensor, rgb_img, img

def download_and_process_image(url):
    """
    Downloads an image from the given URL and processes it.

    Args:
    url (str): URL of the image to download.
    size (tuple): The size to which the image is resized.

    Returns:
    numpy.ndarray: The processed image.
    """
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    # Send a GET request to the image URL
    response = requests.get(url, headers=headers)
    # Check if the request was successful
    if response.status_code == 200:
        # Open the image from the bytes-like object
        img = Image.open(BytesIO(response.content))
        return np.array(img)
    else:
        print(f"Failed to download the image. Status code: {response.status_code}")
        return None

def draw_detections(boxes,colors,names,img):
    
    for box, color, name in zip(boxes, colors, names):
        xmin, ymin, xmax, ymax = box
        # Draw rectangle (bounding box)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        # Put text (object name)
        cv2.putText(img, name, (xmin, ymin - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, lineType=cv2.LINE_AA)
    return img

# Function to renormalize CAM within bounding boxes
def renormalize_cam_in_bounding_boxes(boxes, colors, names, image_float_np, grayscale_cam):
    """Normalize the CAM to be in the range [0, 1] 
    inside every bounding boxes, and zero outside of the bounding boxes."""
    
    renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
    
    for x1, y1, x2, y2 in boxes:
        # Normalize CAM within each bounding box
        renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy()) 
           
    renormalized_cam = scale_cam_image(renormalized_cam)  # Scale the entire CAM
    # Overlay the CAM on the image
    eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
    # Draw the bounding boxes on the CAM overlay image
    image_with_bounding_boxes = draw_detections(boxes, colors, names, eigencam_image_renormalized)
    return image_with_bounding_boxes

def save_image(img,name):
    """Takes as input a numpy value and stores an image"""
    # Display image with detections
    detections_image = Image.fromarray(img)  
    # save image
    detections_image.save(name)
    
    
def gkern(klen, nsig):
    """Returns a Gaussian kernel array.
    Convolution with it results in image blurring."""
    # create nxn zeros
    inp = np.zeros((klen, klen))
    # set element at the middle to one, a dirac delta
    inp[klen//2, klen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((3, 3, klen, klen))
    kern[0, 0] = k
    kern[1, 1] = k
    kern[2, 2] = k
    return torch.from_numpy(kern.astype('float32'))

def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)
