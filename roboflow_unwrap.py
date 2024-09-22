import requests
import json
import cv2
import numpy as np
from unwrap_label import *

def call_roboflow_workflow(image_url):
    workflow_url = "https://detect.roboflow.com/infer/workflows/sriram-kalki/custom-workflow"
    headers = {"Content-Type": "application/json"}
    data = {
        "api_key": "qTssMHObChVIcD3e2Yw8",
        "inputs": {
            "image": {"type": "url", "value": image_url}
        }
    }

    response = requests.post(workflow_url, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        return response.json()
    else:
        print("Error:", response.status_code, response.text)
        return None

def crop_label(image, x, y, width, height, image_scale_x, image_scale_y):
    x = int(x * image_scale_x)
    y = int(y * image_scale_y)
    width = int(width * image_scale_x)
    height = int(height * image_scale_y)
    label_crop = image[int(y - height / 2):int(y + height / 2), int(x - width / 2):int(x + width / 2)]

    return label_crop

def threshold_label(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(binary_image)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    return mask

def find_corner(image, start_x, start_y, x_step, y_step):
    x, y = start_x, start_y
    while 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
        if image[y, x] == 255:
            return (x, y)
        x += x_step
        y += 2 * y_step
    return (start_x,start_y)

def find_midpoints(image):
    mid_col = image.shape[1] // 2 
    first_white_point = None
    last_white_point = None
    
    for y in range(image.shape[0]):
        if image[y, mid_col] == 255:
            first_white_point = (mid_col, y)
            break
    
    for y in range(image.shape[0] - 1, -1, -1):
        if image[y, mid_col] == 255:
            last_white_point = (mid_col, y)
            break
    
    return first_white_point, last_white_point

def process_image(image_url, image_path):
    response_data = call_roboflow_workflow(image_url)
    
    if response_data:
        image = cv2.imread(image_path)
        if len(response_data['outputs'][0]['model_predictions']['predictions']['predictions']) == 0:
            return None
        prediction = response_data['outputs'][0]['model_predictions']['predictions']['predictions'][0]
        width = prediction['width']
        height = prediction['height']
        x = prediction['x']
        y = prediction['y']

        original_width = response_data['outputs'][0]['model_predictions']['predictions']['image']['width']
        original_height = response_data['outputs'][0]['model_predictions']['predictions']['image']['height']
        
        local_image_height, local_image_width, _ = image.shape
        scale_x = local_image_width / original_width
        scale_y = local_image_height / original_height

        cropped_label = crop_label(image, x, y, width, height, scale_x, scale_y)
        
        thresholded_label_image = threshold_label(cropped_label)
        thresholded_label_path = "./thresh.jpg"
        top_left_diagonal = find_corner(thresholded_label_image, 0, 0, 1, 1)
        top_right_diagonal = find_corner(thresholded_label_image, thresholded_label_image.shape[1] - 1, 0, -1, 1)
        bottom_left_diagonal = find_corner(thresholded_label_image, 0, thresholded_label_image.shape[0] - 1, 1, -1)
        bottom_right_diagonal = find_corner(thresholded_label_image, thresholded_label_image.shape[1] - 1, thresholded_label_image.shape[0] - 1, -1, -1)

        top_midpoint, bottom_midpoint = find_midpoints(thresholded_label_image)


        old_points = [top_left_diagonal,top_midpoint,top_right_diagonal,bottom_right_diagonal,bottom_midpoint,bottom_left_diagonal]
        new_points = []
        for c in old_points:
            new_points.append([c[0]/cropped_label.shape[1],c[1]/cropped_label.shape[0]])
            
        print(new_points)
        
        unwrapped_label_image = unwarp_label(cropped_label,new_points)
        return new_points, unwrapped_label_image
    
def unwrap(image_url,image_path):
    return process_image(image_url, image_path)    
    



