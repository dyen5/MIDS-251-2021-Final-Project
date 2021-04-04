# ------------------------------------------------------------------------
## Libraries
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import models, transforms
import cv2
import PIL
import os
import boto3
from datetime import datetime

# -------------------------------------------------------------------------
## **USER UPDATE**
## Path to Pretrained Model 
PATH = '/apps/w251_transfer_learning_weights'

## Path to saved images
PATH_save = '/apps/saved_images/'
if not os.path.exists(PATH_save):
    os.makedirs(PATH_save)

## s3 bucket
bucket = 'w251-covidx-ct/Inference/'

# -------------------------------------------------------------------------
## Load Pretrained Model
# to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load pretratined model into resnet18
model_tl = models.resnet18(pretrained=True)
num_ftrs = model_tl.fc.in_features

# Load pretrained weights into FC layer
model_tl.fc = nn.Linear(num_ftrs, 3)
model_tl.load_state_dict(torch.load(PATH))

# Push model to GPU
model_tl = model_tl.to(device)

#---------------------------------------------------------------------------
## Define Prediction Function
def prediction(model, image):
    
    # Index for classes
    class_names = ['Covid', 'Normal', 'Pneumonia']
    
    # Put model to evaluation mode
    model.eval()  

    # Transform image for the model
    data_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    inputs = data_transform(image) 
    
    # Make 4D for Resnet: (batch, channel, width, height)
    inputs = inputs.cuda().float().unsqueeze(0)   
    
    # To Record inference time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    # Prediction with model - outputs class and probability
    with torch.no_grad():
        start.record()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        prob = nn.functional.softmax(outputs, dim=1)
        end.record()
        
        # Waits for everything to finish running
        torch.cuda.synchronize()
        print('results....................................................')
        print('inference time: %0.0f ms' %start.elapsed_time(end))
        print(class_names[preds[0]], ': %0.3f' %prob[0][preds[0]].item())
        
        return class_names[preds[0]], round(prob[0][preds[0]].item(), 3)

# --------------------------------------------------------------------------------
## Padding for Image
def isolate_image(img):
    changes = []
    first_value = np.mean(img[0])
    for i in range(img.shape[0]):
        second_value = np.mean(img[i])
        change = second_value - first_value
        changes.append(change)
        first_value = second_value

    top_border_index = np.argmin(changes)
    bottom_border_index = np.argmax(changes)

    changes = []
    first_value = np.mean(img[:,0])
    for i in range(img.shape[1]):
        second_value = np.mean(img[:,i])
        change = second_value - first_value
        changes.append(change)
        first_value = second_value

    left_border_index = np.argmin(changes)
    right_border_index = np.argmax(changes)

    return top_border_index, bottom_border_index, left_border_index, right_border_index

# ------------------------------------------------------------------------------------
## Process Image
while True:
    entry = input('enter "s" to scan image for prediction or "q" to quit: ')
    while entry not in ['s','q']:
        entry = input('enter "s" to scan image for prediction or "q" to quit: ')
    if entry == 's':
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
#        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
#        cv2.imshow('image', gray)
        cv2.imshow('image', frame)
        cv2.waitKey(0)
        cap.release()
        cv2.destroyAllWindows()
#        response = input('Is image acceptable for processing?  Enter "y" for Yes or "n" for No ')
#        while response not in ['y','n']:
#            response = input('Is image acceptable for processing?  Enter "y" for Yes or "n" for No ')
#        if response == 'y':
#            try:
#                top_border_index, bottom_border_index, left_border_index, right_border_index = #isolate_image(gray)
#                cropped = gray[top_border_index:bottom_border_index, 
#        else:
#            continue
#        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
#        cv2.imshow('image', cropped)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
        response2 = input('Is image acceptable for resizing and prediction?  Enter "y" for Yes or "n" for No ')
        while response2 not in ['y','n']:
            response2 = input('Is image acceptable for resizing and prediction?  Enter "y" for Yes or "n" for No ')
        if response2 == 'y':
        
            # Loads image from camera and performs prediction
            img = PIL.Image.fromarray(frame).convert('RGB')
            pred, prob = prediction(model_tl, img)
            results = str(pred) + ': ' + str(prob)
            
            # Stores image to s3 with datetime 
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            filename = dt_string + "-" + pred + '.png'
            filepath = PATH_save + filename
            
            # Save image locally
            cv2.imwrite(filepath, frame)
            
            # To s3                       
            s3 = boto3.client('s3')
            s3.upload_file(filepath, bucket, filename)
            print("results uploaded to s3......................................")
            
            # Display results locally
            cv2.namedWindow(results, cv2.WINDOW_NORMAL)
            cv2.imshow(str(results), frame)
            cv2.waitKey(0)
            cap.release()
            cv2.destroyAllWindows()

        else:
            continue

    else:
        break
        
        
        
        
        
