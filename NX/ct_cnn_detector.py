import numpy as np
import pickle
import cv2

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
import torch.nn.functional as F
device = torch.device("cuda")



class CNN(nn.Module):

    def __init__(self,
                 filter_sizes,
                 num_filters,
                 fc1_neurons,
                 fc2_neurons):

        super(CNN, self).__init__()

        # Conv Network
        self.conv2d_list = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])

        # RELU activation function
        self.relu =  nn.ReLU()

        # Fully-connected layers
        self.fc1 = nn.Linear(np.sum(num_filters), fc1_neurons)

        if fc2_neurons != None:

            # print('fc2=None')

            self.fc2 = nn.Linear(fc1_neurons, fc2_neurons)

            self.fc3 = nn.Linear(fc2_neurons, 3)

        else:

            self.fc2 = nn.Linear(fc1_neurons, 3)

        #softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs):

        # Apply CNN and ReLU.
        x_conv_list = [self.relu(conv2d(inputs)) for conv2d in self.conv2d_list]



        # Max pooling.
        x_pool_list = [F.max_pool2d(x_conv, kernel_size=x_conv.shape[2])
            for x_conv in x_conv_list]

        # print(x_pool_list[0].shape)
        # print(x_pool_list[0].squeeze(32).shape)
        # print(x_pool_list[0].squeeze(3).shape)
        # Concatenate x_pool_list to feed the fully connected layer.
        x_fc = torch.cat([x_pool.squeeze(2).squeeze(2) for x_pool in x_pool_list],
                         dim=1)

        x = self.fc1(x_fc)
        x = self.relu(x)

        x = self.fc2(x)

        try:
            if self.fc3 != None:

                x = self.relu(x)
                x = self.fc3(x)

        except:
            pass


        probs = self.softmax(x)

        return probs

def predict_CNN(model, image):
  
    """After the completion of each training epoch, measure the model's
    performance on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled
    # during the test time.
    model.eval()

    image_reshaped = torch.from_numpy(image.reshape(1,1,256,256)).float().cuda()

    # Compute logits
    with torch.no_grad():
        probs = model(image_reshaped)
            
        # Get the predictions
        prediction = torch.argmax(probs, dim=1).flatten()

    if prediction == 0:
        prediction_label = 'Diagnosis: Normal'
    elif prediction == 1:
        prediction_label = 'Diagnosis: Pneumonia'
    else:
        prediction_label = 'Diagnosis: COVID-19'

    print(prediction_label, '; Probability of Diagnosis:', np.exp(probs[0][prediction][0].cpu().item()) )


#Model Initializiation
CNN_Model = CNN(filter_sizes = [2, 3, 4], 
                num_filters = [25,25,25],
                fc1_neurons = 200,
                fc2_neurons = 50
                )

# push the model to GPU
CNN_Model = CNN_Model.to(device)



#load weights of best model
CNN_Model.load_state_dict(torch.load('CNN_Weights_Run_1'))



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


while True:
    entry = input('enter "s" to scan image for prediction or "q" to quit: ')
    while entry not in ['s','q']:
        entry = input('enter "s" to scan image for prediction or "q" to quit: ')
    if entry == 's':
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', gray)
        cv2.waitKey(0)
        cap.release()
        cv2.destroyAllWindows()
        response = input('Is image acceptable for processing?  Enter "y" for Yes or "n" for No ')
        while response not in ['y','n']:
            response = input('Is image acceptable for processing?  Enter "y" for Yes or "n" for No ')
        if response == 'y':
            try:
                top_border_index, bottom_border_index, left_border_index, right_border_index = isolate_image(gray)
                cropped = gray[top_border_index:bottom_border_index, left_border_index:right_border_index]
            except:
                print('image could not be cropped correctly')
        else:
            continue
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        response2 = input('Is image acceptable for resizing and prediction?  Enter "y" for Yes or "n" for No ')
        while response2 not in ['y','n']:
            response2 = input('Is image acceptable for resizing and prediction?  Enter "y" for Yes or "n" for No ')
        if response2 == 'y':
            resized = cv2.resize(cropped, (256, 256), interpolation = cv2.INTER_AREA)
            predict_CNN(CNN_Model, resized)
        else:
            continue

    else:
        break


