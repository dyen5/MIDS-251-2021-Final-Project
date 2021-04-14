
Step 1:
Train model in aws.  This will output model weights.  Any changes to model architecture need to be reflected in the ct_cnn_detector.py program.  Once done with training, move the weights into the NX folder.  The ct_cnn_detector.py program should already be there.

Step 2:
Create folder working folder on NX.  Load Dockerfile in this Repo (NX folder) into that working directory.  Change into that working folder and build docker image: docker build -t final_project -f Dockerfile .

Step 3:
Move model weights and python program which build out model, take picture of CT scan using webcam, and then runs inference on that image:
scp -i key_pair_13_jan.pem ubuntu@<public_ip_address>:/home/ubuntu/MIDS-251-2021-Final-Project/NX/CNN_Weights_Run_1 ~/v2/Final_Project/v2
scp -i key_pair_13_jan.pem ubuntu@<public_ip_address>:/home/ubuntu/MIDS-251-2021-Final-Project/NX/ct_cnn_detector.py ~/v2/Final_Project/v2
Note: For this to work, the pem file will need to be added as an ssh key on the NX (see HW2)

Step 4:
On NX, run: xhost +

Step 5: On NX, spin up docker instance: docker run -it --rm --runtime nvidia -p 8888:8888 --privileged -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix  --name scanner final_project bash

Step 6: 
Execute inference program: python3 ct_cnn_detector.py

Step 7:
There are several points where text input is required to proceed with the program.  The first one is to cause the webcam to take a snapshot of a CT scan.  This should cause a window to come up with the screenshot.  If it looks, good, click on the window and press and key to close the window, and then enter "y" to proceed with cropping out any whitespace around the image.  Another window will pop up after it performs that operation.  If it looks like it performed that correctly, again, click on the window and press any key to close, and then type "y" to have the image resized and then run the resized image through the model.  The model should ouput a classificaiton with the associated probability and then prompt for another scan (so it's possible to keep scanning images).


Here's some code to pad an image of a CT scan in a jupyter notebook.  This will help the cropping program I wrote up which isolates the ct scan in a webcam snapshot:
def pad_image(img, pad_size, pixel_brightness):
  stacked1 = np.vstack( (np.full((pad_size,img.shape[1]), pixel_brightness), img, np.full((pad_size,img.shape[1]), pixel_brightness)) )
  stacked2 = np.hstack( (np.full((stacked1.shape[0],pad_size*2), pixel_brightness), stacked1, np.full((stacked1.shape[0],pad_size*2), pixel_brightness) ) ) 
  return stacked2
