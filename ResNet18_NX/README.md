# Instructions for Inference on Nvidia Jetson NX

## Clone the repo to NX
```
https://github.com/sli0111/MIDS-251-2021-Final-Project.git
```

## Build docker image
```
docker build -t final_project -f Dockerfile .
```

## Enable xhost on NX to allow USB camera to take pictures inside Docker container later
```
xhost +
```

## Spin up the docker container
```
docker run -it --rm --runtime nvidia -p 8888:8888 --privileged -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix  --name scanner final_project bash
```

## Allow s3 bucket access 
Images will be classified on stored on target s3 bucket.  Configure aws
```
aws configure
```

## Update bucket name
```
vim.tiny ResNet18_detector.py
```

## Run detector
Show the camerma an image --> press s ---> click on image ---> y/n
```
python 3 ResNet18_detector.py
```

