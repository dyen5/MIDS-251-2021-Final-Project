FROM w251/l4t-ml:r32.4.4-py3

<<<<<<< HEAD
RUN apt update && apt install -y vim
=======
RUN apt update && install -y vim
>>>>>>> 152e99dcc5d1e8df1b0f9acaa331d88195d8b6ae

RUN pip3 install facenet-pytorch
RUN pip3 install numpy
RUN pip3 install torch
RUN pip3 install torchvision==0.2.2
RUN pip3 install awscli
RUN pip3 install boto3

RUN git clone https://github.com/timesler/facenet-pytorch.git facenet_pytorch

WORKDIR /apps
#COPY *.xml ./
COPY * ./
CMD export NO_AT_BRIDGE=1
