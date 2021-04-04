FROM w251/l4t-ml:r32.4.4-py3

RUN pip3 install facenet-pytorch
RUN pip3 install numpy
RUN pip3 install torch

RUN git clone https://github.com/timesler/facenet-pytorch.git facenet_pytorch

WORKDIR /apps
#COPY *.xml ./
COPY * ./
CMD export NO_AT_BRIDGE=1
