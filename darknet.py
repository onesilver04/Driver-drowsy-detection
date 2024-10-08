# 필요한 library import
from __future__ import division
# util.py작성 후
from util import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

def get_test_input():
    #img = cv2.imread("dog-cycle-car.png")
    img = cv2.imread("phone1.jpeg")
    img = cv2.resize(img, (416,416))          # 입력 크기로 resize 합니다.
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR - > RGB, HxWxC -> CxHxW
    img_ = img_[np.newaxis,:,:,:]/255.0       # (배치를 위한) 0 채널 추가하고 normalize 합니다.
    img_ = torch.from_numpy(img_).float()     # 정수로 변환합니다.
    img_ = Variable(img_)                     # 변수로 변환합니다.
    return img_

# parse_cfg 함수 정의하기, 구성 파일의 경로를 입력으로 받습니다.
def parse_cfg(cfgfile):
    """
    Takes a configuration file

    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list

    """

    file = open(cfgfile, 'r')
    lines = file.read().split('\n')                        # store the lines in a list
    lines = [x for x in lines if len(x) > 0]               # get read of the empty lines
    lines = [x for x in lines if x[0] != '#']              # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]           # get rid of fringe whitespaces

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":               # This marks the start of a new block
            if len(block) != 0:          # If block is not empty, implies it is storing values of previous block.
                blocks.append(block)     # add it the blocks list
                block = {}               # re-init the block
            block["type"] = line[1:-1].rstrip()
        else:
            key,value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks

def create_modules(blocks):
    net_info = blocks[0]     #Captures the information about the input and pre-processing
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        # block의 type을 확인합니다.
        # block에 대한 새로운 module을 생성합니다.
        # module_list에 append 합니다.

        if (x["type"] == "convolutional"):
            # layer에 대한 정보를 얻습니다.
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters= int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # convolutional layer를 추가합니다.
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv)

            # Batch Norm Layer를 추가합니다.
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            # activation을 확인합니다.
            # YOLO에서 Leaky ReLU 또는 Linear 입니다.
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(index), activn)

            # upsampling layer 입니다.
            # Bilinear2dUpsampling을 사용합니다.
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor = 2, mode = "nearest")
            module.add_module("upsample_{}".format(index), upsample)

        # route layer 입니다.
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            # route 시작
            start = int(x["layers"][0])
            # 1개만 존재하면 종료
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            # 양수인 경우
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            # 음수 인 경우
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters= output_filters[index + start]

        # skip connection에 해당하는 shortcut
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

        # YOLO는 detection layer입니다.
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)
        # loop의 마지막에 bookkeeping을 함.
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
    # loop의 전체 종료
    return (net_info, module_list)
class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}   # route layer에 대한 출력값을 저장합니다.

        write = 0
        for i, module in enumerate(modules):
            module_type = (module["type"])

            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)

            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]

                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)


            elif  module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]

            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                # 입력 차원을 얻습니다.
                inp_dim = int (self.net_info["height"])

                # 클래스의 수를 얻습니다.
                num_classes = int (module["classes"])

                # util.py에 있는 predict_transform 함수를 이용하여 변환
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:              # collector가 초기화 되지 않은 경우
                    detections = x
                    write = 1

                else:
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x

        return detections

    def load_weights(self, weightfile):
        # weights file 열기
        fp = open(weightfile, "rb")

        # 첫 5개 값은 header 정보
        # 1. Magor version number
        # 2. Minor version number
        # 3. Subversion number
        # 4, 5. (training 동안) 신경망에 의하여 학습된 이미지
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(fp, dtype = np.float32)

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            # 만약 module_type이 convolutional이면 weights를 불러옵니다.
            # 그렇지 않으면 무시합니다.

            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]


                if (batch_normalize):
                    bn = model[1]

                    # Batch Norm layer의 weight의 수를 얻습니다.
                    num_bn_biases = bn.bias.numel()

                    # weights를 불러옵니다.
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    # 불러온 weights를 모델 weights의 차원으로 변환합니다.
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # data를 model에 복사합니다.
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                # 만약 batch_norm이 not true이면, convolutional layer의 biases를 불러옵니다.
                else:
                    # biases의 수
                    num_biases = conv.bias.numel()

                    # weights 불러오기
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    # 불러온 weights를 model weight의 차원에 맞게 reshape 합니다.
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # 마지막으로 data를 복사합니다.
                    conv.bias.data.copy_(conv_biases)

                # Convolutional layer에 대한 weights를 불러옵니다.
                num_weights = conv.weight.numel()

                # weights를 위와 같이 똑같이 합니다.
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)
                
class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()
