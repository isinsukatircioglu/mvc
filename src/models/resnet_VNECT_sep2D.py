import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0,'../')

import training
import IPython

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetTwoStream(nn.Module):

    def __init__(self, block, layers, input_key='img_crop', output_keys=['3D','2D_heat'],
                 num_scalars=1000, num_maps=17, input_width=256, num_classes=17*3, 
                 net_type="simple", dropout=[0,0]):
        self.output_keys = output_keys
        if net_type  in ["simple","high_res"]:
            self.independentStream = True
            print("Creating simple network with splitted streams")
        else:
            self.independentStream = False
            print("Creating complex network with fusion")
        self.input_key = input_key
        
        self.inplanes = 64
        self.dropout = dropout
        super(ResNetTwoStream, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block,  64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=1)
        self.layer4_reg = self._make_layer(block, 256, layers[3], stride=1)
#        self.avgpool = nn.AvgPool2d(7)
#        self.fc = nn.Linear(512 * block.expansion, num_classes)

        
#         self.toMaps = nn.Sequential(
#                         nn.Conv2d(256* block.expansion, 256, kernel_size=1, stride=1, padding=0, bias=True),
#                         nn.BatchNorm2d(256),
#                         nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
#                         nn.BatchNorm2d(128),
#                         nn.ReLU(inplace=True),
#                         nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False),
#                         nn.BatchNorm2d(256),
#                         nn.ConvTranspose2d(256,128, kernel_size=4, stride=2, padding=1, bias=False),
#                         nn.BatchNorm2d(128),
#                         nn.ReLU(inplace=True),
#                         nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
#                         nn.BatchNorm2d(128),
#                         nn.ReLU(inplace=True),
#                         nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
#                         nn.BatchNorm2d(64),
#                         nn.ReLU(inplace=True),
#                         nn.Conv2d(64, num_maps, kernel_size=3, stride=1, padding=1, bias=False),
#                         nn.BatchNorm2d(num_maps),
#                         nn.ReLU(inplace=True)
#                         )

        self.toMapsLarge = nn.Sequential(
                        nn.Conv2d(256* block.expansion, 256, kernel_size=1, stride=1, padding=0, bias=True),
                        nn.BatchNorm2d(256),
                        nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False),
                        nn.BatchNorm2d(256),
                        nn.ConvTranspose2d(256,128, kernel_size=4, stride=2, padding=1, bias=False), # upsampling
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.ConvTranspose2d(128,128, kernel_size=4, stride=2, padding=1, bias=False), # upsampling
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64, num_maps, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(num_maps),
                        nn.ReLU(inplace=True)
                        )

        if not self.independentStream:
            self.mapToVec = nn.Sequential(
                            nn.Conv2d(num_maps, 56, kernel_size=5, stride=2, padding=0, bias=False),
                            nn.BatchNorm2d(56),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(56, 128, kernel_size=5, stride=2, padding=0, bias=False),
                            nn.BatchNorm2d(128),
                            nn.Sigmoid(),
                            )
            
    
            self.l4_heat_toVec = nn.Sequential(
                            nn.Conv2d(256* block.expansion, 512, kernel_size=3, stride=1, padding=0, bias=True),
                            nn.BatchNorm2d(512),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(512, 128, kernel_size=5, stride=2, padding=0, bias=False),
                            nn.BatchNorm2d(128),
                            nn.Sigmoid(),
                )
        self.l4_reg_toVec = nn.Sequential(
                        nn.Conv2d(256* block.expansion, 512, kernel_size=3, stride=1, padding=0, bias=True),
                        nn.BatchNorm2d(512),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(512, 128, kernel_size=5, stride=2, padding=0, bias=False),
                        nn.BatchNorm2d(128),
                        nn.Sigmoid(),
            )
         
        # size computation of fc input: /16 in resnet, /2 in toMap, -3 in map since no padding but 3x3 (-1) and 5x5 (-2) kernels
        l4_vec_width     = int(input_width/32)-3 
        l4_vec_dimension = 128*l4_vec_width*l4_vec_width
        heat_vec_width     = int(input_width/32)-3
        heat_vec_dimension = 128*heat_vec_width*heat_vec_width

        if self.independentStream:
            self.fc = nn.Linear(l4_vec_dimension, num_classes)
        else:
            self.fc = nn.Linear(2*l4_vec_dimension + heat_vec_dimension, num_classes)
           
#         self.fc = nn.Sequential(
#             nn.Dropout(p=0.5),
#             nn.Linear(2*l4_vec_dimension + heat_vec_dimension, num_classes)
#         )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x_dict):
        x = x_dict[self.input_key]
        
        x = self.conv1(x) # size /2
        x = self.bn1(x)
        x = self.relu(x) 
        x = self.maxpool(x) # size /2

        x = self.layer1(x) 
        x = self.layer2(x)# size /2
        x = self.layer3(x)# size /2
        
        # heatmap stream
        x_h = self.layer4(x)
        #h = self.toMaps(x_h)
        h = self.toMapsLarge(x_h)
                
        # regression stream
        x_r = self.layer4_reg(x)
        
        if self.independentStream:
            if self.dropout is not None and self.dropout[0]>0:
                x_r = F.dropout(x_r, p=self.dropout[0], training = self.training)
            f_r = self.l4_reg_toVec(x_r)
            f_lin = f_r.view(f_r.size(0), -1) # 1D per batch
            
            # dropout
            if self.dropout is not None and self.dropout[1]>0:
                f_lin = F.dropout(f_lin, p=self.dropout[1], training = self.training)
           
            p = self.fc(f_lin)
            #print('f_lin.size()',f_lin.size())
            return {self.output_keys[0]: p, self.output_keys[1]: h} #{'3D': p, '2d_heat': h}
        else:
            # fusion part
            f_r = self.l4_reg_toVec(x_r)
            f_h = self.l4_heat_toVec(x_h)
            f_m = self.mapToVec(h)
            f = torch.cat([f_r, f_h, f_m],1)
    
    #        x = self.avgpool(x)
    #        x = x.view(x.size(0), -1)
    #        x = self.fc(x)
    
            # fully connected part
            f = f.view(f.size()[0],-1)
            p = self.fc(f)

        return {self.output_keys[0]: p, self.output_keys[1]: h} #{'3D': p, '2d_heat': h}


def resnet50(pretrained=False, path_unsup=None, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
#    model = ResNet(Bottleneck, [3, 4, 6, 1], **kwargs)
    model = ResNetTwoStream(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        if path_unsup is not None:
            print("RESNET 50 vnect")
            print("Loading Unsupervised Pretraining weights...")
            unsup_pretrained_dict = torch.load(path_unsup)
            training.utils.transfer_partial_weights(unsup_pretrained_dict, model, submodule=1)
        else:
            print("Loading image net weights...")
            training.utils.transfer_partial_weights(model_zoo.load_url(model_urls['resnet50']), model)
            print("Done image net weights...")
            #model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetTwoStream(Bottleneck, [3, 4, 23, 1], **kwargs)
    if pretrained:
        print("Loading image net weights...")
        training.utils.transfer_partial_weights(model_zoo.load_url(model_urls['resnet101']), model)
        print("Done image net weights...")
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetTwoStream(Bottleneck, [3, 8, 36, 1], **kwargs)
    if pretrained:
        print("Loading image net weights...")
        training.utils.transfer_partial_weights(model_zoo.load_url(model_urls['resnet152']), model)
        print("Done image net weights...")
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
