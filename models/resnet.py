import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
from utils import ReverseLayerF
from collections import OrderedDict

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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

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

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    # modify the structure of the model.
    num_of_feature_map = model.fc.in_features
    model.fc = nn.Linear(num_of_feature_map, 200)

    return model


def resnet34(pretrained=False,args=1):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    num_class_old = args.numclass_old

    print('the num of class old is', num_class_old)
    print('the num of class new is', args.numclass_new)
    model = ResNet(block=BasicBlock, layers=[3, 4, 6, 3], num_classes=num_class_old)
    if args.pretrained_model:
        print('load the pretrained model', args.pretrained_model)
        checkpoint = torch.load(args.pretrained_model)
        state_dict = checkpoint['state_dict']
        new_state_dict = OrderndDict()   
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    elif pretrained:
        print('begin to load the ImageNet pretrained model')
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    if args.newfc:
        num_of_feature_map = model.fc.in_features
        print('the classifier parameters is going to be re-inilize')
        model.fc = nn.Linear(num_of_feature_map, args.numclass_new)	
    return model

class Share_convs(nn.Module):
    def __init__(self, resnet_conv, convout_dimension, num_class, num_domain, args):
        super(Share_convs, self).__init__()
        self.base_conv = resnet_conv
        for_std1 = pow(3, 1 / 2)
        self.fc = nn.Linear(convout_dimension, num_class)
        self.args = args
        if args.domain_feature == 'original':
            self.domain_classifier = nn.Linear(convout_dimension, num_domain)
        elif args.domain_feature == 'full_bilinear':
            self.softmax = nn.Softmax()
            self.domain_classifier = nn.Linear(convout_dimension * num_class, num_domain)
        elif args.domain_feature == 'random_bilinear':
            self.softmax = nn.Softmax()
            self.domain_classifier = nn.Linear(4096, num_domain)
            self.bilinear_project1 = nn.Linear(convout_dimension, 4096)
            self.bilinear_project2 = nn.Linear(num_class, 4096)
            self.bilinear_project1.weight.data.uniform_(-for_std1, for_std1)
            self.bilinear_project2.weight.data.uniform_(-for_std1, for_std1)
            self.bilinear_project1.bias.data.uniform_(-for_std1, for_std1)
            self.bilinear_project2.bias.data.uniform_(-for_std1, for_std1)
        else:
            raise ValueError('un-recognized domain feature:', args.domain_feature)

    def forward(self, x, alpha):
        x = self.base_conv(x)
        classes = self.fc(x)

        if self.args.domain_feature == 'original':
            x = ReverseLayerF.apply(x, alpha)
            do_classes = self.domain_classifier(x)
        elif self.args.domain_feature == 'full_bilinear':
            x_transform = torch.unsqueeze(x, 2)
            # probability = classes
            probability = self.softmax(classes)
            probability_transform = torch.unsqueeze(probability, 1)
            do_feature = torch.bmm(x_transform, probability_transform)
            do_feature = do_feature.view(do_feature.size(0), -1)
            ############## the normalization for the bilinear operation
            # do_feature = torch.mul(torch.sign(do_feature), torch.sqrt(torch.abs(do_feature)))
            do_feature = torch.sqrt(torch.abs(do_feature) + 1e-12)
            do_feature = F.normalize(do_feature, p=2, dim=1)
            do_feature = ReverseLayerF.apply(do_feature, alpha)
            do_classes = self.domain_classifier(do_feature)
        elif self.args.domain_feature == 'random_bilinear':
            project_x = self.bilinear_project1(x)
            probability = self.softmax(classes)
            project_pro = self.bilinear_project2(probability)
            project_feature = project_x * project_pro
            project_feature = torch.sum(project_feature.view(project_feature.size(0), project_feature.size(1), -1),
                                        dim=2)
            project_feature = torch.mul(torch.sign(project_feature), torch.sqrt(torch.abs(project_feature) + 1e-12))
            project_feature = F.normalize(project_feature, p=2, dim=1)
            project_feature = ReverseLayerF.apply(project_feature, alpha)
            do_classes = self.domain_classifier(project_feature)

        return classes, do_classes

def resnet50(pretrained=False, args=1, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    # # modify the structure of the model.
    # num_of_feature_map = model.fc.in_features
    # model.fc = nn.Linear(num_of_feature_map, args.numclass)

    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        if args.pretrained_checkpoint == '':
            pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
        else:
            print('load the pretrained_checkpoint ,', args.pretrained_checkpoint)
            state_dict_temp = torch.load(args.pretrained_checkpoint)['state_dict']
            pretrained_dict = {}
            for k, v in state_dict_temp.items():
                pretrained_dict[k.replace('module.', '')] = v

    model_dict = model.state_dict()
    pretrained_dict_temp = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict_temp)
    model.load_state_dict(model_dict)

    # domain_classifier = Share_convs(model, 2048, args.numclass_da)
    # source_model = Share_convs(model, 2048, args.numclass_s)
    source_model  = Share_convs(model, 2048, args.numclass_s, args.numclass_da, args)

    if args.pretrained_fc and args.pretrained:
        source_model_dict = source_model.state_dict()
        pretrained_dict_temp1 = {k: v for k, v in pretrained_dict.items() if k in source_model_dict}
        source_model_dict.update(pretrained_dict_temp1)
        source_model.load_state_dict(source_model_dict)

        #target_model.load_state_dict(source_model_dict)

    return source_model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    # modify the structure of the model.
    num_of_feature_map = model.fc.in_features
    model.fc = nn.Linear(num_of_feature_map, 200)

    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    # modify the structure of the model.
    num_of_feature_map = model.fc.in_features
    model.fc = nn.Linear(num_of_feature_map, 200)

    return model


def resnet(args, **kwargs):  # Only the ResNet34 is supported.
    print("==> creating model '{}' ".format(args.arch))
    if args.arch == 'resnet18':
        return resnet18(args.pretrained)
    elif args.arch == 'resnet34':
        return resnet34(args.pretrained, args)
    elif args.arch == 'resnet50':
        return resnet50(args.pretrained, args)
    elif args.arch == 'resnet101':
        return resnet101(args.pretrained)
    elif args.arch == 'resnet152':
        return resnet152(args.pretrained)
    else:
        raise ValueError('Unrecognized model architecture', args.arch)
