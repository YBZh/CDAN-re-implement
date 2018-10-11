import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch

__all__ = ['AlexNet', 'alexnet']

model_urls = {
    'alexnet': 'http://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class Alexnet(nn.Module):
    def __init__(self, args=False, num_classes=1000):
        super(AlexNetFc, self).__init__()
        model_alexnet = alexnet(args=args)
        #self.features = model_alexnet.features
        self.features_fix = nn.Sequential()
        for i in range(8):
            self.features_fix.add_module("extractor_fix" + str(i), model_alexnet.features[i])
        self.features_finetune = nn.Sequential()
        for i in range(9,12):
            self.features_finetune.add_module("extractor_finetune" + str(i), model_alexnet.features[i])
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier" + str(i), model_alexnet.classifier[i])
        self.__in_features = model_alexnet.classifier[6].in_features
        self.nfc = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.features_fix(x)
        x = self.features_finetune(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        x = self.nfc(x)
        return x



def alexnet(args, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if args.pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, args.numclass)
    return model