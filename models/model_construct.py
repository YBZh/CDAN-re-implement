from models.alexnet import alexnet
from models.resnet import resnet
def Model_Construct(args):
    if args.arch.find('alexnet') == 0:  ## the required model is vgg structure
        model = alexnet(args)
        return model
    elif args.arch.find('resnet') == 0:
        model = resnet(args)
        return model
    else:
        raise ValueError('the request model is not exist')
