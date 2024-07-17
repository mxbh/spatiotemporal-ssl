# pverride forward of timms resnet to get the right output format
from timm.models import ResNet, BasicBlock
from timm.models.helpers import build_model_with_cfg

class TimmResNet(ResNet):
    def extract(self, x):
        '''
        For compatibility with usb models.
        '''
        return self.forward_features(x)
    
    def forward(self, x):
        x = self.forward_features(x)
        out = self.forward_head(x)
        return {'logits': out, 'feat': x}


def _create_resnet(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(TimmResNet, variant, pretrained, **kwargs)


def timm_resnet18(num_classes, pretrained=False, **kwargs):
    # kwargs are just for compatibility and ignored
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes)
    return _create_resnet('resnet18', pretrained, **model_args)


def timm_resnet34(num_classes, pretrained=False, **kwargs):
    # kwargs are just for compatibility and ignored
    model_args = dict(block=BasicBlock, layers=[3, 4, 6, 3], num_classes=num_classes)
    return _create_resnet('resnet34', pretrained, **model_args)


def timm_resnet50(num_classes, pretrained=False, **kwargs):
    # kwargs are just for compatibility and ignored
    model_args = dict(block=BasicBlock, layers=[3, 4, 6, 3], num_classes=num_classes)
    return _create_resnet('resnet50', pretrained, **model_args)


def timm_resnet101(num_classes, pretrained=False, **kwargs):
    # kwargs are just for compatibility and ignored
    model_args = dict(block=BasicBlock, layers=[3, 4, 23, 3], num_classes=num_classes)
    return _create_resnet('resnet101', pretrained, **model_args)
