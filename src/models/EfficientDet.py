"""
Efficientdet Backbone

"""

import math

import torch
from torch import nn

from models.efficientdet_utils.model import BiFPN, Regressor, Classifier, EfficientNet
from models.efficientdet_utils.utils import Anchors, BBoxTransform, ClipBoxes, invert_affine
from models.efficientdet_utils.utils import postprocess as postprocess_
from models.efficientdet_utils.utils import preprocess as preprocess_


class EfficientDetBackbone(nn.Module):
    def __init__(self, num_classes=80, compound_coef=0, load_weights=False, **kwargs):
        super(EfficientDetBackbone, self).__init__()
        self.compound_coef = compound_coef

        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6]
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8]
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5]
        self.anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5.]
        self.aspect_ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])
        self.num_scales = len(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
        conv_channel_coef = {
            # the channels of P3/P4/P5.
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
        }

        num_anchors = len(self.aspect_ratios) * self.num_scales

        self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters[self.compound_coef],
                    conv_channel_coef[compound_coef],
                    True if _ == 0 else False,
                    attention=True if compound_coef < 6 else False)
              for _ in range(self.fpn_cell_repeats[compound_coef])])

        self.num_classes = num_classes
        self.regressor = Regressor(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                   num_layers=self.box_class_repeats[self.compound_coef])
        self.classifier = Classifier(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                     num_classes=num_classes,
                                     num_layers=self.box_class_repeats[self.compound_coef])

        self.anchors = Anchors(anchor_scale=self.anchor_scale[compound_coef], **kwargs)

        self.backbone_net = EfficientNet(self.backbone_compound_coef[compound_coef], load_weights)

        self.threshold = 0.6
        self.iou_threshold = 0.5
        if("threshold" in kwargs):
            self.threshold = kwargs["threshold"]
        if("iou_threshold" in kwargs):
            self.iou_threshold = kwargs["iou_threshold"]

        return

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs, preprocess=True, postprocess=True, threshold=None,
                iou_threshold=None):

        if(threshold is not None):
            self.threshold = threshold
        if(iou_threshold is not None):
            self.iou_threshold = iou_threshold

        # preprocessing inputs to have a fixed size (e.g., (3, 512, 512))
        if(preprocess):
            if(torch.is_tensor(inputs)):  # converting torch tensor to list of array
                inputs = [inputs[i].cpu().numpy() for i in range(inputs.shape[0])]
            original_inputs, inputs, framed_metas = preprocess_(inputs, max_size=512)
        max_size = inputs.shape[-1]

        # forward pass through the feature extractor backbone
        _, p3, p4, p5 = self.backbone_net(inputs)
        features = (p3, p4, p5)
        features = self.bifpn(features)

        # forward pass through semantic classification and detection coordinates
        # regression head
        regression = self.regressor(features)
        classification = self.classifier(features)
        anchors = self.anchors(inputs, inputs.dtype)

        if(postprocess == False):
            return features, regression, classification, anchors

        # posprocessing outputs (NMS, reshaping, ...)
        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        out = postprocess_(inputs, anchors, regression, classification, regressBoxes,
                           clipBoxes, threshold=self.threshold,
                           iou_threshold=self.iou_threshold)
        out = invert_affine(framed_metas, out)

        # converting to our standarized format
        outs = []
        for o in out:
            # we add 1 to the labels to match coco annotations, where 0 corresponds
            # to the background and 1 to person
            out = {
                'boxes': torch.Tensor(o["rois"]),
                'labels': torch.Tensor(o["class_ids"]).int() + 1,
                'scores': torch.Tensor(o["scores"])
            }
            outs.append(out)
        return outs

    def init_backbone(self, path):
        state_dict = torch.load(path)
        try:
            ret = self.load_state_dict(state_dict, strict=False)
            print(ret)
        except RuntimeError as e:
            print('Ignoring ' + str(e) + '"')
