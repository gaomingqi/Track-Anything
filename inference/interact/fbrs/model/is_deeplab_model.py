import torch
import torch.nn as nn

from .ops import DistMaps
from .modeling.deeplab_v3 import DeepLabV3Plus
from .modeling.basic_blocks import SepConvHead


def get_deeplab_model(backbone='resnet50', deeplab_ch=256, aspp_dropout=0.5,
                      norm_layer=nn.BatchNorm2d, backbone_norm_layer=None,
                      use_rgb_conv=True, cpu_dist_maps=False,
                      norm_radius=260):
    model = DistMapsModel(
        feature_extractor=DeepLabV3Plus(backbone=backbone,
                                        ch=deeplab_ch,
                                        project_dropout=aspp_dropout,
                                        norm_layer=norm_layer,
                                        backbone_norm_layer=backbone_norm_layer),
        head=SepConvHead(1, in_channels=deeplab_ch, mid_channels=deeplab_ch // 2,
                         num_layers=2, norm_layer=norm_layer),
        use_rgb_conv=use_rgb_conv,
        norm_layer=norm_layer,
        norm_radius=norm_radius,
        cpu_dist_maps=cpu_dist_maps
    )

    return model


class DistMapsModel(nn.Module):
    def __init__(self, feature_extractor, head, norm_layer=nn.BatchNorm2d, use_rgb_conv=True,
                 cpu_dist_maps=False, norm_radius=260):
        super(DistMapsModel, self).__init__()

        if use_rgb_conv:
            self.rgb_conv = nn.Sequential(
                nn.Conv2d(in_channels=5, out_channels=8, kernel_size=1),
                nn.LeakyReLU(negative_slope=0.2),
                norm_layer(8),
                nn.Conv2d(in_channels=8, out_channels=3, kernel_size=1),
            )
        else:
            self.rgb_conv = None

        self.dist_maps = DistMaps(norm_radius=norm_radius, spatial_scale=1.0,
                                  cpu_mode=cpu_dist_maps)
        self.feature_extractor = feature_extractor
        self.head = head

    def forward(self, image, points):
        coord_features = self.dist_maps(image, points)

        if self.rgb_conv is not None:
            x = self.rgb_conv(torch.cat((image, coord_features), dim=1))
        else:
            c1, c2 = torch.chunk(coord_features, 2, dim=1)
            c3 = torch.ones_like(c1)
            coord_features = torch.cat((c1, c2, c3), dim=1)
            x = 0.8 * image * coord_features + 0.2 * image

        backbone_features = self.feature_extractor(x)
        instance_out = self.head(backbone_features[0])
        instance_out = nn.functional.interpolate(instance_out, size=image.size()[2:],
                                                 mode='bilinear', align_corners=True)

        return {'instances': instance_out}

    def load_weights(self, path_to_weights):
        current_state_dict = self.state_dict()
        new_state_dict = torch.load(path_to_weights, map_location='cpu')
        current_state_dict.update(new_state_dict)
        self.load_state_dict(current_state_dict)

    def get_trainable_params(self):
        backbone_params = nn.ParameterList()
        other_params = nn.ParameterList()

        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'backbone' in name:
                    backbone_params.append(param)
                else:
                    other_params.append(param)
        return backbone_params, other_params


