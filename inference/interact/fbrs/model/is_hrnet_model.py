import torch
import torch.nn as nn

from .ops import DistMaps
from .modeling.hrnet_ocr import HighResolutionNet


def get_hrnet_model(width=48, ocr_width=256, small=False, norm_radius=260,
                    use_rgb_conv=True, with_aux_output=False, cpu_dist_maps=False,
                    norm_layer=nn.BatchNorm2d):
    model = DistMapsHRNetModel(
        feature_extractor=HighResolutionNet(width=width, ocr_width=ocr_width, small=small,
                                            num_classes=1, norm_layer=norm_layer),
        use_rgb_conv=use_rgb_conv,
        with_aux_output=with_aux_output,
        norm_layer=norm_layer,
        norm_radius=norm_radius,
        cpu_dist_maps=cpu_dist_maps
    )

    return model


class DistMapsHRNetModel(nn.Module):
    def __init__(self, feature_extractor, use_rgb_conv=True, with_aux_output=False,
                 norm_layer=nn.BatchNorm2d, norm_radius=260, cpu_dist_maps=False):
        super(DistMapsHRNetModel, self).__init__()
        self.with_aux_output = with_aux_output

        if use_rgb_conv:
            self.rgb_conv = nn.Sequential(
                nn.Conv2d(in_channels=5, out_channels=8, kernel_size=1),
                nn.LeakyReLU(negative_slope=0.2),
                norm_layer(8),
                nn.Conv2d(in_channels=8, out_channels=3, kernel_size=1),
            )
        else:
            self.rgb_conv = None

        self.dist_maps = DistMaps(norm_radius=norm_radius, spatial_scale=1.0, cpu_mode=cpu_dist_maps)
        self.feature_extractor = feature_extractor

    def forward(self, image, points):
        coord_features = self.dist_maps(image, points)

        if self.rgb_conv is not None:
            x = self.rgb_conv(torch.cat((image, coord_features), dim=1))
        else:
            c1, c2 = torch.chunk(coord_features, 2, dim=1)
            c3 = torch.ones_like(c1)
            coord_features = torch.cat((c1, c2, c3), dim=1)
            x = 0.8 * image * coord_features + 0.2 * image

        feature_extractor_out = self.feature_extractor(x)
        instance_out = feature_extractor_out[0]
        instance_out = nn.functional.interpolate(instance_out, size=image.size()[2:],
                                                 mode='bilinear', align_corners=True)
        outputs = {'instances': instance_out}
        if self.with_aux_output:
            instance_aux_out = feature_extractor_out[1]
            instance_aux_out = nn.functional.interpolate(instance_aux_out, size=image.size()[2:],
                                                         mode='bilinear', align_corners=True)
            outputs['instances_aux'] = instance_aux_out

        return outputs

    def load_weights(self, path_to_weights):
        current_state_dict = self.state_dict()
        new_state_dict = torch.load(path_to_weights)
        current_state_dict.update(new_state_dict)
        self.load_state_dict(current_state_dict)

    def get_trainable_params(self):
        backbone_params = nn.ParameterList()
        other_params = nn.ParameterList()
        other_params_keys = []
        nonbackbone_keywords = ['rgb_conv', 'aux_head', 'cls_head', 'conv3x3_ocr', 'ocr_distri_head']

        for name, param in self.named_parameters():
            if param.requires_grad:
                if any(x in name for x in nonbackbone_keywords):
                    other_params.append(param)
                    other_params_keys.append(name)
                else:
                    backbone_params.append(param)
        print('Nonbackbone params:', sorted(other_params_keys))
        return backbone_params, other_params
