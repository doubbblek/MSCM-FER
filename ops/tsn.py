import torch 
import torch.nn as nn 
import torchvision
from .basic_ops import ConsensusModule
from .transforms import *
from .network import resnet18_msc


######################################################################
# TSN
######################################################################

class TSN(nn.Module):
    def __init__(self, num_class, num_segments,
                 base_model='resnet18',
                 consensus_type='avg',
                 partial_bn=True,
                 print_spec=True, pretrain='imagenet',
                 fc_lr5=False):
        super(TSN, self).__init__()
        self.num_segments = num_segments
        self.reshape = True
        self.consensus_type = consensus_type
        self.pretrain = pretrain
        self.num_class = num_class

        self.base_model_name = base_model
        self.fc_lr5 = fc_lr5

        if print_spec:
            print(("""
    Initializing TSN with base model: {}.
    TSN Configurations:
        num_segments:       {}
        consensus_module:   {}
            """.format(base_model, self.num_segments, consensus_type)))

        self._prepare_base_model(base_model)

        self.consensus = ConsensusModule(consensus_type)

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_base_model(self, base_model):
        print('Add both TSM and MPM')
        if self.base_model_name == 'resnet18':
            print('Using ResNet18')
            self.base_model = resnet18_msc(n_segment=self.num_segments,
                                           pretrained=True if self.pretrain == 'imagenet' else False,
                                           num_classes=self.num_class)
        elif self.base_model_name == 'resnet50':
            print('Using ResNet50')
            self.base_model = resnet50_mpm(n_segment=self.num_segments,
                                           pretrained=True if self.pretrain == 'imagenet' else False)

        self.input_size = 224
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn and mode:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        custom_ops = []
        c3d_weight = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose2d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Conv3d):
                ps = list(m.parameters())
                c3d_weight.append(ps[0])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                if self.fc_lr5:
                    lr5_weight.append(ps[0])
                else:
                    normal_weight.append(ps[0])
                if len(ps) == 2:
                    if self.fc_lr5:
                        lr10_bias.append(ps[1])
                    else:
                        normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
             {'params': c3d_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "conv3d_weight"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
            {'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1,
             'name': "custom_ops"},
            # for fc
            {'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1,
             'name': "lr5_weight"},
            {'params': lr10_bias, 'lr_mult': 10, 'decay_mult': 0,
             'name': "lr10_bias"},
        ]

    def forward(self, input):
        base_out = self.base_model(input.view((-1, 3) + input.size()[-2:]))

        base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
        output = self.consensus(base_out)
        return output.squeeze(1)

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self, flip=True):
        if flip:
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                    GroupRandomHorizontalFlip(is_flow=False)])
        else:
            print('#' * 20, 'NO FLIP!!!')
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66])])


if __name__ == '__main__':
    n = TSN(7, 8,
            base_model='resnet18',
            consensus_type='avg',
            partial_bn=False,
            pretrain='imagenet',
            fc_lr5=False,)
    # n = torch.nn.DataParallel(n)
    # n.load_state_dict(torch.load('checkpoint/mc_1gate_dfew_resnet18_avg_segment8_e40/ckpt.best.pth.tar', map_location='cpu')['state_dict'])
    
    a = n(torch.rand((16, 3, 224, 224)))
    print(a.size())
    n.get_optim_policies()
    # macs, params = profile(n, inputs=(torch.rand(8, 3, 224, 224),))
    # print(n)
    # print(macs/1000000)