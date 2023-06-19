from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from spc.resnetD import ResNetD
import os
import sys
class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out
class ReparamLargeKernelConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups,
                 small_kernel,
                 small_kernel_merged=False):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        # We assume the conv does not change the feature map size, so padding = k//2. Otherwise, you may configure padding as you wish, and change the padding of small_conv accordingly.
        padding = kernel_size // 2
        if small_kernel_merged:
            # self.lkb_reparam =DDFPack(in_channels=in_channels,kernel_size=kernel_size)
            # print(small_kernel_merged)
            self.lkb_reparam = get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=padding, dilation=1, groups=groups, bias=True)
        else:
            self.lkb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=1, groups=groups)
            # self.lkb_origin=DDFPack(in_channels=in_channels,kernel_size=kernel_size)
            if small_kernel is not None:
                assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=small_kernel,
                                             stride=stride, padding=small_kernel//2, groups=groups, dilation=1)

    def forward(self, inputs):
        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs)
        return out

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, 'small_conv'):
            small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            #   add to the central part
            eq_k += nn.functional.pad(small_k, [(self.kernel_size - self.small_kernel) // 2] * 4)
        return eq_k, eq_b

    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = get_conv2d(in_channels=self.lkb_origin.conv.in_channels,
                                     out_channels=self.lkb_origin.conv.out_channels,
                                     kernel_size=self.lkb_origin.conv.kernel_size, stride=self.lkb_origin.conv.stride,
                                     padding=self.lkb_origin.conv.padding, dilation=self.lkb_origin.conv.dilation,
                                     groups=self.lkb_origin.conv.groups, bias=True)
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')
class pw(nn.Module):
    def __init__(self, in_channels, out_channels, ):
        super(pw, self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels , kernel_size=1,stride=1)
        self.nonlin = nn.ELU()
    def forward(self,x):
        out=self.conv(x)
        out=self.nonlin(out)
        return out
class dw(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size ):
        super(dw, self).__init__()
        self.padd = kernel_size // 2
        # self.conv=DDFPack(in_channels,kernel_size=kernel_size)
        # self.conv =DynamicDWConv(in_channels, kernel_size=kernel_size,stride=1,groups=in_channels,padding=self.padd)
        self.conv = nn.Conv2d(in_channels,out_channels , kernel_size=kernel_size,stride=1,groups=in_channels,padding=self.padd)
        self.nonlin = nn.ELU()

    def forward(self,x):
        out=self.conv(x)
        out=self.nonlin(out)
        return out
def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result
def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    if type(kernel_size) is int:
        use_large_impl = kernel_size > 5
    else:
        assert len(kernel_size) == 2 and kernel_size[0] == kernel_size[1]
        use_large_impl = kernel_size[0] > 5
    has_large_impl = 'LARGE_KERNEL_CONV_IMPL' in os.environ
    if has_large_impl and in_channels == out_channels and out_channels == groups and use_large_impl and stride == 1 and padding == kernel_size // 2 and dilation == 1:
        sys.path.append(os.environ['LARGE_KERNEL_CONV_IMPL'])
        #   Please follow the instructions https://github.com/DingXiaoH/RepLKNet-pytorch/blob/main/README.md
        #   export LARGE_KERNEL_CONV_IMPL=absolute_path_to_where_you_cloned_the_example (i.e., depthwise_conv2d_implicit_gemm.py)
        # TODO more efficient PyTorch implementations of large-kernel convolutions. Pull-requests are welcomed.
        # Or you may try MegEngine. We have integrated an efficient implementation into MegEngine and it will automatically use it.
        from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
        return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)
    else:

        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=groups, bias=bias)
def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    if padding is None:
        padding = kernel_size // 2
    result = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, groups=groups, dilation=dilation)
    result.add_module('nonlinear', nn.ELU())
    return result

class Decoder_4(nn.Module):
    """ Encodes the Image into low-dimensional feature representation

    Attributes
    ----------
    num_ch_enc : list
        channels used by the ResNet Encoder at different layers

    Methods
    -------
    forward(x, ):
        Processes input image features into output occupancy maps/layouts
    """

    def __init__(self, num_ch_enc, num_class=8, type=''):
        super(Decoder_4, self).__init__()

        self.num_output_channels = num_class
        self.num_ch_enc = num_ch_enc
        # self.num_ch_dec = np.array([16,32, 32, 64, 128, 256])
        self.num_ch_dec = np.array([16,32,64, 128, 256])
        # decoder
        self.convs = OrderedDict()
        # self.dw = nn.Conv2d(128, 128, kernel_size=7, groups=128,
        #                                      padding=int(7 / 2))
        # self.pw = nn.Conv2d(128, 128, kernel_size=1, groups=1)
        for i in range(2, -1, -1):
            # upconv_0
            if type == 'transform_decoder':
                num_ch_in = 128if i ==2 else self.num_ch_dec[i + 1]
            else:
                num_ch_in =128if i == 2 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            # self.convs["offset",i,0]=ConvOffset2D(num_ch_in)
            self.convs[("upconv", i, 0)] = nn.Conv2d(
                num_ch_in, num_ch_out, 3, 1, 1)
            self.convs[("norm", i, 0)] = nn.BatchNorm2d(num_ch_out)
            self.convs[("relu", i, 0)] = nn.ReLU(True)

            # upconv_1
            # self.convs["offset", i, 1] = ConvOffset2D(num_ch_out)
            self.convs[("upconv", i, 1)] = nn.Conv2d(
                num_ch_out, num_ch_out, 3, 1, 1)
            self.convs[("norm", i, 1)] = nn.BatchNorm2d(num_ch_out)

        self.convs["topview"] = Conv3x3(
            self.num_ch_dec[0], self.num_output_channels)
        self.dropout = nn.Dropout3d(0.2)
        self.decoder = nn.ModuleList(list(self.convs.values()))

    def forward(self, x, is_training=True):
        """

        Parameters
        ----------
        x : torch.FloatTensor
            Batch of encoded feature tensors
            | Shape: (batch_size, 128, occ_map_size/2^5, occ_map_size/2^5)
        is_training : bool
            whether its training or testing phase

        Returns
        -------
        x : torch.FloatTensor
            Batch of output Layouts
            | Shape: (batch_size, 2, occ_map_size, occ_map_size)
        """
        h_x = 0
        # x=self.dw(x)
        # x=self.pw(x)

        for i in range(2, -1, -1):

            x = self.convs[("upconv", i, 0)](x)
            x = self.convs[("norm", i, 0)](x)
            x = self.convs[("relu", i, 0)](x)
            x = upsample(x)

            x = self.convs[("upconv", i, 1)](x)
            x = self.convs[("norm", i, 1)](x)

        if is_training:
            x = self.convs["topview"](x)
        else:
            softmax = nn.Softmax2d()
            x = softmax(self.convs["topview"](x))
            print('not_train')

        return x
class ConvFFN(nn.Module):

    def __init__(self, in_channels, internal_channels, out_channels, drop_path):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.preffn_bn = nn.BatchNorm2d(in_channels)
        self.pw1 = conv_bn(in_channels=in_channels, out_channels=internal_channels, kernel_size=1, stride=1, padding=0, groups=1)
        self.pw2 = conv_bn(in_channels=internal_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, groups=1)
        self.nonlinear = nn.GELU()

    def forward(self, x):
        out = self.preffn_bn(x)
        out = self.pw1(out)
        out = self.nonlinear(out)
        out = self.pw2(out)
        return x + self.drop_path(out)
class RepLKBlock(nn.Module):

    def __init__(self, in_channels, dw_channels, block_lk_size, small_kernel, drop_path, small_kernel_merged=False):
        super().__init__()
        self.pw1 = conv_bn_relu(in_channels, dw_channels, 1, 1, 0, groups=1)
        self.pw2 = conv_bn(dw_channels, in_channels, 1, 1, 0, groups=1)
        self.large_kernel = ReparamLargeKernelConv(in_channels=dw_channels, out_channels=dw_channels, kernel_size=block_lk_size,
                                                  stride=1, groups=dw_channels, small_kernel=small_kernel, small_kernel_merged=small_kernel_merged)
        self.lk_nonlinear = nn.ReLU()
        self.prelkb_bn = nn.BatchNorm2d(in_channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # print('drop path:', self.drop_path)

    def forward(self, x):
        out = self.prelkb_bn(x)
        out = self.pw1(out)
        out = self.large_kernel(out)
        out = self.lk_nonlinear(out)
        out = self.pw2(out)
        return x + self.drop_path(out)
class Encoder_res_big(nn.Module):
    """ Encodes the Image into low-dimensional feature representation

    Attributes
    ----------
    num_layers : int
        Number of layers to use in the ResNet
    img_ht : int
        Height of the input RGB image
    img_wt : int
        Width of the input RGB image
    pretrained : bool
        Whether to initialize ResNet with pretrained ImageNet parameters

    Methods
    -------
    forward(x, is_training):
        Processes input image tensors into output feature tensors
    """

    def __init__(self, num_layers, img_ht, img_wt, pretrained=True):
        super(Encoder_res_big, self).__init__()

        self.resnet_encoder = ResNetD('18')
        self.resnet_encoder.load_state_dict(torch.load('resnetd18.pth', map_location='cuda'), strict=False)
        # self.resnet_encoder = ResidualNet(network_type='ImageNet',depth=18,num_classes=2,att_type="TripletAttention")
        # self.resnet_encoder.load_state_dict(torch.load('pcam_resnet18.pth.tar', map_location='cuda'), strict=False)
        # self.d_encoder=DilatedEncoder()
        num_ch_enc = np.array([64, 64, 128, 256, 512])
        # convolution to reduce depth and size of features before fc
        self.conv1 = Conv3x3(num_ch_enc[-1], 128)
        self.conv2 = Conv3x3(128, 128)
        self.pool = nn.MaxPool2d(2)
        # self.big_kernel_0 = RepLKBlock(in_channels=256, dw_channels=256, block_lk_size=7, drop_path=0.2, small_kernel=5)
        # self.big_kernel_0_1 = RepLKBlock(in_channels=256, dw_channels=256, block_lk_size=9, drop_path=0.2, small_kernel=5)
        self.big_kernel_1 = RepLKBlock(in_channels=256, dw_channels=256, block_lk_size=13, drop_path=0.1,
                                       small_kernel=3)
        self.convffn_1 = ConvFFN(in_channels=256, internal_channels=256, out_channels=256, drop_path=0.1)
        self.big_kernel_2 = RepLKBlock(in_channels=256, dw_channels=256, block_lk_size=13, drop_path=0.1,
                                       small_kernel=3)
        self.convffn_2 = ConvFFN(in_channels=256, internal_channels=256, out_channels=256, drop_path=0.1)
        self.big_kernel_3 = RepLKBlock(in_channels=128, dw_channels=128, block_lk_size=13, drop_path=0.1,
                                       small_kernel=3)
        self.convffn_3 = ConvFFN(in_channels=128, internal_channels=128, out_channels=128, drop_path=0.1)
        self.big_kernel_4 = RepLKBlock(in_channels=128, dw_channels=128, block_lk_size=13, drop_path=0.1,
                                       small_kernel=3)
        self.convffn_4 = ConvFFN(in_channels=128, internal_channels=128, out_channels=128, drop_path=0.1)
        self.big_kernel_5 = RepLKBlock(in_channels=128, dw_channels=128, block_lk_size=13, drop_path=0.1,
                                       small_kernel=3)
        self.convffn_5 = ConvFFN(in_channels=128, internal_channels=128, out_channels=128, drop_path=0.1)

        self.pw_0 = pw(in_channels=256, out_channels=256)
        self.dw_0 = dw(in_channels=256, out_channels=256, kernel_size=3)
        self.pw_1 = pw(in_channels=256, out_channels=128)
        self.dw_1 = dw(in_channels=128, out_channels=128, kernel_size=3)


    def forward(self, x):
        """

        Parameters
        ----------
        x : torch.FloatTensor
            Batch of Image tensors
            | Shape: (batch_size, 3, img_height, img_width)

        Returns
        -------
        x : torch.FloatTensor
            Batch of low-dimensional image representations
            | Shape: (batch_size, 128, img_height/128, img_width/128)
        """

        x = self.resnet_encoder(x)[-2]
        # x=self.pool(x)
        # x=self.pw_3(x)
        # x = self.dw_3(x)
        # x=self.big_kernel_0(x)
        # x=self.big_kernel_0_1(x)
        x = self.pool(x)
        x = self.pw_0(x)
        x = self.dw_0(x)
        batch_size, c, h, w = x.shape
        # pose_1 = torch.randn(batch_size, h ,  w).cuda()
        x = self.big_kernel_1(x)
        x = self.convffn_1(x)
        x = self.big_kernel_2(x)
        x1 = self.convffn_2(x)

        # x=self.hanet_1(x,x1,pos=pose_1)
        x = self.pool(x1)

        x = self.pw_1(x)
        x = self.dw_1(x)
        x = self.big_kernel_3(x)
        x = self.convffn_3(x)
        x = self.big_kernel_4(x)
        x = self.convffn_4(x)

        x = self.pool(x)


        return x, x1