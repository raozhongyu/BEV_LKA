import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from collections import OrderedDict

from spc.vista_attention import Cross_Attention_Douple
from spc.project_2 import SPattenion_block
class ConvAttentionLayer_Decouple(nn.Module):
    def __init__(self, input_channels: int, numhead: int = 1, reduction_ratio=4):

        super().__init__()

        self.q_sem_conv = nn.Conv2d(input_channels,
                                    input_channels // reduction_ratio, 3, 1, 1)

        self.k_sem_conv = nn.Conv2d(input_channels,
                                    input_channels // reduction_ratio, 3, 1, 1)
        self.k_geo_conv = nn.Conv2d(input_channels,
                                    input_channels // reduction_ratio, 3, 1, 1)

        self.v_conv = nn.Conv2d(input_channels, input_channels, 1, 1, 0)
        self.out_sem_conv = nn.Conv2d(input_channels, input_channels, 1, 1)
        self.out_geo_conv = nn.Conv2d(input_channels, input_channels, 1, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.channels = input_channels // reduction_ratio
        self.numhead = numhead
        self.head_dim = self.channels // numhead
        self.sem_norm = nn.LayerNorm(input_channels)
        self.geo_norm = nn.LayerNorm(input_channels)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                q_pos_emb: Optional[torch.Tensor] = None,
                k_pos_emb: Optional[torch.Tensor] = None):

        view = value + 0
        input_channel = view.shape[1]
        if q_pos_emb is not None:
            query += q_pos_emb
        if k_pos_emb is not None:
            key += k_pos_emb

        # to qkv forward
        # q = self.q_conv(query)
        q_sem = self.q_sem_conv(query)
        # q_geo = self.q_geo_conv(query)
        qs = q_sem
        # k = self.k_conv(key)
        k_sem = self.k_sem_conv(key)
        # k_geo = self.k_geo_conv(key)
        ks = k_sem
        v = self.v_conv(value)
        vs = v
        out_conv = self.out_sem_conv
        norm = self.sem_norm
        outputs = []
        # attentions = []

        # for q, k, v, out_conv, norm in zip(qs, ks, vs, out_convs, norms):
            # read shape of qkv
        bs = qs.shape[0]
        qk_channel = qs.shape[1]  # equal to the channel of `k`
        v_channel = vs.shape[1]  # channel of `v`
        h_q, w_q = qs.shape[2:]  # height and weight of query map
        h_kv, w_kv = ks.shape[2:]  # height and weight of key and value map
        numhead = self.numhead
        qk_head_dim = qk_channel // numhead
        v_head_dim = v_channel // numhead

        # scale query
        scaling = float(self.head_dim) ** -0.5
        q = qs * scaling

        # reshape(sequentialize) qkv
        q = rearrange(q, "b c h w -> b c (h w)", b=bs, c=qk_channel, h=h_q, w=w_q)
        q = rearrange(q, "b (n d) (h w) -> (b n) (h w) d", b=bs,
                      n=numhead, h=h_q, w=w_q, d=qk_head_dim)
        q = q.contiguous()
        k = rearrange(ks, "b c h w -> b c (h w)", b=bs, c=qk_channel, h=h_kv, w=w_kv)
        k = rearrange(k, "b (n d) (h w) -> (b n) (h w) d", b=bs,
                      n=numhead, h=h_kv, w=w_kv, d=qk_head_dim)
        k = k.contiguous()
        v = rearrange(v, "b c h w -> b c (h w)", b=bs, c=v_channel, h=h_kv, w=w_kv)
        v = rearrange(v, "b (n d) (h w) -> (b n) (h w) d", b=bs,
                      n=numhead, h=h_kv, w=w_kv, d=v_head_dim)
        v = v.contiguous()

        # get the attention map
        energy = torch.bmm(q, k.transpose(1, 2))  # [h_q*w_q, h_kv*w_kv]
        attention = F.softmax(energy, dim=-1)  # [h_q*w_q, h_kv*w_kv]
        # attentions.append(attention)
        # get the attention output
        r = torch.bmm(attention, v)  # [bs * nhead, h_q*w_q, C']
        r = rearrange(r, "(b n) (h w) d -> b (n d) h w", b=bs,
                      n=numhead, h=h_q, w=w_q, d=v_head_dim)
        r = r.contiguous()
        r = out_conv(r)


        # residual
        temp_view = view + r
        temp_view = temp_view.view(bs, input_channel, -1).permute(2, 0, 1).contiguous()
        temp_view = norm(temp_view)
        outputs=(temp_view)
        return outputs, attention


class self_attention(nn.Module):
    def __init__(self, input_channels: int, numhead: int = 1, reduction_ratio=4):

        super().__init__()
        # self.q_conv = nn.Conv2d(input_channels, input_channels // reduction_ratio, 3, 1, 1)
        self.q_sem_conv = nn.Conv2d(input_channels,
                                    input_channels // reduction_ratio, 3, 1, 1)
        self.q_geo_conv = nn.Conv2d(input_channels,
                                    input_channels // reduction_ratio, 3, 1, 1)
        # self.k_conv = nn.Conv2d(input_channels, input_channels // reduction_ratio, 3, 1, 1)
        self.k_sem_conv = nn.Conv2d(input_channels,
                                    input_channels // reduction_ratio, 3, 1, 1)
        self.k_geo_conv = nn.Conv2d(input_channels,
                                    input_channels // reduction_ratio, 3, 1, 1)
        # self.v_conv = nn.Conv2d(input_channels, input_channels, 3, 1, 1)
        self.v_conv = nn.Conv2d(input_channels, input_channels, 1, 1, 0)
        self.out_sem_conv = nn.Conv2d(input_channels, input_channels, 1, 1)
        self.out_geo_conv = nn.Conv2d(input_channels, input_channels, 1, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.channels = input_channels // reduction_ratio
        self.numhead = numhead
        self.head_dim = self.channels // numhead
        self.sem_norm = nn.LayerNorm(input_channels)
        self.geo_norm = nn.LayerNorm(input_channels)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                q_pos_emb: Optional[torch.Tensor] = None,
                k_pos_emb: Optional[torch.Tensor] = None):


        view = value + 0  # NOTE: a funny method to deepcopy
        input_channel = view.shape[1]
        if q_pos_emb is not None:
            query += q_pos_emb
        if k_pos_emb is not None:
            key += k_pos_emb

        # to qkv forward
        # q = self.q_conv(query)
        q_sem = self.q_sem_conv(query)
        # q_geo = self.q_geo_conv(query)
        qs = q_sem
        # k = self.k_conv(key)
        k_sem = self.k_sem_conv(key)
        # k_geo = self.k_geo_conv(key)
        ks = k_sem
        v = self.v_conv(value)
        vs = v
        out_conv = self.out_sem_conv
        norm = self.sem_norm
        # outputs = []
        # attentions = []

        # for q, k, v, out_conv, norm in zip(qs, ks, vs, out_convs, norms):
        # read shape of qkv
        bs = qs.shape[0]
        qk_channel = qs.shape[1]  # equal to the channel of `k`
        v_channel = vs.shape[1]  # channel of `v`
        h_q, w_q = qs.shape[2:]  # height and weight of query map
        h_kv, w_kv = ks.shape[2:]  # height and weight of key and value map
        numhead = self.numhead
        qk_head_dim = qk_channel // numhead
        v_head_dim = v_channel // numhead

        # scale query
        scaling = float(self.head_dim) ** -0.5
        q = qs * scaling

        # reshape(sequentialize) qkv
        q = rearrange(q, "b c h w -> b c (h w)", b=bs, c=qk_channel, h=h_q, w=w_q)
        q = rearrange(q, "b (n d) (h w) -> (b n) (h w) d", b=bs,
                      n=numhead, h=h_q, w=w_q, d=qk_head_dim)
        q = q.contiguous()
        k = rearrange(ks, "b c h w -> b c (h w)", b=bs, c=qk_channel, h=h_kv, w=w_kv)
        k = rearrange(k, "b (n d) (h w) -> (b n) (h w) d", b=bs,
                      n=numhead, h=h_kv, w=w_kv, d=qk_head_dim)
        k = k.contiguous()
        v = rearrange(v, "b c h w -> b c (h w)", b=bs, c=v_channel, h=h_kv, w=w_kv)
        v = rearrange(v, "b (n d) (h w) -> (b n) (h w) d", b=bs,
                      n=numhead, h=h_kv, w=w_kv, d=v_head_dim)
        v = v.contiguous()

        # get the attention map
        energy = torch.bmm(q, k.transpose(1, 2))  # [h_q*w_q, h_kv*w_kv]
        attention = F.softmax(energy, dim=-1)  # [h_q*w_q, h_kv*w_kv]
        # attentions.append(attention)
        # get the attention output
        r = torch.bmm(attention, v)  # [bs * nhead, h_q*w_q, C']
        r = rearrange(r, "(b n) (h w) d -> b (n d) h w", b=bs,
                      n=numhead, h=h_q, w=w_q, d=v_head_dim)
        r = r.contiguous()
        r = out_conv(r)

        # residual
        temp_view =  r+view
        temp_view = temp_view.view(bs, input_channel, -1).permute(2, 0, 1).contiguous()
        temp_view = norm(temp_view)
        outputs = (temp_view)
        return outputs

class FeedFowardLayer(nn.Module):
    def __init__(self,
                 input_channel: int,
                 hidden_channel: int = 2048):
        super().__init__()
        self.linear1 = nn.Linear(input_channel, hidden_channel)
        self.linear2 = nn.Linear(hidden_channel, input_channel)

        self.norm = nn.LayerNorm(input_channel)

        self.activation = nn.ReLU()  # con be modify as GeLU or GLU

    def forward(self, view: torch.Tensor):
        ffn = self.linear2(self.activation((self.linear1(view))))
        view = view + ffn
        view = self.norm(view)
        return view


class PositionEmbeddingLearned(nn.Module):
    def __init__(self, res: Sequence[int], num_pos_feats=384):
        r"""
        Absolute pos embedding, learned.
        Args:
            res (Sequence[int]): resolution (height and width)
            num_pos_feats (int, optional): the number of feature channel. Defaults to 384.
        """
        super().__init__()
        h, w = res
        num_pos_feats = num_pos_feats // 2
        self.row_embed = nn.Embedding(h, num_pos_feats)
        self.col_embed = nn.Embedding(w, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x: torch.Tensor):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).contiguous().unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


class CrossAttenBlock_Decouple(nn.Module):
    def __init__(self,
                 input_channels: int,
                 numhead: int = 1,
                 hidden_channel: int = 128,
                 reduction_ratio=2):
        r"""
        Block of cross attention (one cross attention layer and one ffn)
        Args:
            input_channels (int): input channel of conv attention
            numhead (int, optional): the number of attention heads. Defaults to 1.
            hidden_channel (int, optional): channel of ffn. Defaults to 2048.
        """
        super().__init__()
        self.cross_atten = ConvAttentionLayer_Decouple(input_channels, numhead, reduction_ratio)
        self.self_attn=self_attention(input_channels, numhead, reduction_ratio)


    def forward(self,
                view_1: torch.Tensor,
                view_2: torch.Tensor,
                pos_emb_1: Optional[torch.Tensor] = None,
                pos_emb_2: Optional[torch.Tensor] = None):

        B, C, H, W = view_1.shape
        view_1,atten_maps=self.cross_atten(view_1,view_2, view_1, pos_emb_1, pos_emb_2)
        view_1_feature=view_1.view(H, W, B, C).permute(2, 3, 0, 1).contiguous()

        view_2=self.self_attn( view_1_feature,view_2,  view_1_feature)
        view_2_feature = view_2.view(H, W, B, C).permute(2, 3, 0, 1).contiguous()
        # view_1_a, atten_map_1= self.cross_atten(view_1, view_2, view_2, pos_emb_1, pos_emb_2)
        # view_2_a, atten_map_2 = self.cross_atten(view_2, view_1, view_1, pos_emb_2, pos_emb_1)
        # ffns = [self.ffn_sem, self.ffn_geo]
        # views=[view_1_a, view_2_a]
        # atten_maps=[atten_map_1,atten_map_2]
        # outputs = []
        # for i in range(len(views)):
        #     ffn = ffns[i]
        #     view = ffn(views[i])
        #     view = view.view(H, W, B, C).permute(2, 3, 0, 1).contiguous()
        #     outputs.append(view)
        return  view_2_feature

def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")
class CrossAttenBlock_all(nn.Module):
    def __init__(self):
        super(CrossAttenBlock_all, self).__init__()
        self.num_ch_dec=[128,128]
        num_ch_in=128
        self.convs = OrderedDict()
        self.conv1=nn.Conv2d(
                256, 128, 3, 1, 1)
        for i in range(2):
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = nn.Conv2d(
                num_ch_in, num_ch_out, 3, 1, 1)
            self.convs[("norm", i, 0)] = nn.BatchNorm2d(num_ch_out)
            self.convs[("relu", i, 0)] = nn.ReLU(True)

            # upconv_1
            # self.convs["offset", i, 1] = ConvOffset2D(num_ch_out)
            self.convs[("upconv", i, 1)] = nn.Conv2d(
                num_ch_out, num_ch_out, 3, 1, 1)
            self.convs[("norm", i, 1)] = nn.BatchNorm2d(num_ch_out)
        # self.pos_1=PositionEmbeddingLearned((32,32),num_pos_feats=128)
        # self.pos_2 = PositionEmbeddingLearned((32, 32), num_pos_feats=128)
        # self.se=CrossAttenBlock_Decouple(input_channels=128)
        self.se=CrossAttenBlock_Decouple(input_channels=128,numhead=2,reduction_ratio=4)
        self.decoder = nn.ModuleList(list(self.convs.values()))
        # self.pw=nn.Conv2d(128,128,1,1)
        # self.dw = nn.Conv2d(128,128,kernel_size=7,padding=3,groups=128)
        # self.pw = nn.Conv2d(128, 128, 1, 1)
        self.cbam= SPattenion_block(channel=128,kernel_size=7)



    def forward(self,x, x_ori):


        for i in range(2):
            x = self.convs[("norm", i, 0)](x)
            x = self.convs[("relu", i, 0)](x)
            x = upsample(x)
            # x = self.convs[("offset", i, 1)](x)
            x = self.convs[("upconv", i, 1)](x)
            x = self.convs[("norm", i, 1)](x)
        # position_1 =  self.pos_1(x)
        x1=x
        x = self.cbam(x)
        x2=x



        outputs= self.se(x2,x1)
        return outputs,x2,x1



if __name__ == '__main__':
    input_1 = torch.randn(2, 128, 32, 32)
    input_2 = torch.randn(2, 128, 32, 32)
    pos_1=PositionEmbeddingLearned((32,32),num_pos_feats=128)
    pos_2 = PositionEmbeddingLearned((32, 32), num_pos_feats=128)
    position=pos_1(input_1)
    position_2=pos_2(input_1)
    se =CrossAttenBlock_Decouple(input_channels=128)
    outputs,atten_maps = se(input_1,input_2,position,position_2)
    print(outputs.shape)