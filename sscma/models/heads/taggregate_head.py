# copyright Copyright (c) Seeed Technology Co.,Ltd.
import torch
import torch.nn as nn
from mmcls.models.builder import HEADS


@HEADS.register_module()
class TAggregate(nn.Module):
    def __init__(self, clip_length=None, embed_dim=64, n_layers=6, nhead=6, n_classes=None, dim_feedforward=512):
        super(TAggregate, self).__init__()
        self.num_tokens = 1
        drop_rate = 0.1
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, activation='gelu', dim_feedforward=dim_feedforward, dropout=drop_rate
        )
        self.transformer_enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers, norm=nn.LayerNorm(embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, clip_length + self.num_tokens, embed_dim))
        self.fc = nn.Linear(embed_dim, n_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    # nn.init.constant_(m.weight, 1)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Parameter):
            with torch.no_grad():
                m.weight.data.normal_(0.0, 0.02)
                # nn.init.orthogonal_(m.weight)

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed
        x.transpose_(1, 0)
        o = self.transformer_enc(x)
        pred = self.fc(o[0])
        return pred
