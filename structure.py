"""
VPT     Script  ver： Apr 21th 14:00

based on
timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
"""

import paddle
import paddle.nn as nn

from ViTmodel.ViT import VisionTransformer, PatchEmbed


class VPT_ViT(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, class_num=1000, embed_dim=768, depth=12,
                 num_heads=8, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer='nn.LayerNorm',
                 act_layer=None, weight_init='', Prompt_Token_num=1, VPT_type="Shallow", basic_state_dict=None):

        # Recreate ViT
        super().__init__(img_size, patch_size, in_chans, class_num, embed_dim, depth, num_heads, mlp_ratio, qkv_bias,
                         qk_scale=None,
                         drop_rate=drop_rate,
                         attn_drop_rate=attn_drop_rate,
                         drop_path_rate=drop_path_rate,
                         norm_layer=norm_layer,
                         epsilon=1e-5)

        # load basic state_dict
        if basic_state_dict is not None:
            self.load_state_dict(basic_state_dict, False)

        self.VPT_type = VPT_type
        if VPT_type == "Deep":
            self.Prompt_Tokens = nn.Parameter(paddle.zeros(depth, Prompt_Token_num, embed_dim))
        else:  # "Shallow"
            self.Prompt_Tokens = nn.Parameter(paddle.zeros(1, Prompt_Token_num, embed_dim))

    def New_CLS_head(self, new_classes=15):
        self.head = nn.Linear(self.embed_dim, new_classes)

    def Freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        self.Prompt_Tokens.requires_grad = True
        try:
            for param in self.head.parameters():
                param.requires_grad = True
        except:
            pass

    def obtain_prompt(self):
        prompt_state_dict = {'head': self.head.state_dict(),
                             'Prompt_Tokens': self.Prompt_Tokens}
        # print(prompt_state_dict)
        return prompt_state_dict

    def load_prompt(self, prompt_state_dict):
        self.head.load_state_dict(prompt_state_dict['head'])
        self.Prompt_Tokens = prompt_state_dict['Prompt_Tokens']

    def forward_features(self, x):
        x = self.patch_embed(x)
        # print(x.shape,self.pos_embed.shape)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        # concatenate CLS token
        x = paddle.concat((cls_token, x), axis=1)
        x = self.pos_drop(x + self.pos_embed)

        if self.VPT_type == "Deep":

            Prompt_Token_num = self.Prompt_Tokens.shape[1]

            for i in range(len(self.blocks)):
                # concatenate Prompt_Tokens
                Prompt_Tokens = self.Prompt_Tokens[i].unsqueeze(0)
                # firstly concatenate
                x = paddle.concat((x, Prompt_Tokens.expand(x.shape[0], -1, -1)), axis=1)
                num_tokens = x.shape[1]
                # lastly remove, a genius trick
                x = self.blocks[i](x)[:, :num_tokens - Prompt_Token_num]

        else:  # self.VPT_type == "Shallow"
            num_tokens = x.shape[1]
            Prompt_Token_num = self.Prompt_Tokens.shape[1]

            # concatenate Prompt_Tokens
            Prompt_Tokens = self.Prompt_Tokens.expand(x.shape[0], -1, -1)
            x = paddle.concat((x, Prompt_Tokens), axis=1)
            # Sequntially procees，lastly remove prompt tokens
            x = self.blocks(x)[:, :num_tokens - Prompt_Token_num]

        x = self.norm(x)
        return x

    def forward(self, x):

        x = self.forward_features(x)

        # use cls token for cls head
        x = self.pre_logits(x[:, 0, :])
        x = self.head(x)
        return x
