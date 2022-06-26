import paddle
import paddle.nn as nn

from SwinT import trunc_normal_, Identity, zeros_
from ViT import VisionTransformer


class DistilledVisionTransformer(VisionTransformer):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 class_num=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=False,
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-5,
                 **kwargs):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            class_num=class_num,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            epsilon=epsilon,
            **kwargs)
        self.pos_embed = self.create_parameter(
            shape=(1, self.patch_embed.num_patches + 2, self.embed_dim),
            default_initializer=zeros_)
        self.add_parameter("pos_embed", self.pos_embed)

        self.dist_token = self.create_parameter(
            shape=(1, 1, self.embed_dim), default_initializer=zeros_)
        self.add_parameter("cls_token", self.cls_token)

        self.head_dist = nn.Linear(
            self.embed_dim,
            self.class_num) if self.class_num > 0 else Identity()

        trunc_normal_(self.dist_token)
        trunc_normal_(self.pos_embed)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        B = paddle.shape(x)[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand((B, -1, -1))
        dist_token = self.dist_token.expand((B, -1, -1))
        x = paddle.concat((cls_tokens, dist_token, x), axis=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        return (x + x_dist) / 2


def DeiT_tiny_patch16_224(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    return model