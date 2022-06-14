import paddle
import paddle.nn as nn
import numpy as np


# from droppath import DropPath
# from einops import rearrange

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Layer):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Mlp(nn.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=None, dropout=0.):
        # act_layer 仅为与原文传参保持一致。请修改 self.act 处。
        super(Mlp, self).__init__()
        w_attr_1, b_attr_1 = self._init_weights()
        self.fc1 = nn.Linear(in_features,
                             hidden_features,
                             weight_attr=w_attr_1,
                             bias_attr=b_attr_1)

        w_attr_2, b_attr_2 = self._init_weights()
        self.fc2 = nn.Linear(hidden_features,
                             in_features,
                             weight_attr=w_attr_2,
                             bias_attr=b_attr_2)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


def window_partition(x, window_size):
    """ partite windows into window_size x window_size
    Args:
        x: Tensor, shape=[b, h, w, c]
        window_size: int, window size
    Returns:
        x: Tensor, shape=[num_windows*b, window_size, window_size, c]
    """
    B, H, W, C = x.shape
    x = x.reshape([B, H // window_size, window_size, W // window_size, window_size, C])
    x = x.transpose([0, 1, 3, 2, 4, 5])
    x = x.reshape([-1, window_size, window_size, C])
    return x


def window_reverse(windows, window_size, H, W):
    """ Window reverse
    Args:
        windows: (n_windows * B, window_size, window_size, C)
        window_size: (int) window size
        H: (int) height of image
        W: (int) width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape([B, H // window_size, W // window_size, window_size, window_size, -1])
    x = x.reshape([B, H, W, -1])  # (bs,num_windows*window_size, num_windows*window_size, C)
    return x


class WindowAttention(nn.Layer):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attention_dropout=0., dropout=0.):

        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scale = qk_scale or self.dim_head ** -0.5
        #		print('This is windows_size[0] ->', window_size[0])
        #		print('This is type(windows_size[0]) ->', type(window_size[0]))
        ws = window_size[0]
        # define a parameter table of relative position bias
        self.relative_position_bias_table = paddle.create_parameter(
            shape=[(2 * ws - 1) * (2 * ws - 1), num_heads],
            dtype='float32',
            default_initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        # relative position index for each token inside window
        coords_h = paddle.arange(self.window_size[0])
        coords_w = paddle.arange(self.window_size[1])
        coords = paddle.stack(paddle.meshgrid([coords_h, coords_w]))  # [2, Window_h, Window_w]
        coords_flatten = paddle.flatten(coords, 1)  # [2, Windows_h*Window_w]
        # [2, window_h * window w, window_h * window w]
        relative_coords = coords_flatten.unsqueeze(2) - coords_flatten.unsqueeze(1)  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.transpose([1, 2, 0])  # [Window_h*Window_w, Window_h*Window_w, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        w_attr_1, b_attr_1 = self._init_weights()
        self.qkv = nn.Linear(dim, dim * 3, weight_attr=w_attr_1, bias_attr=b_attr_1 if qkv_bias else False)
        self.attn_dropout = nn.Dropout(attention_dropout)

        w_attr_2, b_attr_2 = self._init_weights()
        self.proj = nn.Linear(dim, dim, weight_attr=w_attr_2, bias_attr=b_attr_2)
        self.proj_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(axis=-1)

    # This is Function only paddle --

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def transpose_multihead(self, x):
        new_shape = x.shape[:-1] + [self.num_heads, self.dim_head]
        x = x.reshape(new_shape)
        x = x.transpose([0, 2, 1, 3])
        return x

    def get_relative_pos_bias_from_pos_index(self):
        # relative_position_bias_table is a ParamBase object
        # https://github.com/PaddlePaddle/Paddle/blob/067f558c59b34dd6d8626aad73e9943cf7f5960f/python/paddle/fluid/framework.py#L5727
        table = self.relative_position_bias_table  # N x num_heads
        # index is a tensor
        index = self.relative_position_index.reshape([-1])  # window_h*window_w * window_h*window_w
        # NOTE: paddle does NOT support indexing Tensor by a Tensor
        relative_position_bias = paddle.index_select(x=table, index=index)
        return relative_position_bias

    # This is Function only paddle --!

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        #		B_, N, C = x.shape
        #		qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        #		q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        #
        #		q = q * self.scale
        #		attn = (q @ k.transpose(-2, -1))

        qkv = self.qkv(x).chunk(3, axis=-1)  # {list:3}
        q, k, v = map(self.transpose_multihead, qkv)
        # [512,3,49,32] -> [128,6,49,32]-> [32,12,49,32]->[8,24,49,32]
        q = q * self.scale
        attn = paddle.matmul(q, k, transpose_y=True)
        # [512,3,49,49] -> [128,6,49,49] -> [32,12,49,49] -> [8,24,49,49]

        relative_position_bias = self.get_relative_pos_bias_from_pos_index()
        # [2401,3]->[2401,6]->[2401,12]->[2401,24]

        relative_position_bias = relative_position_bias.reshape(
            [self.window_size[0] * self.window_size[1],
             self.window_size[0] * self.window_size[1],
             -1])
        # [49,49,3]->[49,49,6]->[49,49,12]->[49,49,24]

        # nH, window_h*window_w, window_h*window_w
        relative_position_bias = relative_position_bias.transpose([2, 0, 1])
        # [3,49,49]->[6,49,49]->[12,49,49]->[24,49,49]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape([x.shape[0] // nW, nW, self.num_heads, x.shape[1], x.shape[1]])
            attn += mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape([-1, self.num_heads, x.shape[1], x.shape[1]])
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_dropout(attn)  # [512,3,49,49]->[128,6,49,49]->[32,12,49,49]->[8,24,49,49]

        z = paddle.matmul(attn, v)  # [512,3,49,32]->[128,6,49,32]->[32,12,49,32]->[8,24,49,32]
        z = z.transpose([0, 2, 1, 3])
        new_shape = z.shape[:-2] + [self.dim]
        z = z.reshape(new_shape)
        z = self.proj(z)
        z = self.proj_dropout(z)  # [512,49,96]->[128,49,192]->[32,49,384]->[8,49,768]

        return z


#	def extra_repr(self) -> str:
#		return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'
#
#	def flops(self, N):
#		# calculate flops for 1 window with token length of N
#		flops = 0
#		# qkv = self.qkv(x)
#		flops += N * self.dim * 3 * self.dim
#		# attn = (q @ k.transpose(-2, -1))
#		flops += self.num_heads * N * (self.dim // self.num_heads) * N
#		#  x = (attn @ v)
#		flops += self.num_heads * N * N * (self.dim // self.num_heads)
#		# x = self.proj(x)
#		flops += N * self.dim * self.dim
#		return flops

class SwinTransformerBlock(nn.Layer):
    """ Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, dropout=0., attention_dropout=0., droppath=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        # act_layer 、norm_layer 是为了对齐 SwinUnet 代码，未使用。
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        #		assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        w_attr_1, b_attr_1 = self._init_weights_layernorm()
        self.norm1 = nn.LayerNorm(dim, weight_attr=w_attr_1, bias_attr=b_attr_1)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attention_dropout=attention_dropout, dropout=dropout)
        self.drop_path = DropPath(drop_path) if droppath > 0. else None

        w_attr_2, b_attr_2 = self._init_weights_layernorm()
        self.norm2 = nn.LayerNorm(dim, weight_attr=w_attr_2, bias_attr=b_attr_2)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), dropout=dropout)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = paddle.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.reshape((-1, self.window_size * self.window_size))
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = paddle.where(attn_mask != 0, paddle.ones_like(attn_mask) * float(-100.0), attn_mask)
            attn_mask = paddle.where(attn_mask == 0, paddle.zeros_like(attn_mask), attn_mask)

        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def _init_weights_layernorm(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(1.0))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        #		assert L == H * W, "input feature has wrong size"

        h = x
        x = self.norm1(x)  # [bs, H*W, C]
        new_shape = [B, H, W, C]
        x = x.reshape(new_shape)  # [bs,H,W,C]

        #		shortcut = x
        #		x = self.norm1(x)
        #		x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = paddle.roll(x, shifts=(-self.shift_size, -self.shift_size), axis=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.reshape([-1, self.window_size * self.window_size, C])  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.reshape([-1, self.window_size, self.window_size, C])

        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = paddle.roll(shifted_x, shifts=(self.shift_size, self.shift_size), axis=(1, 2))
        else:
            x = shifted_x
        x = x.reshape([B, H * W, C])

        # FFN
        if self.drop_path is not None:
            x = h + self.drop_path(x)
        else:
            x = h + x
        h = x  # [bs,H*W,C]
        x = self.norm2(x)  # [bs,H*W,C]
        x = self.mlp(x)  # [bs,H*W,C]
        if self.drop_path is not None:
            x = h + self.drop_path(x)
        else:
            x = h + x

        # FFN
        #		x = shortcut + self.drop_path(x)
        #		x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


#	def extra_repr(self) -> str:
#		return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
#				f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
#
#	def flops(self):
#		flops = 0
#		H, W = self.input_resolution
#		# norm1
#		flops += self.dim * H * W
#		# W-MSA/SW-MSA
#		nW = H * W / self.window_size / self.window_size
#		flops += nW * self.attn.flops(self.window_size * self.window_size)
#		# mlp
#		flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
#		# norm2
#		flops += self.dim * H * W
#		return flops

class PatchMerging(nn.Layer):
    """ Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super(PatchMerging, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        w_attr_1, b_attr_1 = self._init_weights()
        self.reduction = nn.Linear(4 * dim, 2 * dim, weight_attr=w_attr_1, bias_attr=False)
        w_attr_2, b_attr_2 = self._init_weights_layernorm()
        self.norm = nn.LayerNorm(4 * dim, weight_attr=w_attr_2, bias_attr=b_attr_2)

    def _init_weights_layernorm(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(1.0))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, x):
        """
        x: B, H*W, C
        """
        h, w = self.input_resolution
        b, _, c = x.shape
        x = x.reshape([b, h, w, c])

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = paddle.concat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.reshape([b, -1, 4 * c])  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpand(nn.Layer):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim) if dim_scale == 2 else None
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
            x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, _, C = x.shape
        #		assert L == H * W, "input feature has wrong size"

        x = x.reshape([B, H, W, C])
        #		x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.reshape([B, H * 2, W * 2, C // 4])
        x = x.reshape([B, -1, C // 4])
        x = self.norm(x)

        return x


class FinalPatchExpand_X4(nn.Layer):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        w_attr_1, b_attr_1 = self._init_weights()
        self.expand = nn.Linear(dim, 16 * dim, weight_attr=w_attr_1, bias_attr=False)
        self.output_dim = dim
        w_attr_2, b_attr_2 = self._init_weights_layernorm()
        self.norm = norm_layer(self.output_dim, weight_attr=w_attr_2, bias_attr=b_attr_2)

    def _init_weights_layernorm(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(1.0))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        #		assert L == H * W, "input feature has wrong size"

        x = x.reshape([B, H, W, C])
        #		x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.reshape([B, H * self.dim_scale, W * self.dim_scale, C // self.dim_scale ** 2])
        x = x.reshape([B, -1, self.output_dim])
        x = self.norm(x)

        return x


class BasicLayer(nn.Layer):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super(BasicLayer, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        #		self.blocks = nn.ModuleList([
        #			SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
        #								num_heads=num_heads, window_size=window_size,
        #								shift_size=0 if (i % 2 == 0) else window_size // 2,
        #								mlp_ratio=mlp_ratio,
        #								qkv_bias=qkv_bias, qk_scale=qk_scale,
        #								drop=drop, attn_drop=attn_drop,
        #								drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
        #								norm_layer=norm_layer)
        self.blocks = nn.LayerList()
        for i in range(depth):
            self.blocks.append(
                SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                                     window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2,
                                     mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, dropout=drop,
                                     attention_dropout=attn_drop,
                                     droppath=drop_path[i] if isinstance(drop_path, list) else drop_path)
            )

        #			for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        #			if self.use_checkpoint:
        #				x = checkpoint.checkpoint(blk, x)
        #			else:
        #				x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class BasicLayer_up(nn.Layer):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        #		self.use_checkpoint = use_checkpoint

        # build blocks
        #		self.blocks = nn.ModuleList([
        #			SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
        #								num_heads=num_heads, window_size=window_size,
        #								shift_size=0 if (i % 2 == 0) else window_size // 2,
        #								mlp_ratio=mlp_ratio,
        #								qkv_bias=qkv_bias, qk_scale=qk_scale,
        #								drop=drop, attn_drop=attn_drop,
        #								drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
        #								norm_layer=norm_layer)
        #			for i in range(depth)])

        self.blocks = nn.LayerList()
        for i in range(depth):
            self.blocks.append(
                SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                                     window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2,
                                     mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, dropout=drop,
                                     attention_dropout=attn_drop,
                                     droppath=drop_path[i] if isinstance(drop_path, list) else drop_path)
            )

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class PatchEmbedding(nn.Layer):
    """ Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super(PatchEmbedding, self).__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv2D(in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size,
                                     stride=patch_size)
        w_attr, b_attr = self._init_weights_layernorm()
        self.norm = nn.LayerNorm(embed_dim,
                                 weight_attr=w_attr,
                                 bias_attr=b_attr)

    def _init_weights_layernorm(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(1.0))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    #	def forward(self, x):
    #		B, C, H, W = x.shape
    #		# FIXME look at relaxing size constraints
    #		assert H == self.img_size[0] and W == self.img_size[1], \
    #			f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
    #		x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
    #		if self.norm is not None:
    #			x = self.norm(x)
    #		return x
    def forward(self, x):
        x = self.patch_embed(x)  # [batch, embed_dim, h, w] h,w = patch_resolution
        x = x.flatten(start_axis=2, stop_axis=-1)  # [batch, embed_dim, h*w] h*w = num_patches
        x = x.transpose([0, 2, 1])  # [batch, h*w, embed_dim]
        x = self.norm(x)  # [batch, num_patches, embed_dim]
        return x


class SwinTransformerSys(nn.Layer):
    r""" Swin Transformer
                A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
                    https://arxiv.org/pdf/2103.14030
        Args:
                img_size (int | tuple(int)): Input image size. Default 224
                patch_size (int | tuple(int)): Patch size. Default: 4
                in_chans (int): Number of input image channels. Default: 3
                num_classes (int): Number of classes for classification head. Default: 1000
                embed_dim (int): Patch embedding dimension. Default: 96
                depths (tuple(int)): Depth of each Swin Transformer layer.
                num_heads (tuple(int)): Number of attention heads in different layers.
                window_size (int): Window size. Default: 7
                mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
                qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
                qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
                drop_rate (float): Dropout rate. Default: 0
                attn_drop_rate (float): Attention dropout rate. Default: 0
                drop_path_rate (float): Stochastic depth rate. Default: 0.1
                norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
                ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
                patch_norm (bool): If True, add normalization after patch embedding. Default: True
                use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop=0., droppath=0.,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        super(SwinTransformerSys, self).__init__()

        print(
            "SwinTransformerSys expand initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{}".format(
                depths, depths_decoder, drop_rate, num_classes))

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        #				self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        self.patch_norm = patch_norm
        # split image into non-overlapping patches
        self.patch_embedding = PatchEmbedding(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embedding.num_patches
        #				patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = self.patch_embedding.patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_positional_embedding = paddle.nn.ParameterList([
                paddle.create_parameter(
                    shape=[1, num_patches, self.embed_dim], dtype='float32',
                    default_initializer=paddle.nn.initializer.TruncatedNormal(std=.02))])

        self.position_dropout = nn.Dropout(drop_rate)
        #		self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        #		dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        dpr = [x.item() for x in paddle.linspace(0, droppath, sum(depths))]

        # build encoder and bottleneck layers
        self.layers = nn.LayerList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                                 self.patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        # build decoder layers
        self.layers_up = nn.LayerList()
        self.concat_back_dim = nn.LayerList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      int(embed_dim * 2 ** (self.num_layers - 1 - i_layer))) if i_layer > 0 else None
            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      self.patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                         input_resolution=(
                                         self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                         self.patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                         depth=depths[(self.num_layers - 1 - i_layer)],
                                         num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                         window_size=window_size,
                                         mlp_ratio=self.mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop_rate, attn_drop=attn_drop,
                                         drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                             depths[:(self.num_layers - 1 - i_layer) + 1])],
                                         norm_layer=norm_layer,
                                         upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                         use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        w_attr_1, b_attr_1 = self._init_weights_layernorm()
        self.norm = nn.LayerNorm(self.num_features, weight_attr=w_attr_1, bias_attr=b_attr_1)
        w_attr_2, b_attr_2 = self._init_weights_layernorm()
        self.norm_up = nn.LayerNorm(self.embed_dim, weight_attr=w_attr_2, bias_attr=b_attr_2)

        if final_upsample == "expand_first":
            print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(input_resolution=(img_size // patch_size, img_size // patch_size),
                                          dim_scale=4, dim=embed_dim)
            self.output = nn.Conv2D(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias_attr=None)

    #		self.apply(self._init_weights)

    #	def _init_weights(self, m):
    #			if isinstance(m, nn.Linear):
    #				trunc_normal_(m.weight, std=.02)
    #				if isinstance(m, nn.Linear) and m.bias is not None:
    #					nn.init.constant_(m.bias, 0)
    #			elif isinstance(m, nn.LayerNorm):
    #				nn.init.constant_(m.bias, 0)
    #				nn.init.constant_(m.weight, 1.0)

    #		@torch.jit.ignore
    #		def no_weight_decay(self):
    #			return {'absolute_pos_embed'}
    #
    #		@torch.jit.ignore
    #		def no_weight_decay_keywords(self):
    #			return {'relative_position_bias_table'}

    # Encoder and Bottleneck
    def forward_features(self, x):
        x = self.patch_embedding(x)
        if self.ape:
            x = x + self.absolute_position_embedding
        x = self.position_dropout(x)
        x_downsample = []

        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)

        x = self.norm(x)  # B L C

        return x, x_downsample

    def _init_weights_layernorm(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(1.0))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    # Dencoder and Skip connection
    def forward_up_features(self, x, x_downsample):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = paddle.concat([x, x_downsample[3 - inx]], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

        x = self.norm_up(x)  # B L C

        return x

    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        if self.final_upsample == "expand_first":
            x = self.up(x)
            x = x.reshape([B, 4 * H, 4 * W, -1])
            x = x.transpose([0, 3, 1, 2])  # B,C,H,W
            x = self.output(x)

        return x

    def forward(self, x):
        x, x_downsample = self.forward_features(x)
        x = self.forward_up_features(x, x_downsample)
        x = self.up_x4(x)
        return x


def main():
    vit = SwinTransformerSys()
    print(vit)
    paddle.summary(vit, (8, 3, 224, 224))  # must be tuple


if __name__ == "__main__":
    main()