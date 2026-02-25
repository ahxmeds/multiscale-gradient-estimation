import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from abc import abstractmethod

# ============================================================================
# Utility Functions and Classes
# ============================================================================

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

def conv_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def linear(*args, **kwargs):
    return nn.Linear(*args, **kwargs)

def avg_pool_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

def normalization(channels):
    return GroupNorm32(16, channels)

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def checkpoint(func, inputs, params, flag):
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)

class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors
    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads

# ============================================================================
# ResNet Implementation
# ============================================================================

class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim, hidden_dim=32):
        super(TimeEmbedding, self).__init__()
        self.sin_cos_embed_dim = embed_dim // 2
        self.embed_dim = embed_dim
        self.out_dim = embed_dim - embed_dim // 2
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, self.out_dim)

    def forward(self, t):
        half_dim = self.sin_cos_embed_dim // 2
        emb = (torch.arange(half_dim, dtype=torch.float32) / half_dim).to(t.device)
        emb = t[:, None] * torch.exp(-math.log(10000) * emb)
        sin_cos_emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        t_trainable = t.unsqueeze(-1)
        x = F.relu(self.fc1(t_trainable))
        x = F.relu(self.fc2(x))
        trainable_emb = self.fc3(x)
        combined_emb = torch.cat([sin_cos_emb, trainable_emb], dim=-1)
        return combined_emb

class resnet_block(nn.Module):
    def __init__(self, in_channels, out_channels, hid_channels, time_embed=True):
        super(resnet_block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Conv1 = nn.Conv2d(in_channels, hid_channels, kernel_size=3, padding=1)
        self.Conv2 = nn.Conv2d(hid_channels, out_channels, kernel_size=3, padding=1)
        if time_embed:
            self.te = TimeEmbedding(hid_channels, hidden_dim=32)
        self.time_embed = time_embed

    def forward(self, x, t=[]):
        q = self.Conv1(x)
        if self.time_embed:
            te = self.te(t).unsqueeze(-1).unsqueeze(-1)
            q = q + te
        q = F.layer_norm(q, (q.shape[1], q.shape[2], q.shape[3]))
        q = F.leaky_relu(q, negative_slope=0.2)
        out = self.Conv2(q)
        return out

class resnet(nn.Module):
    def __init__(self, in_channels, out_channels, hid_channels, nlayers, time_embed=True):
        super(resnet, self).__init__()
        self.Open = nn.Conv2d(in_channels, hid_channels, kernel_size=1)
        self.blocks = nn.ModuleList()
        for i in range(nlayers):
            ly = resnet_block(hid_channels, hid_channels, hid_channels, time_embed)
            self.blocks.append(ly)
        self.Close = nn.Conv2d(hid_channels, out_channels, kernel_size=1)
        self.Close.weight.data = 1e-3 * self.Close.weight.data

    def forward(self, x, t=[]):
        x = self.Open(x)
        for B in self.blocks:
            dx = B(x, t)
            x = x + dx
        x = self.Close(x)
        return x

# ============================================================================
# UNet with Attention Implementation
# ============================================================================

class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        pass

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class Upsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class ResBlock(TimestepBlock):
    def __init__(self, channels, emb_channels, dropout, out_channels=None, use_conv=False,
                 use_scale_shift_norm=False, dims=2, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)

class QKVAttention(nn.Module):
    def forward(self, qkv):
        ch = qkv.shape[1] // 3
        q, k, v = torch.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        return torch.einsum("bts,bcs->bct", weight, v)

class UNetModel(nn.Module):
    def __init__(self, in_channels, model_channels, out_channels, num_res_blocks,
                 attention_resolutions, dropout=0, channel_mult=(1, 2, 4, 8),
                 conv_resample=True, dims=2, num_classes=None, use_checkpoint=False,
                 num_heads=1, num_heads_upsample=-1, use_scale_shift_norm=False):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(ch, time_embed_dim, dropout, out_channels=mult * model_channels,
                             dims=dims, use_checkpoint=use_checkpoint,
                             use_scale_shift_norm=use_scale_shift_norm)
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims)))
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint,
                     use_scale_shift_norm=use_scale_shift_norm),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads),
            ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint,
                     use_scale_shift_norm=use_scale_shift_norm),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(ch + input_block_chans.pop(), time_embed_dim, dropout,
                             out_channels=model_channels * mult, dims=dims,
                             use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads_upsample))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    @property
    def inner_dtype(self):
        return next(self.input_blocks.parameters()).dtype

    def forward(self, x, timesteps, y=None):
        assert (y is not None) == (self.num_classes is not None), \
            "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
        h = h.type(x.dtype)
        return self.out(h)


# ============================================================================
# Super-Resolution Network
# ============================================================================

class SRParaConvNet(nn.Module):
    """
    Super-resolution network for 2x upscaling.
    Uses residual learning: output = model(input) + bilinear_upsample(input)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        hid_channels: int = 64,
        upscale_factor: int = 2,
    ):
        super().__init__()

        # Build convolutional layers
        # Layer configs: (in_channels, out_channels, kernel_size, padding)
        layer_configs = [
            (in_channels, hid_channels, 5, 2),
            (hid_channels, hid_channels, 3, 1),
            (hid_channels, 32, 3, 1),
            (32, 32, 3, 1),
        ]
        
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=1, padding=p)
            for in_ch, out_ch, k, p in layer_configs
        ])

        # Final 1x1 projection to output channels
        self.output_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        self.output_conv.weight.data = 1e-3 * torch.randn(out_channels, 32, 1, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with orthogonal initialization."""
        relu_gain = nn.init.calculate_gain("relu")
        for i, layer in enumerate(self.conv_layers):
            gain = relu_gain if i < len(self.conv_layers) - 1 else 1.0
            nn.init.orthogonal_(layer.weight, gain)

    @staticmethod
    def _conv_block(x: torch.Tensor, conv: nn.Module) -> torch.Tensor:
        """Apply convolution + layer norm + leaky ReLU."""
        x = conv(x)
        x = F.layer_norm(x, x.shape[1:])
        x = F.leaky_relu(x, negative_slope=0.2)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Input should already be upsampled by 2x.
        Returns residual to be added to the upsampled input.
        """
        for conv in self.conv_layers:
            x = self._conv_block(x, conv)

        return self.output_conv(x)
