import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from .data import VOCAB


class Block(nn.Module):
    def __init__(self, width, heads=4):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.ff1 = nn.Linear(width, width * 2)
        self.ff2 = nn.Linear(width * 2, width)

    def forward(self, x):
        batch, length, width = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(batch, length, 3, self.heads, width // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.projection(attention.transpose(1, 2).reshape(batch, length, width))
        return x + self.ff2(F.gelu(self.ff1(self.norm2(x))))


class TinyLM(nn.Module):
    def __init__(self, width=64, layers=2, max_length=64):
        super().__init__()
        self.token = nn.Embedding(len(VOCAB), width)
        self.position = nn.Embedding(max_length, width)
        self.blocks = nn.ModuleList([Block(width) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, len(VOCAB))

    def forward(self, inputs):
        if inputs.shape[1] > self.position.num_embeddings:
            raise ValueError("Context length exceeded")
        x = self.token(inputs) + self.position(torch.arange(inputs.shape[1], device=inputs.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.empty(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        self.scale = 1 / math.sqrt(rank)
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.a), self.b) * self.scale


def adapt(model, rank=8):
    result = copy.deepcopy(model)
    for parameter in result.parameters():
        parameter.requires_grad_(False)

    def replace(module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, rank))
            else:
                replace(child)

    replace(result)
    return result


class Int8Linear(nn.Module):
    """Per-output-channel int8 weight storage; dequantized float32 CPU execution."""

    def __init__(self, linear):
        super().__init__()
        weight = linear.weight.detach()
        scale = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127
        self.register_buffer(
            "weight_int8", (weight / scale).round().clamp(-127, 127).to(torch.int8)
        )
        self.register_buffer("scale", scale)
        self.register_buffer(
            "bias", linear.bias.detach().clone() if linear.bias is not None else None
        )

    def forward(self, x):
        return F.linear(x, self.weight_int8.float() * self.scale, self.bias)


def quantize(model):
    result = copy.deepcopy(model).eval()

    def replace(module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, name, Int8Linear(child))
            else:
                replace(child)

    replace(result)
    return result
