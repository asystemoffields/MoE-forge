"""Compute-in-compressed-form helpers.

The verifier measures RESIDENT bytes = the decompressed module's param+buffer footprint, so a
method only wins the memory axis (which, for bandwidth-bound local decode, is also the first-order
speed axis) if it actually RUNS while holding weights compressed. These modules hold packed weights
as buffers and dequantize transiently in forward.

In decompress(): build the model with from_config, replace its nn.Linear layers with Int4Linear
(or CompressedLinear for arbitrary formats), load the non-matmul params (embeddings/norms) at low
precision too -- everything resident counts. Casting the model to bfloat16 keeps embeddings small
and runs on CPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def pack_int4_grouped(weight, group_size=128):
    """weight [out, in] -> (packed uint8 [out, ceil(in_padded/2)], scales fp16 [out, n_groups], in)."""
    W = weight.detach().float()
    out_f, in_f = W.shape
    pad = (-in_f) % group_size
    if pad:
        W = torch.cat([W, torch.zeros(out_f, pad)], dim=1)
    in_p = W.shape[1]
    n_groups = in_p // group_size
    Wg = W.view(out_f, n_groups, group_size)
    scales = (Wg.abs().amax(dim=2, keepdim=True) / 7.0).clamp(min=1e-8)
    codes = (torch.clamp(torch.round(Wg / scales), -7, 7) + 8).view(out_f, in_p).to(torch.uint8)
    packed = (codes[:, 0::2] << 4) | (codes[:, 1::2] & 0xF)
    return packed.to(torch.uint8), scales.squeeze(2).to(torch.float16), in_f


def dequant_int4_grouped(packed, scales_f16, in_f, group_size=128):
    out_f = packed.shape[0]
    in_p = packed.shape[1] * 2
    codes = torch.empty(out_f, in_p, dtype=torch.float32)
    codes[:, 0::2] = ((packed >> 4) & 0xF).float()
    codes[:, 1::2] = (packed & 0xF).float()
    codes -= 8.0
    n_groups = in_p // group_size
    W = (codes.view(out_f, n_groups, group_size) * scales_f16.float().unsqueeze(2)).view(out_f, in_p)
    return W[:, :in_f]


class Int4Linear(nn.Module):
    """Holds int4-grouped packed weight (uint8 buffer) + fp16 scales; dequantizes in forward."""

    def __init__(self, packed, scales, out_features, in_features, bias=None, group_size=128):
        super().__init__()
        self.register_buffer("packed", packed)
        self.register_buffer("scales", scales)
        self.out_features = int(out_features)
        self.in_features = int(in_features)
        self.group_size = int(group_size)
        self.register_buffer("bias", bias)

    @classmethod
    def from_weight(cls, weight, bias=None, group_size=128):
        packed, scales, in_f = pack_int4_grouped(weight, group_size)
        b = bias.detach().to(torch.float16) if bias is not None else None
        return cls(packed, scales, weight.shape[0], in_f, b, group_size)

    def forward(self, x):
        W = dequant_int4_grouped(self.packed, self.scales, self.in_features, self.group_size).to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, W, b)


class CompressedLinear(nn.Module):
    """General holder for arbitrary formats: keeps packed state tensors as buffers (counted in
    resident) plus a dequant function (state -> weight [out, in]); dequantizes transiently."""

    def __init__(self, state, dequant, out_features, in_features, bias=None):
        super().__init__()
        self._dequant = dequant
        self._keys = list(state.keys())
        for k, v in state.items():
            self.register_buffer(k, v if torch.is_tensor(v) else torch.as_tensor(v))
        self.out_features = int(out_features)
        self.in_features = int(in_features)
        self.register_buffer("bias", bias)

    def forward(self, x):
        W = self._dequant({k: getattr(self, k) for k in self._keys})
        if not torch.is_tensor(W):
            W = torch.as_tensor(W)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, W.to(x.dtype), b)


def replace_linears(model, build):
    """Replace every nn.Linear in model with build(full_dotted_name, linear_module). Returns model."""
    for name, module in model.named_modules():
        for cname, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                full = f"{name}.{cname}" if name else cname
                setattr(module, cname, build(full, child))
    return model
