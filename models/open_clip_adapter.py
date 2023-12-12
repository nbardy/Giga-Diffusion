# https://github.com/lucidrains/gigagan-pytorch/blob/3df2a5e23ee60ed7d1f01083d879a525e38c8667/gigagan_pytorch/gigagan_pytorch.py#L1398
#
# A lot of this comes from lucidrains' gigagan-pytorch repo
import torch
from torch import nn, einsum
import torch.nn.functional as F
import open_clip

from einops import rearrange

from beartype import beartype
from beartype.typing import List, Optional
from functools import partial
from beartype import beartype
from beartype.typing import List, Optional, Tuple, Dict, Union, Iterable

from einops import rearrange, pack, unpack, repeat, reduce
from einops.layers.torch import Rearrange, Reduce

from collections import namedtuple
from pathlib import Path
from math import log2, sqrt
from random import random
from functools import partial

from torchvision import utils

import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor

from beartype import beartype
from beartype.typing import List, Optional, Dict, Union, Iterable

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from open_clip import create_model_from_pretrained, get_tokenizer


def exists(val):
    return val is not None


def l2norm(t):
    return F.normalize(t, dim=-1)


apple_model_large = "apple/DFN5B-CLIP-ViT-H-14-378"
open_clip_apple_pretrained = "hf-hub:" + apple_model_large
tokenizer_name = "ViT-L-14"

big_timm_siglip = "timm/ViT-B-16-SigLIP-512"
hub_siglip = "hf-hub:" + big_timm_siglip


class OpenClipAdapter(nn.Module):
    @beartype
    def __init__(
        self,
        name="ViT-B/32",
        pretrained="laion400m_e32",
        tokenizer_name="ViT-B-32-quickgelu",
        eos_id=49407,
    ):
        super().__init__()

        tokenizer = open_clip.get_tokenizer(tokenizer_name)

        clip_model, preprocess = create_model_from_pretrained(pretrained)
        tokenizer = get_tokenizer(tokenizer_name)

        self.clip = clip_model
        self.tokenizer = tokenizer
        self.eos_id = eos_id

        # hook for getting final text representation

        text_attention_final = self.find_layer("ln_final")
        self._dim_latent = text_attention_final.weight.shape[0]
        self.text_handle = text_attention_final.register_forward_hook(self._text_hook)

        # hook for getting final image representation
        # this is for vision-aided gan loss

        self._dim_image_latent = self.find_layer("visual.ln_post").weight.shape[0]

        num_visual_layers = len(clip_model.visual.transformer.resblocks)
        self.image_handles = []

        for visual_layer in range(num_visual_layers):
            image_attention_final = self.find_layer(
                f"visual.transformer.resblocks.{visual_layer}"
            )

            handle = image_attention_final.register_forward_hook(self._image_hook)
            self.image_handles.append(handle)

        # normalize fn

        self.clip_normalize = preprocess.transforms[-1]
        self.cleared = False

    @property
    def device(self):
        return next(self.parameters()).device

    def find_layer(self, layer):
        print("find layer", layer)
        print(self.clip)
        # debugger()
        # import pdb

        # pdb.set_trace()

        modules = dict(
            {
                "text": [*self.clip.transformer.named_modules()],
                "visual": [*self.clip.visual.named_modules()],
            }
        )
        print(modules)
        return modules.get(layer, None)

    def clear(self):
        if self.cleared:
            return

        self.text_handle()
        self.image_handle()

    def _text_hook(self, _, inputs, outputs):
        self.text_encodings = outputs

    def _image_hook(self, _, inputs, outputs):
        if not hasattr(self, "image_encodings"):
            self.image_encodings = []

        self.image_encodings.append(outputs)

    @property
    def dim_latent(self):
        return self._dim_latent

    @property
    def image_size(self):
        image_size = self.clip.visual.image_size
        if isinstance(image_size, tuple):
            return max(image_size)
        return image_size

    @property
    def image_channels(self):
        return 3

    @property
    def max_text_len(self):
        return self.clip.positional_embedding.shape[0]

    @beartype
    def embed_texts(self, texts: List[str]):
        ids = self.tokenizer(texts)
        ids = ids.to(self.device)
        ids = ids[..., : self.max_text_len]

        is_eos_id = ids == self.eos_id
        text_mask_excluding_eos = is_eos_id.cumsum(dim=-1) == 0
        text_mask = F.pad(text_mask_excluding_eos, (1, -1), value=True)
        text_mask = text_mask & (ids != 0)
        assert not self.cleared

        text_embed = self.clip.encode_text(ids)
        text_encodings = self.text_encodings
        text_encodings = text_encodings.masked_fill(~text_mask[..., None], 0.0)
        del self.text_encodings
        return l2norm(text_embed.float()), text_encodings.float()

    def embed_images(self, images):
        if images.shape[-1] != self.image_size:
            images = F.interpolate(images, self.image_size)

        assert not self.cleared
        images = self.clip_normalize(images)
        image_embeds = self.clip.encode_image(images)

        image_encodings = rearrange(self.image_encodings, "l n b d -> l b n d")
        del self.image_encodings

        return l2norm(image_embeds.float()), image_encodings.float()

    @beartype
    def contrastive_loss(
        self,
        images,
        texts: Optional[List[str]] = None,
        text_embeds: Optional[torch.Tensor] = None,
    ):
        assert exists(texts) ^ exists(text_embeds)

        if not exists(text_embeds):
            text_embeds, _ = self.embed_texts(texts)

        image_embeds, _ = self.embed_images(images)

        n = text_embeds.shape[0]

        temperature = self.clip.logit_scale.exp()
        sim = einsum("i d, j d -> i j", text_embeds, image_embeds) * temperature

        labels = torch.arange(n, device=sim.device)

        return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2


class RandomFixedProjection(nn.Module):
    def __init__(self, dim, dim_out, channel_first=True):
        super().__init__()
        weights = torch.randn(dim, dim_out)
        nn.init.kaiming_normal_(weights, mode="fan_out", nonlinearity="linear")

        self.channel_first = channel_first
        self.register_buffer("fixed_weights", weights)

    def forward(self, x):
        if not self.channel_first:
            return x @ self.fixed_weights

        return einsum("b c ..., c d -> b d ...", x, self.fixed_weights)


def default(*vals):
    for val in vals:
        if exists(val):
            return val
    return None


def leaky_relu(neg_slope=0.2):
    return nn.LeakyReLU(neg_slope)


def conv2d_3x3(dim_in, dim_out):
    return nn.Conv2d(dim_in, dim_out, 3, padding=1)


# adaptive conv
# the main novelty of the paper - they propose to learn a softmax weighted sum of N convolutional kernels, depending on the text embedding


def get_same_padding(size, kernel, dilation, stride):
    return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2


class AdaptiveConv2DMod(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        kernel,
        *,
        demod=True,
        stride=1,
        dilation=1,
        eps=1e-8,
        num_conv_kernels=1,  # set this to be greater than 1 for adaptive
    ):
        super().__init__()
        self.eps = eps

        self.dim_out = dim_out

        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.adaptive = num_conv_kernels > 1

        self.weights = nn.Parameter(
            torch.randn((num_conv_kernels, dim_out, dim, kernel, kernel))
        )

        self.demod = demod

        nn.init.kaiming_normal_(
            self.weights, a=0, mode="fan_in", nonlinearity="leaky_relu"
        )

    def forward(
        self, fmap, mod: Optional[Tensor] = None, kernel_mod: Optional[Tensor] = None
    ):
        """
        notation

        b - batch
        n - convs
        o - output
        i - input
        k - kernel
        """

        b, h = fmap.shape[0], fmap.shape[-2]

        # account for feature map that has been expanded by the scale in the first dimension
        # due to multiscale inputs and outputs

        if mod.shape[0] != b:
            mod = repeat(mod, "b ... -> (s b) ...", s=b // mod.shape[0])

        if exists(kernel_mod):
            kernel_mod_has_el = kernel_mod.numel() > 0

            assert self.adaptive or not kernel_mod_has_el

            if kernel_mod_has_el and kernel_mod.shape[0] != b:
                kernel_mod = repeat(
                    kernel_mod, "b ... -> (s b) ...", s=b // kernel_mod.shape[0]
                )

        # prepare weights for modulation

        weights = self.weights

        if self.adaptive:
            weights = repeat(weights, "... -> b ...", b=b)

            # determine an adaptive weight and 'select' the kernel to use with softmax

            assert exists(kernel_mod) and kernel_mod.numel() > 0

            kernel_attn = kernel_mod.softmax(dim=-1)
            kernel_attn = rearrange(kernel_attn, "b n -> b n 1 1 1 1")

            weights = reduce(weights * kernel_attn, "b n ... -> b ...", "sum")

        # do the modulation, demodulation, as done in stylegan2

        mod = rearrange(mod, "b i -> b 1 i 1 1")

        weights = weights * (mod + 1)

        if self.demod:
            inv_norm = (
                reduce(weights**2, "b o i k1 k2 -> b o 1 1 1", "sum")
                .clamp(min=self.eps)
                .rsqrt()
            )
            weights = weights * inv_norm

        fmap = rearrange(fmap, "b c h w -> 1 (b c) h w")

        weights = rearrange(weights, "b o ... -> (b o) ...")

        padding = get_same_padding(h, self.kernel, self.dilation, self.stride)
        fmap = F.conv2d(fmap, weights, padding=padding, groups=b)

        return rearrange(fmap, "1 (b o) ... -> b o ...", b=b)


class VisionAidedDiscriminator(nn.Module):
    """the vision-aided gan loss"""

    @beartype
    def __init__(
        self,
        *,
        depth=2,
        dim_head=64,
        heads=8,
        clip: Optional[OpenClipAdapter] = None,
        layer_indices=(-1, -2, -3),
        conv_dim=None,
        text_dim=None,
        unconditional=False,
        num_conv_kernels=2,
    ):
        super().__init__()

        if not exists(clip):
            clip = OpenClipAdapter()

        self.clip = clip
        dim = clip._dim_image_latent

        self.unconditional = unconditional
        text_dim = default(text_dim, dim)
        conv_dim = default(conv_dim, dim)

        self.layer_discriminators = nn.ModuleList([])
        self.layer_indices = layer_indices

        conv_klass = (
            partial(AdaptiveConv2DMod, kernel=3, num_conv_kernels=num_conv_kernels)
            if not unconditional
            else conv2d_3x3
        )

        for _ in layer_indices:
            self.layer_discriminators.append(
                nn.ModuleList(
                    [
                        RandomFixedProjection(dim, conv_dim),
                        conv_klass(conv_dim, conv_dim),
                        nn.Linear(text_dim, conv_dim) if not unconditional else None,
                        nn.Linear(text_dim, num_conv_kernels)
                        if not unconditional
                        else None,
                        nn.Sequential(
                            conv2d_3x3(conv_dim, 1), Rearrange("b 1 ... -> b ...")
                        ),
                    ]
                )
            )

    def parameters(self):
        return self.layer_discriminators.parameters()

    @property
    def total_params(self):
        return sum([p.numel() for p in self.parameters()])

    @beartype
    def forward(
        self,
        images,
        texts: Optional[List[str]] = None,
        text_embeds: Optional[Tensor] = None,
        return_clip_encodings=False,
    ):
        assert self.unconditional or (exists(text_embeds) ^ exists(texts))

        with torch.no_grad():
            if not self.unconditional and exists(texts):
                self.clip.eval()
                text_embeds = self.clip.embed_texts

        _, image_encodings = self.clip.embed_images(images)

        logits = []

        for layer_index, (
            rand_proj,
            conv,
            to_conv_mod,
            to_conv_kernel_mod,
            to_logits,
        ) in zip(self.layer_indices, self.layer_discriminators):
            image_encoding = image_encodings[layer_index]

            cls_token, rest_tokens = image_encoding[:, :1], image_encoding[:, 1:]
            height_width = int(sqrt(rest_tokens.shape[-2]))  # assume square

            img_fmap = rearrange(rest_tokens, "b (h w) d -> b d h w", h=height_width)

            img_fmap = img_fmap + rearrange(
                cls_token, "b 1 d -> b d 1 1 "
            )  # pool the cls token into the rest of the tokens

            img_fmap = rand_proj(img_fmap)

            if self.unconditional:
                img_fmap = conv(img_fmap)
            else:
                assert exists(text_embeds)

                img_fmap = conv(
                    img_fmap,
                    mod=to_conv_mod(text_embeds),
                    kernel_mod=to_conv_kernel_mod(text_embeds),
                )

            layer_logits = to_logits(img_fmap)

            logits.append(layer_logits)

        if not return_clip_encodings:
            return logits

        return logits, image_encodings
