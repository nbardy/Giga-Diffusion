from beartype import beartype
from beartype.typing import List, Optional, Tuple, Dict, Union, Iterable

# bear
from torch import nn, einsum, Tensor

from models.open_clip_adapter import OpenClipAdapter, VisionAidedDiscriminator
from models.sam_discriminator import ViTDiscriminator, MultiscaleViTDiscriminator
from train_scripts.utils.distributed import all_gather

import open_clip
import lpips

import torch.nn as nn
from torch.nn import functional as F


def exists(val):
    return val is not None


@beartype
def aux_clip_loss(
    clip: OpenClipAdapter,
    images: Tensor,
    texts: Optional[List[str]] = None,
    text_embeds: Optional[Tensor] = None,
):
    assert exists(texts) ^ exists(text_embeds)

    images, batch_sizes = all_gather(images, 0, None)

    if exists(texts):
        text_embeds, _ = clip.embed_texts(texts)
        text_embeds, _ = all_gather(text_embeds, 0, batch_sizes)

    return clip.contrastive_loss(images=images, text_embeds=text_embeds)


def load_lpips_models():
    loss_fn_alex = lpips.LPIPS(net="alex")  # best forward scores
    loss_fn_vgg = lpips.LPIPS(
        net="vgg"
    )  # closer to "traditional" perceptual loss, when used for optimization

    return loss_fn_alex, loss_fn_vgg


def lpips_loss(image_batch_1, image_batch_2, lpips_models, args):
    loss_fn_alex, loss_fn_vgg = lpips_models
    l1 = loss_fn_alex(image_batch_1, image_batch_2)
    l2 = loss_fn_vgg(image_batch_1, image_batch_2)

    return l1 * args.lpips_alex_scale + l2 * args.lpips_vgg_scale


# min side
lpips_scales = [128, 256, 512]


def multi_scale_lpips_loss(image_batch_1, image_batch_2, lpips_models, args):
    total = 0
    for scale in lpips_scales:
        image_1_scaled = F.interpolate(image_batch_1, size=(scale, scale))
        image_2_scaled = F.interpolate(image_batch_2, size=(scale, scale))
        loss_fn_alex, loss_fn_vgg = lpips_models
        l1 = loss_fn_alex(image_1_scaled, image_2_scaled)
        l2 = loss_fn_vgg(image_1_scaled, image_2_scaled)

        scale_loss = l1 * args.lpips_alex_scale + l2 * args.lpips_vgg_scale
        total += scale_loss

    return total / len(lpips_scales)


def discriminator_hinge_loss(real, fake):
    return (F.relu(1 + real) + F.relu(1 - fake)).mean()


class FullLoss(nn.Module):
    # Uses Lpips, discriminator, clip, mse, l1, teacher
    def __init__(self, args):
        super().__init__()

        # Pretrained LPIPS model for various similarity metrics
        self.lpips_models = load_lpips_models()

        # Train a ViT Discriminator on top of a pretrained ViT model
        # Use featurs of SAM model features of frozen ViT model with linear discriminator on top
        self.vit_discriminator = ViTDiscriminator()

        self.multi_scalevit_discriminator = MultiscaleViTDiscriminator(
            vit=self.vit_discriminator.vit
        )

        # Train discriminator on top of CLIP
        # open_coca = OpenClipAdapter(
        #     name="coca_ViT-L-14", pretrained="mscoco_finetuned_laion2B-s13B-b90k"
        # )
        apple_clip_pretrained = "hf-hub:apple/DFN5B-CLIP-ViT-H-14-384"
        name = "ViT-H/14-384"
        clip_adapter = OpenClipAdapter(pretrained=apple_clip_pretrained)
        self.clip_discriminator = VisionAidedDiscriminator(clip=clip_adapter)
        self.clip_model = clip_adapter

        # clip_model, preprocess = create_model_from_pretrained()
        # tokenizer = get_tokenizer('ViT-H-14')

    def foward(self, prompts, real_images, fake_images, teacher_images, args):
        # Discriminator loss
        real_logits = self.vit_discriminator.loss(real_images)
        fake_logits = self.vit_discriminator.loss(fake_images)

        vit_disc_loss = discriminator_hinge_loss(real_logits, fake_logits)

        # Multi scale discriminator loss
        if self.multi_scalevit_discriminator > 0:
            real_logits = self.multi_scalevit_discriminator.loss(real_images)
            fake_logits = self.vit_discriminator.loss(fake_images)

            scaled_vit_disc_loss = discriminator_hinge_loss(real_logits, fake_logits)

        # clip aided discriminator loss
        fake_logits = self.clip_discriminator.loss(fake_images)
        real_logits = self.clip_discriminator.loss(real_images)

        clip_discrim_loss = discriminator_hinge_loss(real_logits, fake_logits)

        # Similarity loss
        lpips_loss = (
            lpips_loss(real_images, fake_images, self.lpips_models, args)
            if args.lpips_loss_scale != 0
            else 0
        )
        multi_scale_lpips_loss = (
            multi_scale_lpips_loss(real_images, fake_images, self.lpips_models, args)
            if args.multi_scale_lpips_loss_scale != 0
            else 0
        )
        mse_loss = (
            F.mse_loss(real_images, fake_images) if args.mse_loss_scale != 0 else 0
        )
        l1_loss = F.l1_loss(real_images, fake_images) if args.l1_loss_scale != 0 else 0

        # Teacher losses
        # if scale is zero skip
        teacher_mse_loss = (
            F.mse_loss(real_images, teacher_images)
            if args.teacher_mse_loss_scale != 0
            else 0
        )
        teacher_l1_loss = (
            F.l1_loss(real_images, teacher_images)
            if args.teacher_l1_loss_scale != 0
            else 0
        )
        teacher_lpips_loss = (
            lpips_loss(real_images, teacher_images, self.lpips_models, args)
            if args.teacher_lips_loss_scale != 0
            else 0
        )

        # Contrastive clip loss
        # We use clip feature source for a discriminator and as an auxiliary loss for contrasting the
        # fit of the prompt itself. This is amore of a boosting auxiliary loss
        clip_contrastive_loss = (
            aux_clip_loss(
                clip=self.clip_model.clip,
                texts=prompts,
                images=fake_images,
            )
            if args.contrastive_loss_scale != 0
            else 0
        )

        loss_dict = {
            "vit_discriminator_loss": vit_disc_loss,
            "multi_scale_vit_discriminator_loss": scaled_vit_disc_loss,
            "clip_discrim_loss": clip_discrim_loss,
            "lpips_loss": lpips_loss,
            "multi_scale_lpips_loss": multi_scale_lpips_loss,
            "mse_loss": mse_loss,
            "l1_loss": l1_loss,
            "teacher_mse_loss": teacher_mse_loss,
            "teacher_l1_loss": teacher_l1_loss,
            "teacher_lpips_loss": teacher_lpips_loss,
            "clip_contrastive_loss": clip_contrastive_loss,
        }

        return (
            args.vit_disc_loss_scale * vit_disc_loss
            + args.scaled_vit_disc_loss_scale * scaled_vit_disc_loss
            + args.clip_loss_scale * clip_discrim_loss
            + args.lpips_loss_scale * lpips_loss
            + args.multi_scale_lpips_loss_scale * multi_scale_lpips_loss
            + args.mse_loss_scale * mse_loss
            + args.l1_loss_scale * l1_loss
            + args.teacher_mse_loss_scale * teacher_mse_loss
            + args.teacher_l1_loss_scale * teacher_l1_loss
            + args.teacher_lpips_loss_scale * teacher_lpips_loss
            + args.contrastive_loss_scale * clip_contrastive_loss,
            loss_dict,
        )
