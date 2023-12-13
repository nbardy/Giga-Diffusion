import random
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline
from diffusers.utils import load_image
import torch

from datasets import concatenate_datasets, load_dataset
from dataset_loaders.multiscale_ar_text_to_image import MultiScaleTextToImage
from train_scripts.utils.gan_loss import FullLoss
from prodigyopt import Prodigy
import wandb


# Accelerator
from accelerate import Accelerator

import argparse


# shuffle arg should default true
# Shuffle seed 32
# Adds multi scale and aspect ratio cropping
def get_dataset(args):
    datasets = []
    for dataset_name in args.dataset_names:
        datasets.append(load_dataset(dataset_name, split="train"))

    dataset_all = concatenate_datasets(datasets)
    if args.shuffle_dataset:
        dataset_all = dataset_all.shuffle(seed=args.shuffle_seed)

    # Each batch will be a different size
    dataset_all = MultiScaleTextToImage(dataset_all)

    return dataset_all


class DiffusionModel:
    # Defaults to sd-turbo, can also load picsart-xl
    def _init_(self, args):
        base = DiffusionPipeline.from_pretrained(
            args.base_model_name,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        base.to(accelerator.device)
        self.pipeline = base

    # Single step diffusion
    def forward(self, image_batch=None, image_strength=0, prompts=None):
        assert (
            image_batch is not None or prompts is not None
        ), "Must pass either image batch or prompts"

        return self.pipeline(
            prompts=prompts, images=image_batch, strength=image_strength, steps=1
        ).images

    # Will call a full step at 0 strength, then at 0.5 stregnth, then at 0.75 strength, etc..
    def multi_step_render(self, prompts=None, step_schedule=[1.0, 0.5, 0.25, 0.125]):
        assert prompts is not None, "Must pass prompts"

        results = []
        image_batch = None
        for step_strength in step_schedule:
            image_batch = self.forward(
                image_batch, image_strength=step_strength, prompts=prompts
            )
            results.append(image_batch)

        return results


# accepts a dict of losses
# Should load all prompts from file and render once at a single step.
def log_validation_samples(model, args, step):
    # Log examples for each prompt at 1, 2, 3, 4 steps
    # Should create a dictionary of wanbdb.Image arrays
    # {prompt-" render 1": [wandb.Image at step 1, 2, 3, 4....], ...(should repreat many steps for a render)}
    all_images = {}
    for prompt in args.prompts:
        for render_index in range(args.num_renders):
            images = model.multi_step_render(
                prompt, step_schedule=[1.0, 0.5, 0.25, 0.125]
            )
            images = [wandb.Image(image) for image in images]

            all_images[f"{prompt} render {render_index}"] = images

    wandb.log(all_images, step=step)
    return


# Should do 0-1 with ^3 exp boost
def get_random_strength():
    x = random.random()
    return x**3


# Load SDXL model
# This uses no gradient this is only in inference

# Reduffuse the image by 0.2 with 8 steps or 0.3 with 12 steps or 0.4 with 16 steps
# Teacher model is slow base model
teacher_step_types = {"0.2": 8, "0.3": 12, "0.4": 16}


## Diffusion teacher uses a larger bathc size pre-trained model full step count model
class DiffusionTeacher:
    def __init__(self, args):
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to("cuda")

    # use types get one of the keys
    def get_random_strength(self):
        # key from step_types
        strength = random.choice(list(teacher_step_types.keys()))
        return strength

    def forward(self, image):
        # Pass the strength image through the pipeline
        strength = get_random_strength()
        output = self.pipeline(
            image=image, strength=strength, steps=teacher_step_types[strength]
        )

        return output


# Try prodigy optimizer.
#
# Autotuning optimizer from META
def get_opt(model, args):
    betas_tuple = tuple(args.prodigy_betas)

    return Prodigy(
        model.parameters(),
        lr=1.0,
        weight_decay=args.weight_decay,
        safeguard_warmup=args.prodigy_safeguard_warmup,
        use_bias_correction=args.prodify_use_bias_correction,
        betas=betas_tuple,
    )


# Train script employs GAN losses from giga-GAN, clip contrastive loss, lpips, distance and teacher losses
# GigaGan Paper showsed stable single step high res sampling with thes loss
def train(args, accelerator):
    loss_model = FullLoss(args)
    model = DiffusionModel(args)
    teacher_model = DiffusionTeacher(args)
    dataset = get_dataset(args)

    # 	model trainable
    model.train()
    loss_model.train()

    opt_gen = get_opt(model, args)
    opt_disc = get_opt(loss_model, args)

    step = 0

    for batch in dataset:
        with accelerator.accumulate(model, loss_model):
            step += 1
            # Append real and fake photos with labels an
            random_strength = get_random_strength()

            denoised_images = model.forward(
                prompts=batch["prompts"],
                images=batch["images"],
                strength=random_strength,
            )

            # Teacher images
            # With no grad
            with torch.no_grad():
                teacher_images = teacher_model.forward(
                    prompts=batch["prompts"],
                    images=batch["images"],
                    strength=random_strength,
                )

            # Calulate loss
            loss, loss_dict = loss_model.forward(
                real_images=batch["images"],
                fake_images=denoised_images,
                teacher_images=teacher_images,
                args=args,
            )

            loss.backward()

            opt_disc.step()
            opt_gen.step()
            # zero grad
            opt_disc.zero_grad()
            opt_gen.zero_grad()

            wandb.log(loss_dict)

            # validation
            if step % args.validation_interval == 0:
                log_validation_samples(model, args, step)


default_base_model_name = "runwayml/stable-diffusion-v1-5"

def get_args():
    parser = argparse.ArgumentParser(description="Training script for SD Turbo")
    parser = argparse.ArgumentParser(
        "--base_model_name", type=str, default=default_base_model_name, help="Base model name"
    )

    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of epochs for training"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0001,
        help="Weight decay for the optimizer",
    )
    parser.add_argument(
        "--lpips_alex_scale",
        type=float,
        default=0.5,
        help="Scale factor for LPIPS AlexNet loss",
    )
    parser.add_argument(
        "--lpips_vgg_scale",
        type=float,
        default=0.5,
        help="Scale factor for LPIPS VGG loss",
    )
    parser.add_argument(
        "--lpips_scale", type=float, default=1.0, help="Scale factor for LPIPS loss"
    )
    parser.add_argument(
        "--text_adherance_scale",
        type=float,
        default=1.0,
        help="Scale factor for text adherance loss",
    )
    parser.add_argument(
        "--discriminator_scale",
        type=float,
        default=1.0,
        help="Scale factor for discriminator loss",
    )
    parser.add_argument(
        "--multi_scale_multi_res",
        type=bool,
        default=False,
        help="Flag for multi-scale multi-resolution training",
    )
    # gradient accumulation steps
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--dataset_names",
        type=list,
        default=["lambdalabs/pokemon-blip-captions"],
        help="Dataset names",
    )
    parser.add_argument(
        "--safeguard_warmup",
        default=True,
        type=bool,
        help="Safeguard warmup",
    )
    parser.add_argument(
        "--prodify_use_bias_correction",
        default=True,
        type=bool,
        help="Use bias correction",
    )
    parser.add_argument(
        "--prodigy_betas",
        default=[0.9, 0.99],
        type=list,
        help="Betas for prodigy optimizer",
    )

    args = parser.parse_args()
    return args


# Run
if __name__ == "__main__":
    args = get_args()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    train(args, accelerator)
