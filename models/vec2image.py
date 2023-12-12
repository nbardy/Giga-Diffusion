# This file is for testing the multimodal pipeline
# 
# The base 'encode_prompt' function is copied from diffusers base SDXL
# It can only encode a single text prompt to embeddings
# 
# We want to create a new MultiModalPipeline class that can be used to encode
# a list of text and image prompts to embeddings
from PIL import Image

import torch
import diffusers
import transformers


small_model = "openai/clip-vit-large-patch14"
big_model = "laion/CLIP-ViT-g-14-laion2B-s12B-b42K"

# Example code for calling and loading clip
# 
# from PIL import Image
# import requests

# from transformers import CLIPProcessor, CLIPModel

# model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

# outputs = model(**inputs)
# logits_per_image = outputs.logits_per_image # this is the image-text similarity score
# probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

def tokenize_prompt(tokenizer, prompt):
    return tokenizer(
        [prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )


# Encode prompt is before refactoring
def encode_prompt_old(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            **text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds



# Refactored encode_promp that splits out the inner loop for clarity
def encode_for_model(model, tokenizer, prompt, text_input_ids=None):
    if tokenizer is not None:
        text_input_ids = tokenize_prompt(tokenizer, prompt)
    else:
        assert text_input_ids is not None

    prompt_embeds = model(
        **text_input_ids.to(model.device),
        output_hidden_states=True,
    )

    # We are only ALWAYS interested in the pooled output of the final text encoder
    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    bs_embed, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)

    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds

# Refactored encode_prompt that can encode a list of prompts
def encode_prompt(models, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []

    for i, model in enumerate(models):
        prompt_embeds, pooled_prompt_embeds = encode_for_model(model, tokenizers[i], prompt, text_input_ids_list[i] if text_input_ids_list else None)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
    return prompt_embeds, pooled_prompt_embeds




# Works with both SDXL text encoders
class EncodeMultiModalPipelineSDXL:
    def __init__(self):
        self.tokenizers = []

        self.text_encoders = []
        self.vision_encoders = []

        self.encoders = []

        # Load both tokenizers
        self.tokenizers.append(
            transformers.AutoTokenizer.from_pretrained(small_model)
        )
        self.tokenizers.append(
            transformers.AutoTokenizer.from_pretrained(big_model)
        ) 

        self.processors = []

        # Load both processors
        self.processors.append(
            transformers.CLIPProcessor.from_pretrained(small_model)
        )
        self.processors.append(
            transformers.CLIPProcessor.from_pretrained(big_model)
        )

        #  Load both clip models
        self.text_encoders.append(
            transformers.CLIPTextModel.from_pretrained(small_model).eval()
        )
        self.text_encoders.append(
            transformers.CLIPTextModel.from_pretrained(big_model).eval()
        )

        self.vision_encoders.append(
            transformers.CLIPVisionModel.from_pretrained(small_model).eval()
        )
        self.vision_encoders.append(
            transformers.CLIPVisionModel.from_pretrained(big_model).eval()
        )

        # Load as encoders
        # self.encoders.append(
        #     transformers.CLIPModel.from_pretrained(small_model).eval()
        # )
        # self.encoders.append(
        #     transformers.CLIPModel.from_pretrained(big_model).eval()
        # )



    # TxI + T + I => feature
    def combine_text_and_image_features(self, text, image):
        # Multiply the text and image features together
        combined = text * image + text + image
        # normalize based on text and image features 
        # Avg text and image norm
        basis = (text.norm() + image.norm()) / 2
        normalized  = combined / basis

        return normalized
    
    # Give a list of features mean pool them
    def combine_feature_list(self, feature_list):
        return torch.mean(torch.stack(feature_list), dim=0)


    # Refactor encode_text to encode_for_model
    def encode_text(self, model, item):
        opts = {
            "output_hidden_states": True,
            "return_dict": True,
        }

        text_model = self.text_encoders[model]
        tokenizer = self.tokenizers[model]

        text = item["text"]
        text_inputs = tokenizer([text], return_tensors="pt", padding=True)
        features = text_model(**text_inputs, **opts)

        pooled_prompt_embeds = features[0]
        prompt_embeds = features.hidden_states[-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
        return prompt_embeds, pooled_prompt_embeds
    
    def encode_image(self, model, item):
        opts = {
            "output_hidden_states": True,
            "return_dict": True,
        }

        vision_model = self.vision_encoders[model]
        processor = self.processors[model]

        image = item["image"]
        image_inputs = processor(images=image, return_tensors="pt", padding=True)
        features = vision_model(**image_inputs, **opts)

        pooled_prompt_embeds = features[0]
        prompt_embeds = features.hidden_states[-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
        return prompt_embeds, pooled_prompt_embeds
    
    # new encode for model that uses image/text fns
    def encode_for_model(self, model, item):
        if "text" in item and "image" in item:
            text_embeds = self.encode_text(model, item)
            image_embeds = self.encode_image(model, item)

            # Print shape of embeds to combine for debug
            print("text_embeds 0", text_embeds[0].shape)
            print("image_embeds 0", image_embeds[0].shape)
            print("text_embeds 1", text_embeds[1].shape)
            print("image_embeds 1", image_embeds[1].shape)



            # combine pooled and non pooled
            embeds = self.combine_text_and_image_features(text_embeds[0], image_embeds[0])
            pooled_embeds = self.combine_text_and_image_features(text_embeds[1], image_embeds[1])

            return embeds, pooled_embeds
        elif "text" in item:
            return self.encode_text(model, item)
        elif "image" in item:
            return self.encode_image(model, item)
        else:
            raise Exception("Invalid input, no text or image key")


    # Prompts is a list of dicts
    # Each dict contains either a text or image or both
    def encode_inputs(self, prompts):
        # We store all the singular embeds in a list to combine later
        prompt_embeds = []
        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = []

        for item in prompts:
            embeds = []
            result_one = self.encode_for_model(0, item)
            result_two = self.encode_for_model(1, item)

            # Use the append and concat as encode_prompt
            embeds.append(result_one[0])
            embeds.append(result_two[0])
            prompt_embeds.append(torch.cat(embeds, dim=-1))

            # We only care about the pooled output for the final text encoder
            pooled_prompt_embeds.append(result_one[1])

        # Combine all the embeds into a single tensor
        prompt_embeds_combined = self.combine_feature_list(prompt_embeds)
        # Combine all the pooled embeds into a single tensor
        pooled_prompt_embeds_combined = self.combine_feature_list(pooled_prompt_embeds)

        return prompt_embeds_combined, pooled_prompt_embeds_combined



pipeline = EncodeMultiModalPipelineSDXL()


## Load models
test_tokenizers = [
    transformers.AutoTokenizer.from_pretrained(small_model),
    transformers.AutoTokenizer.from_pretrained(big_model)
]

test_encoders = [
    transformers.CLIPTextModel.from_pretrained(small_model).eval(),
    transformers.CLIPTextModel.from_pretrained(big_model).eval()
]

## Input Examples

# Multimodal
example_input = [
    {"text": "a dog", "image": Image.open("cat.jpg")},
    {"text": "photorealistic style"},
    {"image": Image.open("cat.jpg")},
]

# Text only
test_prompt = "a dog"


encoded_inputs = encode_prompt_old(test_encoders, test_tokenizers, test_prompt)
refactored_encoded_inputs = encode_prompt(test_encoders, test_tokenizers, test_prompt)
# Test that our encoding pipeline returns the same shape as the original
encoded_inputs_multimodal = pipeline.encode_inputs(example_input)

print("Shape of encoded inputs: ", encoded_inputs[0].shape)
print("Shape of refactored encoded inputs: ", refactored_encoded_inputs[0].shape)
print("Shape of multimodal encoded inputs: ", encoded_inputs_multimodal[0].shape)

# Print error for each one not equal to encoded_inputs

## check that the outputs are the same
print("Refactored encoded inputs: ", refactored_encoded_inputs[0].shape == encoded_inputs[0].shape)
print("Multimodal encoded inputs: ", encoded_inputs_multimodal[0].shape == encoded_inputs[0].shape)