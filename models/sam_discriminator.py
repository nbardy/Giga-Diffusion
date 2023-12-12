# Use segment anything model as pre-trained discriminator
from transformers import AutoModel
import torch.nn as nn
import torch.nn.functional as F
import torch

DEFAULT_MODEL = "facebook/sam-vit-large"


class ViTDiscriminator(nn.Module):
    def __init__(self, vit=None):
        super().__init__()
        self.vit = vit or AutoModel.from_pretrained(DEFAULT_MODEL)

        # Freeze the weights of the base model
        for param in self.vit.parameters():
            param.requires_grad = False

        # Add a binary classification head
        # Assuming the last hidden size of the model is 1024
        self.classifier = nn.Linear(256, 2)
        # Make classifier trainable
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        # Get the output from the ViT model
        outputs = self.vit(x)
        # We use the [CLS] token output (assumed at index 0) for classification
        cls_output = outputs.last_hidden_state[:, 0, :]
        # Pass through the classifier
        logits = self.classifier(cls_output)
        return F.softmax(logits, dim=1)


## Multi scale ViT Discriminator
class MultiscaleViTDiscriminator(nn.Module):
    def __init__(self, vit=None):
        super().__init__()

        # Initialize the base ViT model
        self.vit = vit or AutoModel.from_pretrained("facebook/sam-vit-large")

        # Add a binary classification head for each scale
        self.classifier_small = nn.Linear(256, 2)
        self.classifier_medium = nn.Linear(256, 2)
        self.classifier_large = nn.Linear(256, 2)

        # Make  base model frozen
        for param in self.vit.parameters():
            param.requires_grad = False

        # Make classifiers trainable
        for classifier in [
            self.classifier_small,
            self.classifier_medium,
            self.classifier_large,
        ]:
            for param in classifier.parameters():
                param.requires_grad = True

    def forward(self, x_small, x_medium, x_large):
        # Get the output from the ViT model at different scales
        outputs_small = self.vit(x_small)
        outputs_medium = self.vit(x_medium)
        outputs_large = self.vit(x_large)

        # Pass through the corresponding classifier
        logits_small = self.classifier_small(outputs_small)
        logits_medium = self.classifier_medium(outputs_medium)
        logits_large = self.classifier_large(outputs_large)

        # Concatenate the logits from each scale into a single batch
        logits = torch.cat(
            (logits_small, logits_medium, logits_large), dim=0
        )  # (B_small + B_medium + B_large)x2

        return F.softmax(logits, dim=1)  # (B_small + B_medium + B_large)x2
