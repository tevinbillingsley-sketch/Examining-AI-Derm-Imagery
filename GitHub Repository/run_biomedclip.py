import os
import re
import torch
import pandas as pd

from PIL import Image
from tqdm import tqdm

import open_clip

print("Loading BioMedCLIP...")

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess_train, preprocess = open_clip.create_model_and_transforms(
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)

tokenizer = open_clip.get_tokenizer(
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)

model = model.to(device)

print("BioMedCLIP loaded.")

DIAGNOSES = [

    "atopic dermatitis",

    "basal cell carcinoma",

    "focal acral hyperkeratosis",

    "molluscum contagiosum",

    "psoriasis vulgaris"

]

DATASETS = [

    "Gemini",

    "ChatGPT"

]



