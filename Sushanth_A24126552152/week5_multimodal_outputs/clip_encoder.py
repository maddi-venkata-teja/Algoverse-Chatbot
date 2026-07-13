"""
week5_multimodal_outputs/clip_encoder.py
────────────────────────────────────────────
Week 5 — Multimodal Research Integration
Contribution by: Sushanth (A24126552152)

Provides joint image-text embeddings using CLIP, enabling
cross-modal retrieval: a text query like "binary tree traversal"
can match an uploaded diagram image, and vice versa.

Usage:
    python week5_multimodal_outputs/clip_encoder.py \
        --image data/binary_tree_diagram.png \
        --query "binary tree traversal"
"""

import argparse

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"


class ClipEncoder:
    def __init__(self, model_name: str = CLIP_MODEL_NAME):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def embed_image(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            return self.model.get_image_features(**inputs)

    def embed_text(self, text: str) -> torch.Tensor:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        with torch.no_grad():
            return self.model.get_text_features(**inputs)

    @staticmethod
    def similarity(emb_a: torch.Tensor, emb_b: torch.Tensor) -> float:
        return torch.nn.functional.cosine_similarity(emb_a, emb_b).item()


def main():
    parser = argparse.ArgumentParser(description="CLIP cross-modal similarity demo")
    parser.add_argument("--image", required=True)
    parser.add_argument("--query", required=True)
    args = parser.parse_args()

    encoder = ClipEncoder()
    image_emb = encoder.embed_image(args.image)
    text_emb = encoder.embed_text(args.query)

    sim = encoder.similarity(image_emb, text_emb)
    print(f"🖼️  Image:  {args.image}")
    print(f"📝 Query:  {args.query}")
    print(f"📐 Similarity: {sim:.4f}")


if __name__ == "__main__":
    main()
