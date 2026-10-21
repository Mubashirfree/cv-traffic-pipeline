"""
Phase 3: OpenCLIP ViT-B/32 wrapper — image and text embedding functions.
"""

import torch
import open_clip
from PIL import Image

_model = None
_preprocess = None
_tokenizer = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def _load():
    global _model, _preprocess, _tokenizer
    if _model is None:
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32-quickgelu", pretrained="openai"
        )
        _tokenizer = open_clip.get_tokenizer("ViT-B-32")
        _model.to(_device).eval()
    return _model, _preprocess, _tokenizer


def embed_image(image) -> list[float]:
    """image: PIL.Image or path string."""
    model, preprocess, _ = _load()
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(_device)
    with torch.no_grad():
        feat = model.encode_image(tensor)
        feat /= feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).cpu().tolist()


def embed_text(text: str) -> list[float]:
    model, _, tokenizer = _load()
    tokens = tokenizer([text]).to(_device)
    with torch.no_grad():
        feat = model.encode_text(tokens)
        feat /= feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).cpu().tolist()


DEFAULT_ENSEMBLE_TEMPLATES = [
    "{q}",
    "a traffic camera photo of {q}",
    "a video frame showing {q}",
    "surveillance footage of {q}",
]


def embed_text_ensemble(text: str, templates: list[str] = None) -> list[float]:
    """
    Prompt ensembling: embed the query under several phrasings, average the
    normalized embeddings, and renormalize. Averaging several phrasings of the
    same query smooths out CLIP's sensitivity to exact wording, at the cost of
    one extra forward pass per template (batched, so it's still a single call).
    Does not require re-embedding any indexed video -- only the text side changes.
    """
    templates = templates or DEFAULT_ENSEMBLE_TEMPLATES
    model, _, tokenizer = _load()
    prompts = [t.format(q=text) for t in templates]
    tokens = tokenizer(prompts).to(_device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)   # normalize each template's embedding
        avg = feats.mean(dim=0)
        avg = avg / avg.norm()                             # renormalize the average
    return avg.cpu().tolist()


if __name__ == "__main__":
    v = embed_text("a red car on a highway")
    print("Text embedding dim:", len(v))
    v2 = embed_text_ensemble("a red car on a highway")
    print("Ensemble embedding dim:", len(v2))
