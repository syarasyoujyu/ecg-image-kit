import argparse
import os
import random
import re
from pathlib import Path
from sys import platform
from typing import Iterable, List, Optional, Union

import numpy as np
import requests
from bs4 import BeautifulSoup, Comment
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

try:
    import spacy
except ImportError:  # pragma: no cover - optional dependency
    spacy = None

try:
    import validators
except ImportError:  # pragma: no cover - optional dependency
    validators = None


MEDICAL_TERMS = {
    "ecg",
    "ekg",
    "sinus",
    "rhythm",
    "arrhythmia",
    "tachycardia",
    "bradycardia",
    "afib",
    "flutter",
    "stemi",
    "nstemi",
    "ischemia",
    "infarction",
    "ventricular",
    "atrial",
    "bundle",
    "block",
    "axis",
    "deviation",
    "hypertrophy",
    "qrs",
    "pr",
    "qt",
    "qtc",
    "twave",
    "troponin",
    "lead",
    "leads",
    "paced",
    "pacemaker",
    "myocardial",
    "cardiac",
    "cardiomyopathy",
    "palpitation",
    "murmur",
    "syncope",
    "chest",
    "pain",
    "acute",
    "normal",
    "abnormal",
}


def get_parser():
    description = "Create handwritten-like text overlays for ECG images"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-l", "--link", type=str, required=True)
    parser.add_argument("-n", "--num_words", type=int, required=True)
    parser.add_argument("-x_offset", dest="x_offset", type=int, default=0)
    parser.add_argument("-y_offset", dest="y_offset", type=int, default=0)
    parser.add_argument("-hws", dest="handwriting_size_factor", type=float, default=0.2)
    parser.add_argument("-s", dest="source_dir", type=str, required=True)
    parser.add_argument("-i", dest="input_file", type=str, required=True)
    parser.add_argument("-o", dest="output_dir", type=str, required=True)
    parser.add_argument("--model", dest="model_path", type=str, default=None)
    parser.add_argument("--text", dest="text", type=str, default=None)
    parser.add_argument("--style", dest="style", type=int, default=None)
    parser.add_argument("--bias", dest="bias", type=float, default=1.0)
    parser.add_argument("--force", dest="force", action="store_true", default=False)
    parser.add_argument("--animation", dest="animation", action="store_true", default=False)
    parser.add_argument("--noinfo", dest="info", action="store_false", default=True)
    parser.add_argument("--save", dest="save", type=str, default=None)
    return parser


def _is_url(value: str) -> bool:
    if not value:
        return False
    if validators is not None:
        return bool(validators.url(value))
    return value.startswith(("http://", "https://"))


def _load_text_source(link: str) -> str:
    if _is_url(link):
        response = requests.get(link, timeout=15)
        response.raise_for_status()

        parser = "html5lib" if platform == "darwin" else "lxml"
        soup = BeautifulSoup(response.content, parser)
        body = soup.body or soup
        chunks = []
        for text in body.find_all(string=True):
            if text.parent.name in ["script", "meta", "link", "style"]:
                continue
            if isinstance(text, Comment):
                continue
            stripped = text.strip()
            if stripped:
                chunks.append(stripped)
        return " ".join(chunks)

    if not link:
        link = "HandwrittenText/Biomedical.txt"

    with open(link, "r") as file:
        return " ".join(line.strip() for line in file if line.strip())


def _extract_with_spacy(text: str) -> List[str]:
    if spacy is None:
        return []

    try:
        nlp = spacy.load("en_core_sci_sm")
    except Exception:
        return []

    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        normalized = re.sub(r"\s+", " ", ent.text.strip())
        if normalized:
            entities.append(normalized)
    return entities


def _extract_with_regex(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-_/]{1,}", text.lower())
    candidates = []
    for token in tokens:
        if token in MEDICAL_TERMS or len(token) >= 6:
            candidates.append(token)
    return candidates


def _choose_words(text: str, num_words: int, explicit_text: Optional[str] = None) -> List[str]:
    if explicit_text:
        words = [chunk.strip() for chunk in explicit_text.split(",") if chunk.strip()]
        if words:
            return words[:num_words]

    candidates = _extract_with_spacy(text)
    if not candidates:
        candidates = _extract_with_regex(text)
    if not candidates:
        candidates = ["sinus rhythm", "normal ecg", "qrs narrow", "st changes"]

    return random.choices(candidates, k=max(1, num_words))


def _load_font(font_size: int) -> Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]:
    font_candidates = [
        Path("Fonts") / "Times_New_Roman.ttf",
        Path("Fonts") / "Arial_Italic.ttf",
        Path("Fonts") / "Verdana_Italic.ttf",
    ]

    for font_path in font_candidates:
        if font_path.exists():
            try:
                return ImageFont.truetype(str(font_path), size=font_size)
            except OSError:
                continue
    return ImageFont.load_default()


def _render_handwritten_text(words: Iterable[str], width: int, height: int) -> Image.Image:
    canvas = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(canvas)
    base_size = max(18, min(width, height) // 8)
    font = _load_font(base_size)
    y = max(8, base_size // 3)

    for word in words:
        text = str(word).strip()
        if not text:
            continue

        line_font = _load_font(max(16, int(base_size * random.uniform(0.9, 1.15))))
        bbox = draw.textbbox((0, 0), text, font=line_font)
        text_width = max(1, bbox[2] - bbox[0])
        text_height = max(1, bbox[3] - bbox[1])
        x = random.randint(5, max(5, width - text_width - 5))
        jittered = Image.new("L", (text_width + 20, text_height + 20), color=255)
        jitter_draw = ImageDraw.Draw(jittered)

        # Multiple slightly shifted strokes produce a handwritten-looking pen trace.
        for _ in range(random.randint(2, 4)):
            dx = random.randint(-1, 1)
            dy = random.randint(-1, 1)
            jitter_draw.text((10 + dx, 10 + dy), text, font=line_font, fill=random.randint(0, 45))

        angle = random.uniform(-8, 8)
        jittered = jittered.rotate(angle, expand=True, fillcolor=255, resample=Image.Resampling.BICUBIC)
        jittered = jittered.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.6)))

        paste_y = min(y, max(0, height - jittered.height))
        paste_x = min(x, max(0, width - jittered.width))
        canvas.paste(ImageChops.darker(canvas.crop((paste_x, paste_y, paste_x + jittered.width, paste_y + jittered.height)), jittered), (paste_x, paste_y))

        y += jittered.height + random.randint(4, max(6, base_size // 3))
        if y >= height - base_size:
            break

    return canvas


def _overlay_text(base_image: Image.Image, mask: Image.Image, x_offset: int, y_offset: int) -> Image.Image:
    base = np.asarray(base_image.convert("RGB")).copy()
    mask_rgb = np.repeat(np.asarray(mask)[..., None], 3, axis=2)
    binary_mask = (mask_rgb >= 250).astype(np.uint8)

    x_start = max(0, x_offset)
    y_start = max(0, y_offset)
    x_end = min(base.shape[0], x_offset + binary_mask.shape[0])
    y_end = min(base.shape[1], y_offset + binary_mask.shape[1])

    if x_start >= x_end or y_start >= y_end:
        return Image.fromarray(base)

    mask_x_start = x_start - x_offset
    mask_y_start = y_start - y_offset
    mask_x_end = mask_x_start + (x_end - x_start)
    mask_y_end = mask_y_start + (y_end - y_start)

    crop = binary_mask[mask_x_start:mask_x_end, mask_y_start:mask_y_end, :]
    base[x_start:x_end, y_start:y_end, :] *= crop
    return Image.fromarray(base)


def get_handwritten(
    link,
    num_words,
    input_file,
    output_dir,
    x_offset=0,
    y_offset=0,
    handwriting_size_factor=0.2,
    model_path=None,
    text=None,
    style=None,
    bias=1.0,
    force=False,
    animation=False,
    noinfo=True,
    save=None,
    bbox=False,
):
    del model_path, style, bias, force, animation, noinfo, save, bbox

    source_text = _load_text_source(link)
    words = _choose_words(source_text, num_words=num_words, explicit_text=text)

    img_ecg = Image.open(input_file).convert("RGB")
    overlay_width = max(32, int(np.floor(img_ecg.size[0] * handwriting_size_factor)))
    overlay_height = max(32, int(np.floor(img_ecg.size[1] * handwriting_size_factor)))

    handwritten_mask = _render_handwritten_text(words, width=overlay_width, height=overlay_height)
    img_final = _overlay_text(img_ecg, handwritten_mask, x_offset=x_offset, y_offset=y_offset)

    _, tail = os.path.split(input_file)
    outfile = os.path.join(output_dir, tail)
    img_final.save(outfile)
    return outfile
