import base64
import io
import json
import numpy as np
import torch
from PIL import Image


MAX_RESOLUTION = 16384
REGIONS = (
    ("red", "RED", (1.0, 0.0, 0.0)),
    ("blue", "BLUE", (0.0, 0.0, 1.0)),
    ("yellow", "YELLOW", (1.0, 1.0, 0.0)),
    ("green", "GREEN", (0.0, 1.0, 0.0)),
    ("magenta", "MAGENTA", (1.0, 0.0, 1.0)),
)
STANDARD_PROMPT_DEFAULTS = {
    "quality_prompt": "masterpiece, absurdres, score_7, anime style",
    "scene_prompt": "",
    "red_prompt": "",
    "blue_prompt": "",
    "yellow_prompt": "",
    "green_prompt": "",
    "magenta_prompt": "",
    "negative_prompt": "worst quality, low quality, blurry, bad anatomy",
}

def _text_input(default=""):
    return ("STRING", {"multiline": True, "dynamicPrompts": True, "default": default})


def _canvas_input():
    return ("STRING", {"multiline": True, "default": ""})


def _encode_text(clip, text):
    if clip is None:
        raise RuntimeError("CLIP input is required.")
    tokens = clip.tokenize(text or "")
    if hasattr(clip, "encode_from_tokens_scheduled"):
        return clip.encode_from_tokens_scheduled(tokens)

    if hasattr(clip, "encode_from_tokens"):
        try:
            encoded = clip.encode_from_tokens(tokens, return_pooled=True)
        except TypeError:
            encoded = clip.encode_from_tokens(tokens)
        if isinstance(encoded, tuple):
            cond = encoded[0]
            pooled = encoded[1] if len(encoded) > 1 else None
            meta = {"pooled_output": pooled} if pooled is not None else {}
            return [[cond, meta]]
        return [[encoded, {}]]

    raise RuntimeError("CLIP object does not support token encoding.")


def _intermediate_device():
    try:
        import comfy.model_management as model_management

        return model_management.intermediate_device()
    except Exception:
        return torch.device("cpu")


def _intermediate_dtype():
    try:
        import comfy.model_management as model_management

        return model_management.intermediate_dtype()
    except Exception:
        return torch.float32


def _conditioning_set_values(conditioning, values):
    updated = []
    for item in conditioning:
        entry = [item[0], item[1].copy()]
        for key, value in values.items():
            entry[1][key] = value
        updated.append(entry)
    return updated


def _nearest_resample():
    resampling = getattr(Image, "Resampling", None)
    return getattr(resampling, "NEAREST", Image.NEAREST)


def _set_mask(conditioning, mask, strength, set_area_to_bounds=False):
    if len(mask.shape) < 3:
        mask = mask.unsqueeze(0)
    return _conditioning_set_values(
        conditioning,
        {
            "mask": mask,
            "set_area_to_bounds": set_area_to_bounds,
            "mask_strength": strength,
        },
    )


def _set_default(conditioning):
    return _conditioning_set_values(conditioning, {"default": True})


def _latent(width, height, batch_size):
    width, height = _latent_size(width, height)
    samples = torch.zeros(
        [batch_size, 4, height // 8, width // 8],
        device=_intermediate_device(),
        dtype=_intermediate_dtype(),
    )
    return {"samples": samples, "downscale_ratio_spacial": 8}


def _latent_size(width, height):
    width = min(MAX_RESOLUTION, max(16, int(width))) // 8 * 8
    height = min(MAX_RESOLUTION, max(16, int(height))) // 8 * 8
    return width, height


def _resize_image_tensor(image, width, height):
    width, height = _latent_size(width, height)
    src = image[:, :, :, :3].float()
    if src.shape[1:3] == (height, width):
        return src
    return torch.nn.functional.interpolate(
        src.movedim(-1, 1),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).movedim(1, -1)


def _regional_image_input(image, batch_size):
    if image is None or image.ndim != 4 or image.shape[-1] < 3:
        raise ValueError("image must be a ComfyUI IMAGE tensor in BHWC format.")
    width, height = _latent_size(image.shape[2], image.shape[1])
    source = image[:1, :, :, :3].detach().float()
    if source.shape[1:3] != (height, width):
        source = torch.nn.functional.interpolate(
            source.movedim(-1, 1),
            size=(height, width),
            mode="nearest",
        ).movedim(1, -1)
    source = source.to(device=_intermediate_device(), dtype=_intermediate_dtype())
    if batch_size > 1:
        source = source.repeat(batch_size, 1, 1, 1)
    return source, width, height


def _grow_mask(mask, amount):
    mask = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
    amount = max(0, int(amount))
    if amount <= 0:
        return mask.round()
    mask = mask.round()
    kernel_size = amount * 2 + 1
    return torch.nn.functional.max_pool2d(mask, kernel_size, stride=1, padding=amount)


def _inpaint_latent(vae, pixels, mask, grow_mask_by):
    downscale_ratio = vae.spacial_compression_encode() if hasattr(vae, "spacial_compression_encode") else 8
    height = (pixels.shape[1] // downscale_ratio) * downscale_ratio
    width = (pixels.shape[2] // downscale_ratio) * downscale_ratio
    mask = torch.nn.functional.interpolate(
        mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
        size=(pixels.shape[1], pixels.shape[2]),
        mode="bilinear",
    )
    pixels = pixels.clone()
    if pixels.shape[1] != height or pixels.shape[2] != width:
        y_offset = (pixels.shape[1] % downscale_ratio) // 2
        x_offset = (pixels.shape[2] % downscale_ratio) // 2
        pixels = pixels[:, y_offset:height + y_offset, x_offset:width + x_offset, :]
        mask = mask[:, :, y_offset:height + y_offset, x_offset:width + x_offset]

    noise_mask = _grow_mask(mask, grow_mask_by)[:, :, :height, :width].round()
    keep = (1.0 - mask.round()).squeeze(1)
    for channel in range(3):
        pixels[:, :, :, channel] -= 0.5
        pixels[:, :, :, channel] *= keep
        pixels[:, :, :, channel] += 0.5

    return {"samples": vae.encode(pixels), "noise_mask": noise_mask}


def _image_from_canvas(canvas_data, width, height, batch_size):
    width, height = _latent_size(width, height)
    image = None
    if canvas_data:
        try:
            payload = json.loads(canvas_data)
            data_url = payload.get("data_url", "")
            if "," in data_url:
                data_url = data_url.split(",", 1)[1]
            raw = base64.b64decode(data_url)
            image = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            image = None

    if image is None:
        image = Image.new("RGB", (width, height), (255, 255, 255))
    elif image.size != (width, height):
        image = image.resize((width, height), _nearest_resample())

    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).to(
        device=_intermediate_device(),
        dtype=_intermediate_dtype(),
    )
    if batch_size > 1:
        tensor = tensor.repeat(batch_size, 1, 1, 1)
    return tensor


def _canvas_payload_has_image(canvas_data):
    if not canvas_data:
        return False
    try:
        return bool(json.loads(canvas_data).get("data_url"))
    except Exception:
        return False


def _mask_preview_image(mask_image, base_image=None, alpha=0.45):
    mask = mask_image[:, :, :, :3].float()
    if base_image is None:
        base = torch.ones_like(mask)
    else:
        base = base_image[:, :, :, :3].float()
        if base.shape[1:3] != mask.shape[1:3]:
            base = torch.nn.functional.interpolate(
                base.movedim(-1, 1),
                size=mask.shape[1:3],
                mode="bilinear",
                align_corners=False,
            ).movedim(1, -1)
        if base.shape[0] != mask.shape[0]:
            base = base[:1].repeat(mask.shape[0], 1, 1, 1)
    painted = (mask < 0.98).any(dim=-1, keepdim=True).float()
    return torch.clamp(base * (1.0 - painted * alpha) + mask * (painted * alpha), 0.0, 1.0)


def _extract_masks(image, threshold=0.15):
    src = image[0].detach()
    r, g, b = src[..., 0], src[..., 1], src[..., 2]
    masks = {
        "red": ((r >= 1 - threshold) & (g < threshold) & (b < threshold)).float(),
        "blue": ((r < threshold) & (g < threshold) & (b >= 1 - threshold)).float(),
        "yellow": ((r >= 1 - threshold) & (g >= 1 - threshold) & (b < threshold)).float(),
        "green": ((r < threshold) & (g >= 1 - threshold) & (b < threshold)).float(),
        "magenta": ((r >= 1 - threshold) & (g < threshold) & (b >= 1 - threshold)).float(),
        "white": ((r >= 1 - threshold) & (g >= 1 - threshold) & (b >= 1 - threshold)).float(),
    }
    union = torch.zeros_like(masks["white"])
    for key, _, _ in REGIONS:
        union = torch.maximum(union, masks[key])
    masks["base"] = torch.clamp(1.0 - union, 0.0, 1.0)
    return masks


def _normalize_sam_masks(masks):
    if not torch.is_tensor(masks):
        raise ValueError("masks must be a ComfyUI MASK tensor.")
    masks = masks.detach().float()
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)
    elif masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    elif masks.ndim == 4 and masks.shape[-1] == 1:
        masks = masks[..., 0]
    if masks.ndim != 3:
        raise ValueError(f"Expected MASK shape [N,H,W], got {tuple(masks.shape)}.")
    return masks.clamp(0.0, 1.0)


def _mask_centroid(mask):
    points = torch.nonzero(mask, as_tuple=False)
    if points.numel() == 0:
        return 0.0, 0.0
    yx = points.float().mean(dim=0)
    return float(yx[1].item()), float(yx[0].item())


def _select_two_character_masks(masks, threshold, minimum_area_percent):
    binary = _normalize_sam_masks(masks) >= float(threshold)
    image_area = binary.shape[-2] * binary.shape[-1]
    minimum_area = max(1.0, image_area * float(minimum_area_percent) / 100.0)
    candidates = [mask for mask in binary if float(mask.sum().item()) >= minimum_area]
    candidates.sort(key=lambda mask: float(mask.sum().item()), reverse=True)

    selected = []
    for candidate in candidates:
        duplicate = False
        for existing in selected:
            intersection = (candidate & existing).sum().float()
            union = (candidate | existing).sum().float().clamp_min(1.0)
            if float((intersection / union).item()) >= 0.85:
                duplicate = True
                break
        if not duplicate:
            selected.append(candidate)
        if len(selected) == 2:
            break

    if len(selected) < 2:
        raise ValueError(
            "独立した人物MASKを2枚取得できませんでした。"
            " 検出閾値、MASK閾値、最小面積を調整してください。"
        )
    return selected


def _order_and_separate_masks(selected, assignment):
    first, second = selected
    first_x, first_y = _mask_centroid(first)
    second_x, second_y = _mask_centroid(second)
    if assignment == "auto":
        assignment = "left-right" if abs(first_x - second_x) >= abs(first_y - second_y) else "top-bottom"

    axis = 1 if assignment == "left-right" else 0
    first_position = first_x if axis == 1 else first_y
    second_position = second_x if axis == 1 else second_y
    red, blue = (first, second) if first_position <= second_position else (second, first)

    overlap = red & blue
    if overlap.any():
        red_x, red_y = _mask_centroid(red)
        blue_x, blue_y = _mask_centroid(blue)
        midpoint = ((red_x + blue_x) / 2.0) if axis == 1 else ((red_y + blue_y) / 2.0)
        height, width = red.shape
        coordinates = torch.arange(width if axis == 1 else height, device=red.device)
        red_side = coordinates <= midpoint
        red_side = red_side.unsqueeze(0).expand(height, width) if axis == 1 else red_side.unsqueeze(1).expand(height, width)
        red = (red & ~overlap) | (overlap & red_side)
        blue = (blue & ~overlap) | (overlap & ~red_side)
    return red, blue, assignment


def _prompt_value(kwargs, name, default=""):
    # Empty strings are explicit values; only absent/None inputs fall back.
    external = kwargs.get(f"{name}_in")
    if external is not None:
        return external
    local = kwargs.get(name)
    return default if local is None else local


def _prompts(kwargs):
    legacy_base = _prompt_value(kwargs, "base_prompt")
    result = {
        "quality": _prompt_value(kwargs, "quality_prompt", legacy_base),
        "scene": _prompt_value(kwargs, "scene_prompt"),
        "negative": _prompt_value(kwargs, "negative_prompt"),
    }
    for key, label, _ in REGIONS:
        result[key] = _prompt_value(kwargs, f"{key}_prompt")
    return result


def _positive_prompt_text(prompts):
    return "\n\n".join(
        text.strip()
        for text in [prompts["quality"], prompts["scene"], *(prompts[key] for key, _, _ in REGIONS)]
        if text and text.strip()
    )


def _positive_prompt_string(prompts):
    return ", ".join(
        text.strip()
        for text in [prompts["quality"], prompts["scene"], *(prompts[key] for key, _, _ in REGIONS)]
        if text and text.strip()
    )


def _global_prompt_text(prompts):
    return "\n\n".join(
        text.strip()
        for text in [prompts["quality"], prompts["scene"]]
        if text and text.strip()
    )


def _region_prompt_text(prompts, key):
    return "\n\n".join(
        text.strip()
        for text in [prompts["quality"], prompts["scene"], prompts[key]]
        if text and text.strip()
    )


def _conditioning(clip, prompts, masks, strength, enabled=True):
    global_text = _global_prompt_text(prompts)

    if not enabled:
        positive = _encode_text(clip, _positive_prompt_text(prompts))
        negative = _encode_text(clip, prompts["negative"])
        return positive, negative

    global_conditioning = _encode_text(clip, global_text)
    positive = _set_default(global_conditioning)
    active_regions = 0
    for key, _, _ in REGIONS:
        text = prompts[key].strip()
        if not text:
            continue
        mask = masks.get(key)
        if mask is None or torch.max(mask).item() <= 0:
            continue
        positive.extend(_set_mask(_encode_text(clip, _region_prompt_text(prompts, key)), mask, strength, False))
        active_regions += 1

    if active_regions == 0:
        positive = global_conditioning

    negative = _encode_text(clip, prompts["negative"])
    return positive, negative


def _metadata(prompts, width, height, mode, regional_enabled, region_strength):
    return json.dumps(
        {
            "node": "Anima Regional Canvas",
            "mode": mode,
            "width": width,
            "height": height,
            "regional_enabled": bool(regional_enabled),
            "region_strength": float(region_strength),
            "regions": {label: prompts[key] for key, label, _ in REGIONS},
            "prompt": _positive_prompt_text(prompts),
            "quality": prompts["quality"],
            "scene": prompts["scene"],
            "base": _global_prompt_text(prompts),
            "negative": prompts["negative"],
        },
        ensure_ascii=False,
    )


class AnimaRegionalCanvasImageInput:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "width": ("INT", {"default": 1024, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
            "height": ("INT", {"default": 1024, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            "brush_size": ("INT", {"default": 92, "min": 1, "max": 512, "step": 1}),
            "region_strength": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 10.0, "step": 0.01}),
            "quality_prompt": _text_input(STANDARD_PROMPT_DEFAULTS["quality_prompt"]),
            "scene_prompt": _text_input(STANDARD_PROMPT_DEFAULTS["scene_prompt"]),
            "red_prompt": _text_input(STANDARD_PROMPT_DEFAULTS["red_prompt"]),
            "blue_prompt": _text_input(STANDARD_PROMPT_DEFAULTS["blue_prompt"]),
            "yellow_prompt": _text_input(STANDARD_PROMPT_DEFAULTS["yellow_prompt"]),
            "green_prompt": _text_input(STANDARD_PROMPT_DEFAULTS["green_prompt"]),
            "magenta_prompt": _text_input(STANDARD_PROMPT_DEFAULTS["magenta_prompt"]),
            "negative_prompt": _text_input(STANDARD_PROMPT_DEFAULTS["negative_prompt"]),
            "canvas_data": _canvas_input(),
            "regional_enabled": ("BOOLEAN", {"default": True}),
            "use_painted_canvas": ("BOOLEAN", {"default": False}),
        }
        optional = {
            "image": ("IMAGE",),
            "quality_prompt_in": ("STRING", {"forceInput": True}),
            "scene_prompt_in": ("STRING", {"forceInput": True}),
            "red_prompt_in": ("STRING", {"forceInput": True}),
            "blue_prompt_in": ("STRING", {"forceInput": True}),
            "yellow_prompt_in": ("STRING", {"forceInput": True}),
            "green_prompt_in": ("STRING", {"forceInput": True}),
            "magenta_prompt_in": ("STRING", {"forceInput": True}),
            "negative_prompt_in": ("STRING", {"forceInput": True}),
        }
        return {"required": required, "optional": optional}

    RETURN_TYPES = (
        "IMAGE", "MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "STRING", "IMAGE",
        "STRING", "STRING",
    )
    RETURN_NAMES = (
        "IMAGE", "MODEL", "POSITIVE", "NEGATIVE", "LATENT", "METADATA", "MASK_PREVIEW",
        "POSITIVE_TEXT", "NEGATIVE_TEXT",
    )
    FUNCTION = "execute"
    CATEGORY = "Anima/Regional"

    def execute(self, model, clip, width, height, batch_size, brush_size, region_strength, canvas_data="", **kwargs):
        batch_size = max(1, int(batch_size))
        source_image = kwargs.get("image")
        if source_image is not None:
            image, width, height = _regional_image_input(source_image, batch_size)
            if kwargs.get("use_painted_canvas", False):
                image = _image_from_canvas(canvas_data, width, height, batch_size)
                mode = "image-painted"
            else:
                mode = "image-input"
        else:
            width, height = _latent_size(width, height)
            image = _image_from_canvas(canvas_data, width, height, batch_size)
            mode = "canvas"
        masks = _extract_masks(image)
        prompts = _prompts(kwargs)
        regional_enabled = kwargs.get("regional_enabled", True)
        positive, negative = _conditioning(clip, prompts, masks, region_strength, regional_enabled)
        latent = _latent(width, height, batch_size)
        metadata = _metadata(prompts, width, height, mode, regional_enabled, region_strength)
        preview = _mask_preview_image(image)
        positive_text = _positive_prompt_string(prompts)
        negative_text = prompts["negative"].strip()
        return (
            image, model, positive, negative, latent, metadata, preview,
            positive_text, negative_text,
        )


class SAM3MasksToRedBlueImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "assignment": (["auto", "left-right", "top-bottom"], {"default": "auto"}),
                "mask_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "minimum_area_percent": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 25.0, "step": 0.1}),
            },
            "optional": {
                "anime_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "STRING")
    RETURN_NAMES = ("RED_BLUE_IMAGE", "RED_MASK", "BLUE_MASK", "INFO")
    FUNCTION = "convert"
    CATEGORY = "Anima/Regional"

    def convert(self, masks, assignment="auto", mask_threshold=0.5, minimum_area_percent=0.1, anime_mask=None):
        normalized = _normalize_sam_masks(masks)
        selected = _select_two_character_masks(normalized, mask_threshold, minimum_area_percent)
        red, blue, resolved_assignment = _order_and_separate_masks(selected, assignment)

        anime_red_added = 0
        anime_blue_added = 0
        if anime_mask is not None:
            anime = _normalize_sam_masks(anime_mask).to(device=red.device)
            anime = anime.amax(dim=0, keepdim=True).unsqueeze(0)
            if anime.shape[-2:] != red.shape:
                anime = torch.nn.functional.interpolate(
                    anime,
                    size=red.shape,
                    mode="bilinear",
                    align_corners=False,
                )
            anime = anime[0, 0] >= float(mask_threshold)

            red_x, red_y = _mask_centroid(red)
            blue_x, blue_y = _mask_centroid(blue)
            axis = 1 if resolved_assignment == "left-right" else 0
            midpoint = ((red_x + blue_x) / 2.0) if axis == 1 else ((red_y + blue_y) / 2.0)
            height, width = red.shape
            coordinates = torch.arange(width if axis == 1 else height, device=red.device)
            red_side = coordinates <= midpoint
            red_side = (
                red_side.unsqueeze(0).expand(height, width)
                if axis == 1
                else red_side.unsqueeze(1).expand(height, width)
            )

            # IS-Net Anime is a union foreground mask. SAM3 keeps ownership of the
            # two characters; the auxiliary mask only fills pixels SAM3 missed.
            unassigned = anime & ~(red | blue)
            red_addition = unassigned & red_side
            blue_addition = unassigned & ~red_side
            anime_red_added = int(red_addition.sum().item())
            anime_blue_added = int(blue_addition.sum().item())
            red = red | red_addition
            blue = blue | blue_addition

        height, width = red.shape
        image = torch.ones((1, height, width, 3), device=normalized.device, dtype=torch.float32)
        image[0, red] = torch.tensor([1.0, 0.0, 0.0], device=image.device)
        image[0, blue] = torch.tensor([0.0, 0.0, 1.0], device=image.device)
        info = json.dumps(
            {
                "assignment": resolved_assignment,
                "red_pixels": int(red.sum().item()),
                "blue_pixels": int(blue.sum().item()),
                "source_masks": int(normalized.shape[0]),
                "anime_refinement": anime_mask is not None,
                "anime_red_added_pixels": anime_red_added,
                "anime_blue_added_pixels": anime_blue_added,
            },
            ensure_ascii=False,
        )
        return image, red.unsqueeze(0).float(), blue.unsqueeze(0).float(), info


class AnimeInstanceMasksToRedBlueImage(SAM3MasksToRedBlueImage):
    """Compatibility alias with a source-agnostic name for anime instance masks."""

    pass

class AnimaRegionalInpaintCanvas:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "width": ("INT", {"default": 1024, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
            "height": ("INT", {"default": 1024, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            "brush_size": ("INT", {"default": 92, "min": 1, "max": 512, "step": 1}),
            "region_strength": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 10.0, "step": 0.01}),
            "quality_prompt": _text_input(STANDARD_PROMPT_DEFAULTS["quality_prompt"]),
            "scene_prompt": _text_input(STANDARD_PROMPT_DEFAULTS["scene_prompt"]),
            "red_prompt": _text_input(STANDARD_PROMPT_DEFAULTS["red_prompt"]),
            "blue_prompt": _text_input(STANDARD_PROMPT_DEFAULTS["blue_prompt"]),
            "yellow_prompt": _text_input(STANDARD_PROMPT_DEFAULTS["yellow_prompt"]),
            "green_prompt": _text_input(STANDARD_PROMPT_DEFAULTS["green_prompt"]),
            "magenta_prompt": _text_input(STANDARD_PROMPT_DEFAULTS["magenta_prompt"]),
            "negative_prompt": _text_input(STANDARD_PROMPT_DEFAULTS["negative_prompt"]),
            "canvas_data": _canvas_input(),
            "regional_enabled": ("BOOLEAN", {"default": True}),
            "grow_mask_by": ("INT", {"default": 6, "min": 0, "max": 64, "step": 1}),
        }
        optional = {
            "vae": ("VAE",),
            "quality_prompt_in": ("STRING", {"forceInput": True}),
            "scene_prompt_in": ("STRING", {"forceInput": True}),
            "red_prompt_in": ("STRING", {"forceInput": True}),
            "blue_prompt_in": ("STRING", {"forceInput": True}),
            "yellow_prompt_in": ("STRING", {"forceInput": True}),
            "green_prompt_in": ("STRING", {"forceInput": True}),
            "magenta_prompt_in": ("STRING", {"forceInput": True}),
            "negative_prompt_in": ("STRING", {"forceInput": True}),
            "image": ("IMAGE",),
        }
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("IMAGE", "MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "MASK", "STRING")
    RETURN_NAMES = ("IMAGE", "MODEL", "POSITIVE", "NEGATIVE", "INPAINT_LATENT", "INPAINT_MASK", "METADATA")
    FUNCTION = "execute"
    CATEGORY = "Anima/Regional"

    def execute(self, model, clip, width, height, batch_size, brush_size, region_strength, grow_mask_by, canvas_data="", **kwargs):
        width, height = _latent_size(width, height)
        batch_size = max(1, int(batch_size))
        mask_image = _image_from_canvas(canvas_data, width, height, batch_size)
        masks = _extract_masks(mask_image)
        prompts = _prompts(kwargs)
        regional_enabled = kwargs.get("regional_enabled", True)
        positive, negative = _conditioning(clip, prompts, masks, region_strength, regional_enabled)
        inpaint_mask = torch.clamp(1.0 - masks["base"], 0.0, 1.0)
        inpaint_mask = inpaint_mask.unsqueeze(0).repeat(batch_size, 1, 1)
        source_image = kwargs.get("image")
        vae = kwargs.get("vae")
        if source_image is not None and vae is not None:
            pixels = _resize_image_tensor(source_image, width, height)
            if pixels.shape[0] != batch_size:
                pixels = pixels[:1].repeat(batch_size, 1, 1, 1)
            latent = _inpaint_latent(vae, pixels, inpaint_mask, grow_mask_by)
        else:
            latent = _latent(width, height, batch_size)
        metadata = _metadata(prompts, width, height, "inpaint", regional_enabled, region_strength)
        return (mask_image, model, positive, negative, latent, inpaint_mask, metadata)

NODE_CLASS_MAPPINGS = {
    "AnimaRegionalCanvasImageInput": AnimaRegionalCanvasImageInput,
    "SAM3MasksToRedBlueImage": SAM3MasksToRedBlueImage,
    "AnimeInstanceMasksToRedBlueImage": AnimeInstanceMasksToRedBlueImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaRegionalCanvasImageInput": "Anima Regional Canvas（IMAGE入力対応）",
    "SAM3MasksToRedBlueImage": "Anima: SAM3 MASKを赤青IMAGE化",
    "AnimeInstanceMasksToRedBlueImage": "Anima: Anime Instance MASKを赤青IMAGE化",
}
