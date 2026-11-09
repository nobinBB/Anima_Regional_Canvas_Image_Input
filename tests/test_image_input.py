import importlib
import base64
import io
import json
from pathlib import Path
import sys
import unittest

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
PACKAGE = importlib.import_module(ROOT.name)


class RecordingClip:
    def tokenize(self, text):
        return text

    def encode_from_tokens_scheduled(self, tokens):
        return [[tokens, {}]]


class ImageInputCanvasTests(unittest.TestCase):
    def test_exports_image_input_canvas_and_sam3_adapter(self):
        self.assertEqual(
            list(PACKAGE.NODE_CLASS_MAPPINGS),
            [
                "AnimaRegionalCanvasImageInput",
                "SAM3MasksToRedBlueImage",
                "AnimeInstanceMasksToRedBlueImage",
            ],
        )

    def test_anime_instance_adapter_reuses_two_mask_conversion(self):
        sam_node = PACKAGE.NODE_CLASS_MAPPINGS["SAM3MasksToRedBlueImage"]
        anime_node = PACKAGE.NODE_CLASS_MAPPINGS["AnimeInstanceMasksToRedBlueImage"]
        self.assertTrue(issubclass(anime_node, sam_node))

    def test_schema_has_image_input(self):
        node = PACKAGE.NODE_CLASS_MAPPINGS["AnimaRegionalCanvasImageInput"]
        self.assertIn("image", node.INPUT_TYPES()["optional"])
        self.assertIn("use_painted_canvas", node.INPUT_TYPES()["required"])

    def test_connected_image_drives_size_and_regions(self):
        node = PACKAGE.NODE_CLASS_MAPPINGS["AnimaRegionalCanvasImageInput"]()
        image = torch.ones((1, 64, 96, 3), dtype=torch.float32)
        image[:, 8:56, 8:40] = torch.tensor([1.0, 0.0, 0.0])
        image[:, 8:56, 56:88] = torch.tensor([0.0, 0.0, 1.0])
        output = node.execute(
            model="model",
            clip=RecordingClip(),
            width=1024,
            height=1024,
            batch_size=1,
            brush_size=92,
            region_strength=0.95,
            image=image,
            quality_prompt="quality",
            scene_prompt="scene",
            red_prompt="red character",
            blue_prompt="blue character",
            yellow_prompt="",
            green_prompt="",
            magenta_prompt="",
            negative_prompt="negative",
            regional_enabled=True,
        )
        regional_image, _, positive, _, latent, metadata, preview, positive_text, negative_text = output
        self.assertEqual(tuple(regional_image.shape), (1, 64, 96, 3))
        self.assertTrue(torch.equal(regional_image, image))
        self.assertEqual(tuple(preview.shape), tuple(image.shape))
        self.assertEqual(tuple(latent["samples"].shape), (1, 4, 8, 12))
        self.assertEqual(json.loads(metadata)["mode"], "image-input")
        self.assertEqual(positive_text, "quality, scene, red character, blue character")
        self.assertEqual(negative_text, "negative")
        masks = [entry[1].get("mask") for entry in positive if "mask" in entry[1]]
        self.assertEqual(len(masks), 2)

        by_prompt = {entry[0]: entry[1]["mask"] for entry in positive if "mask" in entry[1]}
        red_mask = by_prompt["quality\n\nscene\n\nred character"]
        blue_mask = by_prompt["quality\n\nscene\n\nblue character"]
        self.assertEqual(float(red_mask[0, 16, 16]), 1.0)
        self.assertEqual(float(red_mask[0, 16, 64]), 0.0)
        self.assertEqual(float(blue_mask[0, 16, 16]), 0.0)
        self.assertEqual(float(blue_mask[0, 16, 64]), 1.0)

    def test_string_outputs_follow_conditioning_outputs(self):
        node = PACKAGE.NODE_CLASS_MAPPINGS["AnimaRegionalCanvasImageInput"]
        self.assertEqual(len(node.RETURN_TYPES), 9)
        self.assertEqual(node.RETURN_TYPES[7:9], ("STRING", "STRING"))
        self.assertEqual(node.RETURN_NAMES[7:9], ("POSITIVE_TEXT", "NEGATIVE_TEXT"))

    def test_imported_paint_canvas_overrides_connected_image(self):
        node = PACKAGE.NODE_CLASS_MAPPINGS["AnimaRegionalCanvasImageInput"]()
        connected = torch.ones((1, 32, 48, 3), dtype=torch.float32)
        connected[:, 4:28, 4:20] = torch.tensor([1.0, 0.0, 0.0])

        painted = Image.new("RGB", (48, 32), (255, 255, 255))
        for y in range(4, 28):
            for x in range(28, 44):
                painted.putpixel((x, y), (0, 0, 255))
        buffer = io.BytesIO()
        painted.save(buffer, format="PNG")
        canvas_data = json.dumps({
            "version": 2,
            "width": 48,
            "height": 32,
            "data_url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
        })

        output = node.execute(
            model="model", clip=RecordingClip(), width=1024, height=1024,
            batch_size=1, brush_size=92, region_strength=0.95,
            canvas_data=canvas_data, image=connected, use_painted_canvas=True,
            quality_prompt="quality", scene_prompt="scene", red_prompt="red",
            blue_prompt="blue", yellow_prompt="", green_prompt="", magenta_prompt="",
            negative_prompt="negative", regional_enabled=True,
        )
        regional_image, _, _, _, _, metadata, _, _, _ = output
        self.assertEqual(json.loads(metadata)["mode"], "image-painted")
        self.assertTrue(torch.equal(regional_image[0, 10, 34], torch.tensor([0.0, 0.0, 1.0])))
        self.assertTrue(torch.equal(regional_image[0, 10, 10], torch.tensor([1.0, 1.0, 1.0])))

    def test_cleared_paint_canvas_stays_blank_instead_of_using_connected_image(self):
        node = PACKAGE.NODE_CLASS_MAPPINGS["AnimaRegionalCanvasImageInput"]()
        connected = torch.zeros((1, 32, 48, 3), dtype=torch.float32)
        connected[:, :, :] = torch.tensor([1.0, 0.0, 0.0])
        blank_canvas_data = json.dumps({
            "version": 2,
            "width": 48,
            "height": 32,
            "data_url": "",
        })

        output = node.execute(
            model="model", clip=RecordingClip(), width=1024, height=1024,
            batch_size=1, brush_size=92, region_strength=0.95,
            canvas_data=blank_canvas_data, image=connected, use_painted_canvas=True,
            quality_prompt="quality", scene_prompt="scene", red_prompt="red",
            blue_prompt="blue", yellow_prompt="", green_prompt="", magenta_prompt="",
            negative_prompt="negative", regional_enabled=True,
        )
        regional_image, _, positive, _, _, metadata, _, _, _ = output
        self.assertEqual(json.loads(metadata)["mode"], "image-painted")
        self.assertTrue(torch.equal(regional_image, torch.ones_like(regional_image)))
        self.assertFalse(any("mask" in entry[1] for entry in positive))

    def test_sam3_masks_become_left_red_and_right_blue_image(self):
        node = PACKAGE.NODE_CLASS_MAPPINGS["SAM3MasksToRedBlueImage"]()
        masks = torch.zeros((3, 48, 80), dtype=torch.float32)
        masks[0, 8:42, 48:72] = 1.0
        masks[1, 6:44, 7:34] = 1.0
        masks[2, 1:3, 1:3] = 1.0

        image, red, blue, info = node.convert(masks, "auto", 0.5, 0.1)
        self.assertEqual(tuple(image.shape), (1, 48, 80, 3))
        self.assertTrue(torch.equal(image[0, 20, 15], torch.tensor([1.0, 0.0, 0.0])))
        self.assertTrue(torch.equal(image[0, 20, 60], torch.tensor([0.0, 0.0, 1.0])))
        self.assertTrue(torch.equal(image[0, 46, 40], torch.tensor([1.0, 1.0, 1.0])))
        self.assertEqual(float((red * blue).sum()), 0.0)
        self.assertEqual(json.loads(info)["assignment"], "left-right")

    def test_anime_foreground_fills_missing_silhouette_without_merging_masks(self):
        node = PACKAGE.NODE_CLASS_MAPPINGS["SAM3MasksToRedBlueImage"]()
        masks = torch.zeros((2, 48, 80), dtype=torch.float32)
        masks[0, 12:42, 48:70] = 1.0
        masks[1, 12:42, 9:32] = 1.0
        anime_mask = torch.zeros((1, 48, 80), dtype=torch.float32)
        anime_mask[0, 5:43, 6:35] = 1.0
        anime_mask[0, 4:43, 45:73] = 1.0

        image, red, blue, info = node.convert(
            masks, "auto", 0.5, 0.1, anime_mask=anime_mask,
        )
        details = json.loads(info)
        self.assertTrue(details["anime_refinement"])
        self.assertGreater(details["anime_red_added_pixels"], 0)
        self.assertGreater(details["anime_blue_added_pixels"], 0)
        self.assertEqual(float(red[0, 6, 15]), 1.0)
        self.assertEqual(float(blue[0, 6, 55]), 1.0)
        self.assertEqual(float((red * blue).sum()), 0.0)
        self.assertTrue(torch.equal(image[0, 6, 15], torch.tensor([1.0, 0.0, 0.0])))
        self.assertTrue(torch.equal(image[0, 6, 55], torch.tensor([0.0, 0.0, 1.0])))


if __name__ == "__main__":
    unittest.main()
