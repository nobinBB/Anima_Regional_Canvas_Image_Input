import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const source = fs.readFileSync(path.join(here, "..", "web", "anima_regional_canvas_image_input.js"), "utf8");

if (!source.includes('"AnimaRegionalCanvasImageInput"')) throw new Error("new node id is missing");
if (!source.includes('slot.name === "image"')) throw new Error("image connection handling is missing");
if (!source.includes('nodeData.name === "AnimaRegionalCanvasImageInput"')) throw new Error("image preview loading is missing");
if (!source.includes('makeButton("IMAGE → Paint"')) throw new Error("IMAGE to paint import button is missing");
if (!source.includes('setWidgetValue(node, "use_painted_canvas", usePaintedCanvasW, true)')) throw new Error("painted canvas activation is missing");
const pointerDownBlock = source.slice(
  source.indexOf('canvas.addEventListener("pointerdown"'),
  source.indexOf('canvas.addEventListener("pointermove"'),
);
if (!pointerDownBlock.includes('setWidgetValue(node, "use_painted_canvas", usePaintedCanvasW, true)')) {
  throw new Error("drawing must activate the painted canvas");
}
const clearBlock = source.slice(
  source.indexOf('clear.addEventListener("click"'),
  source.indexOf('function restoreCanvasFromText'),
);
if (!clearBlock.includes('setWidgetValue(node, "use_painted_canvas", usePaintedCanvasW, true)')) {
  throw new Error("clear must keep the intentionally blank painted canvas active");
}
if (clearBlock.includes('loadConnectedImage(true)')) {
  throw new Error("clear must not reload the connected image");
}
if (!source.includes('usePaintedCanvasW?.value === true') || !source.includes('&& !maskHasPaint()')) {
  throw new Error("automatic image refresh must preserve an intentionally blank cleared canvas");
}
if (!source.includes('usePaintedCanvas: Boolean(usePaintedCanvasW?.value)')) throw new Error("history must save painted-canvas mode");
if (!source.includes('Boolean(prev.usePaintedCanvas)')) throw new Error("undo must restore painted-canvas mode");
if (!source.includes('vueNode: "never"')) throw new Error("Nodes 2.0 hidden-widget policy is missing");
if (!source.includes('widget.computeLayoutSize = () => ({ minHeight: 0, maxHeight: 0, minWidth: 0 })')) throw new Error("Nodes 2.0 hidden-widget layout is missing");
if (!source.includes('getHeight: () => EDITOR_MIN_HEIGHT')) throw new Error("Nodes 2.0 editor height is missing");
if (!source.includes('node.arrange?.();')) throw new Error("Nodes 2.0 socket layout refresh is missing");
if (!source.includes('strengthLabel.textContent = "Region Strength"')) throw new Error("visible Region Strength control is missing");
if (!source.includes('setWidgetValue(node, "region_strength", regionStrengthW, value)')) throw new Error("Region Strength control is not connected to backend value");
if (source.includes('new Set(["AnimaRegionalCanvas",')) throw new Error("original node id must not be registered");

console.log("IMAGE-input canvas frontend contract checks passed");
