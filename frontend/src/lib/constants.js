export const CLASSES = ["car", "bus", "truck", "auto_rickshaw", "motorcycle", "bicycle"];

export const CLASS_NAMES = {
  car: ["car", "cars"],
  bus: ["bus", "buses"],
  truck: ["truck", "trucks"],
  auto_rickshaw: ["auto-rickshaw", "auto-rickshaws"],
  motorcycle: ["motorcycle", "motorcycles"],
  bicycle: ["bicycle", "bicycles"],
};

export const EXAMPLES = [
  "more than 3 trucks at night",
  "empty road in daylight",
  "heavy traffic with buses",
  "auto-rickshaws and motorcycles",
  "no buses, only cars",
  "crowded junction at night",
];

export const DATASETS = [
  { value: "indian_video", label: "Indian videos" },
  { value: "indian", label: "Indian street photos" },
  { value: "detrac", label: "Highway camera" },
];

export const MODES = [
  { value: "both", label: "Scene + vehicles (best)" },
  { value: "frame", label: "Whole scene" },
  { value: "crop", label: "Individual vehicles" },
];

export const MODEL_LABELS = {
  coco_baseline: "Generic detector (before training)",
  detrac_ft: "Trained on highway footage",
  indian_ft: "Trained on Indian streets",
};
