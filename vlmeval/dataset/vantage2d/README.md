# vlmeval/dataset/vantage2d/

Image benchmark implementations for VANTAGE-Bench. Three independent tasks, each with its own dataset class.

---

## Benchmarks

| Task | Class | Dataset key(s) | Data location |
|------|-------|----------------|---------------|
| Referring expression grounding | `VANTAGE_2DGroundingDataset` | `VANTAGE_2DGrounding`, `VANTAGE_2DGrounding_val`, `VANTAGE_2DGrounding_small` | `~/LMUData/datasets/VANTAGE_2DGrounding/` |
| Person detection on aerial imagery | `Astro2DDetectionDataset` | `Astro2D` | `~/LMUData/datasets/Astro2D/` |
| Spatial pointing (MCQ) | `VANTAGE_2DPointing` | `VANTAGE_2DPointing` | `~/LMUData/datasets/VANTAGE_2DPointing/` |

---

## Files

| File | Purpose |
|------|---------|
| `grounding_2d_dataset.py` | `VANTAGE_2DGroundingDataset` — referring expression grounding. Accepts RefCOCO JSON or JSONL annotation formats. |
| `astro_2d_dataset.py` | `Astro2DDetectionDataset` — person detection on aerial/overhead imagery. Labels in KITTI format. |
| `pointing_dataset.py` | `VANTAGE_2DPointing` — spatial pointing multiple-choice benchmark. |
| `datasets.yaml` | Per-dataset path config. Maps dataset name → class and `data_root`. Loaded by `utils.load_dataset_config()`. |
| `utils.py` | Shared helpers: `load_dataset_config`, `scale_bbox`, `compute_2d_iou`, `parse_kitti_label`, `parse_bbox_2d_from_text`. |
| `__init__.py` | Re-exports `VANTAGE_2DGroundingDataset`, `Astro2DDetectionDataset`. |

---

## Data layout

```
~/LMUData/datasets/
├── VANTAGE_2DGrounding/
│   ├── images/                   # VisDrone images (1503 files)
│   └── annotations.json          # RefDrone referring expressions (no GT bboxes in public release)
├── VANTAGE_2DGrounding_val/
│   ├── images/
│   └── annotations.json
└── Astro2D/
    ├── images/                   # aerial imagery frames
    └── labels/                   # KITTI-format label files (empty in public release)
```

`VANTAGE_2DPointing` data lives under `~/LMUData/datasets/VANTAGE_2DPointing/` as a TSV + `images_annotated/` folder.

---

## Config resolution

`datasets.yaml` is loaded at runtime by `utils.load_dataset_config(dataset_name)`. The YAML key must match the string passed to the class constructor's `dataset=` argument. If the key is absent, the class falls back to `LMUDataRoot()/datasets/<dataset_name>/`.

---

## Adding a new 2D benchmark

1. Create `<name>_dataset.py` with a class that extends `ImageBaseDataset`.
2. Implement `supported_datasets()`, `__init__()`, `build_prompt()`, and `evaluate()`.
3. Add the dataset name and `data_root` to `datasets.yaml`.
4. Import and add to `IMAGE_DATASET` in `vlmeval/dataset/__init__.py`.
