"""2D Grounding Dataset for images in RefCOCO or JSONL format.

This dataset evaluates
2D visual grounding using Accuracy@IoU metrics.

Supported annotation formats:

1. RefCOCO format (JSON):
   - images/ directory containing image files
   - A JSON annotation file with referring expressions and bounding boxes

2. JSONL format (one JSON per line):
   {"image_path": "path/to/image.jpg", "gt": {"car": [[x1,y1,x2,y2], ...], ...}, 
    "categories": ["car", "truck", ...], "dataset_name": "...", "task_name": "..."}

The question for each sample:
f"Locate {object description} in the provided image and output its bbox coordinates using JSON format"

For evaluation, common grounding metrics are used:
- Acc@0.5: Accuracy at IoU threshold 0.5
- Acc@0.25: Accuracy at IoU threshold 0.25
- Mean IoU: Average IoU across all samples

Note: One expression can be associated with one or multiple bounding boxes.
When multiple GT boxes exist, a prediction is correct if it matches ANY of them.
"""

import os
import re
import json
import numpy as np
import pandas as pd
import PIL.Image
import yaml
from PIL import ImageOps
from ..image_base import ImageBaseDataset
from ...smp import get_logger, load, LMUDataRoot, dump
from .utils import load_dataset_config, scale_bbox, compute_2d_iou


def convert_bbox_xywh_to_xyxy(bbox, image_width=None, image_height=None):
    """
    Convert bbox from xywh to xyxy format if needed.

    Args:
        bbox: Bounding box [x, y, w, h] or [x1, y1, x2, y2]
        image_width: Image width for heuristic detection
        image_height: Image height for heuristic detection

    Returns:
        Bounding box in [x1, y1, x2, y2] format
    """
    if len(bbox) != 4:
        return bbox

    # Heuristic: if bbox[2] and bbox[3] are small relative to image size,
    # it's likely xywh format
    max_dim = max(image_width or 10000, image_height or 10000)
    if bbox[2] < max_dim / 2 and bbox[3] < max_dim / 2:
        # xywh format -> convert to xyxy
        x1, y1, w, h = bbox
        return [x1, y1, x1 + w, y1 + h]
    return bbox


def parse_refcoco_annotations(annotation_path):
    """
    Parse RefCOCO format annotation file.

    Expected JSON format:
    {
        "images": [
            {"id": 1, "file_name": "image1.jpg", "width": 640, "height": 480},
            ...
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "bbox": [x, y, width, height],  # COCO format (xywh) - single box
                "bboxes": [[x1,y1,w1,h1], [x2,y2,w2,h2]],  # Multiple boxes (optional)
                "sentence": "the red car on the left",
                "category": "car"  # optional
            },
            ...
        ]
    }

    Alternative simplified format:
    [
        {
            "image": "image1.jpg",
            "bbox": [x1, y1, x2, y2],  # xyxy format - single box
            "bboxes": [[x1,y1,x2,y2], [x1,y1,x2,y2]],  # Multiple boxes (optional)
            "sentence": "the red car on the left"
        },
        ...
    ]

    Note: One expression can be associated with one or multiple bounding boxes.
    Use 'bbox' for single box or 'bboxes' for multiple boxes.

    Returns:
        List of dicts with 'image', 'bboxes' (list of [x1, y1, x2, y2]), 'sentence'
    """
    if not os.path.exists(annotation_path):
        return []

    with open(annotation_path, 'r') as f:
        data = json.load(f)

    annotations = []

    # Handle COCO-style format
    if isinstance(data, dict) and 'images' in data and 'annotations' in data:
        # Build image_id to filename mapping
        id_to_image = {img['id']: img for img in data['images']}

        for ann in data['annotations']:
            image_id = ann['image_id']
            if image_id not in id_to_image:
                continue

            image_info = id_to_image[image_id]
            img_w = image_info.get('width')
            img_h = image_info.get('height')

            # Handle multiple bboxes per expression
            bboxes = []
            if 'bboxes' in ann and ann['bboxes']:
                # Multiple boxes provided
                for bbox in ann['bboxes']:
                    bbox = convert_bbox_xywh_to_xyxy(bbox, img_w, img_h)
                    bboxes.append(bbox)
            elif 'bbox' in ann:
                # Single box
                bbox = convert_bbox_xywh_to_xyxy(ann['bbox'], img_w, img_h)
                bboxes.append(bbox)

            # if not bboxes:
            #     continue

            # Handle multiple sentences per annotation
            sentences = ann.get('sentences', [])
            if not sentences and 'sentence' in ann:
                sentences = [ann['sentence']]
            elif not sentences and 'raw' in ann:
                sentences = [ann['raw']]

            for sentence in sentences:
                if isinstance(sentence, dict):
                    sentence = sentence.get('raw', sentence.get('sent', str(sentence)))

                annotations.append({
                    'image': image_info['file_name'],
                    'image_id': image_id,
                    'bboxes': bboxes,  # List of bboxes
                    'sentence': sentence,
                    'category': ann.get('category', ann.get('category_name', '')),
                    'width': img_w,
                    'height': img_h,
                })

    # Handle simplified list format
    elif isinstance(data, list):
        for ann in data:
            img_w = ann.get('width')
            img_h = ann.get('height')

            # Handle multiple bboxes per expression
            bboxes = []
            if 'bboxes' in ann and ann['bboxes']:
                for bbox in ann['bboxes']:
                    bboxes.append(bbox)  # Assume already in xyxy format
            elif 'bbox' in ann:
                bboxes.append(ann['bbox'])  # Assume already in xyxy format

            # if not bboxes:
            #     continue

            annotations.append({
                'image': ann.get('image', ann.get('file_name', '')),
                'image_id': ann.get('image_id', ''),
                'bboxes': bboxes,  # List of bboxes
                'sentence': ann.get('sentence', ann.get('expression', '')),
                'category': ann.get('category', ''),
                'width': img_w,
                'height': img_h,
            })

    return annotations


def parse_jsonl_annotations(annotation_path, data_root=None):
    """
    Parse JSONL format annotation file where each line is a JSON object.

    Expected JSONL format (one JSON object per line):
    {
        "image_path": "visdrone/image.jpg",  # relative path to image
        "gt": {
            "car": [[x1, y1, x2, y2], [x1, y1, x2, y2], ...],  # category -> list of bboxes
            "truck": [[x1, y1, x2, y2], ...],
            ...
        },
        "categories": ["car", "truck", "bus", "van"],
        "dataset_name": "VisDrone",
        "task_name": "common_object_detection"
    }

    This function creates one annotation entry per category per image.
    Each category's bboxes become the ground truth for that grounding query.

    Args:
        annotation_path: Path to JSONL annotation file
        data_root: Root directory for resolving relative image paths

    Returns:
        List of dicts with 'image', 'image_path', 'bboxes', 'sentence', 'category'
    """
    if not os.path.exists(annotation_path):
        return []

    annotations = []

    with open(annotation_path, 'r') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num + 1}: {e}")
                continue

            image_path = data.get('image_path', '')
            gt = data.get('gt', {})
            categories = data.get('categories', [])

            if not image_path or not gt:
                continue

            # Resolve full image path
            if data_root and not os.path.isabs(image_path):
                full_image_path = os.path.join(data_root, image_path)
            else:
                full_image_path = image_path

            # Create one annotation entry per category
            for category in categories:
                if category not in gt:
                    continue

                bboxes = gt[category]
                # if not bboxes:
                #     continue

                # Ensure bboxes is a list of lists
                if bboxes and isinstance(bboxes[0], (int, float)):
                    # Single bbox, wrap it
                    bboxes = [bboxes]

                annotations.append({
                    'image': image_path,  # Relative path
                    'image_path': full_image_path,  # Full path
                    'image_id': f"{line_num}",  # TODO: add category? _{category}
                    'bboxes': bboxes,  # List of bboxes in xyxy format
                    'sentence': category,  # Category name as the grounding query
                    'category': category,
                    'dataset_name': data.get('dataset_name', ''),
                    'task_name': data.get('task_name', ''),
                    'width': None,
                    'height': None,
                })
    return annotations


def detect_annotation_format(annotation_path):
    """
    Detect the format of the annotation file.

    Returns:
        'jsonl' for JSONL format, 'json' for JSON format
    """
    if annotation_path.endswith('.jsonl'):
        return 'jsonl'

    # Try to detect by reading first line
    try:
        with open(annotation_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line.startswith('{') and not first_line.endswith('}'):
                # Could be multi-line JSON
                return 'json'
            if first_line.startswith('{'):
                # Try to parse as single JSON object (JSONL line)
                data = json.loads(first_line)
                # Check for JSONL-specific keys
                if 'gt' in data and 'categories' in data:
                    return 'jsonl'
            if first_line.startswith('['):
                return 'json'
    except Exception:
        pass

    return 'json'  # Default to JSON


def _extract_bbox_from_dict(d: dict) -> list:
    """Extract a single [x1, y1, x2, y2] from a dict, or return None."""
    if not isinstance(d, dict):
        return None
    for key in ['bbox', 'bbox_2d', 'bounding_box', 'box', 'coordinates']:
        if key in d:
            val = d[key]
            if isinstance(val, list):
                if len(val) >= 4 and all(isinstance(x, (int, float)) for x in val[:4]):
                    return val[:4]
                # Nested list of bboxes, e.g. {"bbox_2d": [[x1,y1,x2,y2], ...]}
                if len(val) > 0 and isinstance(val[0], list):
                    return [b[:4] for b in val if len(b) >= 4 and all(isinstance(x, (int, float)) for x in b[:4])]
    # Gemini uses 'box_2d' with [y1, x1, y2, x2] (yxyx) — swap to xyxy
    if 'box_2d' in d:
        val = d['box_2d']
        if isinstance(val, list) and len(val) >= 4 and all(isinstance(x, (int, float)) for x in val[:4]):
            return [val[1], val[0], val[3], val[2]]
    if all(k in d for k in ['x1', 'y1', 'x2', 'y2']):
        return [d['x1'], d['y1'], d['x2'], d['y2']]
    if all(k in d for k in ['xmin', 'ymin', 'xmax', 'ymax']):
        return [d['xmin'], d['ymin'], d['xmax'], d['ymax']]
    return None


def _extract_bbox_nums_by_regex(text: str) -> list:
    """Extract all [x, y, x, y] 4-number bracket patterns from text.

    Handles integers and floats. Used as a fallback when JSON parsing fails.
    Returns a list of [x1, y1, x2, y2] boxes, preserving order.
    """
    pattern = r'\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]'
    results = []
    for m in re.finditer(pattern, text):
        try:
            coords = [float(m.group(i)) for i in range(1, 5)]
            coords = [int(c) if c == int(c) else c for c in coords]
            results.append(coords)
        except (ValueError, OverflowError):
            continue
    return results


def parse_bbox_from_text(text: str) -> list:
    """
    Parse bounding boxes from model response text.

    Handles multiple formats including malformed Cosmos JSON.
    Args:
        text: Model response text containing JSON with bbox information

    Returns:
        List of bboxes [[x1, y1, x2, y2], ...] or empty list if parsing fails.
    """
    # First try regex extraction (most robust for malformed JSON from Cosmos)
    regex_bboxes = _extract_bbox_nums_by_regex(text)
    if regex_bboxes and len(regex_bboxes) > 0:
        return regex_bboxes
    
    try:
        # Find JSON content
        if "```json" in text:
            start_idx = text.find("```json")
            end_idx = text.find("```", start_idx + 7)
            if end_idx != -1:
                json_str = text[start_idx + 7:end_idx].strip()
            else:
                json_str = text[start_idx + 7:].strip()
        else:
            start_idx = text.find("{")
            if start_idx == -1:
                start_idx = text.find("[")
            end_idx = text.rfind("}")
            if end_idx == -1:
                end_idx = text.rfind("]")

            if start_idx != -1 and end_idx != -1:
                json_str = text[start_idx:end_idx + 1]
            else:
                return regex_bboxes

        bbox_data = json.loads(json_str)

        # --- Cosmos format: [{"query": [...bboxes...]}] ---
        if isinstance(bbox_data, list) and len(bbox_data) > 0 and isinstance(bbox_data[0], dict):
            first_dict = bbox_data[0]
            for key, val in first_dict.items():
                if isinstance(val, list):
                    if len(val) > 0 and isinstance(val[0], list):
                        return [b[:4] for b in val if len(b) >= 4 and all(isinstance(x, (int, float)) for x in b[:4])]
                    elif len(val) >= 4 and all(isinstance(x, (int, float)) for x in val[:4]):
                        return [val[:4]]

        # --- list input [[x1,y1,x2,y2], ...] or [x1,y1,x2,y2] ---
        if isinstance(bbox_data, list):
            if len(bbox_data) == 4 and all(isinstance(x, (int, float)) for x in bbox_data):
                return [bbox_data]
            if len(bbox_data) > 0 and isinstance(bbox_data[0], list):
                return [b[:4] for b in bbox_data if len(b) >= 4 and all(isinstance(x, (int, float)) for x in b[:4])]
            if len(bbox_data) > 0 and isinstance(bbox_data[0], dict):
                results = []
                for item in bbox_data:
                    extracted = _extract_bbox_from_dict(item)
                    if extracted is None:
                        continue
                    if isinstance(extracted[0], list):
                        results.extend(extracted)
                    else:
                        results.append(extracted)
                if results:
                    return results
            if len(bbox_data) > 0 and isinstance(bbox_data[0], str):
                results = []
                for s in bbox_data:
                    results.extend(_extract_bbox_nums_by_regex(s))
                if results:
                    return results

        # --- dict input ---
        if isinstance(bbox_data, dict):
            if 'bboxes' in bbox_data:
                val = bbox_data['bboxes']
                if isinstance(val, list):
                    if val and isinstance(val[0], list):
                        return [b[:4] for b in val if len(b) >= 4 and all(isinstance(x, (int, float)) for x in b[:4])]
                    if len(val) == 4 and all(isinstance(x, (int, float)) for x in val):
                        return [val]
            extracted = _extract_bbox_from_dict(bbox_data)
            if extracted is not None:
                if isinstance(extracted[0], list):
                    return extracted
                return [extracted]

        return regex_bboxes

    except (json.JSONDecodeError, IndexError, KeyError, TypeError, ValueError):
        pass

    return regex_bboxes


class VANTAGE_2DGroundingDataset(ImageBaseDataset):
    """Dataset class for 2D visual grounding evaluation in RefCOCO format."""

    TYPE = 'GROUNDING'  # Use VQA type so predictions are treated as text
    MODALITY = 'IMAGE'

    # Prompt template for referring-expression grounding.
    # Uses grounding framing (find the referred instance) rather than detection
    # framing (find all instances of the category).
    PROMPT_TEMPLATE = (
        "As an AI visual assistant, your task is to identify and locate specific objects in the provided image.\n\n"
        "Supplied Description: {description}\n\n"
        "Task:\n"
        "Based on the description and the image content, identify the key groups of objects mentioned. "
        "For each group, provide a descriptive label and the precise bounding box coordinates for every individual instance in that group.\n\n"
        "Coordinates must be normalized to a 0-1000 scale in [x1, y1, x2, y2] format.\n\n"
        "Output Format:\n"
        "For each group of objects, output one line in exactly this format:\n"
        "The [object description]: [[x1, y1, x2, y2], [x3, y3, x4, y4]]\n\n"
        "Example:\n"
        "The blue cars parked on the right: [[579, 454, 690, 636], [342, 441, 435, 608]]"
    )

    @classmethod
    def supported_datasets(cls):
        return ['VANTAGE_2DGrounding', 'VANTAGE_2DGrounding_val', 'VANTAGE_2DGrounding_small']

    def __init__(self, dataset='VANTAGE_2DGrounding', data_root=None, annotation_file=None, custom_prompt=None,
                 limit=None, random_state=None, **kwargs):
        """
        Args:
            dataset: Dataset name (used to look up config in datasets.yaml)
            data_root: Root directory containing 'images' subdirectory.
            annotation_file: Path to RefCOCO format annotation JSON file
            custom_prompt: Custom prompt for grounding (overrides default prompt template if provided)
        """
        self.dataset_name = dataset
        self.data_root = data_root
        self.annotation_file = annotation_file
        self.custom_prompt = custom_prompt

        # --- 1. RESOLVE DATA ROOT ---
        if self.data_root is None:
            try:
                # Attempt to find the root via LMUDataRoot environment
                base_root = LMUDataRoot()
                self.data_root = os.path.join(base_root, 'datasets', dataset)
            except Exception as e:
                print(f"DEBUG: LMUDataRoot resolution failed: {e}")
                self.data_root = None

        # --- 2. LOAD CONFIG FROM YAML ---
        # If still missing info, try the datasets.yaml config
        if self.data_root is None or self.annotation_file is None:
            dataset_cfg = load_dataset_config(dataset, task='grounding')
            if dataset_cfg:
                if self.data_root is None:
                    self.data_root = dataset_cfg.get('data_root')
                if self.annotation_file is None:
                    self.annotation_file = dataset_cfg.get('annotation_file')

        # Critical check for data_root before proceeding to S3 or file lookups
        if self.data_root is None:
            raise ValueError(
                f"data_root must be specified for dataset '{dataset}'. "
                "Run: python scripts/run_lmudata.py --task grounding --lmu-root ~/LMUData"
            )

        
        self.img_root = self.data_root
        self.images_dir = os.path.join(self.data_root, 'images')

        # --- 4. FIND ANNOTATION FILE (The Bug Fix Area) ---
        if self.annotation_file is None:
            # List possible locations/names to check automatically
            # Added 'annotations/coco.json' based on common RefCOCO structures
            search_candidates = [
                os.path.join(self.data_root, 'annotations/coco.json'),
                os.path.join(self.data_root, 'annotations.json'),
                os.path.join(self.data_root, 'grounding.json'),
                os.path.join(self.data_root, 'val.json')
            ]
            for path in search_candidates:
                if os.path.exists(path):
                    self.annotation_file = path
                    break

        # REFIXED LOGIC: Check if it is STILL None OR if the path doesn't actually exist
        # This avoids the TypeError: stat: path should be string... not NoneType
        if self.annotation_file is None or not os.path.exists(self.annotation_file):
            raise ValueError(f"Annotation file not found. Resolved path: {self.annotation_file}")

        # --- 5. INITIALIZE DATA ---
        self.annotation_format = detect_annotation_format(self.annotation_file)
        if self.annotation_format == 'jsonl':
            self.annotations = parse_jsonl_annotations(self.annotation_file, self.data_root)
        else:
            self.annotations = parse_refcoco_annotations(self.annotation_file)
        
        #assert len(self.annotations) > 0, f"No annotations found in {self.annotation_file}"
        with open(self.annotation_file, 'r') as file:
            data = json.load(file)

        first_entry = data[0]
        target_key = "bboxes"
        if target_key in first_entry:
            self.data = self._build_data_structure_gt()
        else:
            self.data = self._build_data_structure()

        try:
            self.post_build(self.dataset_name)
        except Exception:
            pass

        if limit is not None and limit > 0 and hasattr(self, 'data'):
            original_size = len(self.data)
            if limit <= 1.0:
                sample_num = max(1, int(limit * len(self.data)))
            else:
                sample_num = min(int(limit), len(self.data))
            self.data = self.data.sample(n=sample_num, random_state=random_state)
            print(f"Applied limit: using {len(self.data)} of {original_size} samples")


    def _build_data_structure_gt(self):
        """Build the data structure for VLMEvalKit format."""
        logger = get_logger('VANTAGE_2DGrounding')
        logger.info(f"Loaded {len(self.annotations)} annotations from {self.annotation_file} "
                    f"(format: {self.annotation_format})")

        data_list = []
        skipped_count = 0

        for idx, ann in enumerate(self.annotations):
            # Handle image path based on annotation format
            if 'image_path' in ann and ann['image_path']:
                # JSONL format: image_path is already resolved
                image_path = ann['image_path']
                image_filename = ann.get('image', os.path.basename(image_path))
            else:
                # RefCOCO format: construct path from images_dir
                image_filename = ann['image']
                image_path = os.path.join(self.images_dir, image_filename)

            # Skip if image doesn't exist
            if not os.path.exists(image_path):
                skipped_count += 1
                if skipped_count <= 5:
                    logger.warning(f"Image not found: {image_path}")
                elif skipped_count == 6:
                    logger.warning("Suppressing further 'Image not found' warnings...")
                continue

            sentence = ann['sentence']
            gt_bboxes = ann['bboxes']  # List of bboxes

            if self.custom_prompt is not None:
                question = self.custom_prompt.format(description=sentence)
            else:
                # Generate the question/prompt
                question = self.PROMPT_TEMPLATE.format(description=sentence)

            row = {
                'index': str(idx),
                'image_path': image_path,
                'image_filename': image_filename,
                'question': question,
                'sentence': sentence,
                'gt_bboxes': json.dumps(gt_bboxes),  # Store list of bboxes as JSON string
                'num_gt_bboxes': len(gt_bboxes),
                'category': ann.get('category', ''),
                'image_width': ann.get('width'),
                'image_height': ann.get('height'),
            }
            data_list.append(row)

        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count} annotations due to missing images")

        logger.info(f"Built dataset with {len(data_list)} samples")
        return pd.DataFrame(data_list)


    def _build_data_structure(self):
        """Build the data structure for VLMEvalKit format."""
        logger = get_logger('VANTAGE_2DGrounding')
        logger.info(f"Loaded {len(self.annotations)} annotations from {self.annotation_file} "
                    f"(format: {self.annotation_format})")

        data_list = []
        skipped_count = 0

        for idx, ann in enumerate(self.annotations):
            # Handle image path based on annotation format
            if 'image_path' in ann and ann['image_path']:
                # JSONL format: image_path is already resolved
                image_path = ann['image_path']
                image_filename = ann.get('image', os.path.basename(image_path))
            else:
                # RefCOCO format: construct path from images_dir
                image_filename = ann['image']
                image_path = os.path.join(self.images_dir, image_filename)

            # Skip if image doesn't exist
            if not os.path.exists(image_path):
                skipped_count += 1
                if skipped_count <= 5:
                    logger.warning(f"Image not found: {image_path}")
                elif skipped_count == 6:
                    logger.warning("Suppressing further 'Image not found' warnings...")
                continue

            sentence = ann['sentence']
            #gt_bboxes = ann['bboxes']  # List of bboxes

            if self.custom_prompt is not None:
                question = self.custom_prompt.format(description=sentence)
            else:
                # Generate the question/prompt
                question = self.PROMPT_TEMPLATE.format(description=sentence)

            row = {
                'index': str(idx),
                'image_path': image_path,
                'image_filename': image_filename,
                'question': question,
                'sentence': sentence,
                #'gt_bboxes': json.dumps(gt_bboxes),  # Store list of bboxes as JSON string
                #'num_gt_bboxes': len(gt_bboxes),
                'category': ann.get('category', ''),
                #'image_width': ann.get('width'),
                #'image_height': ann.get('height'),
            }
            data_list.append(row)

        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count} annotations due to missing images")

        logger.info(f"Built dataset with {len(data_list)} samples")
        return pd.DataFrame(data_list)

    def build_prompt(self, line):
        """Build prompt for visual grounding."""
        if isinstance(line, int):
            line = self.data.iloc[line]

        image_path = line['image_path']
        question = line['question']

        msgs = [
            dict(type='image', value=image_path),
            dict(type='text', value=question)
        ]

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        """
        Evaluate grounding predictions using standard metrics.

        Metrics:
        - Acc@0.5: Accuracy at IoU threshold 0.5 (standard RefCOCO metric)
        - Acc@0.25: Accuracy at IoU threshold 0.25
        - Acc@0.75: Accuracy at IoU threshold 0.75
        - Mean IoU: Average IoU across all samples

        Note: When multiple GT boxes exist for one expression, a prediction is
        considered correct if it matches ANY of the GT boxes (max IoU is used).
        """
        logger = get_logger('VANTAGE_2DGrounding')

        # Try different file extensions if the specified one doesn't exist
        if not os.path.exists(eval_file):
            base = os.path.splitext(eval_file)[0]
            for ext in ['.xlsx', '.tsv', '.json', '.pkl']:
                candidate = base + ext
                if os.path.exists(candidate):
                    eval_file = candidate
                    logger.info(f'Using alternate file format: {eval_file}')
                    break
            else:
                logger.error(f'No prediction file found with base: {base}')
                return {
                    'Acc@0.5': 0.0,
                    'Acc@0.25': 0.0,
                    'Acc@0.75': 0.0,
                    'Mean_IoU': 0.0,
                    'error': 'Prediction file not found'
                }

        try:
            data = load(eval_file)
            logger.info(f'Loaded {len(data)} predictions from {eval_file}')
        except Exception as e:
            logger.error(f'Failed to load predictions: {e}')
            return {'acc_at_0_5': 0.0, 'acc_at_0_25': 0.0, 'acc_at_0_75': 0.0, 'mean_iou': 0.0, 'error': str(e)}

        import os as _os
        from vlmeval.dataset.utils.vantagebench.emit import emit_submission
        _suffix = eval_file.split('.')[-1]
        submission_path = eval_file.replace(f'.{_suffix}', '_submission.jsonl')
        emit_submission(data, _os.path.splitext(_os.path.basename(eval_file))[0], submission_path, task='grounding')
        print(f"Submission written to: {submission_path}")

        has_gt = 'gt_bboxes' in data.columns and data['gt_bboxes'].notna().any()
        if not has_gt and 'gt_bboxes' in self.data.columns:
            has_gt = self.data['gt_bboxes'].notna().any()
        if not has_gt:
            return {}

        # Evaluation metrics
        iou_thresholds = [0.25, 0.5, 0.75]
        correct_at_threshold = {t: 0 for t in iou_thresholds}
        all_ious = []
        valid_count = 0
        total_count = len(data)
        parse_fail_count = 0
        invalid_box_count = 0

        for idx, row in data.iterrows():
            # Parse prediction
            pred_text = str(row.get('prediction', ''))
            pred_bbox = parse_bbox_from_text(pred_text)
            if len(pred_bbox) == 0 and pred_text.strip() not in ('', 'nan'):
                parse_fail_count += 1

            # Get ground truth bboxes from the merged prediction row (preferred)
            # or fall back to GT dataframe lookup by index column.
            gt_bboxes_str = row.get('gt_bboxes', None)
            if gt_bboxes_str is None:
                # Fallback: match by index column to avoid positional misalignment
                sample_idx = str(row.get('index', idx))
                gt_row = self.data[self.data['index'] == sample_idx]
                gt_bboxes_str = gt_row.iloc[0]['gt_bboxes'] if len(gt_row) > 0 else '[]'
            try:
                gt_bboxes = json.loads(gt_bboxes_str)
            except (json.JSONDecodeError, TypeError):
                gt_bboxes = []

            # Ensure gt_bboxes is a list of bboxes
            if gt_bboxes and isinstance(gt_bboxes[0], (int, float)):
                # Single bbox stored as flat list, wrap it
                gt_bboxes = [gt_bboxes]

            if len(pred_bbox) == 0 or len(gt_bboxes) == 0:
                all_ious.append(0.0)
                continue

            valid_count += 1

            # Get image dimensions from row (stored in dataframe), avoid opening image
            img_width = row.get('image_width', None)
            img_height = row.get('image_height', None)
            if img_width is None or img_height is None:
                image_path = row.get('image_path', '')
                try:
                    pil_image = PIL.Image.open(image_path)
                    pil_image = ImageOps.exif_transpose(pil_image)
                    img_width, img_height = pil_image.size
                except Exception:
                    img_width = img_width or 1000
                    img_height = img_height or 1000

            img_width = float(img_width)
            img_height = float(img_height)

            # Scale predictions from 0-1000 normalized → pixel coords to match GT
            pred_bboxes_scaled_raw = [
                [p[0] * img_width / 1000, p[1] * img_height / 1000,
                 p[2] * img_width / 1000, p[3] * img_height / 1000]
                for p in pred_bbox
            ]

            # Clip to image bounds, then filter degenerate/inverted boxes.
            pred_bboxes_scaled = []
            for b in pred_bboxes_scaled_raw:
                x1 = max(0.0, min(b[0], img_width))
                y1 = max(0.0, min(b[1], img_height))
                x2 = max(0.0, min(b[2], img_width))
                y2 = max(0.0, min(b[3], img_height))
                if x2 > x1 and y2 > y1:
                    pred_bboxes_scaled.append([x1, y1, x2, y2])
                else:
                    invalid_box_count += 1

            if len(pred_bboxes_scaled) == 0:
                all_ious.append(0.0)
                continue

            # Best 1-to-1 matching: for each pred bbox find its best GT IoU,
            # then take the overall maximum across all pred bboxes.
            max_iou = 0.0
            for pb_scaled in pred_bboxes_scaled:
                for gt_bbox in gt_bboxes:
                    if len(gt_bbox) >= 4:
                        iou = compute_2d_iou(pb_scaled, gt_bbox)
                        max_iou = max(max_iou, iou)

            all_ious.append(max_iou)

            # Check accuracy at each threshold (using max IoU)
            for thresh in iou_thresholds:
                if max_iou >= thresh:
                    correct_at_threshold[thresh] += 1

        # Compute metrics
        mean_iou = np.mean(all_ious) if len(all_ious) > 0 else 0.0

        acc_25 = float(correct_at_threshold[0.25] / total_count) if total_count > 0 else 0.0
        acc_50 = float(correct_at_threshold[0.5] / total_count) if total_count > 0 else 0.0
        acc_75 = float(correct_at_threshold[0.75] / total_count) if total_count > 0 else 0.0

        logger.info(f"Acc@0.25: {acc_25:.4f}")
        logger.info(f"Acc@0.5: {acc_50:.4f}")
        logger.info(f"Acc@0.75: {acc_75:.4f}")
        logger.info(f"Mean IoU: {mean_iou:.4f}")
        logger.info(f"Valid predictions: {valid_count}/{total_count} ({valid_count / total_count if total_count > 0 else 0:.2%})")
        logger.info(f"Parse failures: {parse_fail_count}, invalid boxes filtered: {invalid_box_count}")

        result = {
            'acc_at_0_5': acc_50,
            'acc_at_0_25': acc_25,
            'acc_at_0_75': acc_75,
            'mean_iou': float(mean_iou),
        }
        suffix = eval_file.split('.')[-1]
        metrics_file = eval_file.replace(f'.{suffix}', '_metrics.json')
        dump({**result, 'total_samples': total_count, 'valid_predictions': valid_count,
              'valid_rate': valid_count / total_count if total_count > 0 else 0.0,
              'parse_failures': parse_fail_count, 'invalid_boxes_filtered': invalid_box_count}, metrics_file)

        return result
