"""
2D Detection Dataset for images in KITTI format.

The validation data should be in KITTI format with:
- images/ directory containing image files
- labels/ directory containing label files (KITTI format)

For evaluation, predicted labels and groundtruth labels are mapped to "person" 
and evaluated using F1 score.
"""

import os
import random
import numpy as np
import pandas as pd
import PIL.Image
from PIL import ImageOps
from ..image_base import ImageBaseDataset
from ...smp import *
from .utils import (
    load_dataset_config,
    scale_bbox,
    parse_kitti_label,
    parse_bbox_2d_from_text,
    compute_2d_iou,
)


# Categories that map to "person" for evaluation
PERSON_CATEGORIES = {'person', "Person", "people", "People", "pedestrian", "Pedestrian"}

# Default minimum bbox area in pixels (used when not set per-dataset)
DEFAULT_MIN_BBOX_AREA = 0

# Final path segment that uses MIN_BBOX_AREA=400 by default
SEQ_WITH_NOISE = 'IVA-0009-KPI-05_190916_10ft-60-deg.mp4'


def _normalize_pred_bbox(bbox, sample_id=None):
    """
    Normalize a parsed prediction bbox to a plain [x1, y1, x2, y2] float list.

    Accepted input shapes:
      list/tuple  : [x1, y1, x2, y2]  (numeric or numeric-string elements)
      dict (nested): {"bbox"|"box"|"pred_bbox"|"bbox_2d"|"box_2d": [x1,y1,x2,y2]}
      dict (named) : {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                     {"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2}

    Returns [x1, y1, x2, y2] as floats, or None if the shape is unsupported.
    """
    logger = get_logger('Astro2D')
    tag = f' (sample {sample_id})' if sample_id is not None else ''

    def _to_float(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    if isinstance(bbox, (list, tuple)):
        if len(bbox) >= 4:
            vals = [_to_float(bbox[i]) for i in range(4)]
            if all(v is not None for v in vals):
                return vals
        logger.warning(f'Unsupported bbox list/tuple shape{tag}: {bbox!r}')
        return None

    if isinstance(bbox, dict):
        for key in ('bbox', 'box', 'pred_bbox', 'bbox_2d', 'box_2d'):
            if key in bbox:
                val = bbox[key]
                if isinstance(val, (list, tuple)) and len(val) >= 4:
                    vals = [_to_float(val[i]) for i in range(4)]
                    if all(v is not None for v in vals):
                        return vals
        if all(k in bbox for k in ('x1', 'y1', 'x2', 'y2')):
            vals = [_to_float(bbox[k]) for k in ('x1', 'y1', 'x2', 'y2')]
            if all(v is not None for v in vals):
                return vals
        if all(k in bbox for k in ('xmin', 'ymin', 'xmax', 'ymax')):
            vals = [_to_float(bbox[k]) for k in ('xmin', 'ymin', 'xmax', 'ymax')]
            if all(v is not None for v in vals):
                return vals
        logger.warning(f'Unsupported bbox dict shape{tag}: {bbox!r}')
        return None

    logger.warning(f'Unrecognized bbox type {type(bbox).__name__}{tag}: {bbox!r}')
    return None


def compute_bbox_area(bbox):
    """
    Compute the area of a bounding box.
    
    Args:
        bbox: [x1, y1, x2, y2] format

    Returns:
        Area in pixels
    """
    x1, y1, x2, y2 = bbox[:4]
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    return width * height


# Default prompt for detection
DETECTION_PROMPT = (
    "Locate every instance that belongs to the following categories: 'person'. "
    'Report bbox coordinates as JSON: [{"bbox_2d": [x1, y1, x2, y2], "label": "..."}]. '
    "Coordinates normalized to 0-1000."
)


def map_label_to_person(label):
    """
    Map various person labels to 'person'.

    Args:
        label: Original label string

    Returns:
        'person' if label is a person type, otherwise original label
    """
    if label is None or isinstance(label, (list, tuple, dict)):
        return 'person'

    label = str(label).strip().lower()

    person_categories = {str(x).strip().lower() for x in PERSON_CATEGORIES}
    if label in person_categories:
        return 'person'
    return label


class Astro2DDetectionDataset(ImageBaseDataset):
    """Dataset class for 2D object detection evaluation in KITTI format."""

    TYPE = 'Detection'  # Use VQA type so predictions are treated as text
    MODALITY = 'IMAGE'

    @classmethod
    def supported_datasets(cls):
        return ['Astro2D']

    def __init__(self, dataset='Astro2D', data_root=None, custom_prompt=None, limit=None, random_state=None, **kwargs):
        """
        Args:
            dataset: Dataset name (used to look up config in datasets.yaml)
            data_root: Root directory containing 'images' and 'labels' subdirectories.
                       If None, will be loaded from datasets.yaml based on dataset name.
            min_bbox_area: Optional. Min bbox area in pixels; bboxes smaller than this are
                filtered out. If not set in datasets.yaml or kwargs, defaults to 400 when
                the path ends with "IVA-0009-KPI-05_190916_10ft-60-deg.mp4", else 0.
        """
        self.dataset_name = dataset
        self.data_root = data_root
        self.custom_prompt = custom_prompt

        
        # FORCED PATH RESOLUTION
        if data_root is None:
            # Use the environment's LMU root and manually append the path
            try:
                base_root = LMUDataRoot()
                self.data_root = os.path.join(base_root, 'datasets', dataset)
                print(f"DEBUG: Successfully resolved data_root to: {self.data_root}")
            except Exception as e:
                print(f"DEBUG: LMUDataRoot failed: {e}")
                self.data_root = None
        else:
            self.data_root = data_root

        # Keep a dummy config so the rest of the script doesn't break
        dataset_cfg = {}
        if data_root is None:
            # Try to load from datasets.yaml
            dataset_cfg = load_dataset_config(dataset) or {}
            if dataset_cfg and 'data_root' in dataset_cfg:
                self.data_root = dataset_cfg['data_root']
        else:
            dataset_cfg = load_dataset_config(dataset) or {}

        if self.data_root is None:
            raise ValueError(
                f"data_root must be specified or configured in datasets.yaml for dataset '{dataset}'"
            )

        if self.data_root.startswith('s3://'):
            raise ValueError(
                f"S3 data_root ('{self.data_root}') is not supported. "
                "Copy the dataset to a local path and set data_root to that path."
            )
        else:
            self.img_root = self.data_root
            # Prefer images_hres/labels_hres; fall back to images/labels
            if os.path.isdir(os.path.join(self.data_root, 'images_hres')):
                self.images_dir = os.path.join(self.data_root, 'images_hres')
                self.labels_dir = os.path.join(self.data_root, 'labels_hres')
            else:
                self.images_dir = os.path.join(self.data_root, 'images')
                self.labels_dir = os.path.join(self.data_root, 'labels')

        # min_bbox_area: from config, or 400 if path ends with IVA-0009-KPI-05_190916_10ft-60-deg.mp4, else 0
        if 'min_bbox_area' in dataset_cfg:
            self.min_bbox_area = int(dataset_cfg['min_bbox_area'])
        elif kwargs.get('min_bbox_area') is not None:
            self.min_bbox_area = int(kwargs['min_bbox_area'])
        else:
            final_name = self.data_root.rstrip('/').split('/')[-1]
            self.min_bbox_area = 400 if final_name == SEQ_WITH_NOISE else DEFAULT_MIN_BBOX_AREA

        # Cache for ground truth
        self._gt_cache = {}

        # Build data structure
        self.data = self._build_data_structure()

        # Call post build hook for compatibility
        try:
            self.post_build(self.dataset_name)
        except Exception:
            pass

        if limit is not None and limit > 0 and hasattr(self, 'data'):
            original_size = len(self.data)
            sample_num = max(1, int(limit * len(self.data))) if limit <= 1.0 else min(int(limit), len(self.data))
            self.data = self.data.sample(n=sample_num, random_state=random_state)
            print(f'Applied limit: using {len(self.data)} of {original_size} samples')

        if limit is not None and limit > 0 and hasattr(self, 'data'):
            original_size = len(self.data)
            sample_num = max(1, int(limit * len(self.data))) if limit <= 1.0 else min(int(limit), len(self.data))
            self.data = self.data.sample(n=sample_num, random_state=random_state)
            print(f'Applied limit: using {len(self.data)} of {original_size} samples')

        if limit is not None and limit > 0 and hasattr(self, 'data'):
            original_size = len(self.data)
            sample_num = max(1, int(limit * len(self.data))) if limit <= 1.0 else min(int(limit), len(self.data))
            self.data = self.data.sample(n=sample_num, random_state=random_state)
            print(f'Applied limit: using {len(self.data)} of {original_size} samples')

    def get_config_dict(self):
        safe_dict = {}

        for k, v in self.__dict__.items():

            # Keep only simple types
            if isinstance(v, (int, float, str, bool, type(None), dict, list)):
                safe_dict[k] = v
            else:
                safe_dict[k] = str(type(v))  # fallback

        return safe_dict

    def _get_image_files(self):
        """Get list of image files from the images directory."""
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(
                f"Images directory not found: {self.images_dir}. "
                "Run: python scripts/run_lmudata.py --task astro2d --lmu-root ~/LMUData"
            )

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []

        for filename in sorted(os.listdir(self.images_dir)):
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_extensions:
                image_files.append(filename)

        return image_files

    def _load_ground_truth(self, image_filename):
        """
        Load ground truth labels for an image.

        Args:
            image_filename: Name of the image file

        Returns:
            List of ground truth objects with 'label' and 'bbox'
        """
        if image_filename in self._gt_cache:
            return self._gt_cache[image_filename]

        # Construct label path (same name as image, but .txt extension)
        base_name = os.path.splitext(image_filename)[0]
        label_path = os.path.join(self.labels_dir, base_name + '.txt')

        # Fall back to a suffix match if exact name not found (e.g. "C0065-QA_m4v_000003.txt")
        if not os.path.exists(label_path):
            suffix = f'_{base_name}.txt'
            for fname in os.listdir(self.labels_dir):
                if fname.endswith(suffix):
                    label_path = os.path.join(self.labels_dir, fname)
                    break

        gt_objects = parse_kitti_label(label_path)

        # Map labels to 'person'
        for obj in gt_objects:
            obj['original_label'] = obj['label']
            obj['label'] = map_label_to_person(obj['label'])

        self._gt_cache[image_filename] = gt_objects
        return gt_objects

    def _build_data_structure(self):
        """Build the data structure for VLMEvalKit format."""
        logger = get_logger('Astro2D')

        image_files = self._get_image_files()
        logger.info(f"Found {len(image_files)} images in {self.images_dir}")

        data_list = []
        for idx, image_filename in enumerate(image_files):
            image_path = os.path.join(self.images_dir, image_filename)

            # Load ground truth to check if there are objects
            # Skip images with no exact-match label file
            base_name = os.path.splitext(image_filename)[0]
            label_path = os.path.join(self.labels_dir, base_name + ".txt")
            if not os.path.exists(label_path):
                continue
            gt_objects = self._load_ground_truth(image_filename)

            if self.custom_prompt is not None:
                row = {
                    'index': str(idx),
                    'image_path': image_path,
                    'image_filename': image_filename,
                    'question': self.custom_prompt,
                    'num_gt_objects': len(gt_objects),
                }
            else:
                row = {
                    'index': str(idx),
                    'image_path': image_path,
                    'image_filename': image_filename,
                    'question': DETECTION_PROMPT,
                    'num_gt_objects': len(gt_objects),
                }
            data_list.append(row)

        logger.info(f"Built dataset with {len(data_list)} samples")
        return pd.DataFrame(data_list)

    def build_prompt(self, line):
        """Build prompt for 2D detection."""
        if isinstance(line, int):
            line = self.data.iloc[line]

        image_path = line['image_path']

        question = line['question']

        msgs = [
            dict(type='image', value=image_path),
            dict(type='text', value=question)
        ]
        
        # print(f"DEBUG: Msgs {msgs}")
        return msgs

    def _compute_f1_at_iou(self, all_predictions, all_gt_boxes, iou_threshold):
        """
        Compute F1 score at a specific IoU threshold.

        Args:
            all_predictions: List of (pred_boxes_person, gt_boxes_person) tuples per image
            all_gt_boxes: Not used, kept for API compatibility
            iou_threshold: IoU threshold for matching

        Returns:
            Tuple of (precision, recall, f1, tp, fp, fn)
        """
        total_tp = 0
        total_fp = 0
        total_gt = 0

        for pred_boxes_person, gt_boxes_person in all_predictions:
            total_gt += len(gt_boxes_person)

            # Sort predictions by confidence (descending) and shuffle for tie-breaking
            pred_boxes_sorted = sorted(pred_boxes_person, key=lambda x: x.get('score', 1.0), reverse=True)
            # random.shuffle(pred_boxes_sorted)

            # Match predictions to ground truth at IoU threshold
            gt_matched = [False] * len(gt_boxes_person)

            for pred in pred_boxes_sorted:
                pred_bbox = pred['bbox']
                best_iou = 0.0
                best_gt_idx = -1

                for gt_idx, gt in enumerate(gt_boxes_person):
                    if gt_matched[gt_idx]:
                        continue
                    iou = compute_2d_iou(pred_bbox, gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    total_tp += 1
                    gt_matched[best_gt_idx] = True
                else:
                    total_fp += 1

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / total_gt if total_gt > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1, total_tp, total_fp, total_gt - total_tp

    def evaluate(self, eval_file, **judge_kwargs):
        """
        Evaluate predictions using Precision, Recall, and F1 score at multiple IoU thresholds.
        """
        logger = get_logger('Astro2D')

        from vlmeval.dataset.utils.vantagebench.emit import emit_submission

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
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'f1_0.95': 0.0,
                    'f1_mIOU': 0.0,
                    'total_predictions': 0,
                    'valid_bbox_predictions': 0,
                    'error': 'Prediction file not found'
                }

        try:
            data = load(eval_file)
            logger.info(f'Loaded {len(data)} predictions from {eval_file}')
        except Exception as e:
            logger.error(f'Failed to load predictions: {e}')
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'f1_at_0_95': 0.0, 'f1_miou': 0.0, 'error': str(e)}

        _suffix = eval_file.split('.')[-1]
        submission_path = eval_file.replace(f'.{_suffix}', '_submission.jsonl')
        emit_submission(data, os.path.splitext(os.path.basename(eval_file))[0], submission_path, task='astro')
        print(f"Submission written to: {submission_path}")

        has_gt = hasattr(self, 'labels_dir') and os.path.isdir(self.labels_dir) and bool(os.listdir(self.labels_dir))
        if not has_gt:
            return {}

        # Collect all predictions and ground truths for multi-threshold evaluation
        all_predictions = []  # List of (pred_boxes_person, gt_boxes_person) tuples
        total_pred = 0
        total_gt = 0
        valid_count = 0
        total_gt_filtered = 0  # GT boxes filtered due to small size
        total_pred_filtered = 0  # Pred boxes filtered due to small size

        # Process each image
        for idx, row in data.iterrows():
            # Parse prediction
            pred_text = str(row.get('prediction', ''))
            pred_boxes_raw = parse_bbox_2d_from_text(pred_text)

            image_path = row.get('image_path', '')
            # Load image
            try:
                pil_image = PIL.Image.open(image_path)
                pil_image = pil_image.convert('RGB')
                pil_image = ImageOps.exif_transpose(pil_image)
                width, height = pil_image.size
            except Exception as e:
                logger.error(f"Failed to load image {image_path}: {e}")
                width, height = 640, 480

            # Normalize prediction format
            pred_boxes = []
            for pred in pred_boxes_raw:
                if isinstance(pred, (list, tuple)):
                    # Plain [x1, y1, x2, y2] coordinate list
                    bbox = _normalize_pred_bbox(pred, sample_id=idx)
                    if bbox is not None:
                        bbox = scale_bbox(bbox, height, width, scale_factor=1000)
                    if bbox is not None and len(bbox) >= 4:
                        pred_boxes.append({
                            'bbox': bbox[:4],
                            'label': 'person',
                            'score': 1.0,
                        })
                elif isinstance(pred, dict):
                    # Handle different bbox key names
                    # Gemini returns 'box_2d'; also accept 'bbox_2d' and 'bbox'
                    bbox = None
                    key = None
                    for k in ['bbox_2d', 'box_2d', 'bbox']:
                        if k in pred:
                            bbox = pred[k]
                            key = k
                            break

                    if bbox is not None:
                        bbox = _normalize_pred_bbox(bbox, sample_id=idx)
                    if bbox is not None:
                        # Gemini's native 'box_2d' is [y1, x1, y2, x2] in 0-1000 space;
                        # scale_bbox expects [x1, y1, x2, y2], so swap axes.
                        if key == 'box_2d':
                            bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]
                        bbox = scale_bbox(bbox, height, width, scale_factor=1000)

                    if bbox is not None and len(bbox) >= 4:
                        pred_boxes.append({
                            'bbox': bbox[:4],
                            'label': map_label_to_person(pred.get('label', 'person')),
                            'score': pred.get('score', pred.get('confidence', 1.0))
                        })

            if len(pred_boxes) > 0:
                valid_count += 1

            # Get ground truth
            image_filename = row.get('image_filename', '')
            if not image_filename:
                image_filename = os.path.basename(image_path)

            gt_boxes = self._load_ground_truth(image_filename)

            # Filter to only 'person' category
            gt_boxes_person_raw = [gt for gt in gt_boxes if gt['label'] == 'person']
            pred_boxes_person_raw = [p for p in pred_boxes if p['label'] == 'person']

            # Filter out small bboxes (area < min_bbox_area)
            gt_boxes_person = [gt for gt in gt_boxes_person_raw if compute_bbox_area(gt['bbox']) >= self.min_bbox_area]
            pred_boxes_person = [p for p in pred_boxes_person_raw if compute_bbox_area(p['bbox']) >= self.min_bbox_area]

            total_gt_filtered += len(gt_boxes_person_raw) - len(gt_boxes_person)
            total_pred_filtered += len(pred_boxes_person_raw) - len(pred_boxes_person)

            total_gt += len(gt_boxes_person)
            total_pred += len(pred_boxes_person)

            all_predictions.append((pred_boxes_person, gt_boxes_person))

        # Compute F1 at multiple IoU thresholds
        iou_thresholds = [0.5 + 0.05 * i for i in range(10)]  # 0.5, 0.55, ..., 0.95
        f1_scores = {}

        for iou_thresh in iou_thresholds:
            precision, recall, f1, tp, fp, fn = self._compute_f1_at_iou(all_predictions, None, iou_thresh)
            f1_scores[iou_thresh] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }

        # Get metrics at IoU=0.5
        metrics_05 = f1_scores[0.5]
        precision = metrics_05['precision']
        recall = metrics_05['recall']
        f1_05 = metrics_05['f1']
        total_tp = metrics_05['tp']
        total_fp = metrics_05['fp']

        # Get F1 at IoU=0.95
        f1_095 = f1_scores[0.95]['f1']

        # Compute mean F1 across all thresholds (F1@mIOU)
        f1_mIOU = np.mean([f1_scores[t]['f1'] for t in iou_thresholds])

        debug_info = {
            'total_predictions': len(data),
            'valid_bbox_predictions': valid_count,
            'valid_rate': valid_count / len(data) if len(data) > 0 else 0,
            'total_gt_objects': total_gt,
            'total_pred_objects': total_pred,
            'true_positives': total_tp,
            'false_positives': total_fp,
            'false_negatives': total_gt - total_tp,
            'gt_filtered_small': total_gt_filtered,
            'pred_filtered_small': total_pred_filtered,
        }

        if valid_count == 0:
            logger.warning("NO valid bbox predictions found (valid_count=0). "
                           "Check that the prediction JSON uses a recognized key: "
                           "bbox_2d, box_2d, or bbox.")

        logger.info(f"Precision@IoU=0.5: {precision:.4f}")
        logger.info(f"Recall@IoU=0.5: {recall:.4f}")
        logger.info(f"F1@IoU=0.5: {f1_05:.4f}")
        logger.info(f"F1@IoU=0.95: {f1_095:.4f}")
        logger.info(f"F1@mIOU (0.5:0.05:0.95): {f1_mIOU:.4f}")
        logger.info(f"TP: {total_tp}, FP: {total_fp}, FN: {total_gt - total_tp}")
        logger.info(f"Valid predictions: {valid_count}/{len(data)} ({debug_info['valid_rate']:.2%})")
        logger.info(f"Filtered small bboxes - GT: {total_gt_filtered}, Pred: {total_pred_filtered}")

        suffix = eval_file.split('.')[-1]
        score_file = eval_file.replace(f'.{suffix}', '_metrics.json')
        dump({**debug_info, 'f1': f1_05, 'precision': precision, 'recall': recall,
              'f1_at_0_95': f1_095, 'f1_miou': f1_mIOU}, score_file)
        logger.info(f"Metrics saved to {score_file}")

        return {
            'f1': float(f1_05),
            'precision': float(precision),
            'recall': float(recall),
            'f1_at_0_95': float(f1_095),
            'f1_miou': float(f1_mIOU),
        }
