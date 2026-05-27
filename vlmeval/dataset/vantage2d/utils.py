"""
Common utilities for 2D detection datasets.

This module contains shared functions for parsing, evaluation, and configuration
used by various 2D detection datasets (VANTAGE_2D, Astro2D, etc.).
"""

import os
import json
import numpy as np
import yaml


def load_dataset_config(dataset_name, config_path=None, task='detection'):
    """
    Load dataset configuration from datasets.yaml.

    Args:
        dataset_name: Name of the dataset to look up
        config_path: Path to datasets.yaml. If None, uses default location.

    Returns:
        dict with dataset configuration or None if not found
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'datasets.yaml')

    if not os.path.exists(config_path):
        return None

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Look in 'detection' section
        if task in config:
            for dataset_cfg in config[task]:
                if dataset_cfg.get('name') == dataset_name:
                    return dataset_cfg

        return None
    except Exception:
        return None


def scale_bbox(bbox, height, width, scale_factor=1000):
    """
    Scale the bounding box to the original image size.

    Args:
        bbox: The bounding box to scale.
        height: The height of the original image.
        width: The width of the original image.
        scale_factor: The scale factor to use.
    """
    abs_x1, abs_y1, abs_x2, abs_y2 = (
        int(bbox[0] / scale_factor * width),
        int(bbox[1] / scale_factor * height),
        int(bbox[2] / scale_factor * width),
        int(bbox[3] / scale_factor * height)
    )

    # Clip the bounding box to the image size
    abs_x1 = max(abs_x1, 0)
    abs_y1 = max(abs_y1, 0)

    abs_x2 = min(abs_x2, width)
    abs_y2 = min(abs_y2, height)

    return abs_x1, abs_y1, abs_x2, abs_y2


def parse_kitti_label(label_path):
    """
    Parse KITTI format label file.

    KITTI format: type truncated occluded alpha bbox(x1,y1,x2,y2) dimensions(h,w,l) location(x,y,z) rotation_y [score]

    Returns:
        List of dicts with 'label' and 'bbox' (x1, y1, x2, y2 in pixel coordinates)
    """
    objects = []
    if not os.path.exists(label_path):
        return objects

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue

            label = parts[0].lower()
            # Skip DontCare or other ignore labels
            if label in ['dontcare', 'misc', 'ignore']:
                continue

            # KITTI bbox format: x1, y1, x2, y2 (0-indexed, pixel coords)
            try:
                x1 = float(parts[4])
                y1 = float(parts[5])
                x2 = float(parts[6])
                y2 = float(parts[7])
            except (ValueError, IndexError):
                continue

            objects.append({
                'label': label,
                'bbox': [x1, y1, x2, y2]
            })

    return objects


def parse_bbox_2d_from_text(text: str) -> list:
    """
    Parse 2D bounding box information from assistant response.

    Args:
        text: Assistant response text containing JSON with bbox information

    Returns:
        List of dictionaries containing bbox data with 'bbox' and optionally 'label'
    """
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
            # Find first [ and last ]
            start_idx = text.find("[")
            end_idx = text.rfind("]")
            if start_idx != -1 and end_idx != -1:
                json_str = text[start_idx:end_idx + 1]
            else:
                return []

        bbox_data = json.loads(json_str)

        # Normalize to list format
        if isinstance(bbox_data, list):
            return bbox_data
        elif isinstance(bbox_data, dict):
            return [bbox_data]
        else:
            return []

    except (json.JSONDecodeError, IndexError, KeyError):
        return []


def compute_2d_iou(box1, box2):
    """
    Compute IoU between two 2D bounding boxes.

    Args:
        box1, box2: [x1, y1, x2, y2] format

    Returns:
        IoU value (float)
    """
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def compute_ap_coco(recalls, precisions):
    """
    Compute Average Precision using COCO-style all-point interpolation.

    The precision at each recall level r is interpolated by taking the maximum
    precision measured at a recall >= r. Then AP is computed as the area under
    this interpolated precision-recall curve.

    Args:
        recalls: numpy array of recall values (sorted in ascending order)
        precisions: numpy array of precision values corresponding to recalls

    Returns:
        AP value (float)
    """
    if len(recalls) == 0:
        return 0.0

    # Prepend (0, 1) and append (1, 0) to the PR curve
    recalls = np.concatenate([[0.0], recalls, [1.0]])
    precisions = np.concatenate([[1.0], precisions, [0.0]])

    # Make precision monotonically decreasing (from right to left)
    # This is the interpolation step: precision at recall r is max precision at recall >= r
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Find points where recall changes
    recall_change_indices = np.where(recalls[1:] != recalls[:-1])[0] + 1

    # Compute area under the interpolated PR curve
    ap = np.sum((recalls[recall_change_indices] - recalls[recall_change_indices - 1]) *
                precisions[recall_change_indices])

    return ap


def compute_ap_from_matches(tp_list, num_gt, confidence_scores):
    """
    Compute AP from detection matches using COCO-style calculation.

    Args:
        tp_list: list of 1s (true positive) or 0s (false positive) for each detection
        num_gt: total number of ground truth objects
        confidence_scores: confidence score for each detection

    Returns:
        AP value, precision array, recall array
    """
    if num_gt == 0:
        return 0.0, np.array([]), np.array([])

    if len(tp_list) == 0:
        return 0.0, np.array([]), np.array([])

    # Sort by confidence (descending)
    sorted_indices = np.argsort(-np.array(confidence_scores))
    tp_sorted = np.array(tp_list)[sorted_indices]

    # Cumulative sums
    tp_cumsum = np.cumsum(tp_sorted)
    fp_cumsum = np.cumsum(1 - tp_sorted)

    # Precision and recall at each threshold
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    recalls = tp_cumsum / num_gt

    # Compute AP using COCO-style interpolation
    ap = compute_ap_coco(recalls, precisions)

    return ap, precisions, recalls


def evaluate_detections(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Evaluate detections against ground truth using the given IoU threshold.

    Args:
        pred_boxes: List of predicted boxes, each with 'bbox' and optional 'score'
        gt_boxes: List of ground truth boxes with 'bbox'
        iou_threshold: IoU threshold for matching (default 0.5 for AP50)

    Returns:
        dict with 'tp', 'fp', 'fn', 'precision', 'recall'
    """
    if len(pred_boxes) == 0:
        return {
            'tp': 0,
            'fp': 0,
            'fn': len(gt_boxes),
            'precision': 0.0,
            'recall': 0.0
        }

    if len(gt_boxes) == 0:
        return {
            'tp': 0,
            'fp': len(pred_boxes),
            'fn': 0,
            'precision': 0.0,
            'recall': 1.0 if len(pred_boxes) == 0 else 0.0
        }

    # Sort predictions by score if available
    if 'score' in pred_boxes[0]:
        pred_boxes = sorted(pred_boxes, key=lambda x: x.get('score', 1.0), reverse=True)

    gt_matched = [False] * len(gt_boxes)
    tp = 0
    fp = 0

    for pred in pred_boxes:
        pred_bbox = pred['bbox']
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(gt_boxes):
            if gt_matched[gt_idx]:
                continue
            iou = compute_2d_iou(pred_bbox, gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold:
            tp += 1
            gt_matched[best_gt_idx] = True
        else:
            fp += 1

    fn = sum(1 for matched in gt_matched if not matched)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall
    }
