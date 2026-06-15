"""
VANTAGE-Bench Temporal: temporal localization in videos.

Data: Place under LMUDataRoot()/datasets/VANTAGE_Temporal/
  - VANTAGE_Temporal.tsv (required)
  - videos/ (required, .mp4 files)
  - annotations/ (optional, for TSV generation)
  - mappings/ (optional, category JSON files)
"""
import argparse
import json
import os
import re
import csv
import glob
import numpy as np
from collections import defaultdict
from typing import Any, Tuple, List, Dict
from pathlib import Path
import pandas as pd
import base64

from ..smp import *
from ..smp.file import get_intermediate_file_path
from .video_base import VideoBaseDataset
from .utils import build_judge, DEBUG_MESSAGE


FAIL_MSG = 'Failed to obtain answer via API.'


class VANTAGE_Temporal(VideoBaseDataset):
    """
    VANTAGE-Bench Temporal dataset for temporal localization in videos.
    Data loaded from LMUDataRoot()/datasets/VANTAGE_Temporal.

    Category Filtering:
    Categories are resolved from category_mapping JSON files or by matching video name against
    include_categories. Available categories (from mapping files) include Smart_Spaces,
    Transportation_real, Transportation_sim, Warehouse, Healthcare, Retail, etc.
    """

    MD5 = ''
    TYPE = 'Video-Temporal-Localization'

    QUESTION_PREFIX = """Localize a series of activity events in the video, output the start and end timestamp for each event. Provide the result in json format with 'mm:ss.ss' format for time depiction for this event. Use keywords 'start' and 'end' in the json output."""

    def __init__(self, dataset='VANTAGE_Temporal', pack=False, nframe=0, fps=0, total_pixels=None, max_pixels=None, max_frames=None, test_mode=False, limit=None, verbose=False, random_state=None, include_categories=None, custom_prompt=None):
        self.test_mode = test_mode
        self.category_mapping = {}
        self.limit = limit
        self.verbose = verbose
        self.max_pixels = max_pixels
        self.max_frames = max_frames
        self.random_state = random_state
        self.include_categories = set(include_categories) if include_categories else None

        if not test_mode:
            super().__init__(dataset=dataset, pack=pack, nframe=nframe, fps=fps, total_pixels=total_pixels, max_pixels=max_pixels, max_frames=max_frames, custom_prompt=custom_prompt)
            # --- ADD THIS FILTER ---
            if hasattr(self, 'data') and len(self.data) > 0:
                # prepare_dataset sets self.data_root to 'LMUDataRoot/datasets/VANTAGE_Temporal/videos'
                video_exists = self.data['video'].apply(
                    lambda x: osp.exists(osp.join(self.data_root, str(x) + '.mp4'))
                )
                before_count = len(self.data)
                self.data = self.data[video_exists].reset_index(drop=True)
                print(f"Video existence check: Kept {len(self.data)}/{before_count} samples.")
            # ------------------------
            original_size = len(self.data) if hasattr(self, 'data') else 0

            if self.include_categories is None and self.verbose and hasattr(self, 'data') and 'video' in self.data.columns:
                all_cats = self.data['video'].apply(self.get_category)
                dist_all = all_cats.value_counts().to_dict()
                print(f"Dataset loaded with ALL categories (no filtering): {len(self.data)} total samples")
                print(f"Category distribution: {dist_all}")

            if self.include_categories is not None and hasattr(self, 'data') and 'video' in self.data.columns:
                derived_cats = self.data['video'].apply(self.get_category)
                before_rows = len(self.data)
                dist_before = derived_cats.value_counts().to_dict()
                if self.verbose:
                    print(f"Category distribution before filter: {dist_before}")
                categories_found = set(dist_before.keys())
                missing_categories = self.include_categories - categories_found
                if missing_categories:
                    print(f"Warning: Requested categories not found in dataset: {sorted(missing_categories)}")
                    print(f"   Available categories: {sorted(categories_found)}")
                self.data = self.data[derived_cats.isin(self.include_categories)]
                if self.verbose or len(self.data) == 0:
                    print(f"Filtered by categories {sorted(self.include_categories)}: {len(self.data)}/{before_rows} rows kept")
                    if self.verbose and len(self.data) > 0:
                        try:
                            dist_after = self.data['video'].apply(self.get_category).value_counts().to_dict()
                            print(f"Category distribution after filter: {dist_after}")
                        except Exception:
                            pass
                if len(self.data) == 0:
                    print(f"Error: No samples found for requested categories: {sorted(self.include_categories)}")

            if self.limit is not None and self.limit > 0 and hasattr(self, 'data'):
                if self.limit <= 1.0:
                    sample_num = max(1, int(self.limit * len(self.data)))
                    self.data = self.data.sample(n=sample_num, random_state=self.random_state)
                else:
                    sample_num = min(int(self.limit), len(self.data))
                    self.data = self.data.sample(n=sample_num, random_state=self.random_state)
                print(f"Applied limit sampling: using {len(self.data)} out of {original_size} samples")

            if hasattr(self, 'data') and 'video' in self.data.columns:
                videos = list(set(self.data['video']))
                videos.sort()
                self.videos = videos
        else:
            self.dataset_name = dataset
            self.nframe = nframe
            self.fps = fps
            self.total_pixels = total_pixels
            self.max_pixels = max_pixels
            self.max_frames = max_frames
            self.custom_prompt = custom_prompt

    def get_config_dict(self):
        safe_dict = {}

        for k, v in self.__dict__.items():

            # Keep only simple types
            if isinstance(v, (int, float, str, bool, type(None), dict, list)):
                safe_dict[k] = v
            else:
                safe_dict[k] = str(type(v))  # fallback

        return safe_dict
    

    @classmethod
    def supported_datasets(cls):
        return ['VANTAGE_Temporal']

    def prepare_dataset(self, dataset_name='VANTAGE_Temporal'):
        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')
            if not osp.exists(data_file):
                return False
            video_dir = osp.join(pth, 'videos')
            if not osp.exists(video_dir) or not os.listdir(video_dir):
                return False
            return True

        local_dir = osp.join(LMUDataRoot(), 'datasets', 'VANTAGE_Temporal')
        if check_integrity(local_dir):
            print(f"Using existing local dataset at: {local_dir}")
            dataset_path = local_dir
        else:
            annotations_dir = osp.join(local_dir, 'annotations')
            video_dir_local = osp.join(local_dir, 'videos')
            if osp.exists(annotations_dir) and os.listdir(annotations_dir) and osp.exists(video_dir_local):
                mapping_dir = osp.join(local_dir, 'mappings')
                if osp.exists(mapping_dir):
                    self.category_mapping = self._load_category_mapping(mapping_dir)
                self._generate_tsv_from_local_annotations(local_dir, dataset_name)
                dataset_path = local_dir
            else:
                raise FileNotFoundError(
                    f"VANTAGE_Temporal data not found under {local_dir}. "
                    "Run: python scripts/run_lmudata.py --task temporal --lmu-root ~/LMUData"
                )

        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')
        if not osp.exists(data_file):
            raise FileNotFoundError(
                f"VANTAGE_Temporal TSV not found: {data_file}. "
                "Run: python scripts/run_lmudata.py --task temporal --lmu-root ~/LMUData"
            )
        mapping_dir = osp.join(dataset_path, 'mappings')
        if osp.exists(mapping_dir):
            self.category_mapping = self._load_category_mapping(mapping_dir)

        return dict(data_file=data_file, root=osp.join(dataset_path, 'videos'))

    def _generate_tsv_from_local_annotations(self, local_dir, dataset_name):
        annotations_dir = osp.join(local_dir, 'annotations')
        annotation_files = glob.glob(osp.join(annotations_dir, '*.json'))
        print(f"Found {len(annotation_files)} annotation files")
        seen_qids = set()
        data_list = []
        for ann_file in sorted(annotation_files):
            with open(ann_file, 'r') as f:
                ann_data = json.load(f)
            if not isinstance(ann_data, list):
                continue
            for item in ann_data:
                qid = item.get('question_id', '')
                if qid and qid in seen_qids:
                    continue
                if qid:
                    seen_qids.add(qid)
                processed = self._process_annotation_item(item)
                if processed:
                    data_list.append(processed)
        data_list.sort(key=lambda x: (x['video'], x.get('qid', '')))
        for idx, item in enumerate(data_list):
            item['index'] = idx
        df = pd.DataFrame(data_list)
        tsv_path = osp.join(local_dir, f'{dataset_name}.tsv')
        df.to_csv(tsv_path, sep='\t', index=False)
        print(f"Generated TSV with {len(data_list)} entries ({len(df['video'].unique())} unique videos) -> {tsv_path}")

    def _get_category_for_video(self, vid):
        if not self.include_categories:
            return 'Other'
        vid_lower = vid.lower()
        for category_name in self.include_categories:
            if category_name.lower() in vid_lower:
                return category_name
        return 'Other'

    def _normalize_category(self, cat: str) -> str:
        if not cat:
            return 'Other'
        cat = str(cat).strip()
        if cat == 'Smart Spaces':
            return 'Smart_Spaces'
        return cat

    def get_category(self, vid: str) -> str:
        mapped = self._get_category(vid)
        if mapped and mapped.strip() and mapped != 'Other':
            return self._normalize_category(mapped)
        return self._get_category_for_video(vid)

    def _process_annotation_item(self, item):
        try:
            video_name = item.get('vid', '')
            if not video_name:
                return None
            question = item.get('question', '')
            if not question:
                return None
            question = self.QUESTION_PREFIX + "\n" + question
            answer_text = item.get('answer', '')
            match = re.match(r'<([\d.]+)>\s*<([\d.]+)>', answer_text)
            if match:
                start_seconds = float(match.group(1))
                end_seconds = float(match.group(2))
                start_str = f"{int(start_seconds // 60):02d}:{start_seconds % 60:05.2f}"
                end_str = f"{int(end_seconds // 60):02d}:{end_seconds % 60:05.2f}"
            else:
                start_str = "00:00.00"
                end_str = "00:10.00"
            return {
                'index': 0,
                'video': video_name,
                'question': question,
                'answer': json.dumps({'start': start_str, 'end': end_str}),
                'duration': item.get('duration', 30.0),
                'category': self.get_category(video_name),
                'qid': item.get('question_id', f"{video_name}_0")
            }
        except Exception as e:
            print(f"Error processing annotation item: {e}")
            return None

    def _load_category_mapping(self, directory):
        merged_mapping = {}
        json_files = sorted(glob.glob(os.path.join(directory, "*.json")))
        if self.verbose:
            print(f"Loading category mappings from {len(json_files)} files...")
        for file in json_files:
            try:
                with open(file) as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        merged_mapping.update(data)
            except (json.JSONDecodeError, OSError) as e:
                print(f"Error loading JSON from {file}: {e}")
        return merged_mapping

    def generate_question(self, base_question: str) -> str:
        if not base_question:
            base_question = "Identify and localize the main activity in this video."
        return self.QUESTION_PREFIX + "\n" + base_question

    def build_prompt(self, line, video_llm=True):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]
        process_video_kwargs = {
            k: v for k, v in dict(
                total_pixels=self.total_pixels,
                max_pixels=self.max_pixels,
                max_frames=self.max_frames,
            ).items() if v is not None
        }
        if self.nframe > 0:
            process_video_kwargs['nframes'] = self.nframe
        if self.fps > 0:
            process_video_kwargs['fps'] = self.fps
        question = line['question']
        if self.custom_prompt is not None:
            question = self.custom_prompt + "\n" + question

        # if self.verbose:
        #     print(f"\n{'='*80}")
        #     print(f"Building prompt for video: {line['video']}")
        #     # print(f"Ground truth: {line['answer']}")
        #     # print(f"Duration: {line.get('duration', 'N/A')} seconds")
        #     print(f"Question: {question}")
        #     if process_video_kwargs:
        #         print(f"Video processing kwargs: {process_video_kwargs}")

        video_path = osp.join(self.data_root, line['video'] + '.mp4')

        # if self.verbose:
        #     print(f"Video path: {video_path}")
        #     print(f"Video exists: {osp.exists(video_path)}")
        #     print(f"video_llm parameter: {video_llm}")

        if video_llm and osp.exists(video_path):
            message = [
                dict(type='video', value=video_path, **process_video_kwargs),
                dict(type='text', value=question)
            ]

            if self.verbose:
                try:
                    video_size_mb = os.path.getsize(video_path) / (1024 * 1024)
                    print(f"Video file size: {video_size_mb:.2f} MB")
                except Exception as e:
                    print(f"Could not get video size: {e}")

                try:
                    import decord
                    vr = decord.VideoReader(video_path)
                    width = vr[0].shape[1]
                    height = vr[0].shape[0]
                    total_frames = len(vr)
                    video_fps = vr.get_avg_fps()
                    duration = total_frames / video_fps

                    print(f"Original video dimensions: {width}x{height}")
                    print(f"Total frames in video: {total_frames}")
                    print(f"Video FPS: {video_fps:.2f}")
                    print(f"Video duration: {duration:.2f} seconds")

                    if self.fps > 0:
                        sampled_frames = int(duration * self.fps)
                        print(f"Frames after fps={self.fps} sampling: ~{sampled_frames}")
                    elif self.nframe > 0:
                        print(f"Frames after nframe={self.nframe} sampling: {self.nframe}")
                    else:
                        print(f"No frame sampling applied (using all frames)")

                except Exception as e:
                    print(f"Could not read video properties with decord: {e}")
            return message
        msgs = []
        if osp.exists(video_path) and self.nframe > 0:
            frames = self.save_video_frames(line['video'])
            for frame in frames:
                msgs.append(dict(type='image', value=frame))
            msgs.append({'type': 'text', 'value': f"You are provided with {len(frames)} frames uniformly sampled from the video."})
        elif osp.exists(video_path):
            msgs.extend(self.read_video(video_path, question))
        msgs.append({'type': 'text', 'value': question})
        #print(f"VANTAGE-Bench Temporal Msgs: {msgs}")
        return msgs

    def read_video(self, video_path: str, query: str, local_file: bool = False) -> List[Dict]:
        if local_file:
            url = Path(video_path).resolve().as_uri()
        else:
            with open(video_path, 'rb') as f:
                video_bytes = f.read()
                video_base64 = base64.b64encode(video_bytes).decode('utf-8')
                url = f"data:video/mp4;base64,{video_base64}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": url}},
                    {"type": "text", "text": str(query)}
                ]
            }
        ]
        return messages

    @staticmethod
    def parse_timestamps_json(text: str, duration: float, strict: bool = False) -> List[float]:
        text_repaired = text.strip()
        if '"start":' in text_repaired and text_repaired.count('"start":') == 1:
            start_idx = text_repaired.find('"start":')
            start_quote_idx = text_repaired.find('"', start_idx + 8)
            if start_quote_idx != -1:
                comma_idx = text_repaired.find(',', start_quote_idx)
                if comma_idx != -1 and text_repaired[comma_idx - 1] != '"':
                    text_repaired = text_repaired[:comma_idx] + '"' + text_repaired[comma_idx:]
        if '"end":' in text_repaired and text_repaired.count('"end":') == 1:
            end_idx = text_repaired.find('"end":')
            end_quote_idx = text_repaired.find('"', end_idx + 6)
            if end_quote_idx != -1:
                next_quote_idx = text_repaired.find('"', end_quote_idx + 1)
                if next_quote_idx == -1 or text_repaired[next_quote_idx - 1] == '\n':
                    text_repaired = text_repaired.rstrip() + '"'
        text_repaired = text_repaired.rstrip()
        if not text_repaired.endswith('}'):
            text_repaired += '}'
        try:
            json_output = json.loads(text_repaired)
            if isinstance(json_output, list) and len(json_output) > 0:
                json_output = json_output[0]
            start = json_output["start"]
            end = json_output["end"]
        except json.JSONDecodeError:
            start_match = re.search(r'"start":\s*"([^"]+)"', text_repaired)
            end_match = re.search(r'"end":\s*"([^"]+)"', text_repaired)
            if start_match and end_match:
                start = start_match.group(1)
                end = end_match.group(1).rstrip('\n}')
            else:
                if strict:
                    raise ValueError(f"Failed to parse timestamps from: {text}")
                return [0, duration]
        start_parts = start.split(":")
        end_parts = end.split(":")
        start_seconds = float(start_parts[0]) * 60 + float(start_parts[1]) if len(start_parts) == 2 else float(start_parts[0])
        end_seconds = float(end_parts[0]) * 60 + float(end_parts[1]) if len(end_parts) == 2 else float(end_parts[0])
        return [start_seconds, end_seconds]

    @staticmethod
    def parse_timestamps(text: str, duration: float, strict: bool = False) -> Tuple[float, float]:
        matches = list(re.finditer(r"\<(?: (?: \d* \.? \d+ ) | (?: \d+ \.? ) )\>", text, re.VERBOSE))
        if strict:
            assert len(matches) >= 2, "Expected at least two timestamps in the text."
        elif len(matches) < 2:
            return [0, duration]
        timestamps = [min(max(float(m.group(0)[1:-1]), 0), duration) for m in matches[:2]]
        return [min(timestamps), max(timestamps)]

    @staticmethod
    def iou(s1: Tuple[float, float], s2: Tuple[float, float]) -> float:
        i = max(min(s1[1], s2[1]) - max(s1[0], s2[0]), 0)
        u = max(s1[1] - s1[0], 0) + max(s2[1] - s2[0], 0) - i
        return i / u if u > 0 else 0

    @staticmethod
    def precision(threshold: float):
        def precision_func(s1: Tuple[float, float], s2: Tuple[float, float]) -> float:
            return float(VANTAGE_Temporal.iou(s1, s2) >= threshold)
        return precision_func

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)

        from vlmeval.dataset.utils.vantagebench.emit import emit_submission
        _suffix = eval_file.split('.')[-1]
        submission_path = eval_file.replace(f'.{_suffix}', '_submission.jsonl')
        emit_submission(data, osp.splitext(osp.basename(eval_file))[0], submission_path, task='temporal')
        print(f"Submission written to: {submission_path}")

        if 'answer' not in self.data.columns:
            return {}

        verbose = judge_kwargs.get('verbose', False) or self.verbose

        if verbose:
            print(f"\n{'='*80}")
            print(f"Starting VANTAGE_Temporal Evaluation")
            print(f"Evaluating {len(data)} predictions from: {eval_file}")
            print(f"{'='*80}")

        outputs = []
        for idx, row in data.iterrows():
            matching = self.data[self.data['index'] == row['index']]
            if len(matching) == 0:
                if verbose:
                    print(f"Warning: index {row['index']} not found in dataset, skipping")
                continue
            gt_item = matching.iloc[0]
            duration = gt_item['duration'] if 'duration' in gt_item.index else 30.0
            try:
                pred_timestamps = self.parse_timestamps_json(row['prediction'], duration)
            except Exception:
                pred_timestamps = self.parse_timestamps(row['prediction'], duration)
            try:
                gt_timestamps = self.parse_timestamps_json(gt_item['answer'], duration, strict=True)
            except Exception:
                gt_timestamps = self.parse_timestamps(gt_item['answer'], duration, strict=True)
            sample_iou = self.iou(pred_timestamps, gt_timestamps)

            if verbose:
                print(f"\n--- Sample {idx + 1}/{len(data)} ---")
                print(f"Video: {gt_item['video']}")
                print(f"Question: {gt_item['question'][:200]}...")
                print(f"Model prediction: {row['prediction']}")
                print(f"Parsed prediction: start={pred_timestamps[0]:.2f}s, end={pred_timestamps[1]:.2f}s")
                print(f"Ground truth: {gt_item['answer']}")
                print(f"Parsed GT: start={gt_timestamps[0]:.2f}s, end={gt_timestamps[1]:.2f}s")
                print(f"Duration: {duration}s")
                print(f"IoU: {sample_iou:.4f}")
                print(f"Precision@0.5: {'pass' if sample_iou >= 0.5 else 'fail'}")

            outputs.append({
                'vid': gt_item['video'],
                'qid': gt_item.get('qid', gt_item['video']),
                'output': f"<{pred_timestamps[0]}> <{pred_timestamps[1]}>",
                'target': f"<{gt_timestamps[0]}> <{gt_timestamps[1]}>",
                'duration': duration,
                'category': self.get_category(gt_item['video']),
                'raw_prediction': row['prediction'],
                'iou': sample_iou
            })
        metric_funcs = {"iou": self.iou, "precision@0.5": self.precision(0.5)}
        results = self._compute_metrics(outputs, eval_file, metric_funcs, verbose=verbose)
        score_file = get_intermediate_file_path(eval_file, '_metrics', 'json')
        dump(results, score_file)
        overall = results.get('overall', {})
        return {
            'miou': overall.get('iou', 0.0),
            'precision_at_0_5': overall.get('precision@0.5', 0.0),
        }

    def _get_category(self, video_id: str) -> str:
        if not self.category_mapping:
            return "Other"
        for key in self.category_mapping:
            key_base = os.path.splitext(key)[0]
            if video_id.startswith(key_base):
                return self.category_mapping[key]
        return "Other"

    def _compute_metrics(self, outputs: List[Dict], eval_file: str, metric_funcs: Dict, verbose: bool = False) -> Dict:
        metrics = {name: defaultdict(list) for name in metric_funcs}
        category_metrics = defaultdict(lambda: defaultdict(list))
        for output in outputs:
            category = output.get("category", "Other")
            for name in metrics:
                try:
                    score = metric_funcs[name](
                        self.parse_timestamps(output["output"], output["duration"], strict=False),
                        self.parse_timestamps(output["target"], output["duration"], strict=True),
                    )
                    metrics[name][output["vid"]].append(score)
                    category_metrics[category][name].append(score)
                except Exception as e:
                    metrics[name][output["vid"]].append(0.0)
                    category_metrics[category][name].append(0.0)
        final_metrics = {}
        category_final_metrics = {}
        print("\nEvaluation Metrics:")
        print(f"{'Category':<30}{'IOU':<15}{'Precision@0.5':<15}{'Count':<10}")
        print("=" * 75)
        for category, metric_dict in sorted(category_metrics.items(), key=lambda x: -len(next(iter(x[1].values()), []))):
            category_iou = np.mean(metric_dict["iou"]) if metric_dict["iou"] else 0.0
            category_precision = np.mean(metric_dict["precision@0.5"]) if metric_dict["precision@0.5"] else 0.0
            count = len(metric_dict["iou"])
            category_final_metrics[category] = {"iou": category_iou, "precision@0.5": category_precision, "count": count}
            print(f"{category:<30}{category_iou:<15.4f}{category_precision:<15.4f}{count:<10}")
        overall_iou = np.mean([np.mean(metrics["iou"][vid]) for vid in metrics["iou"]]) if metrics["iou"] else 0.0
        overall_precision = np.mean([np.mean(metrics["precision@0.5"][vid]) for vid in metrics["precision@0.5"]]) if metrics["precision@0.5"] else 0.0
        total_items = sum(len(metrics["iou"][vid]) for vid in metrics["iou"])
        print(f"{'Overall':<30}{overall_iou:<15.4f}{overall_precision:<15.4f}{total_items:<10}")
        final_metrics["overall"] = {"iou": overall_iou, "precision@0.5": overall_precision, "count": total_items}
        final_metrics["category_metrics"] = category_final_metrics
        csv_path = get_intermediate_file_path(eval_file, '_acc', 'csv')
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Category", "IOU", "Precision@0.5", "Count"])
            for category, values in category_final_metrics.items():
                writer.writerow([category, f"{values['iou']:.4f}", f"{values['precision@0.5']:.4f}", values["count"]])
            writer.writerow(["Overall", f"{final_metrics['overall']['iou']:.4f}", f"{final_metrics['overall']['precision@0.5']:.4f}", final_metrics['overall']['count']])
        print(f"\nMetrics also saved to {csv_path}")

        if verbose:
            print(f"\n{'='*80}")
            print(f"DETAILED ANALYSIS")
            print(f"{'='*80}")

            if outputs and 'iou' in outputs[0]:
                ious = [item['iou'] for item in outputs]
                print(f"\nIoU Distribution:")
                print(f"  Min IoU: {min(ious):.4f}")
                print(f"  Max IoU: {max(ious):.4f}")
                print(f"  Mean IoU: {sum(ious)/len(ious):.4f}")
                print(f"  Std Dev: {np.std(ious):.4f}")
                print(f"  IoU >= 0.5: {sum(1 for iou in ious if iou >= 0.5)}/{len(ious)} ({100*sum(1 for iou in ious if iou >= 0.5)/len(ious):.1f}%)")

                unique_ious = set(round(iou, 4) for iou in ious)
                if len(unique_ious) == 1:
                    print(f"\n  WARNING: All IoU values are identical: {ious[0]:.4f}")
                    print(f"  This suggests consistent placeholder ground truth or systematic model behavior.")

                if abs(overall_iou - 0.3333) < 0.001:
                    print(f"\n  IoU = 1/3 Pattern Detected:")
                    print(f"  This typically occurs when GT=[0,10]s and prediction=[0,30]s -> IoU=10/30=0.333")
                elif abs(overall_iou - 1.0) < 0.001:
                    print(f"\n  Perfect IoU = 1.0! Model predictions perfectly match ground truth.")

            if outputs:
                print(f"\nSample Predictions (first 3):")
                for i, output in enumerate(outputs[:3]):
                    if 'raw_prediction' in output:
                        print(f"\nSample {i+1}:")
                        print(f"  Video: {output['vid']}")
                        print(f"  Raw: {str(output['raw_prediction'])[:200]}")
                        print(f"  Parsed output: {output['output']}")
                        print(f"  Ground truth: {output['target']}")
                        if 'iou' in output:
                            print(f"  IoU: {output['iou']:.4f}")

        return final_metrics


def load_vantage_results(results_dir):
    results = []
    seen_qids = set()
    if os.path.exists(results_dir):
        for root, _, files in os.walk(results_dir):
            for file in files:
                if "jsonl" in file:
                    with open(os.path.join(root, file)) as f:
                        for line in f:
                            result = json.loads(line)
                            qid = result.get("qid")
                            if qid is not None and qid not in seen_qids:
                                results.append(result)
                                seen_qids.add(qid)
                            elif qid is None and result.get("vid") not in seen_qids:
                                results.append(result)
                                seen_qids.add(result.get("vid"))
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate VANTAGE_Temporal dataset')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory containing result JSONL files')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save evaluation metrics')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.results_dir
    outputs = load_vantage_results(args.results_dir)
    if not outputs:
        print("No results found!")
        return
    dataset = VANTAGE_Temporal(test_mode=True, verbose=args.verbose)
    metric_funcs = {"iou": dataset.iou, "precision@0.5": dataset.precision(0.5)}
    results = dataset._compute_metrics(outputs, args.output_dir, metric_funcs, verbose=args.verbose)
    output_path = os.path.join(args.output_dir, "vantage_temporal_metrics.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Final metrics saved to {output_path}")


if __name__ == "__main__":
    main()
