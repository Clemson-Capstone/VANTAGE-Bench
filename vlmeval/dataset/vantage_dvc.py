"""
VANTAGE-Bench DVC (Dense Video Captioning): temporal event localization with captions.

Data: local under LMUDataRoot()/datasets/VANTAGE_DVC/ (VANTAGE_DVC.tsv + videos/).
"""
import argparse
import json
import os
import re
import csv
import numpy as np
from collections import defaultdict
from typing import Tuple, List, Dict
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from ..smp import *
from ..smp.file import get_intermediate_file_path
from .video_base import VideoBaseDataset
from .utils import build_judge, DEBUG_MESSAGE

FAIL_MSG = 'Failed to obtain answer via API.'


def _iou(pred_span: List[float], gt_span: List[float]) -> float:
    i = max(min(pred_span[1], gt_span[1]) - max(pred_span[0], gt_span[0]), 0)
    u = max(pred_span[1] - pred_span[0], 0) + max(gt_span[1] - gt_span[0], 0) - i
    return i / u if u > 0 else 0


def _chased_dp_assignment(scores: np.ndarray) -> Tuple[float, List[Tuple[int, int]]]:
    M, N = scores.shape
    dp = -np.ones((M, N))
    path = np.zeros((M, N), dtype=int)

    # Fill DP table iteratively
    for i in range(M):
        for j in range(N):
            if i == 0 and j == 0:
                state = [-1.0, -1.0, scores[i, j]]
            elif i == 0:
                state = [-1.0, dp[i, j - 1], scores[i, j]]
            elif j == 0:
                state = [dp[i - 1, j], -1.0, scores[i, j]]
            else:
                state = [dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1] + scores[i, j]]
            path[i, j] = int(np.argmax(state))
            dp[i, j] = state[path[i, j]]

    max_score = dp[M - 1, N - 1]

    # Backtrack to recover matched pairs iteratively
    pairs = []
    i, j = M - 1, N - 1
    while True:
        p = np.where(path[i, :j + 1] == 2)[0]
        if i != 0 and len(p) == 0:
            i -= 1
            continue
        if len(p) == 0:
            break
        col = p[-1]
        pairs.append((i, col))
        if i == 0 or col == 0:
            break
        j = col - 1
        i -= 1
    pairs.reverse()

    return max_score, pairs


class VANTAGE_DVC(VideoBaseDataset):
    MD5 = ''
    TYPE = 'Video-DVC'
    DENSE_CAPTION_QUERY = "Describe the notable events in the provided video. Provide the result in json format with 'mm:ss.ff' format for time depiction for each event. Use keywords 'start', 'end' and 'caption' in the json output."

    def __init__(self, dataset='VANTAGE_DVC', pack=False, nframe=0, fps=0, total_pixels=None, max_pixels=None, max_frames=None, test_mode=False, limit=None, random_state=None, include_categories=None, custom_prompt=None):
        self.test_mode = test_mode
        self.limit = limit
        self.random_state = random_state
        self.include_categories = set(include_categories) if include_categories else None
        if not test_mode:
            super().__init__(dataset=dataset, pack=pack, nframe=nframe, fps=fps, total_pixels=total_pixels, max_pixels=max_pixels, max_frames=max_frames, custom_prompt=custom_prompt)
            original_size = len(self.data) if hasattr(self, 'data') else 0
            if self.include_categories is not None and hasattr(self, 'data') and 'category' in self.data.columns:
                self.data = self.data[self.data['category'].isin(self.include_categories)]
            if self.limit is not None and self.limit > 0 and hasattr(self, 'data'):
                sample_num = max(1, int(self.limit * len(self.data))) if self.limit <= 1.0 else min(int(self.limit), len(self.data))
                self.data = self.data.sample(n=sample_num, random_state=self.random_state)
                print(f"Applied limit sampling: using {len(self.data)} out of {original_size} samples")
            if hasattr(self, 'data') and 'video' in self.data.columns:
                self.videos = sorted(set(self.data['video']))
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
        return ['VANTAGE_DVC']

    def prepare_dataset(self, dataset_name='VANTAGE_DVC'):
        def check_integrity(pth):
            return osp.exists(osp.join(pth, f'{dataset_name}.tsv')) and osp.exists(osp.join(pth, 'videos')) and os.listdir(osp.join(pth, 'videos'))
        local_dir = osp.join(LMUDataRoot(), 'datasets', 'VANTAGE_DVC')
        if check_integrity(local_dir):
            print(f"Using existing local dataset at: {local_dir}")
            dataset_path = local_dir
        else:
            raise FileNotFoundError(
                f"VANTAGE_DVC data not found under {local_dir}. "
                "Run: python scripts/run_lmudata.py --task dvc --lmu-root ~/LMUData"
            )
        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')
        if not osp.exists(data_file):
            raise FileNotFoundError(
                f"VANTAGE_DVC TSV not found: {data_file}. "
                "Run: python scripts/run_lmudata.py --task dvc --lmu-root ~/LMUData"
            )
        return dict(data_file=data_file, root=osp.join(dataset_path, 'videos'))

    def build_prompt(self, line, video_llm=True):
        if isinstance(line, int):
            line = self.data.iloc[line]
        process_video_kwargs = {k: v for k, v in dict(total_pixels=self.total_pixels, max_pixels=self.max_pixels, max_frames=self.max_frames).items() if v is not None}
        if self.nframe > 0:
            process_video_kwargs['nframes'] = self.nframe
        if self.fps > 0:
            process_video_kwargs['fps'] = self.fps
        if self.custom_prompt is not None:
            question = self.custom_prompt
        else:
            question = self.DENSE_CAPTION_QUERY
        video_path = osp.join(self.data_root, line['video'].removesuffix('.mp4') + '.mp4')
        if video_llm and osp.exists(video_path):
            # print(f"DEBUG: Message {question}")
            return [dict(type='video', value=video_path, **process_video_kwargs), dict(type='text', value=question)]
        msgs = []
        if osp.exists(video_path) and self.nframe > 0:
            for frame in self.save_video_frames(line['video']):
                msgs.append(dict(type='image', value=frame))
            n_frames = len(msgs)
            msgs.append({'type': 'text', 'value': f"You are provided with {n_frames} frames uniformly sampled from the video."})
        msgs.append({'type': 'text', 'value': question})
        # print(f"DEBUG: Msgs {msgs}")
        return msgs

    @staticmethod
    def parse_timestamp(ts_str) -> float:
        if ts_str is None:
            return 0.0
        if isinstance(ts_str, (int, float)):
            return float(ts_str)
        ts_str = str(ts_str).strip()
        if not ts_str:
            return 0.0
        if ':' in ts_str:
            parts = ts_str.split(':')
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            elif len(parts) == 3:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        return float(ts_str)

    @staticmethod
    def parse_events_from_json(text: str) -> List[Dict]:
        text = text.strip()
        m = re.search(r'\[[\s\S]*\]', text)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        m = re.search(r'\{[\s\S]*\}', text)
        if m:
            try:
                e = json.loads(m.group())
                return [e] if isinstance(e, dict) else e
            except json.JSONDecodeError:
                pass
        out = []
        for start, end, caption in re.findall(r'"start":\s*"([^"]+)".*?"end":\s*"([^"]+)".*?"caption":\s*"([^"]*)"', text, re.DOTALL):
            out.append({"start": start, "end": end, "caption": caption})
        if not out:
            for start, end, caption in re.findall(r'<(\d{2}:\d{2}:\d{2})><(\d{2}:\d{2}:\d{2})>\s*(.+)', text):
                out.append({"start": start, "end": end, "caption": caption.strip()})
        return out

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)

        from vlmeval.dataset.utils.vantagebench.emit import emit_submission
        _suffix = eval_file.split('.')[-1]
        submission_path = eval_file.replace(f'.{_suffix}', '_submission.jsonl')
        emit_submission(data, osp.splitext(osp.basename(eval_file))[0], submission_path, task='dvc')
        print(f"Submission written to: {submission_path}")

        if 'answer' not in self.data.columns:
            return {}

        preds = {}
        gts = {}
        categories = {}
        for idx, row in data.iterrows():
            matching = self.data[self.data['index'] == row['index']]
            if len(matching) == 0:
                continue
            gt_item = matching.iloc[0]
            vid = gt_item['video']
            categories[vid] = gt_item.get('category', 'Unknown')
            pred_events = self.parse_events_from_json(row.get('prediction', ''))
            pred_list = []
            for pe in pred_events:
                start = self.parse_timestamp(pe.get('start') or pe.get('start_time', '0'))
                end = self.parse_timestamp(pe.get('end') or pe.get('end_time', '0'))
                caption = pe.get('caption', '') or pe.get('description', '') or ''
                if caption:
                    pred_list.append({"sentence": caption, "timestamp": [start, end]})
            pred_list = sorted(pred_list, key=lambda x: x["timestamp"][0])
            if pred_list:
                preds[vid] = pred_list
            try:
                gt_events = json.loads(gt_item['answer'])
            except (json.JSONDecodeError, TypeError):
                gt_events = []
            gt_timestamps = []
            gt_sentences = []
            for ge in gt_events:
                start = self.parse_timestamp(ge.get('start', '0'))
                end = self.parse_timestamp(ge.get('end', '0'))
                caption = ge.get('caption', '') or ge.get('description', '') or ''
                if caption:
                    gt_timestamps.append([start, end])
                    gt_sentences.append(caption)
            if gt_timestamps:
                gts[vid] = {"timestamps": gt_timestamps, "sentences": gt_sentences}
        gt_vids = list(set(gts.keys()) & set(preds.keys()))
        if not gt_vids:
            print("Warning: No videos with both predictions and ground truth.")
            return {"overall": {"mIoU": 0.0, "IoU_F1": 0.0, "BertScore_F1": 0.0, "SODA_c": 0.0}}
        print(f"\nEvaluating {len(gt_vids)} videos with SODA-c...")
        bert_endpoint = judge_kwargs.get('bert_score_endpoint') or os.environ.get('BERT_SCORE_ENDPOINT')
        use_remote = bool(bert_endpoint)
        bert_scorer = None
        if use_remote:
            print(f"Using remote BERTScore: {bert_endpoint}")
        else:
            try:
                import torch
                from bert_score import BERTScorer
                # Prefer the job's GPU when one is allocated (a low-tier card is
                # ample for roberta-large); VANTAGE_BERT_DEVICE overrides, and a
                # GPU-less job falls back to CPU automatically.
                bert_device = os.environ.get("VANTAGE_BERT_DEVICE") or (
                    "cuda" if torch.cuda.is_available() else "cpu")
                print(f"BERTScore device: {bert_device}")
                bert_scorer = BERTScorer(model_type="roberta-large", device=bert_device)
            except ImportError:
                print("Warning: bert_score not available, using dummy (F1=0.5). pip install bert-score")
        def bert_remote(cands, refs):
            import requests
            r = requests.post(f"{bert_endpoint}/score", json={"candidates": cands, "references": refs}, timeout=300)
            r.raise_for_status()
            return r.json()["f1"]
        all_iou = []
        iou_fs, bert_fs, combined_fs = [], [], []
        category_results = defaultdict(lambda: {'iou_f': [], 'bert_f': [], 'combined_f': [], 'count': 0})
        for vid in tqdm(gt_vids, desc="SODA"):
            pred, gold = preds[vid], gts[vid]
            iou_mat = np.array([[_iou(p["timestamp"], gt) for p in pred] for gt in gold["timestamps"]])
            pred_s = [p["sentence"] for p in pred]
            gt_s = gold["sentences"]
            if use_remote:
                score_mat = np.zeros((len(gt_s), len(pred_s)))
                for gi, g in enumerate(gt_s):
                    score_mat[gi, :] = np.array(bert_remote(pred_s, [g] * len(pred_s)))
            elif bert_scorer:
                score_mat = np.zeros((len(gt_s), len(pred_s)))
                for gi, g in enumerate(gt_s):
                    _, _, F1 = bert_scorer.score(pred_s, [g] * len(pred_s))
                    score_mat[gi, :] = F1.cpu().numpy()
            else:
                score_mat = np.ones_like(iou_mat) * 0.5
            comb = iou_mat * score_mat
            n_gt, n_pred = iou_mat.shape
            if n_gt > 0 and n_pred > 0:
                max_sc, pairs = _chased_dp_assignment(comb)
                if pairs:
                    r, c = zip(*pairs)
                    iou_sum = np.sum(iou_mat[r, c])
                    bert_sum = np.sum(score_mat[r, c])
                    all_iou.extend(iou_mat[r, c].tolist())
                else:
                    iou_sum = bert_sum = 0.0
                iou_f = 2 * (iou_sum / n_pred) * (iou_sum / n_gt) / ((iou_sum / n_pred) + (iou_sum / n_gt)) if iou_sum > 0 else 0
                bert_f = 2 * (bert_sum / n_pred) * (bert_sum / n_gt) / ((bert_sum / n_pred) + (bert_sum / n_gt)) if bert_sum > 0 else 0
                combined_f = 2 * (max_sc / n_pred) * (max_sc / n_gt) / ((max_sc / n_pred) + (max_sc / n_gt)) if max_sc > 0 else 0
            else:
                iou_f = bert_f = combined_f = 0.0
            iou_fs.append(iou_f)
            bert_fs.append(bert_f)
            combined_fs.append(combined_f)
            cat = categories.get(vid, 'Unknown')
            category_results[cat]['iou_f'].append(iou_f)
            category_results[cat]['bert_f'].append(bert_f)
            category_results[cat]['combined_f'].append(combined_f)
            category_results[cat]['count'] += 1
        mean_iou = np.mean(all_iou) if all_iou else 0.0
        final = {'overall': {'mIoU': mean_iou, 'IoU_F1': np.mean(iou_fs), 'BertScore_F1': np.mean(bert_fs), 'SODA_c': np.mean(combined_fs), 'count': len(gt_vids)}, 'category_metrics': {}}
        for cat, m in sorted(category_results.items(), key=lambda x: -x[1]['count']):
            final['category_metrics'][cat] = {'IoU_F1': np.mean(m['iou_f']), 'BertScore_F1': np.mean(m['bert_f']), 'SODA_c': np.mean(m['combined_f']), 'count': m['count']}
        csv_path = get_intermediate_file_path(eval_file, '_acc', 'csv')
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Category", "mIoU", "IoU_F1", "BertScore_F1", "SODA_c", "Count"])
            for cat, v in final['category_metrics'].items():
                w.writerow([cat, f"{mean_iou:.4f}", f"{v['IoU_F1']:.4f}", f"{v['BertScore_F1']:.4f}", f"{v['SODA_c']:.4f}", v["count"]])
            w.writerow(["Overall", f"{final['overall']['mIoU']:.4f}", f"{final['overall']['IoU_F1']:.4f}", f"{final['overall']['BertScore_F1']:.4f}", f"{final['overall']['SODA_c']:.4f}", final['overall']['count']])
        dump(final, get_intermediate_file_path(eval_file, '_metrics', 'json'))
        overall = final.get('overall', {})
        return {
            'soda_c': overall.get('SODA_c', 0.0),
            'miou': overall.get('mIoU', 0.0),
            'iou_f1': overall.get('IoU_F1', 0.0),
            'bertscore_f1': overall.get('BertScore_F1', 0.0),
        }

if __name__ == "__main__":
    main()
