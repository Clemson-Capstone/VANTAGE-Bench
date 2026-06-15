"""
VANTAGE-Bench VQA: multiple choice video question answering.

Data: Place under LMUDataRoot()/datasets/VANTAGE_VQA/
  - VANTAGE_VQA.tsv (required)
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
import pandas as pd

from ..smp import *
from ..smp.file import get_intermediate_file_path, get_file_extension
from .video_base import VideoBaseDataset
from .utils import build_judge, DEBUG_MESSAGE


FAIL_MSG = 'Failed to obtain answer via API.'


def extract_answer(text):
    """Extract multiple choice answer (A, B, C, D) from model response."""
    if pd.isna(text):
        return None
    text = str(text).strip()

    # 1. Explicit "Answer: X" marker (CoT format)
    explicit = re.search(r'Answer:\s*([A-D])\b', text, re.IGNORECASE)
    if explicit:
        return explicit.group(1).upper()

    # 2. Response starts with "X." or "X:" or "(X)" — model gave direct answer
    leading = re.match(r'^\(?([A-D])\)?[.:\s]', text, re.IGNORECASE)
    if leading:
        return leading.group(1).upper()

    # 3. "the answer is X" / "option X" / "choose X"
    sentence = re.search(r'(?:answer is|option|choose|select|pick)\s*:?\s*\(?([A-D])\)?', text, re.IGNORECASE)
    if sentence:
        return sentence.group(1).upper()

    # 4. Last standalone letter as final fallback
    matches = re.findall(r'\b([A-D])\b', text, re.IGNORECASE)
    if matches:
        return matches[-1].upper()

    return None


class VANTAGE_VQA(VideoBaseDataset):
    """
    VANTAGE-Bench VQA dataset for multiple choice video question answering.
    Data loaded from LMUDataRoot()/datasets/VANTAGE_VQA.
    """

    MD5 = ''
    TYPE = 'VANTAGE-VQA'

    QUESTION_PREFIX = (
        "You are provided with a sequence of video frames depicting a scene\n"
        "Begin with a concise overview of what's happening; keep items conceptual, not implementation-level\n"
        "Answer the question based only on the visual content of the image."
    )

    def __init__(self, dataset='VANTAGE_VQA', pack=False, nframe=0, fps=-1, total_pixels=None,
                 max_pixels=None, max_frames=None, test_mode=False, limit=None, verbose=False,
                 random_state=None, include_categories=None, include_task_types=None, custom_prompt=None):
        self.test_mode = test_mode
        self.category_mapping = {}
        self.limit = limit
        self.verbose = verbose
        self.random_state = random_state
        self.include_categories = set(include_categories) if include_categories else None
        self.include_task_types = set(include_task_types) if include_task_types else None

        if not test_mode:
            super().__init__(dataset=dataset, pack=pack, nframe=nframe, fps=fps, total_pixels=total_pixels, max_pixels=max_pixels, max_frames=max_frames, custom_prompt=custom_prompt)
            # --- ADD THIS FILTER ---
            if hasattr(self, 'data') and len(self.data) > 0:
                # Build absolute path for each video in the TSV
                # Note: prepare_dataset sets self.data_root to 'LMUDataRoot/datasets/VANTAGE_VQA/videos'
                video_exists = self.data['video'].apply(
                    lambda x: osp.exists(osp.join(self.data_root, str(x).removesuffix('.mp4') + '.mp4'))
                )
                before_count = len(self.data)
                self.data = self.data[video_exists].reset_index(drop=True)
                print(f"Video existence check: Kept {len(self.data)}/{before_count} samples.")
            # ------------------------
            original_size = len(self.data) if hasattr(self, 'data') else 0

            if self.include_categories is not None and hasattr(self, 'data'):
                before_rows = len(self.data)
                if 'category' in self.data.columns:
                    derived_cats = self.data['category']
                elif 'video' in self.data.columns:
                    derived_cats = self.data['video'].apply(self.get_category)
                else:
                    derived_cats = None

                if derived_cats is not None:
                    if self.verbose:
                        try:
                            dist_before = derived_cats.value_counts().to_dict()
                            print(f"Category distribution before filter: {dist_before}")
                        except Exception:
                            pass
                    self.data = self.data[derived_cats.isin(self.include_categories)]
                    if self.verbose:
                        print(f"Filtered by categories {sorted(self.include_categories)}: {len(self.data)}/{before_rows} rows kept")

            if self.include_task_types is not None and hasattr(self, 'data') and 'task_type' in self.data.columns:
                before_rows = len(self.data)
                self.data = self.data[self.data['task_type'].isin(self.include_task_types)]
                if self.verbose:
                    print(f"Filtered by task types {sorted(self.include_task_types)}: {len(self.data)}/{before_rows} rows kept")

            if self.limit is not None and self.limit > 0 and hasattr(self, 'data'):
                if self.limit <= 1.0:
                    sample_num = max(1, int(self.limit * len(self.data)))
                    self.data = self.data.sample(n=sample_num, random_state=self.random_state)
                else:
                    sample_num = min(int(self.limit), len(self.data))
                    self.data = self.data.sample(n=sample_num, random_state=self.random_state)
                if self.verbose:
                    print(f"Applied limit sampling: using {len(self.data)} out of {original_size} samples")

            if hasattr(self, 'data') and 'video' in self.data.columns:
                videos = list(set(self.data['video']))
                videos.sort()
                self.videos = videos
        else:
            self.dataset_name = dataset
            self.nframe = nframe
            self.fps = fps
            self.TYPE = self.TYPE

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
        return ['VANTAGE_VQA']

    def prepare_dataset(self, dataset_name='VANTAGE_VQA'):
        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')
            if not osp.exists(data_file):
                return False
            video_dir = osp.join(pth, 'videos')
            if not osp.exists(video_dir) or not os.listdir(video_dir):
                return False
            return True

        local_dir = osp.join(LMUDataRoot(), 'datasets', 'VANTAGE_VQA')
        if check_integrity(local_dir):
            print(f"Using existing local dataset at: {local_dir}")
            dataset_path = local_dir
        else:
            annotations_dir = osp.join(local_dir, 'annotations')
            video_dir = osp.join(local_dir, 'videos')
            if osp.exists(annotations_dir) and os.listdir(annotations_dir) and osp.exists(video_dir):
                print(f"Found local annotations, generating TSV...")
                self._generate_tsv_from_annotations(local_dir, dataset_name)
                dataset_path = local_dir
            else:
                raise FileNotFoundError(
                    f"VANTAGE_VQA data not found under {local_dir}. "
                    "Run: python scripts/run_lmudata.py --task vqa --lmu-root ~/LMUData"
                )

        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')
        if not osp.exists(data_file):
            raise FileNotFoundError(
                f"VANTAGE_VQA TSV not found: {data_file}. "
                "Run: python scripts/run_lmudata.py --task vqa --lmu-root ~/LMUData"
            )
        mapping_dir = osp.join(dataset_path, 'mappings')
        if osp.exists(mapping_dir):
            self.category_mapping = self._load_category_mapping(mapping_dir)

        return dict(data_file=data_file, root=osp.join(dataset_path, 'videos'))

    def _process_annotation_item(self, item):
        try:
            video_name = item.get('q_uid', item.get('vid', ''))
            if not video_name:
                return None
            for ext in ('.json', '.mp4'):
                if video_name.endswith(ext):
                    video_name = video_name[:-len(ext)]
                    break
            question = item.get('question', '')
            if not question:
                return None
            options_raw = item.get('options', [])
            if not options_raw:
                return None
            options = []
            for opt in options_raw:
                if isinstance(opt, str):
                    parts = opt.split(': ', 1)
                    options.append(parts[1] if len(parts) == 2 else opt)
                else:
                    options.append(str(opt))
            task_type = item.get('task_type', '') or item.get('dimension', '')
            formatted_question = self.generate_question(question, options, task_type=task_type)
            gt_option = item.get('gt_option', 'A')
            if gt_option not in ['A', 'B', 'C', 'D']:
                gt_option = 'A'
            answer_idx = ord(gt_option) - ord('A')
            category = item.get('industry', '')
            if not category:
                category = self.get_category(video_name)
            else:
                category = self._normalize_category(category)
            return {
                'index': 0,
                'video': video_name,
                'question': formatted_question,
                'answer': gt_option,
                'answer_idx': answer_idx,
                'options': json.dumps(options),
                'category': category,
                'qid': item.get('question_id', f"{video_name}_0"),
                'task_type': item.get('task_type', '') or item.get('dimension', ''),
                'difficulty': item.get('difficulty', '')
            }
        except Exception as e:
            if self.verbose:
                print(f"Error processing annotation item: {e}")
            return None

    def _load_category_mapping(self, directory):
        merged_mapping = {}
        for file in glob.glob(os.path.join(directory, "*.json")):
            try:
                with open(file) as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        merged_mapping.update(data)
            except (json.JSONDecodeError, OSError) as e:
                print(f"Error loading JSON from {file}: {e}")
        return merged_mapping
    
    def _generate_tsv_from_annotations(self, local_dir, dataset_name):
        data_list = []
        annotations_dir = osp.join(local_dir, 'data_jsons/annotations')
        vqa_files = [
            'metrics_spatial_wo_ss.json',
            'VANTAGE_VQA_Verification_Final_ITS_Data.json',
            'metrics_spatial_ss.json',
            'metrics_temporal_filtered_ss.json',
            'metrics_temporal_wo_ss.json',
        ]
        for vqa_file in vqa_files:
            print(f"Looking for annotation file: {vqa_file} in {annotations_dir}...")
            file_path = osp.join(annotations_dir, vqa_file)
            if osp.exists(file_path):
                with open(file_path, 'r') as f:
                    ann_data = json.load(f)
                    if isinstance(ann_data, list):
                        for item in ann_data:
                            processed = self._process_annotation_item(item)
                            if processed and (self.include_categories is None or processed.get('category') in self.include_categories):
                                data_list.append(processed)
        if not data_list:
            annotation_files = glob.glob(osp.join(annotations_dir, '*.json'))
            for ann_file in annotation_files:
                try:
                    with open(ann_file, 'r') as f:
                        ann_data = json.load(f)
                        if isinstance(ann_data, list):
                            for item in ann_data:
                                processed = self._process_annotation_item(item)
                                if processed and (self.include_categories is None or processed.get('category') in self.include_categories):
                                    data_list.append(processed)
                except json.JSONDecodeError:
                    continue
        if not data_list:
            print("Warning: No valid annotations found")
            return
        data_list.sort(key=lambda x: (x['video'], x.get('qid', '')))
        for idx, item in enumerate(data_list):
            item['index'] = idx
        df = pd.DataFrame(data_list)
        df.to_csv(osp.join(local_dir, f'{dataset_name}.tsv'), sep='\t', index=False)
        print(f"Generated TSV with {len(data_list)} entries ({len(df['video'].unique())} unique videos)")


    def generate_question(self, base_question: str, options: list, task_type: str = '') -> str:
        option_labels = ["A", "B", "C", "D"]
        prefix = self.QUESTION_PREFIX + "\n"
        prefix += "Question: " + base_question + "\n"
        prefix += "Select your answer from the choices below:\n"
        for i, c in enumerate(option_labels[:len(options)]):
            prefix += c + ". " + str(options[i]) + "\n"
        prefix += "Respond with ONLY the letter corresponding to your answer (A, B, C, or D). Do not provide any explanation or other text.\n"
        return prefix

    def _normalize_category(self, cat: str) -> str:
        if not cat:
            return 'Other'
        cat = str(cat).strip()
        if cat == 'Smart Spaces':
            return 'Smart_Spaces'
        return cat

    def _get_category_for_video(self, vid):
        return 'Other'

    def get_category(self, vid: str) -> str:
        mapped = self._get_category(vid)
        if mapped and mapped != 'Other':
            return self._normalize_category(mapped)
        return self._normalize_category(self._get_category_for_video(vid))

    def _get_category(self, video_id: str) -> str:
        if not self.category_mapping:
            return "Other"
        for key in self.category_mapping:
            key_base = os.path.splitext(key)[0]
            if video_id.startswith(key_base):
                return self.category_mapping[key]
        return "Other"

    def _build_vqa_prompt(self, line):
        """Reconstruct the VQA prompt at runtime from TSV parts.

        The stored TSV question may contain an old preamble and/or double-spaced
        options. This method extracts base_question and options from the TSV fields
        and rebuilds the exact intended format, then falls back to the raw
        line['question'] string if reconstruction is not possible.

        Extraction priority for options:
          1. JSON-encoded line['options'] (current TSV schema)
          2. Labeled lines (A. / B. / ...) parsed from line['question'] text
          3. Comma-separated line['options'] string (legacy TSV schema)
        """
        option_labels = ['A', 'B', 'C', 'D']
        raw_question = str(line['question'])

        # --- Extract base_question via regex (handles preambles before "Question:") ---
        base_question = None
        q_match = re.search(r'Question:\s+(.+)', raw_question)
        if q_match:
            base_question = q_match.group(1).strip()

        # --- 1. Try JSON options column ---
        options = []
        try:
            raw_opts = line['options']
            if raw_opts is not None and str(raw_opts) not in ('nan', 'None', ''):
                parsed = json.loads(str(raw_opts))
                if isinstance(parsed, list) and parsed:
                    options = parsed
        except Exception:
            pass

        # --- 2. Parse labeled lines from question text (A. / B. / ...) ---
        if not options:
            labeled = re.findall(r'^[A-D]\.\s+(.+)$', raw_question, re.MULTILINE)
            if labeled:
                options = [t.strip() for t in labeled]

        # --- 3. Comma-separated options column (legacy TSV schema) ---
        if not options:
            try:
                raw_opts = str(line.get('options', ''))
                if raw_opts not in ('nan', 'None', ''):
                    # Split on ', ' but only accept if we get 2-4 parts
                    parts = [p.strip() for p in raw_opts.split(', ') if p.strip()]
                    if 2 <= len(parts) <= 4:
                        options = parts
            except Exception:
                pass

        # --- Fallback: return raw TSV question unchanged ---
        if not options or base_question is None:
            return raw_question, base_question or raw_question, [], raw_question

        prompt = self.QUESTION_PREFIX + '\n'
        prompt += 'Question: ' + base_question + '\n'
        prompt += 'Select your answer from the choices below:\n'
        for i, c in enumerate(option_labels[:len(options)]):
            prompt += c + '. ' + str(options[i]) + '\n'
        prompt += (
            'Respond with ONLY the letter corresponding to your answer '
            '(A, B, C, or D). Do not provide any explanation or other text.\n'
        )
        return prompt, base_question, options, raw_question

    def build_prompt(self, line, video_llm=True):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        process_video_kwargs = {
            k: v for k, v in dict(
                total_pixels=getattr(self, 'total_pixels', None),
                max_pixels=getattr(self, 'max_pixels', None),
                max_frames=getattr(self, 'max_frames', None),
            ).items() if v is not None
        }
        if self.nframe > 0:
            process_video_kwargs['nframes'] = self.nframe
        if self.fps > 0:
            process_video_kwargs['fps'] = self.fps

        question, _base_q, _opts, _raw_q = self._build_vqa_prompt(line)

        video_name = str(line['video'])
        if not video_name.endswith('.mp4'):
            video_name += '.mp4'
        video_path = osp.join(self.data_root, video_name)

        if video_llm and osp.exists(video_path):
            return [
                dict(type='video', value=video_path, **process_video_kwargs),
                dict(type='text', value=question),
            ]
        else:
            msgs = []
            if osp.exists(video_path) and (self.nframe > 0 or self.fps > 0):
                frames = self.save_video_frames(line['video'])
                for frame in frames:
                    msgs.append(dict(type='image', value=frame))
                msgs.append({
                    'type': 'text',
                    'value': f"You are provided with {len(frames)} frames uniformly sampled from the video.",
                })
            msgs.append({'type': 'text', 'value': question})
            return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], \
            'data file should be an supported format (xlsx/json/tsv) file'

        data = load(eval_file)

        from vlmeval.dataset.utils.vantagebench.emit import emit_submission
        _suffix = eval_file.split('.')[-1]
        submission_path = eval_file.replace(f'.{_suffix}', '_submission.jsonl')
        emit_submission(data, osp.splitext(osp.basename(eval_file))[0], submission_path, task='vqa')
        print(f"Submission written to: {submission_path}")

        if 'answer' not in self.data.columns:
            return {}

        verbose = judge_kwargs.get('verbose', False) or self.verbose
        results = {}
        category_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        task_stats = defaultdict(lambda: {"correct": 0, "total": 0})

        for idx, row in data.iterrows():
            matching = self.data[self.data['index'] == row['index']]
            if len(matching) == 0:
                if verbose:
                    print(f"Warning: index {row['index']} not found in dataset, skipping")
                continue
            gt_item = matching.iloc[0]
            gt_answer = gt_item['answer']
            pred_answer = extract_answer(row['prediction'])
            # Use stored category from TSV, fall back to derived
            category = gt_item.get('category', '') or self.get_category(gt_item['video'])
            if not category or category == 'nan':
                category = self.get_category(gt_item['video'])
            task_type = gt_item.get('task_type', '') or 'Unknown'
            if not task_type or str(task_type) == 'nan':
                task_type = 'Unknown'
            correct = pred_answer == gt_answer
            category_stats[category]['total'] += 1
            task_stats[task_type]['total'] += 1
            if correct:
                category_stats[category]['correct'] += 1
                task_stats[task_type]['correct'] += 1

        def _print_breakdown(title, stats):
            print(f"\n{title}")
            print(f"{'Name':<30}{'Accuracy':<12}{'Correct':<10}{'Total':<10}")
            print("=" * 62)
            overall_c, overall_t = 0, 0
            rows = {}
            for name in sorted(stats.keys()):
                s = stats[name]
                acc = s['correct'] / s['total'] if s['total'] > 0 else 0.0
                rows[name] = {'acc': acc, 'correct': s['correct'], 'total': s['total']}
                overall_c += s['correct']
                overall_t += s['total']
                print(f"{name:<30}{acc:<12.4f}{s['correct']:<10}{s['total']:<10}")
            overall_acc = overall_c / overall_t if overall_t > 0 else 0.0
            rows['Overall'] = {'acc': overall_acc, 'correct': overall_c, 'total': overall_t}
            print(f"{'Overall':<30}{overall_acc:<12.4f}{overall_c:<10}{overall_t:<10}")
            return rows

        results = _print_breakdown("Results by Category", category_stats)
        task_results = _print_breakdown("Results by Task Type", task_stats)
        overall_acc = results['Overall']['acc']
        overall_correct = results['Overall']['correct']
        overall_total = results['Overall']['total']

        results_file = get_intermediate_file_path(eval_file, '_results', 'csv')
        pd.DataFrame(results).to_csv(results_file, index=True)
        return {'accuracy': overall_acc}


def main():
    parser = argparse.ArgumentParser(description='Evaluate VANTAGE_VQA dataset')
    parser.add_argument('--eval_file', type=str, required=True, help='Path to evaluation file (TSV/XLSX/JSON with predictions)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    dataset = VANTAGE_VQA(verbose=args.verbose)
    results = dataset.evaluate(args.eval_file, verbose=args.verbose)
    print("\nFinal Results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
