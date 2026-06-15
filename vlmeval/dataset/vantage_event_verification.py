from ..smp import *
from .video_base import VideoBaseDataset
import pandas as pd
import numpy as np
import re


def flatten_dict(d, route=()):
    flat_dict = {}
    for key, value in d.items():
        if isinstance(value, dict):
            for k, v in flatten_dict(value, route + (key,)).items():
                new_key = "--".join(route + (k,))
                flat_dict[new_key] = v
        else:
            new_key = "--".join(route + (key,))
            flat_dict[new_key] = value
    return flat_dict

class VANTAGE_EventVerification(VideoBaseDataset):

    TYPE = 'VANTAGE-EventVerification'

    def __init__(
        self,
        dataset='VANTAGE_EventVerification',
        nframe=0,
        fps=-1,
        total_pixels=8192 * 32 * 32,
        max_pixels=None,
        max_frames=None,
        system_prompt_option='merged',
        nsamples=None,
        custom_prompt=None
    ):
        """
        VANTAGE_EventVerification dataset for event verification in videos.
        The task is to predict a physics correctness score (pc) for each video.
        """
        # Call parent init which will call prepare_dataset
                 # video related
        self.total_pixels = total_pixels
        self.max_pixels = max_pixels
        self.max_frames = max_frames
        self.system_prompt_option = system_prompt_option
        super().__init__(
            dataset=dataset,
            nframe=nframe,
            fps=fps,
            total_pixels=total_pixels,
            custom_prompt=custom_prompt
        )
        if nsamples is not None:
            self.data = self.data.iloc[:nsamples].reset_index(drop=True)

        print(f"Loaded dataset with {len(self.data)} samples.")
        print(f"Sample data:\n{self.data.head(1)}")

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
        return ['VANTAGE_EventVerification']


    @staticmethod
    def _load_items(annotation_json):
        """Normalize annotation JSONs to a consistent list of dicts."""
        with open(annotation_json) as f:
            raw = json.load(f)
        # Format A: {"bcq": [...]} with keys id, video, system_prompt, question, answer
        if isinstance(raw, dict) and 'bcq' in raw:
            return raw['bcq']
        # Format B: plain list with keys id, video_id, question, answer (no system_prompt)
        if isinstance(raw, list):
            normalized = []
            for item in raw:
                normalized.append({
                    'id': item['id'],
                    'video': item.get('video', item.get('video_id')),
                    'system_prompt': item.get('system_prompt', (
                        "You are a warehouse safety monitoring system analyzing surveillance video. "
                        "Determine if a near-miss incident has occurred between a person and a forklift. "
                        "A near-miss is defined as a situation where a person and an operating forklift "
                        "come into dangerously close proximity without a collision occurring — for example, "
                        "a person crossing the path of a moving forklift, a forklift passing close behind "
                        "or in front of a person, or a person narrowly avoiding being struck. "
                        "Answer \"Yes\" if a near-miss is clearly visible. Otherwise, answer \"No\"."
                    )),
                    'question': item['question'],
                    'answer': item['answer'],
                })
            return normalized
        raise ValueError(f"Unrecognized annotation format in {annotation_json}")

    def prepare_dataset(self, dataset_name='VANTAGE_EventVerification'):
        from pathlib import Path
        dataset_dir = Path(LMUDataRoot()) / 'datasets' / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        data_file = dataset_dir / f'{dataset_name}.tsv'
        if not data_file.exists():
            raise FileNotFoundError(
                f"VANTAGE_EventVerification TSV not found: {data_file}. "
                "Run: python scripts/run_lmudata.py --task event_verification --lmu-root ~/LMUData"
            )
        return dict(root=str(dataset_dir), data_file=str(data_file))

    def build_prompt(self, line, video_llm):
        """
        Build the prompt for a given line.
        
        For API models (video_llm), returns OpenAI-style chat format with video.
        For local models, extracts and returns video frames.
        
        Args:
            line: A row from the dataframe or an index
            video_llm: Whether using a video language model API (bool)
            
        Returns:
            list: For API models - OpenAI-style messages with role/content.
                  For local models - List of message dicts with type/value keys.
        """
        if isinstance(line, int):
            line = self.data.iloc[line]
        
        # Get video path — video column stores absolute paths in the combined TSV
        video = line['video']
        video_path = Path(video) if Path(video).is_absolute() else Path(self.data_root) / 'videos' / video
        question = line['question']
        system_prompt = line['system_prompt']
        if self.custom_prompt is not None:
            full_text_payload = f"{self.custom_prompt}\n\nQuestion: {question}"
        else:
            if not isinstance(system_prompt, str) or not system_prompt.strip():
                system_prompt = None
            full_text_payload = f"{system_prompt}\n\nQuestion: {question}" if system_prompt else question

        msgs = []
        if video_llm:
            process_video_kwargs = {
                k: v for k, v in dict(
                    fps=self.fps,
                    total_pixels=self.total_pixels,
                    max_pixels=self.max_pixels,
                    max_frames=self.max_frames,
                ).items() if v is not None and (k != 'fps' or v > 0)
            }
            if self.nframe > 0: 
                process_video_kwargs['nframes'] = self.nframe
            
            # Format strictly for the VLM Base preprocessor
            msgs.append(dict(type='video', value=video_path.as_posix(), **process_video_kwargs))
            msgs.append(dict(type='text', value=full_text_payload))
        else:
            if osp.exists(video_path) and (self.nframe > 0 or self.fps > 0):
                video_path = os.path.join(self.data_root, "videos", line["video"])
                frames = self.save_video_frames(video_path[:-4])
                for frame in frames:
                    msgs.append(dict(type='image', value=frame))
                msgs.append({'type': 'text', 'value': f"You are provided with {len(frames)} frames uniformly sampled from the video."})
            msgs.append({'type': 'text', 'value': question})
        #print(f"DEBUG: Msgs {msgs}")
        return msgs

    def _lookup_gt_row(self, row):
        if 'index' in row and not pd.isna(row['index']):
            matching = self.data[self.data['index'] == row['index']]
            if len(matching):
                return matching.iloc[0]

        if 'id' in row and 'id' in self.data.columns and not pd.isna(row['id']):
            matching = self.data[self.data['id'] == row['id']]
            if len(matching):
                return matching.iloc[0]

        if 'video' in row and 'video' in self.data.columns and not pd.isna(row['video']):
            matching = self.data[self.data['video'] == row['video']]
            if len(matching) == 1:
                return matching.iloc[0]

        return None
    
    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)

        from vlmeval.dataset.utils.vantagebench.emit import emit_submission
        import os.path as _osp
        _suffix = eval_file.split('.')[-1]
        submission_path = eval_file.replace(f'.{_suffix}', '_submission.jsonl')
        emit_submission(data, _osp.splitext(_osp.basename(eval_file))[0], submission_path, task='event_verification')
        print(f"Submission written to: {submission_path}")

        if 'answer' not in self.data.columns:
            return {}

        
        # Extract predictions and ground truth
        predictions = []
        ground_truths = []
        
        for _, row in data.iterrows():
            pred = row.get('prediction', '')
            gt_row = self._lookup_gt_row(row)
            if gt_row is None:
                continue
            gt = gt_row['answer']
            
            # Try to extract answer from prediction
            pred_answer = self._extract_answer(pred)

            if (
                (pred_answer is not None)
                and (pred_answer.strip().lower() in ['yes', 'no'])
            ):
                predictions.append(pred_answer.strip().lower())
                ground_truths.append(str(gt).strip().lower())
        
        if len(predictions) == 0:
            print("Warning: No valid predictions found!")
            return {'macro_f1': 0.0, 'accuracy': 0.0, 'balanced_accuracy': 0.0}

        from sklearn.metrics import classification_report, balanced_accuracy_score
        report = classification_report(ground_truths, predictions, output_dict=True)
        macro_f1 = report.get('macro avg', {}).get('f1-score', 0.0)
        accuracy = report.get('accuracy', 0.0)
        bal_acc = balanced_accuracy_score(ground_truths, predictions)

        print("\n" + "=" * 50)
        print("VANTAGE_EventVerification Evaluation Results")
        print("=" * 50)
        print(f"Macro F1:          {macro_f1:.4f}")
        print(f"Accuracy:          {accuracy:.4f}")
        print(f"Balanced Accuracy: {bal_acc:.4f}")
        print(f"Valid Predictions: {len(predictions)} / {len(data)}")
        print("=" * 50 + "\n")

        result = {
            'macro_f1': float(macro_f1),
            'accuracy': float(accuracy),
            'balanced_accuracy': float(bal_acc),
        }
        suffix = eval_file.split('.')[-1]
        score_file = eval_file.replace(f'.{suffix}', '_acc.json')
        dump(result, score_file)
        print(f"Saved evaluation metrics to {score_file}")
        return result
    
    def _extract_answer(self, text):
        if pd.isna(text):
            return None
        m = re.search(r'\b(yes|no)\b', str(text), re.IGNORECASE)
        return m.group(1).lower() if m else None


def test_vantage_event_verification_dataset():
    """
    Test function to verify that VANTAGE_EventVerification can be built and processed.
    """
    print("Testing VANTAGE_EventVerification dataset...")
    
    try:
        dataset = VANTAGE_EventVerification(dataset='VANTAGE_EventVerification')
        print(f"✓ Dataset loaded successfully!")
        print(f"  Number of samples: {len(dataset)}")
        print(f"  Dataset type: {dataset.TYPE}")
        print(f"  Dataset modality: {dataset.MODALITY}")
        
        # Display first few samples
        print("\nFirst sample:")
        print(dataset.data.head(1).T)
        
        # Test building a prompt
        if len(dataset) > 0:
            print("\nTesting prompt building for first sample...")
            try:
                prompt = dataset.build_prompt(0, video_llm=True)
                num_videos = len([msg for msg in prompt if msg['type'] == 'video'])
                print(f"✓ Built prompt with {num_videos} video(s)")
                print(f"  Text: {[msg['value'] for msg in prompt if msg['type'] == 'text']}")
                print(f"  Prompt: {prompt}")
            except Exception as e:
                print(f"✗ Error building prompt: {e}")
        
        print("\n✓ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    test_vantage_event_verification_dataset()
