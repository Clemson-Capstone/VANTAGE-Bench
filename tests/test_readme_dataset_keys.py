"""Every dataset key the README tells a submitter to pass to --data must resolve.

VANTAGE_2DPointing overrides load_data and has no DATASET_URL entry, so
its name does not come from ImageBaseDataset.supported_datasets(); the
class declares supported_datasets() itself. The tests check that it is
registered and that `python run.py --data VANTAGE_2DPointing --model <M>`
constructs the class.

The README is parsed here rather than hard-coding the keys, so a key added
to the docs without a registration (or the other way round) fails the test.
"""
import os
import os.path as osp
import re

import pandas as pd
import pytest

from vlmeval.dataset import SUPPORTED_DATASETS, build_dataset, supported_video_datasets
from vlmeval.dataset.image_mcq import VANTAGE_2DPointing
from vlmeval.smp import dump

README = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'README.md')
KEY_RE = re.compile(r'^(VANTAGE_[A-Za-z0-9_.]+|Astro2D)$')


def readme_data_keys():
    """Keys named after --data in run.py examples and in the first cell of dataset tables."""
    text = open(README, encoding='utf-8').read()
    keys = set()
    # `python run.py ... --data A B \` with line continuations.
    for block in re.findall(r'python run\.py(?:[^\n]*\\\n)*[^\n]*', text):
        flat = block.replace('\\\n', ' ')
        m = re.search(r'--data\s+((?:[^\s-][^\s]*\s*)+)', flat)
        if m:
            keys.update(t for t in m.group(1).split() if KEY_RE.match(t))
    # Table rows whose first cell is a backticked dataset key.
    for m in re.finditer(r'^\|\s*`([^`]+)`\s*\|', text, flags=re.M):
        if KEY_RE.match(m.group(1)):
            keys.add(m.group(1))
    return sorted(keys)


KEYS = readme_data_keys()


def test_readme_lists_the_eight_task_keys():
    # The parser must find at least one key for the check to be meaningful.
    for k in ['VANTAGE_VQA_8frame', 'VANTAGE_EventVerification_8frame', 'VANTAGE_Temporal_8frame',
              'VANTAGE_DVC_8frame', 'VANTAGE_SOT', 'VANTAGE_2DGrounding', 'VANTAGE_2DPointing', 'Astro2D']:
        assert k in KEYS


@pytest.mark.parametrize('key', KEYS)
def test_readme_data_key_is_registered(key):
    assert key in SUPPORTED_DATASETS or key in supported_video_datasets, (
        f'{key} is documented as a --data key but build_dataset() cannot resolve it')


def test_vantage_2dpointing_is_registered():
    assert 'VANTAGE_2DPointing' in SUPPORTED_DATASETS
    assert VANTAGE_2DPointing.supported_datasets() == ['VANTAGE_2DPointing']


def test_build_dataset_constructs_vantage_2dpointing(tmp_path):
    root = tmp_path / 'VANTAGE_2DPointing'
    root.mkdir()
    df = pd.DataFrame({
        'index': [0, 1],
        'question': ['Point to the car.', 'Point to the tree.'],
        'A': ['[10, 10]', '[20, 20]'],
        'B': ['[30, 30]', '[40, 40]'],
        'C': ['[50, 50]', '[60, 60]'],
        'D': ['[70, 70]', '[80, 80]'],
        'answer': ['A', 'B'],
        'image_path': ['img0.jpg', 'img1.jpg'],
    })
    dump(df, str(root / 'VANTAGE_2DPointing.tsv'))
    ds = build_dataset('VANTAGE_2DPointing', data_root=str(root))
    assert isinstance(ds, VANTAGE_2DPointing)
    assert len(ds) == 2
    assert ds.img_root == str(root)


def test_build_dataset_reports_missing_pointing_tsv(tmp_path):
    # With the key registered the class is constructed and names the missing
    # file.
    with pytest.raises(FileNotFoundError, match='VANTAGE_2DPointing TSV not found'):
        build_dataset('VANTAGE_2DPointing', data_root=str(tmp_path))
