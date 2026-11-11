"""Reproduce a model's leaderboard scores from one config in configs/.

    python scripts/reproduce.py configs/qwen3_vl_8b.json
    python scripts/reproduce.py configs/qwen3_vl_8b.json --tasks VANTAGE_VQA_4fps VANTAGE_SOT
    python scripts/reproduce.py configs/qwen3_vl_8b.json --dry-run

A leaderboard run uses different model settings for some tasks (for example a larger output limit
for dense captioning). run.py applies one set of model settings to every dataset in a config, so
this script groups the tasks that share settings and runs one run.py call per group. Every group
writes to the same <work-dir>/<name>/ folder, so the results package as one submission.

Any argument not listed below (for example --lmudata-root /path) is passed to run.py.
"""
import argparse
import inspect
import json
import os
import os.path as osp
import subprocess
import sys

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))


def build_groups(cfg, tasks):
    """Return [(model_args, {dataset_key: dataset_args})], one entry per distinct model setting."""
    groups = {}
    for key in tasks:
        spec = cfg['tasks'][key]
        model_args = {**cfg['model'], **(spec.get('model') or {})}
        group_key = json.dumps(model_args, sort_keys=True)
        groups.setdefault(group_key, (model_args, {}))[1][key] = dict(spec['data'])
    return list(groups.values())


def check(groups):
    """Validate class and argument names without loading any weights."""
    import vlmeval.api
    import vlmeval.dataset
    import vlmeval.vlm
    problems = []
    for model_args, data in groups:
        args = dict(model_args)
        cls_name = args.pop('class')
        cls = getattr(vlmeval.api, cls_name, None) or getattr(vlmeval.vlm, cls_name, None)
        if cls is None:
            problems.append(f'model class {cls_name} not found')
        else:
            params = inspect.signature(cls.__init__).parameters
            if not any(p.kind == p.VAR_KEYWORD for p in params.values()):
                problems += [f'{cls_name}: unknown argument {k}' for k in args if k not in params]
        for name, d in data.items():
            d = dict(d)
            dcls = getattr(vlmeval.dataset, d.pop('class'), None)
            if dcls is None:
                problems.append(f'{name}: dataset class not found')
                continue
            params = inspect.signature(dcls.__init__).parameters
            if not any(p.kind == p.VAR_KEYWORD for p in params.values()):
                problems += [f'{name}: unknown argument {k}' for k in d if k not in params]
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config', help='a config from configs/')
    ap.add_argument('--tasks', nargs='+', help='dataset keys to run (default: all in the config)')
    ap.add_argument('--work-dir', default='./outputs', help='run.py output folder (default: ./outputs)')
    ap.add_argument('--dry-run', action='store_true', help='write and check the run.py configs, run nothing')
    args, extra = ap.parse_known_args()

    with open(args.config) as f:
        cfg = json.load(f)
    name = cfg['name']
    tasks = args.tasks or list(cfg['tasks'])
    unknown = [t for t in tasks if t not in cfg['tasks']]
    if unknown:
        sys.exit(f'not in {args.config}: {", ".join(unknown)}. Choose from: {", ".join(cfg["tasks"])}')

    groups = build_groups(cfg, tasks)
    sys.path.insert(0, ROOT)
    problems = check(groups)
    if problems:
        sys.exit('config problems:\n  ' + '\n  '.join(problems))

    work_dir = osp.abspath(args.work_dir)
    config_dir = osp.join(work_dir, name, 'reproduce_configs')
    os.makedirs(config_dir, exist_ok=True)
    extra = [x for x in extra if x != '--']
    for i, (model_args, data) in enumerate(groups):
        path = osp.join(config_dir, f'group{i}.json')
        with open(path, 'w') as f:
            json.dump({'model': {name: model_args}, 'data': data}, f, indent=2)
        print(f'[{name}] group {i + 1}/{len(groups)}: {", ".join(data)} -> {path}', flush=True)
        if args.dry_run:
            continue
        cmd = [sys.executable, osp.join(ROOT, 'run.py'), '--config', path, '--work-dir', work_dir] + extra
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            sys.exit(f'run.py failed for group {i + 1} ({", ".join(data)}), exit code {result.returncode}')

    if not args.dry_run:
        print(f'\nResults for {name} are under {osp.join(work_dir, name)}')


if __name__ == '__main__':
    main()
