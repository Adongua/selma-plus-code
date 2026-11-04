"""Compare synchronized full-forward operator diagnostics (LABEL=report.json)."""
import argparse
import json
from pathlib import Path
import statistics

GROUPS = ('attention', 'norm', 'gelu', 'linear', 'conv', 'cat_cls')


def summarize(label, path):
    report = json.loads(Path(path).read_text())
    if report.get('preprocessing_mode') != 'inline_instrumented' or not report.get('operator_profiling'):
        raise ValueError(f'{label}: report requires --inline-preprocessing-timing --profile-operators')
    records = [r for r in report['records'] if not r['warmup']]
    if not records:
        raise ValueError(f'{label}: no measured repetitions')
    selected = [r['operator_profile']['per_rank'][r['operator_profile']['slowest_online_rank']] for r in records]
    signatures = [[(e['operator'], e['input_shapes']) for e in r['operator_profile']['calls_per_rank'][0]] for r in records]
    if any(s != signatures[0] for s in signatures[1:]):
        raise ValueError(f'{label}: operator sequences or shapes differ between repetitions')
    mean = statistics.mean
    groups = {}
    for name in GROUPS:
        groups[name] = {'calls': selected[0]['operators'].get(name, {}).get('calls', 0),
                        'online_seconds': mean(x['operators'].get(name, {}).get('online_estimate_seconds', 0) for x in selected)}
    return {'label': label, 'report': str(Path(path).resolve()), 'status': report['status'],
            'backend': report['backend'], 'world_size': report['world_size'], 'image_sha256': report['image_sha256'],
            'precision': report['precision'], 'truncation_method': report.get('truncation_method'),
            'repeats': len(records), 'groups': groups,
            'online_seconds': mean(x['online_estimate_seconds'] for x in selected),
            'unattributed_seconds': mean(x['unattributed_online_seconds'] for x in selected),
            'slowest_ranks': [r['operator_profile']['slowest_online_rank'] for r in records],
            'operator_input_shapes': [{'operator': name, 'input_shapes': shapes} for name, shapes in signatures[0]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', action='append', required=True, metavar='LABEL=PATH')
    parser.add_argument('--output', required=True, help='new output directory')
    args = parser.parse_args()
    results = []
    for spec in args.report:
        if '=' not in spec:
            parser.error('--report must be LABEL=PATH')
        label, path = spec.split('=', 1)
        if not label or not path:
            parser.error('label and path must be nonempty')
        results.append(summarize(label, path))
    if len({r['label'] for r in results}) != len(results):
        parser.error('report labels must be unique')
    for key in ('backend', 'world_size', 'image_sha256', 'operator_input_shapes'):
        if any(r[key] != results[0][key] for r in results[1:]):
            raise ValueError(f'reports differ in {key}')
    note = ('Synchronized operator diagnostics from a continuous forward; fresh provider and Ring3 preparation are subtracted per rank. '
            'Each repetition uses the rank with the maximum total online estimate; its operator groups and residual add to that total. '
            'Operator boundary synchronizations perturb timings; results are not uninstrumented latency. '
            'Residual includes additions, reshapes, synchronization and instrumentation. Local randomness outside measured preparation remains online. '
            'Precision and correctness gates can differ; failed correctness results are diagnostic only.')
    lines = ['# Full ViT operator timing', '', note, '',
             '| Method | Status | Precision | Repeats | Attention (s) | LayerNorm (s) | GeLU (s) | Linear (s) | Conv (s) | CLS (s) | Residual (s) | Online total (s) |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        values = ' | '.join(f"{r['groups'][g]['online_seconds']:.3f}" for g in GROUPS)
        lines.append(f"| {r['label']} | {r['status']} | {r['precision']} | {r['repeats']} | {values} | {r['unattributed_seconds']:.3f} | {r['online_seconds']:.3f} |")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'comparison.json').write_text(json.dumps({'note': note, 'results': results}, indent=2, allow_nan=False) + '\n')
    text = '\n'.join(lines) + '\n'
    (out / 'comparison.md').write_text(text)
    print(text, end='')
    print(f'Reports: {out}')


if __name__ == '__main__':
    main()
