"""Summarize instrumented step reports; online numbers are inline estimates."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(report, skip):
    rows = report['steps'][skip:]
    if not rows or any('benchmark' not in row for row in rows):
        raise ValueError('missing measured steps; run with --benchmark-step and reduce --skip if needed')
    return {stage: {metric: statistics.mean(row['benchmark'][stage][metric] for row in rows)
                    for metric in ('online_estimate_seconds', 'online_communication_bytes')}
            for stage in ('forward', 'backward', 'optimizer', 'total')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', action='append', required=True, metavar='NAME=PATH')
    parser.add_argument('--skip', type=int, default=1, help='initial measured steps to exclude')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.skip < 0:
        parser.error('--skip must be nonnegative')
    results = []
    for item in args.report:
        name, sep, path = item.partition('=')
        if not sep or not name or not path:
            parser.error('--report must be NAME=PATH')
        report = json.loads(Path(path).read_text())
        model = report.get('model', {})
        options = report.get('options', {})
        model_info = {key: model.get(key) for key in ('dim', 'heads', 'layers', 'patches', 'image_shape', 'fixture')}
        if model_info['layers'] is None and isinstance(model.get('blocks'), list):
            model_info['layers'] = len(model['blocks'])
            if model['blocks']:
                model_info['heads'] = len(model['blocks'][0]['paths'])
        results.append({'method': name, 'status': report['status'],
                        'measured_steps': len(report['steps']) - args.skip,
                        'source': path, 'model': model_info,
                        'model_source': report.get('source'),
                        'precision_bits': options.get('precision_bits', 17 if report.get('backend') == '3pc' else None),
                        'trainable_parameters': report.get('trainable_parameters', report.get('config', {}).get('rank')),
                        'stages': summarize(report, args.skip)})
    lines = ['# Per-step 3PC benchmark', '',
             'Inline online estimates; synchronization and unmeasured preprocessing remain. '
             'Communication is estimated payload, excluding direct control metadata and dealer IPC. '
             'FoT forward includes two queries, input sharing and loss-difference sharing; '
             'SFT/PT forward includes input sharing, one query and logit-gradient sharing. '
             'Evaluation is excluded. Different models, token shapes and precision affect results. '
             'GB = 10^9 bytes; each duration is the maximum across parties per step.', '',
             '| Method | Forward (s) | Forward (GB) | Backward (s) | Backward (GB) | '
             'Optimizer (s) | Optimizer (GB) | Total (s) | Total (GB) |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for result in results:
        vals = []
        for stage in ('forward', 'backward', 'optimizer', 'total'):
            row = result['stages'][stage]
            vals.extend((f'{row["online_estimate_seconds"]:.3f}',
                         f'{row["online_communication_bytes"] / 1e9:.6f}'))
        lines.append('| ' + result['method'] + ' | ' + ' | '.join(vals) + ' |')
    lines.extend(['', 'Model metadata and numeric status:'])
    for result in results:
        lines.append(f'- {result["method"]}: status={result["status"]}; '
                     f'Q{result["precision_bits"]}; model={result["model"]}; '
                     f'model source={result["model_source"]}; '
                     f'trainable parameters={result["trainable_parameters"]}; '
                     f'measured steps={result["measured_steps"]}')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'summary.md').write_text('\n'.join(lines) + '\n')
    (args.output / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
