"""Completion report based exclusively on already-saved evaluation/statistics."""
import json

from fecd.execution.storage import guard_mount
from fecd.training.f1_protocol import ROOT, NAMES


def write_report(directory, report, dry_run):
    guard_mount(directory)
    lines = ['# Stage 4F1 completion', '',
             'SIMULATION ONLY; no formal results.' if dry_run else
             'Engineering PASS. One training seed; new_dev development comparison, not confirmation.',
             '', '## Final Epoch20 metrics', '',
             '| Variant | F1 | IoU | Predicted area | GT area | Dataset-equal F1 |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    if not dry_run:
        for variant in NAMES:
            summary = json.loads((directory/'artifacts/evaluation'/variant/'summary.json').read_text())
            values = summary['domains']['ALL']
            lines.append(f'| {variant} | {values["image_macro_f1"]:.6f} | {values["image_macro_iou"]:.6f} | '
                         f'{values["mean_prediction_positive_fraction"]:.6f} | '
                         f'{values["mean_gt_positive_fraction"]:.6f} | {summary["dataset_equal"]["f1"]:.6f} |')
    lines.extend(['', '## Paired F1 contrasts', '',
                  '| Contrast | Image-weighted delta | 95% CI | Domain direction | Research signal |',
                  '| --- | ---: | --- | --- | --- |'])
    if not dry_run:
        for name, comparison in report['paired_statistics']['comparisons'].items():
            value = comparison['f1']['image_weighted']
            low, high = value['ci95']
            domains = [v['f1']['estimate'] for v in comparison['domains'].values()]
            mixed = min(domains) < 0 < max(domains)
            label = 'NO CLEAR GAIN' if low <= 0 <= high or mixed else (
                     'POSITIVE SIGNAL' if low > 0 else 'NEGATIVE SIGNAL')
            lines.append(f'| {name} | {value["estimate"]:.6f} | [{low:.6f}, {high:.6f}] | '
                         f'{"mixed" if mixed else "same direction"} | {label} |')
    lines.extend(['', 'See completion_report.json and artifacts/analysis/ for full source-group CI, domains, area and failure cases.',
                  'These CIs do not establish training-seed stability. CASIAv2 giant source component and reconstructed NIST16 remain risks.',
                  'Balanced BCE can encourage foreground overprediction; this run does not establish its causal contribution.',
                  'Pixel selection effects do not prove forensic evidence utility. No additional seeds or locked GT are authorized.',
                  'Shutdown is DISABLED; the server remains running for the user to stop manually.' if report.get('no_shutdown') else
                  'SUCCESS_SHUTDOWN_REQUESTED means a stop request; it never confirms platform billing has stopped.', ''])
    with (directory/'completion_report.md').open('x') as stream:
        stream.write('\n'.join(lines))
    with (directory/'stage4f1_report.md').open('x') as stream:
        stream.write('\n'.join(lines))
    if not dry_run:
        with (ROOT/'fecd/docs/stage4f1_report.md').open('x') as stream:
            stream.write('\n'.join(lines))
