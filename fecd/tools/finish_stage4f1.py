"""Wait for the four formal finals, then run frozen evaluation and GT-free audits."""
import os
from pathlib import Path
import subprocess
import sys
import time

from fecd.training.f1_storage import resolve_output_root, parse_output_root


def main(output_root=None):
    output = resolve_output_root(output_root)
    while not (output/'training_execution.json').exists():
        if list(output.glob('BLOCKED*.json')):
            raise RuntimeError('Training is BLOCKED; evaluation will not run')
        time.sleep(10)
    commands = [
        ('strict_evaluation', ['-u','-m','fecd.tools.evaluate_formal_students']),
        ('paired_statistics', ['-u','-m','fecd.tools.summarize_formal_students']),
        ('curves', ['-m','fecd.tools.plot_formal_curves']),
        ('full_tests', ['-m','unittest','discover','-s','fecd/tests']),
        ('validation', ['-u','-m','fecd.tools.check_stage4f1']),
    ]
    for name, args in commands:
        print(f'Starting {name}', flush=True)
        if name != 'full_tests':
            args = [*args, '--output-root', str(output)]
        with (output/'logs'/f'{name}.log').open('x') as stream:
            code = subprocess.call([sys.executable, *args], stdout=stream, stderr=subprocess.STDOUT,
                                    env=os.environ.copy(), cwd=Path(__file__).resolve().parents[2])
        if code:
            raise RuntimeError(f'{name} failed with exit {code}; see saved log')
        print(f'Completed {name}', flush=True)


if __name__ == '__main__':
    main(parse_output_root())
