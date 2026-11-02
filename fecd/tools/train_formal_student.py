"""Run only authorized stage4F1 single-seed formal students."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json

from fecd.training.f1_protocol import NAMES
from fecd.training.f1_runner import run
from fecd.training.f1_storage import resolve_output_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=NAMES, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output-root', help='Absolute output directory; default: stage4f1/storage.json')
    args = parser.parse_args()
    output = resolve_output_root(args.output_root)
    try:
        run(args.variant, args.resume, output_root=output)
    except Exception as error:
        if output.is_dir():
            with (output/f'BLOCKED_{args.variant}.json').open('x') as stream:
                json.dump({'status':'BLOCKED', 'error':repr(error), 'variant':args.variant}, stream)
        raise


if __name__ == '__main__':
    main()
