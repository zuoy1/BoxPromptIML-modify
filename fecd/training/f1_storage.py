"""Output locations are operational metadata, independent of frozen training identity."""
import argparse
import json
from pathlib import Path


POLICY = Path(__file__).resolve().parents[1]/'configs/stage4f1/storage.json'


def resolve_output_root(output_root=None):
    if output_root is None:
        output_root = json.loads(POLICY.read_text())['output_root']
    requested = Path(output_root).expanduser()
    if not requested.is_absolute():
        raise ValueError('4F1 output_root must be an absolute path')
    target = requested.resolve()
    data = Path('/hy-tmp')
    if requested.is_relative_to(data):
        if not target.is_relative_to(data) or data.stat().st_dev == Path('/').stat().st_dev:
            raise ValueError('/hy-tmp is not an independent data filesystem; refusing output writes')
    return target


def parse_output_root():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', help='Absolute 4F1 output directory; default: storage.json')
    return parser.parse_args().output_root

