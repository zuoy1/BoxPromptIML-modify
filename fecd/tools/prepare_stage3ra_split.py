"""Freeze route-A roles after image-only source identity and isolation checks."""
import argparse
import json
from pathlib import Path

from fecd.diagnostics.stage3ra_split import DEFAULT_AUDIT, DEFAULT_SPLIT, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_SPLIT)
    parser.add_argument('--audit-output', type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.audit_output), indent=2))


if __name__ == '__main__':
    main()
