"""Summarize development-only route-A diagnostics without opening labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fecd.diagnostics.stage3ra_aggregate import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-dir', type=Path, default=Path('fecd/assets'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = summarize(args.assets_dir, args.output)
    print(json.dumps({'decision': result['decision'], 'images': result['images'],
                      'plots': len(result['risk_curve_paths'])}, indent=2))


if __name__ == '__main__':
    main()
