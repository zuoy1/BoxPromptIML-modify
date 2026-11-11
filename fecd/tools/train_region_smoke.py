"""阶段5C0独立入口：仅R0/R1探针或一个完整Epoch。"""
import argparse
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')

from fecd.training.region_runner import probe, run_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('R0', 'R1'), required=True)
    parser.add_argument('--mode', choices=('probe', 'epoch'), required=True)
    args = parser.parse_args()
    (probe if args.mode == 'probe' else run_epoch)(args.variant)


if __name__ == '__main__':
    main()
