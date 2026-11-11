"""仅执行授权5C1正式R0/R1；永不自动关机或启动额外seed。"""
import argparse
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')

from fecd.training.region_formal_protocol import OUTPUT, write
from fecd.training.region_formal_runner import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=('R0','R1'),required=True)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    try:
        run(args.variant,args.resume)
    except Exception as exc:
        write(OUTPUT/f'BLOCKED_{args.variant}_{__import__("time").time_ns()}.json',
              {'status':'BLOCKED','variant':args.variant,'reason':str(exc)})
        raise


if __name__ == '__main__':
    main()
