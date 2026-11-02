"""Full new_dev development evaluator; new_confirm has no access path."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from fecd.evaluation.strict import evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    result=evaluate(args.checkpoint,args.output_dir)
    print(json.dumps({'result_hash':result['result_hash'],'metrics':result['domains']['ALL'],
                      'stop_formal_request':result['stop_formal_request']},indent=2))


if __name__=='__main__':
    main()
