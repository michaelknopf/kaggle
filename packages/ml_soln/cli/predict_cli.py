import argparse

from ml_soln.common.predictors import PREDICTORS


def run(args):
    predictor = PREDICTORS[args.predictor]
    predictor.func(args.job_name)


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('predictor',
                        type=str,
                        choices=list(sorted(PREDICTORS.keys())),
                        help='Name of predictor to run')
    parser.add_argument('job_name',
                        type=str,
                        help='Name of training job whose model to use for prediction')
    parser.set_defaults(func=run)


def main():
    parser = argparse.ArgumentParser(prog='mlops_predict',
                                     description='Predict results for a dataset using a trained model')
    add_arguments(parser)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
