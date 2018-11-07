import argparse
import traceback
from lib.util import timeit
from lib.automl import AutoML

from lib.utils import *


@timeit
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['classification', 'regression'])
    parser.add_argument('--model-dir')
    parser.add_argument('--train-csv')
    parser.add_argument('--test-csv')
    parser.add_argument('--prediction-csv')
    args = parser.parse_args()

    automl = AutoML(args.model_dir)

    if args.train_csv is not None:
        automl.train(args.train_csv, args.mode)
        automl.save()
    elif args.test_csv is not None:
        automl.load()
        automl.predict(args.test_csv, args.prediction_csv)
    else:
        exit(1)

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--mode', choices=['classification', 'regression'])
#     parser.add_argument('--model-dir')
#     parser.add_argument('--train-csv')
#     parser.add_argument('--test-csv')
#     parser.add_argument('--prediction-csv')
#     parser.add_argument('--test-target-csv')
#
#     args = parser.parse_args()
#     if args.model_dir is None:
#         tests = {
#             1: 'regression',
#             2: 'regression',
#             3: 'regression',
#             4: 'classification',
#             5: 'classification',
#             6: 'classification',
#             7: 'classification',
#             8: 'classification',
#         }
#
#         for i in tests.keys():
#
#             folder = r'..\check_' + str(i) + '_' + tests[i][0] + '\\'
#             argv = [
#                 '--train-csv', folder + 'train.csv',
#                 '--test-csv', folder + 'test.csv',
#                 '--prediction-csv', folder + 'prediction.csv',
#                 '--test-target-csv', folder + 'test-target.csv',
#                 '--model-dir', '.',
#                 # '--nrows', '5000' if i in [3, 4, 5, 6, 7] else '500' if i in [8] else '-1',
#                 '--mode', tests[i]]
#             args = parser.parse_args(argv)
#
#             logf('processing', folder)
#
#             automl = AutoML(args.model_dir)
#
#             if args.train_csv is not None:
#                 automl.train(args.train_csv, args.mode)
#                 automl.save()
#                 log_trail('-', '\n')
#
#             if args.test_csv is not None:
#                 automl.load()
#                 automl.predict(args.test_csv, args.prediction_csv)
#                 log_trail('-', '\n')
#
#     else:
#         automl = AutoML(args.model_dir)
#
#         if args.train_csv is not None:
#             automl.train(args.train_csv, args.mode)
#             automl.save()
#             log_trail('-', '\n')
#
#         if args.test_csv is not None:
#             automl.load()
#             automl.predict(args.test_csv, args.prediction_csv)
#             log_trail('-', '\n')
#
#     log_trail('=', '\n\n')
#
#

if __name__ == '__main__':
    try:
        main()
    except BaseException as e:
        logf('EXCEPTION:', e)
        logf(traceback.format_exc())
        exit(1)
