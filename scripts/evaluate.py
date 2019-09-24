import os
import csv
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Airbus evaluation script')
    parser.add_argument('ground_truth', type=str, help='ground truth csv file')
    parser.add_argument('predictions', type=str, help='predictions csv file')

    args = parser.parse_args()

    ground_truth = os.path.expanduser(args.ground_truth)
    predictions = os.path.expanduser(args.predictions)

    nb_lines = 0
    true_positive, true_negative = 0, 0
    false_positive, false_negative = 0, 0

    with open(ground_truth) as gt:
        with open(predictions) as preds:
            gt_reader = csv.reader(gt, delimiter=',')
            next(gt_reader)

            preds_reader = csv.reader(preds, delimiter=',')
            next(preds_reader)

            for gt_r, p_r in zip(gt_reader, preds_reader):
                nb_lines += 1
                if int(gt_r[1]) == 0:
                    if int(p_r[1]) == 0:
                        true_negative += 1
                    else:
                        false_positive += 1
                else:
                    if int(p_r[1]) == 0:
                        false_negative += 1
                    else:
                        true_positive += 1

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (false_negative + true_positive)
    print('Precision: {:.2%} || Recall: {:.2%}'.format(precision, recall))

