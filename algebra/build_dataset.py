#!/usr/bin/env python3

import argparse
import csv
import json

import sys
# Link to the socratic-tutor repository so that we can import `evaluation`.
sys.path.append('../../socratic-tutor')
import evaluation

import re


def make_irt_dataset(data, output, drop_corrected=True, drop_freeform=True, normalize=True):
    with open(data) as f:
        rows = list(csv.DictReader(f))

    if drop_corrected:
        rows = [r for r in rows if r['interfaceMode'] != '3']

    if drop_freeform:
        rows = [r for r in rows if r['interfaceMode'] != '4']

    all_rows = [
        { 'problem': r['problemTrace'].split(';')[0].strip(),
          'student': r['user_id'],
          'correct': r['correct'] == '1',
          'errors': 0 if r['invalidSteps'] == '[]' else (1 + r['invalidSteps'].count(',')),
          'timestamp': r['startTime'],
         }
        for r in rows
    ]

    rows = []

    for problem, student in set((r['problem'], r['student']) for r in all_rows):
        rows.append(
            {
                'problem': problem,
                'student': student,
                'correct': sum(r['errors'] for r in all_rows if r['problem'] == problem and r['student'] == student) == 0,
                'timestamp': min(r['timestamp'] for r in all_rows if r['problem'] == problem and r['student'] == student),
            }
        )

    #for r in rows:
    #    print(r['student'], r['problem'], r['correct'])

    # Syntactically normalize solutions using the Racket parser/formatter.
    if True:
        problems = [r['problem'] for r in rows]
        # print('Problems:', problems)
        problems = evaluation.normalize_solutions([problems])[0]
        for r, p in zip(rows, problems):
            r['problem'] = re.sub('[a-z]', 'x', p)

            if normalize:
                r['problem'] = re.sub('[0-9]+', 'C', r['problem'])

    print(len(rows), 'data points.')
    print(len(set(r['student'] for r in rows)), 'students.')
    print(len(set(r['problem'] for r in rows)), 'problems.')
    print(sum(r['correct'] for r in rows) / len(rows), 'average correct.')

    with open(output, 'w') as f:
        json.dump(rows, f)
        print('Wrote', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='Path to csv file containing the dataset (default: random data)')
    parser.add_argument('--output', help='Path to output file.')
    parser.add_argument('--drop-corrected', action='store_true', help='Whether to ignore data collected in the corrected interface.')
    parser.add_argument('--drop-freeform', action='store_true', help='Whether to ignore data collected in the freeform interface.')
    parser.add_argument('--normalize-consts', action='store_true', help='Whether to replace constants by a placeholder, grouping equations having the same structure.')

    opt = parser.parse_args()
    make_irt_dataset(opt.data,
                     opt.output,
                     drop_corrected=opt.drop_corrected,
                     drop_freeform=opt.drop_freeform,
                     normalize=opt.normalize_consts,
                     )
