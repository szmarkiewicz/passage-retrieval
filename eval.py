from argparse import ArgumentParser

import pandas as pd
from sklearn.metrics import ndcg_score

if __name__ == '__main__':
    # Parse provided arguments
    parser = ArgumentParser()
    parser.add_argument('--true', dest='true', help='tsv file with true (question, passage) pairs', metavar='FILE')
    parser.add_argument('--pred', dest='pred', help='tsv file with predicte (question, passage) pairs', metavar='FILE')
    parser.add_argument('--k', type=int, dest='k', default=10)
    args = parser.parse_args()

    true = pd.read_csv(args.true, sep='\t')
    pred = pd.read_csv(args.pred, sep='\t')
    pred['score'] -= pred['score'].min() - 1e-6  # convert to non-negative scores

    # Prepare DataFrame with scores
    scores = true.merge(pred, on=['question-id', 'passage-id'], how='outer', suffixes=('_true', '_pred'))
    scores = scores.fillna(0)
    scores = scores.groupby('question-id').agg(list)
    scores = scores.reset_index()


    def score(r):
        # Check if there is only one true passage and only one result returned, and if the result matches the true
        # value. Then, return 1 as the metric value
        if len(r['score_true']) == 1 and len(r['score_pred']) == 1 and r['passage-id'][0] == \
                pred[pred['question-id'] == r['question-id']].iloc[0]['passage-id']:
            return 1
        return ndcg_score([r['score_true']], [r['score_pred']], k=args.k)


    # Obtain NDCG scores
    ndcg = scores.apply(score, axis=1).mean()

    directories_pred = args.pred.split('/')

    # Display scores and write to 'results.txt' file
    f = open('/'.join(directories_pred[:(len(directories_pred) - 1)]) + "/results.txt", "w")
    f.write(f'NDCG@{args.k}: {ndcg:.3f}')

    print(f'NDCG@{args.k}: {ndcg:.3f}')

    f.close()
