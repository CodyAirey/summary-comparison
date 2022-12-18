import rouge_scoring
from rouge_scoring.rouge_scoring import rouge_n


def compute_score(token, sentence):
    current_score, precision, recall = rouge_n([token], [sentence], n=1)
    return current_score