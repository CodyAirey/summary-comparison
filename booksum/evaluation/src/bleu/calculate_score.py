from nltk import tokenize
from nltk.translate.bleu_score import sentence_bleu


def compute_score(token, sentence):
    token_split = tokenize.word_tokenize(token)
    sent_split = tokenize.word_tokenize(sentence)
    current_score = sentence_bleu([token_split], sent_split, weights=(1,.0,0,0))
    return current_score