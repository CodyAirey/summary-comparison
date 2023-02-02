# from typing import List, Union, Iterable
import numpy as np
import nltk
import math
# from collections import defaultdict
from nltk import tokenize
# from itertools import zip_longest
import json
# import sys
# import getopt
import pathlib
import pandas as pd
# import re
import time

chapter_summaries = dict()
book_summaries = dict()
threshold = .8
summary_comparison_data = []
line_by_line_data = []


# returns summaries located in scripts/finished_summaries
def get_human_summary(summary_path):
    """ Retrieves the summary text from the given path

    Args:
        summary_path (str): filepath to summary

    Returns:
        str: summary text
    """
    try:
        with open("../../scripts/" + summary_path, encoding='utf-8') as f:
            summary_json = json.load(f)
            return summary_json["summary"]
    except Exception as e:
        print("Failed to read summary file: {}".format(e))
        return None


def scan_chapter_summaries():
    """Gets each chapter summary and places all relevant info into a dictionary
    """
    f = open(pathlib.Path(f"../../alignments/chapter-level-summary-alignments/fixed_chapter_summaries_all_final.jsonl"),
             encoding='utf-8')
    for line in f:
        summary = json.loads(line)
        summary_text = get_human_summary(summary['summary_path'])
        if summary_text is not None:
            try:
                chapter_summaries[summary['summary_path']] = {
                    "section_title": summary['corrected_section'],
                    "source": summary['source'],
                    "book": summary['normalized_title'],
                    "summary_text": summary_text
                }
            except:
                continue
    print("Evaluating {} chapter summary documents...".format(len(chapter_summaries)))


def scan_book_summaries():
    """Gets each book summary and places all relevant info into a dictionary
    """
    f = open(pathlib.Path(f"../../alignments/book-level-summary-alignments/fixed_book_summaries_all_final.jsonl"),
             encoding='utf-8')
    for line in f:
        summary = json.loads(line)
        summary_text = get_human_summary(summary['summary_path'])
        if summary_text is not None:
            try:
                book_summaries[summary['summary_path']] = {
                    "book_title": summary['normalized_title'],
                    "source": summary['source'],
                    "summary_text": summary_text
                }
            except:
                continue
    print("Evaluating {} book summary documents...".format(len(chapter_summaries)))


def setup_model(metric):  # there has got to be a better way to do this.
    """Sets up a model if required, using given metric

    Args:
        metric (str): metric used for test

    Returns:
        _type_: _description_
    """
    if metric == "bleu":
        return  # no model reqiured
    elif metric == "bert":
        from bert import calculate_score
        calculate_score.create_model()
    elif metric == "bertscore":
        from bert import calculate_bertscore
        calculate_bertscore.create_model()
    elif metric == "rouge-1n" or metric == "rouge-2n" or metric == "rouge-l":
        return  # no model required
    elif metric == "moverscore":
        return  # no model required
    elif metric == "qaeval":
        from qaeval_scoring import calculate_score
        calculate_score.create_model()
    elif metric == "meteor":
        return  # no model required
    elif metric == "summac":
        from summac_scoring import calculate_score
        calculate_score.create_model()
        return
    elif metric == "bartscore":
        from bartscore import calculate_score
        calculate_score.create_model()
    elif metric == "chrf":
        return  # no model required
    

def calculate_score(metric, threshold):
    """Scores each chapter summary (reference document) against its corresponding book summary
    (hypothesis document) from the same source. Each ref-sentence i is scored against all hyp sentences 0..n
    individually, taking the all scores above the threshold for each ref-sentence i, then using the average 
    of each ref-sentence score(s) as the final scoring.

    metric used depends on the metric given.

    Args:
        metric (_type_): _description_
    """
    for chap_summary in chapter_summaries.values():
        for book_summary in book_summaries.values():
            if chap_summary['source'] == book_summary['source'] and chap_summary['book'] == book_summary['book']:
                ref_doc = tokenize.sent_tokenize(chap_summary['summary_text'])
                hyp_doc = tokenize.sent_tokenize(book_summary['summary_text'])
                ref_sentence_scores = [] # contains the max scores from each singular ref-sentence i, against all possible hyp-sentences 0..n 
                temp_time = time.time()
                for ref_sent_index, ref_sent in enumerate(ref_doc):

                    for hyp_sent_index, hyp_sent in enumerate(hyp_doc):
                        current_score = "!" #initilze value to something error worthy if not changed.
                        precision = "NA"
                        recall = "NA"

                        compute_single_score(metric, ref_sent, hyp_sent)
                        
                        if current_score > threshold:
                            ref_sentence_scores.append(current_score)

                        line_by_line_data.append([chap_summary['section_title'], book_summary['normalized_title'], book_summary['source'], ref_sent_index, hyp_sent_index, current_score, precision, recall])
                        
                    #end hyp-sent forloop
                #end ref-sent forloop
                score = np.mean(ref_sentence_scores)
                summary_comparison_data.append(score, chap_summary['section_title'], book_summary['book_title'], book_summary['source'])
                summaries_count += 1
                print(f"{book_summary['book_title']}, {chap_summary['section_title']}, {book_summary['source']} - Time: {time.time() - temp_time, 3}, seconds.")

                if summaries_count >= 20:
                    return
                

def compute_single_score(metric, ref_sent, hyp_sent):


    """Calculates an f1 score between two sentences depending on the metric used 

    Args:
        metric (str): metric to denote how the calculation is performed
        ref_sent (str): reference sentence
        hyp_sent (str): hypothesis sentence

    Returns:
        float: f1 score based on how similar the ref_sent and hyp_sent are
    """
    # calculate score based on metric, p.s. surely there is a better way to do this.
    if metric == "bleu":
        from bleu import calculate_score
        current_score, precision = calculate_score.compute_score(ref_sent, hyp_sent)
    elif metric == "bert":
        from bert import calculate_score
        current_score = calculate_score.compute_score(ref_sent, hyp_sent)
    elif metric == "bertscore":
        from bert import calculate_bertscore
        current_score, precision, recall = calculate_bertscore.compute_score(ref_sent, hyp_sent)
    elif metric == "rouge-1n":
        from rouge_scoring import calculate_score
        current_score, precision, recall = calculate_score.compute_score_1n(ref_sent, hyp_sent)
    elif metric == "rouge-2n":
        from rouge_scoring import calculate_score
        current_score, precision, recall = calculate_score.compute_score_2n(ref_sent, hyp_sent)
    elif metric == "rouge-l":
        from rouge_scoring import calculate_score
        current_score, precision, recall = calculate_score.compute_score_l(ref_sent, hyp_sent)
    elif metric == "moverscore":
        from moverscore import calculate_score
        current_score = calculate_score.compute_score(ref_sent, hyp_sent)
    elif metric == "qaeval":
        from qaeval_scoring import calculate_score
        current_score = calculate_score.compute_score(ref_sent, hyp_sent)
    elif metric == "meteor":
        from meteor import calculate_score
        current_score = calculate_score.compute_score(ref_sent, hyp_sent)
    elif metric == "summac":
        from summac_scoring import calculate_score
        current_score = calculate_score.compute_score(ref_sent, hyp_sent)
    elif metric == "bartscore":
        from bartscore import calculate_score
        current_score = calculate_score.compute_score(ref_sent, hyp_sent)
    elif metric == "chrf":
        from chrf import calculate_score
        current_score = calculate_score.compute_score(ref_sent, hyp_sent)

    return current_score

def write_results_to_csv(function, split, filename):
    df = pd.DataFrame(summary_comparison_data, columns=[function + "score", ""])