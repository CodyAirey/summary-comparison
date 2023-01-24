from typing import List, Union, Iterable
import numpy as np
import nltk
import math
from collections import defaultdict
from nltk import tokenize
from itertools import zip_longest
import json
import sys
import getopt
import pathlib
import time
import pandas as pd
import re

human_summaries = dict()
summaries_count = 0
data = []
used_files = []
unique_books = set()
unique_used_books = set()
unique_chapters = set()
unique_used_chapters = set()


library = {}

allignment_tables = {}

chapter_numbers = {}

number_of_matches = {
    "bookwolf": dict(),
    "cliffnotes": dict(),
    "gradesaver": dict(),
    "novelguide": dict(),
    "pinkmonkey": dict(),
    "shmoop": dict(),
    "sparknotes": dict(),
    "thebestnotes": dict()
}
# source1 ----------book1 -------total_aggreagte_sections = 0
#           \              \---- total_agg_sections_used = 0
#            \              \ ---total_sections = 0
#             \              \---used_sections = 0
#              \              \-- { section_name_1 { sources : [s1, s2,s3], word_count : 123 },
#                                 , section_name_2 { sources : [s2, s4,s7], word_count : 213 }
#                                  , ...etc.etc  { sources : [s1, s2,s3], word_count : 123 }
#               \              \-
#                \
#                 \-book2 ----- etc ^
# souce2 ---- etc.


average_books = {}

def setup_matches_datastructure(split):
    f = open(pathlib.Path(f"../../alignments/chapter-level-summary-alignments/fixed_chapter_summaries_{split}.jsonl"),
             encoding='utf-8')

    # for sourc in number_of_matches:
    # for book in number_of_matches[sourc]:


    # setup the 'datastructure'.
    for line in f:
        content = json.loads(line)

        original_title = content['book_id']
        # extracts book title as a string
        book = content['book_id'].split(".")[0:1][0].replace(",","")
        # special case.
        if content['book_id'].split(".")[0:3] == ["Dr", " Jekyll and Mr", " Hyde"]:
            book = "Dr. Jekyll and Mr. Hyde"
        if book == "King Henry IV Part 1":
            book = "Henry IV Part 1"

        corrected_title = content['corrected_title']


        source = content['source']



        library[book] = {
            'total_individual_chapters': 0, #total number of chapter/scenes/w.e for the actual book, 
            'total_sections': 0,
            'total_sections_used': 0, # that are found elsewhere / compared with.
            'total_aggregate_sections': 0,
            'total_aggregate_sections_used': 0, #that are found elsewhere / compared with.
            'total_non-aggregate_sections': 0,
            'total_non-aggregate_sections_used': 0, # that are found elsewhere / compared with.
            'non-aggregate_sections_used': dict(),
            'aggregate_sections_used': dict(), #that are found elsewhere / compared with.
            'non-aggregate_sections_not_used': dict(),
            'aggregate_sections_not_used': dict()
        }

        text = get_human_summary(content['summary_path'])
        
        if text is not None:
            try:
                human_summaries[content['summary_path']] = {
                    "book": book,
                    "chapter_title": corrected_title,
                    "source": source,
                    "summary_text": text,
                    "original_title": original_title,
                    'is_aggregate': content['is_aggregate'],
                    'chapter_path': content['chapter_path']
                }

                print(f"Found {content['c_title']}")
            except:
                continue

    print("Evaluating {} summary documents...".format(len(human_summaries)))



def result_printout(function):
    """Prints out the results for summary comparison

    Args:
        function (str): the function used in the test
    """
    print("Unique chapters covered: {}".format(len(unique_chapters)))
    print("Unique chapters used: {}".format(len(unique_used_chapters)))
    FUNC_list = [data_item[0] for data_item in data]
    FUNC_mean = sum(FUNC_list) / len(FUNC_list)
    print(f"Mean {function}: {FUNC_mean}")
    print()


# returns summaries located in scripts/finished_summaries
def get_human_summary(summary_path):
    try:
        with open("../../scripts/" + summary_path, encoding='utf-8') as f:
            summary_json = json.load(f)
            return summary_json["summary"]
    except Exception as e:
        print("Failed to read summary file: {}".format(e))
        return None


def setup_model(function):  # there has got to be a better way to do this.
    """Sets up a model if required, using given function

    Args:
        function (str): function used for test

    Returns:
        _type_: _description_
    """
    if function == "bleu":
        return  # no model reqiured
    elif function == "bert":
        from bert import calculate_score
        calculate_score.create_model()
    elif function == "bertscore":
        from bert import calculate_bertscore
        calculate_bertscore.create_model()
    elif function == "rouge-1n" or function == "rouge-2n" or function == "rouge-l":
        return  # no model required
    elif function == "moverscore":
        return  # no model required
    elif function == "qaeval":
        from qaeval_scoring import calculate_score
        calculate_score.create_model()
    elif function == "meteor":
        return  # no model required
    elif function == "summac":
        from summac_scoring import calculate_score
        calculate_score.create_model()
        return
    elif function == "bartscore":
        from bartscore import calculate_score
        calculate_score.create_model()
    elif function == "chrf":
        return  # no model required


def calculate_F1(function):
    """Calculates the summary score using the given function, this methods a mess, im sorry. 
    (in my defense i took over this code from a previous user)

    Starts by looping over all summaries, and scoring the similarity between reviews from different sources for the same book. 
    E.G. Dracula.chapter1.sparknotes vs Dracula.chapter1.bookwolf, Dracula.chapter1.gradesaver

    chapter is compared one sentence at a time to ensure uniform comparing methods for each function
    (some can't handle multiple at once)

    Args:
        function (str): the metric to use to calculate score

    Returns:
        data (list): data containing the Score, Number of unique sentences, Book Title, Chapter Number, and Source
        summaries_count (int): number of summaries compared
        unique_books (set): all the unique books avaiable
        unique_books_used (set): all the unique books used (books with pairs to compare)

    """
    start_time = time.time()
    summaries_count = 0

    for summary_path, summary in human_summaries.items():

        book = summary['book']
        section_title = summary['chapter_title'] #normalized for roman numerals & aggregate chapter names (chapters1-3 / chapter1-chapter3)
        source = summary['source']
        summary_text = summary['summary_text']
        original_title = summary['original_title']
        is_aggregate = summary['is_aggregate']
        
        unique_books.add(summary['chapter_title']) #add to set.


        section_number = section_title.split("-")[-1]
        # num_chapters = 0
        if section_number.isnumeric():
            library[book]['total_individual_chapters'] = max(library[book]['total_individual_chapters'], int(section_number))
        #     num_chapters = max(int(section_number), num_chapters)
        # if library[book]['total_individual_chapters'] < num_chapters:
        #     library[book]['total_individual_chapters'] = num_chapters

        #grab all summaries with same title but different source (matching sections to compare with one another) aka related.
        related_summaries = list(filter(lambda curr_summary: curr_summary['chapter_title'] == summary[
                'chapter_title'] and curr_summary['source'] != summary['source'], human_summaries.values()))

                
        # Remember which files have been used.
        used_files.extend(related_summaries) #is this needed? #TODO

        word_count = len(summary['summary_text'].split(" "))

        # if book not in number_of_matches_2:
        #     number_of_matches_2[]

        library[summary['book']]['total_sections'] += 1
        if summary['is_aggregate'] == True:
            library[summary['book']]['total_aggregate_sections'] += 1
        else:
            library[summary['book']]['total_non-aggregate_sections'] += 1

        # if there are no related summary documents, then just print.
        if len(related_summaries) == 0:
            print(f"No related summary documents were found for {section_title}.")
            if(is_aggregate):
                library[book]['aggregate_sections_not_used'][section_title] = { source : word_count}
            else:
                library[book]['non-aggregate_sections_not_used'][section_title] = { source : word_count}
            continue #no need to perform calculation

        related_summary_texts = []
        for summary2 in related_summaries: #same title, diff source.
            summary2_word_count = len(summary2['summary_text'].split(" "))
            related_summary_texts.append(summary2['summary_text'])
            with open("../../" + summary['chapter_path'], 'r') as text_file:
                        text_data = text_file.read()
                        split_data = re.split(" |\t|\n", text_data)
                        split_data = list(filter(None, split_data)) #remove empty strings.
                        original_word_count = len(split_data)
            if is_aggregate:
                if summary2['chapter_title'] not in library[book]['aggregate_sections_used']:
                    library[book]['aggregate_sections_used'][summary2['chapter_title']] = {'original': original_word_count}
                library[book]['aggregate_sections_used'][section_title][summary2['source']] = summary2_word_count
                library[book]['total_aggregate_sections_used'] += 1
                library[book]['total_sections_used'] += 1

            else: #not aggregate
                if summary2['chapter_title'] not in library[book]['non-aggregate_sections_used']:
                    library[book]['non-aggregate_sections_used'][summary2['chapter_title']] = {'original': original_word_count}
                    # library[book]['non-aggregate_sections_used'][summary2['chapter_title']] = {}
                library[book]['non-aggregate_sections_used'][section_title][summary2['source']] = summary2_word_count
                library[book]['total_non-aggregate_sections_used'] += 1
                library[book]['total_sections_used'] += 1

            
        # prep text by tokenizing it into sentences
        ref_doc = tokenize.sent_tokenize(summary['summary_text'])
        tokenized_sums = []
        for cursum in related_summary_texts:
            tokenized_sums.append(tokenize.sent_tokenize(cursum))

        temp_time = time.time()

        max_scores = []
        for sentence_list in tokenized_sums:
            sentence_scores = []
            unique_sents = set()
            for i, token in enumerate(ref_doc):
                best_score = -math.inf
                best_score_i = -1
                for j, sentence in enumerate(sentence_list):
                    current_score = "!"

                    # calculate score based on function, p.s. surely there is a better way to do this.
                    if function == "bleu":
                        from bleu import calculate_score
                        current_score = calculate_score.compute_score(
                            token, sentence)
                    elif function == "bert":
                        from bert import calculate_score
                        current_score = calculate_score.compute_score(
                            token, sentence)
                    elif function == "bertscore":
                        from bert import calculate_bertscore
                        current_score = calculate_bertscore.compute_score(
                            token, sentence)
                    elif function == "rouge-1n":
                        from rouge_scoring import calculate_score
                        current_score = calculate_score.compute_score_1n(
                            token, sentence)
                    elif function == "rouge-2n":
                        from rouge_scoring import calculate_score
                        current_score = calculate_score.compute_score_2n(
                            token, sentence)
                    elif function == "rouge-l":
                        from rouge_scoring import calculate_score
                        current_score = calculate_score.compute_score_l(
                            token, sentence)
                    elif function == "moverscore":
                        from moverscore import calculate_score
                        current_score = calculate_score.compute_score(
                            token, sentence)
                    elif function == "qaeval":
                        from qaeval_scoring import calculate_score
                        current_score = calculate_score.compute_score(
                            token, sentence)
                    elif function == "meteor":
                        from meteor import calculate_score
                        current_score = calculate_score.compute_score(
                            token, sentence)
                    elif function == "summac":
                        from summac_scoring import calculate_score
                        current_score = calculate_score.compute_score(
                            token, sentence)
                    elif function == "bartscore":
                        from bartscore import calculate_score
                        current_score = calculate_score.compute_score(
                            token, sentence)
                    elif function == "chrf":
                        from chrf import calculate_score
                        current_score = calculate_score.compute_score(
                            token, sentence)

                    # print("token:", token)
                    # print("sentence: ", sentence)
                    # print("score:", current_score)
                    # print("---")
                    if current_score > best_score:
                        best_score = current_score
                        best_score_i = j
                sentence_scores.append(best_score)
                unique_sents.add(best_score_i)
                # print("NEXT TOKEN")
            max_scores.append(np.mean(sentence_scores))
            print(f"{sentence_scores} => {np.mean(sentence_scores)}")
            print("Unique sentences:", len(unique_sents), "out of",
                  len(ref_doc), "ref sents. :", unique_sents)
            mean_sent_score = np.mean(sentence_scores)
        print(f"{np.mean(max_scores)}")
        mean_max_score = np.mean(max_scores)

        print(summary['chapter_title'], "-", summary['source'],
              "- time:", round((time.time() - temp_time), 3), "seconds.")

        data.append([mean_max_score, len(unique_sents),
                    summary['chapter_title'], summary['source']])

        unique_used_books.add(summary['chapter_title'])
        summaries_count += 1

        if summaries_count >= 200:
            break

    # adjust_individual_chapters()
    # print(len(library["The Mysteries of Udolpho"]['non-aggregate_sections_not_used']))
    

    total_time = (time.time() - start_time)
    print(summaries_count)
    print("time total:", round(total_time, 1), "seconds.",
          "Average:", (total_time / summaries_count))

    return data, summaries_count, unique_books, unique_used_books


def write_summary_count_to_json(split, filename):
    with open(f"../summary_count/chapter-comparison-counts-postfix-{split}-{filename}.json", 'w') as f:
        f.write(json.dumps(library))
            



def write_to_csv(function, split, filename):
    print(filename)
    df = pd.DataFrame(data, columns=[
                      function, "number of Unique sentences", "chapter-title", "source"])
    # Save file.
    df.to_csv(
        f"../csv_results/booksum_summaries/chapter-comparison-results-{split}-{filename}.csv")


def helper(function_list):
    """Prints useful help commands when user uses file with incorrect arguments

    Args:
        function_list (list): list of possible useable functions (metrics currently supported)
    """
    print('Usage: compare_chapters.py -f <function> -o <output-csv-filename> -s <split>')
    print('----')
    print("Functions:", function_list)
    print("Possible Splits: test, train, val    (default is train)")
    print("Example Filename: bart-24-12-2022")


# def adjust_individual_chapters():
#     '''
#     hard coded fix for these outliers, adjusting total number of chapters based on (apendices?)
#     '''
#     # Frankenstein-volume-3-final-letters
#     # Middlemarch-finale-finale
#     # The-Turn-of-the-Screw-prologue
#     # Green-Mansions-prologue
#     # Little-Dorrit-preface
#     # Troilus-and-Cressida-prologue-prologue
#     # Henry-5-prologue
#     # Henry-8-prologue

#     # #Dont need this anymore, we are ignoring prolouge, preface, etc etc.
#     # for book in library:
#     #     if book == 'Henry':
#     #         library[book]['total_individual_chapters'] += 2
#     #     if book in ['Middlemarch', "The Turn of the Screw", "Green Mansions", "Little Dorrit", "Troilus and Cressida", "Frankenstein"]:
#     #         library[book]['total_individual_chapters'] += 1

#     #fixes so that we can just grab max last chapter instead of incrementally adding volumes + chapters.
#     # library["The Mysteries of Udolpho"]['total_individual_chapters'] = 57

#     number_of_matches = {
#         "bookwolf"     : 0,
#         "cliffnotes"   : 0,
#         "gradesaver"   : 0,
#         "novelguide"   : 0,
#         "pinkmonkey"   : 0,
#         "shmoop"       : 0,
#         "sparknotes"   : 0,
#         "thebestnotes" : 0
#     }

#     for book in library:

#         for summary_path, summary in human_summaries.items():
#             if summary['book'] == book:
#                 if summary['is_aggregate']:
#                     split_title = summary['chapter_title'].split("-")
#                     if(summary['chapter_title'].count('scene') == 2):
#                         #last scene number(offset by -1, because 1-3 = 2,not3) 
#                         first_chapter_number = split_title[split_title.index("scene")+1]
#                         split_title.reverse()
#                         second_chapter_number = split_title[split_title.index("scene")-1]
#                         if first_chapter_number.isnumeric() and second_chapter_number.isnumeric(): #used to avoid chapter 3 - chapter finale
#                             chapter_coverage = int(second_chapter_number) - (int(first_chapter_number)-1)
#                             if chapter_coverage <= 0:
#                                 print(f"Reverse Numbers: {summary['chapter_title']}")
#                             else:
#                                 number_of_matches[summary['source']] += chapter_coverage
#                         else:
#                             print(f"Bad Numeric: {summary['chapter_title']}")

#                     elif(summary['chapter_title'].count('chapter') == 2):
#                         first_chapter_number = split_title[split_title.index("chapter")+1]
#                         split_title.reverse()
#                         second_chapter_number = split_title[split_title.index("chapter")-1]
#                         if first_chapter_number.isnumeric() and second_chapter_number.isnumeric(): #used to avoid chapter 3 - chapter finale
#                             chapter_coverage = int(second_chapter_number) - (int(first_chapter_number)-1)
#                             if chapter_coverage <= 0:
#                                 print(f"Reverse Numbers: {summary['chapter_title']}")
#                             else:
#                                 number_of_matches[summary['source']] += chapter_coverage
#                         else:
#                             print(f"Bad Numeric: {summary['chapter_title']}")

#                 else: #not aggregate, thank god!
#                     number_of_matches[summary['source']] += 1
   


# def create_tables():
#     sources = ["bookwolf", "cliffnotes", "gradesaver", "novelguide", "pinkmonkey", "shmoop", "sparknotes", "thebestnotes"]
#     keywords = ['act', 'canto', 'epilogue', 'scene', 'finale', 'prologue', 'chapter', 'volume', 'part', 'book']
#     for book in library:
#         #make table.

#         allignment_tables[book] = pd.DataFrame(columns=(['chapter_number'] + sources))


#         for i in range(0, chapter_numbers[book]):
#             allignment_tables[book].at[i, 'chapter_number'] = i+1 

#         for source in sources:
#             for i in range(0, chapter_numbers[book]): #Do i need to +1 on chapters? ordinal / nominal?

#                 if chapter_coverage > 0:
#                     allignment_tables[book].at[i, source] = 'A' #a for aggregate
#                     chapter_coverage -= 1

                
#                 non_agg_sections = library[book]['non-aggregate_sections_used']
#                 for section in non_agg_sections:
#                     if source in section.keys():
#                         allignment_tables[book].at[i, source] = 1 # 1 for single-non-aggreagte 
                

#                 #all this is to find out how much chapters the aggregate covers.
#                 agg_sections = library[book]['aggregate_sections_used']
#                 for section in agg_sections:
#                     section_t_split = section.split("-")
#                     chapter_coverage = None
#                     for word in keywords:
#                         if section_t_split.count(word) >= 2:

#                             first_chapter_number = section_t_split[section_t_split.index(word)+1] #get numbers (3 & 4) from string: "Dracula-chapter-3-chapter-4"
#                             section_t_split.reverse()
#                             second_chapter_number = section_t_split[section_t_split.index(word)-1] # ^^

#                             if first_chapter_number.isnumeric() and second_chapter_number.isnumeric(): #avoid things like: chapter-1-chapter-finale
#                                 chapter_coverage = int(second_chapter_number) - (int(first_chapter_number)-1) # chapter 1 - chapter 3 should = 3 covered chapters.
#                     #three hardcoded fixes
#                     if section == "Middlemarch-book-8-chapter-84-chapter-finale":
#                         chapter_coverage = 16
#                     if section == "Coriolanus-act-4-5-scene-5-scene-1":
#                         chapter_coverage = 4
#                     if section == "The-Winter's-Tale-act-3-4-scene-3-3":
#                         chapter_coverage = 4 #act3-s3, act4-s1, s2 and, s3

                


def populate_individual_chapters():
    f = open("chapter_numbers_train.jsonl")
    chapter_numbers = json.load(f)

def main(argv):
    """Main method takes arguments for Function, OutputFilename, and Split to use.
    Afterwhich, it calculates the score for all booksum sections and writes the output to a file.

    Args:
        argv (list? of str): 0: filename, 1-3: function, split, outputfilename
    """
    function = None
    outputfile = None
    split = None
    function_list = ["bleu", "bert", "bertscore", "rouge-1n", "rouge-2n", "rouge-l",
                     "moverscore", "qaeval", "meteor", "summac", "bartscore", "chrf"]
    split_list = ["test", "train", "val"]

    if (len(argv) <= 4):
        helper(function_list)
        sys.exit(2)

    # used getopt for first time to handle arguments, works well but feels messy.
    try:
        opts, args = getopt.getopt(
            argv, "hf:o:s:", ["help", "function=", "ofile=", "split="])
    except getopt.GetoptError:
        helper(function_list)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', "--help"):
            helper(function_list)
            sys.exit()
        elif opt in ("-f", "--function"):
            function = arg
            if function not in function_list or function == '' or function == None:
                print("Function not acceptable, please use one of:", function_list)
                sys.exit(2)
        elif opt in ("-o", "--ofile"):
            outputfile = arg
            if outputfile == '' or outputfile == None:
                print("Please provide a output filename")
                sys.exit(2)
        elif opt in ("-s", "--split"):
            split = arg
            if split not in split_list:
                print("Split not acceptable, please use one of:", split_list)
                sys.exit(2)

    print('Function is:', function)
    print('Output file is:', outputfile)
    print('Split is:', split)

    setup_matches_datastructure(split)
    setup_model(function)
    populate_individual_chapters()
    
    # calculate_original_word_count()


    # create_tables()

    data, summaries_count, unique_books, unique_used_books = calculate_F1(
        function)
    
    result_printout(function)
    # write_to_csv(function, split, outputfile)

    

    write_summary_count_to_json(split, outputfile)


if __name__ == "__main__":
    # print(sys.argv[1:])
    # sys.exit(1)
    main(sys.argv[1:])
