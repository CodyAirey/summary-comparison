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


number_of_matches = {
   "bookwolf"     : dict(),
   "cliffnotes"   : dict(),
   "gradesaver"   : dict(),
   "novelguide"   : dict(),
   "pinkmonkey"   : dict(),
   "shmoop"       : dict(),
   "sparknotes"   : dict(),
   "thebestnotes" : dict()
}
# source1 ----------book1 -------total_aggreagte_sections = 0
#           \              \---- total_agg_sections_used = 0
#            \              \ ---total_sections = 0
#             \              \---used_sections = 0
#              \              \-- [ section_name_1 , section_name_2...etc.etc ]
#               \              \-
#                \
#                 \-book2 ----- etc ^


def setup_matches_datastructure(split):
   f = open(pathlib.Path(f"../../alignments/chapter-level-summary-alignments/chapter_summary_aligned_{split}_split.jsonl"),
            encoding='utf-8')

   # for sourc in number_of_matches:
      # for book in number_of_matches[sourc]:

   x = 0

   #setup the 'datastructure'.
   for line in f:
      content = json.loads(line)

      flag = False
      
      original_title = content['book_id']
         
      book = content['book_id'].split(".")[0:1][0] #extracts book title as a string

      # book = content['book_id'].split(content['summary_id'])[0][:-1]

      # print(f"{content['book_id']}")
      id_split = re.split("\.|_|-| ", content['book_id'])
      temp_id = []

      #replace romans to ints, e.g. ACT_II to ACT_2
      for i, each in enumerate(id_split):
         if(romanToInt(each.upper()) >= 1):
               id_split[i] = str(romanToInt(each.upper()))

      #special case.
      if content['book_id'].split(".")[0:3] == ["Dr"," Jekyll and Mr", " Hyde"]:
         book = "Dr. Jekyll and Mr. Hyde"

      i = 0

      # fix instances of "chapters_43-48" to "chapter-43-chapter-48" etc.
      try :
         while i < len(id_split):
            if id_split[i] == "chapters" and i+1 <= len(id_split):
                  temp_id.append("chapter")
                  temp_id.append(id_split[i+1])
                  temp_id.append("chapter")
                  # i += 2
                  i += 2
            elif id_split[i] == "scenes" and i+1 <= len(id_split):
                  temp_id.append("scene")
                  temp_id.append(id_split[i+1])
                  temp_id.append("scene")
                  # i += 2
                  i += 2
            elif i < len(id_split):
                  temp_id.append(id_split[i])
                  i += 1
      except:
         temp_id.append(id_split[i])
         i += 1
         continue

      corrected_title = "-".join(temp_id)


      source = content['source']
      number_of_matches[source][book] = {
         'total_sections' : 0,
         'total_sections_used' : 0,
         'total_aggregate_sections' : 0,
         'total_aggregate_sections_used' : 0,
         'sections_that_exist_elsewhere' : set()
      }

      text = get_human_summary(content['summary_path'])
      if text is not None:
         try:
               human_summaries[content['summary_path']] = {
                  "book" : book,
                  "chapter_title": corrected_title,
                  "source": source,
                  "summary_text": text,
                  "original_title": original_title
               }

               print(f"Found {content['c_title']}")
         except:
               continue

   print("Evaluating {} summary documents...".format(len(human_summaries)))



#yoinked from geeks for geeks | https://www.geeksforgeeks.org/python-program-for-converting-roman-numerals-to-decimal-lying-between-1-to-3999/
def romanToInt(s):

    try:

        translations = {
            "I": 1,
            "V": 5,
            "X": 10,
            "L": 50,
            "C": 100,
            "D": 500,
            "M": 1000
        }
        number = 0
        s = s.replace("IV", "IIII").replace("IX", "VIIII")
        s = s.replace("XL", "XXXX").replace("XC", "LXXXX")
        s = s.replace("CD", "CCCC").replace("CM", "DCCCC")
        for char in s:
            number += translations[char]
        return number
    except:
        return -1


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


def get_human_summary(summary_path): #returns summaries located in scripts/finished_summaries
   try:
      with open("../../scripts/" + summary_path, encoding='utf-8') as f:
         summary_json = json.load(f)
         return summary_json["summary"]
   except Exception as e:
      print("Failed to read summary file: {}".format(e))
      return None


def setup_model(function): #there has got to be a better way to do this.
   """Sets up a model if required, using given function

   Args:
       function (str): function used for test

   Returns:
       _type_: _description_
   """
   if function == "bleu":
      return # no model reqiured
   elif function == "bert":
      from bert import calculate_score
      calculate_score.create_model()
   elif function == "bertscore":
      from bert import calculate_bertscore
      calculate_bertscore.create_model()
   elif function == "rouge-1n" or function == "rouge-2n" or function == "rouge-l":
      return # no model required
   elif function == "moverscore":
      return #no model required
   elif function == "qaeval":
      from qaeval_scoring import calculate_score
      calculate_score.create_model()
   elif function == "meteor":
      return #no model required
   elif function == "summac":
      from summac_scoring import calculate_score
      calculate_score.create_model()
      return
   elif function == "bartscore":
      from bartscore import calculate_score
      calculate_score.create_model()
   elif function == "chrf":
      return #no model required


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

      # Get all related summary documents.
      unique_books.add(summary['chapter_title'])
      # Special case for Around the World in Eighty (80) Days
      if summary['chapter_title'] == "Around-the-World-in-Eighty Days":
         related_summaries = list(filter(
               lambda curr_summary: curr_summary['chapter_title'] == 'Around-the-World-in-80-Days', human_summaries.values()))
      elif summary['chapter_title'] == "Around-the-World-in-80-Days":
         related_summaries = list(filter(
               lambda curr_summary: curr_summary['chapter_title'] == 'Around-the-World-in-Eighty-Days', human_summaries.values()))
      else:
         related_summaries = list(filter(lambda curr_summary: curr_summary['chapter_title'] == summary[
                                    'chapter_title'] and curr_summary['source'] != summary['source'], human_summaries.values()))
      # Remember which files have been used.
      used_files.extend(related_summaries)

      # print(used_files)

      number_of_matches[summary['source']][summary['book']]['total_sections'] += 1

      # if there are no related summary documents, then just print.
      if len(related_summaries) == 0:
         print("No related summary documents were found for {}.".format(
               summary['chapter_title']))
         continue
      related_summary_texts = [curr_summary['summary_text']
                              for curr_summary in related_summaries]

      for path, summary2 in human_summaries.items():
         if summary == summary2: continue
         if summary["chapter_title"] == summary2["chapter_title"]:
            number_of_matches[summary['source']][summary['book']]['sections_that_exist_elsewhere'].add(summary['chapter_title']) #esentially just adding each chapter/section from a source to a set.

      #prep text by tokenizing it into sentences
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
               best_score_i = -1;
               for j, sentence in enumerate(sentence_list):
                  current_score = "!"

                  #calculate score based on function, p.s. surely there is a better way to do this.
                  if function == "bleu":
                     from bleu import calculate_score
                     current_score = calculate_score.compute_score(token, sentence)
                  elif function == "bert":
                     from bert import calculate_score
                     current_score = calculate_score.compute_score(token, sentence)
                  elif function == "bertscore":
                     from bert import calculate_bertscore
                     current_score = calculate_bertscore.compute_score(token, sentence)
                  elif function == "rouge-1n":
                     from rouge_scoring import calculate_score
                     current_score = calculate_score.compute_score_1n(token, sentence)
                  elif function == "rouge-2n":
                     from rouge_scoring import calculate_score
                     current_score = calculate_score.compute_score_2n(token, sentence)
                  elif function == "rouge-l":
                     from rouge_scoring import calculate_score
                     current_score = calculate_score.compute_score_l(token, sentence)
                  elif function == "moverscore":
                     from moverscore import calculate_score
                     current_score = calculate_score.compute_score(token, sentence)
                  elif function == "qaeval":
                     from qaeval_scoring import calculate_score
                     current_score = calculate_score.compute_score(token, sentence)
                  elif function == "meteor":
                     from meteor import calculate_score
                     current_score = calculate_score.compute_score(token, sentence)
                  elif function == "summac":
                     from summac_scoring import calculate_score
                     current_score = calculate_score.compute_score(token, sentence)
                  elif function == "bartscore":
                     from bartscore import calculate_score
                     current_score = calculate_score.compute_score(token, sentence)
                  elif function == "chrf":
                     from chrf import calculate_score
                     current_score = calculate_score.compute_score(token, sentence)

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
         print("Unique sentences:", len(unique_sents), "out of", len(ref_doc), "ref sents. :", unique_sents)
         mean_sent_score = np.mean(sentence_scores)
      print(f"{np.mean(max_scores)}")
      mean_max_score = np.mean(max_scores)

      print(summary['chapter_title'], "-", summary['source'], "- time:", round((time.time() - temp_time), 3), "seconds.")

      data.append([mean_max_score, len(unique_sents), summary['chapter_title'], summary['source']])

      unique_used_books.add(summary['chapter_title'])
      summaries_count += 1

      # if summaries_count >= 10:
      #    break
   update_match_counts()
   total_time = (time.time() - start_time)
   print(summaries_count)
   print("time total:", round(total_time, 1), "seconds.", "Average:", (total_time / summaries_count))

   return data, summaries_count, unique_books, unique_used_books


def write_summary_count_to_json(split,filename):

   with open(f"../summary_count/chapter-comparison-counts-{split}-{filename}.json", 'w') as f:

      for source in number_of_matches:
         for book in number_of_matches[source]:
            number_of_matches[source][book]['sections_that_exist_elsewhere'] = list(number_of_matches[source][book]['sections_that_exist_elsewhere'])
      json.dump(number_of_matches, f)


def update_match_counts():
   for source in number_of_matches:
      for book in number_of_matches[source]:
         number_of_matches[source][book]['total_sections_used'] = len(number_of_matches[source][book]['sections_that_exist_elsewhere'])


def write_to_csv(function, split,filename):
   print(filename)
   df = pd.DataFrame(data, columns=[function, "number of Unique sentences", "chapter-title", "source"])
   # Save file.
   df.to_csv(f"../csv_results/booksum_summaries/chapter-comparison-results-{split}-{filename}.csv")


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

   #used getopt for first time to handle arguments, works well but feels messy.
   try:
      opts, args = getopt.getopt(argv,"hf:o:s:",["help","function=", "ofile=","split="])
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

   data, summaries_count, unique_books, unique_used_books = calculate_F1(function)
   result_printout(function)
   # write_to_csv(function, split, outputfile)
   write_summary_count_to_json(split, outputfile)


if __name__ == "__main__":
   # print(sys.argv[1:])
   # sys.exit(1)
   main(sys.argv[1:])

