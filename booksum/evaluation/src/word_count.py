import pathlib
import json
import re
import sys
from dataclasses import dataclass
import pandas as pd
from nltk.tokenize import word_tokenize
import string

data_type = sys.argv[1]

# @dataclass
# class Book:
#     title: str

@dataclass
class Original_Book_Variation:
    book_title: str
    book_path: str
    book_word_count: int

    def as_dict(self):
        return {
            'Book Title': self.book_title,
            'Book Word Count': self.book_word_count,
            'Book Path': self.book_path
            }


# @dataclass
# class Section:
#     section_title: str

# @dataclass
# class Original_Section_Variation:
#     chapter_path: str
#     section_word_count: int

# @dataclass
# class Section_Summary:
#     summary_path: str
#     section_sum_word_count: int


books = pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)
original_book_variations = pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)
book_summaries = pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)

sections = pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)
original_section_variations = pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)
section_summaries = pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)

obvs = []

book_titles = set()
book_paths = {}
book_summary_paths = set()

section_titles = set()
section_paths = set()
section_summary_paths = set()



book_file = open(pathlib.Path(f"../../alignments/book-level-summary-alignments/fixed_book_summaries_{data_type}_final.jsonl"),
            encoding='utf-8')

chapter_file = open(pathlib.Path(f"../../alignments/chapter-level-summary-alignments/fixed_chapter_summaries_{data_type}_final.jsonl"),
            encoding='utf-8')


for line in book_file:
    summary = json.loads(line)

    if summary['normalized_title'] not in book_titles:
        # book_objs.add(current_book)
        book_titles.add(summary['normalized_title'])

    if summary['book_path'] not in book_paths:
        # book_paths.add(summary['book_path'])
        book_paths[summary['book_path']] = summary['normalized_title']

    if summary['summary_path'] not in book_summary_paths:
        book_summary_paths.add(summary['summary_path'])

for line in chapter_file:
    summary = json.loads(line)

    if summary['normalized_title'] not in section_titles:
        section_titles.add(summary['normalized_title'])

    if summary['chapter_path'] not in section_paths:
        section_paths.add(summary['chapter_path'])

    if summary['summary_path'] not in section_summary_paths:
        section_summary_paths.add(summary['summary_path'])


books = pd.DataFrame(data=book_titles, columns=["Book Title"])

for path in book_paths:

    with open("../../" + path, 'r') as f:
        data = f.read()
        tokens = word_tokenize(data)
        words = [''.join(letter for letter in word if letter not in string.punctuation) for word in tokens]
        words = list(filter(None, words))
        word_count = len(words)

        obvs.append( Original_Book_Variation(book_paths[path], path, word_count))

print(books)

print("_----")



original_book_variations = pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)

print()
