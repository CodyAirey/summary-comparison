import pathlib
import json
import re
import sys
from dataclasses import dataclass
import pandas as pd

data_type = sys.argv[1]

@dataclass
class Book:
    title: str

@dataclass
class Original_Book_Variation:
    source_book: Book
    book_path: str
    book_word_count: int


@dataclass
class Section:
    section_title: str

@dataclass
class Original_Section_Variation:
    chapter_path: str
    section_word_count: int

@dataclass
class Section_Summary:
    summary_path: str
    section_sum_word_count: int


books = pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)
original_book_variations = pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)
book_summaries = pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)

sections = pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)
original_section_variations = pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)
section_summaries = pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)


book_objs = set()
original_book_variation_objs = set()


book_titles = set()
book_paths = set()
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
        books.append(pd.DataFrame(data=[summary['normalized_title']], columns=["Book Title"]))


    if summary['book_path'] not in book_paths:

        # corrolated_book = [book for book in book_objs if book.title == summary['normalized_title']]

        # current_original_book_var = Original_Book_Variation(summary['book_path'], 0, corrolated_book)
        # original_book_variation_objs.add(current_book)
        book_paths.add(summary['book_path'])

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


# books = pd.DataFrame(data=book_titles, columns=["Book Title"])
print(books)
