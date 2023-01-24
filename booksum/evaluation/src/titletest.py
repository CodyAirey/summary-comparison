import json
import pathlib



#B for book level
b = open(pathlib.Path(f"../../alignments/book-level-summary-alignments/book_summaries_aligned_train.jsonl"),
        encoding='utf-8')

#C for chapter level
c = open(pathlib.Path(f"../../alignments/chapter-level-summary-alignments/fixed_chapter_summaries_train_final.jsonl"),
             encoding='utf-8')

chapter_level_json_titles = set()
book_level_json_titles = set()

ctitles_not_in_book_json = set()
btitles_not_in_chapt_json = set()


# #Do chapter titles not in book level json
# for line in b:
#     content = json.loads(line)
#     book_level_json_titles.add(content['title'])

# for line in c:
#     content = json.loads(line)
#     book_title = content['book_id'].split(".")[0:1][0].replace(",","").replace("!","") #extracts book title as a string
#     if content['book_id'].split(".")[0:3] == ["Dr"," Jekyll and Mr", " Hyde"]:
#         book_title = "Dr. Jekyll and Mr. Hyde"
#     if(book_title not in book_level_json_titles):
#         ctitles_not_in_book_json.add(book_title)
#         # print(book_title)


#Do book titles not in chapter level json
for line in c:
    content = json.loads(line) #add all titles to from chapter level json to a set.
    chapter_level_json_titles.add(content['normalized_title'])

for line in b:
    content = json.loads(line) #if book title not found in chapter level json, add to new set.
    
    new_title = content['title'].lower()

    if new_title not in chapter_level_json_titles:
        btitles_not_in_chapt_json.add(new_title)


# print("CHAPTER titles NOT in the BOOK level JSON")
# for e in ctitles_not_in_book_json:
#     print(e)
# print("==========")
# print("OTEHR WAY ARAOUND")
# print("==========")
print("BOOK titles NOT in the CHAPTER level JSON")
for e in btitles_not_in_chapt_json:
    print(e)