import csv
from difflib import SequenceMatcher
import sys
import enchant
from nltk.stem import WordNetLemmatizer
import re
import string
from collections import defaultdict

l = WordNetLemmatizer()
d = enchant.DictWithPWL("en_US", "technical-vocab.txt")
PUNCTUATION = set(string.punctuation) | {"–", "”"}
word_counts = {}
total_words = 0
total_count = 0
with open('word-frequency.txt') as fin:
    for line in fin:
        word, count = line.split()
        count = int(count)
        word_counts[word.lower().strip()] = count
        total_count += count
        total_words += 1

def probability(word):
    word = word.strip()
    if word in word_counts:
        return word_counts[word]/total_count
    
    lemmatized_word = l.lemmatize(word)
    if lemmatized_word in word_counts:
        return word_counts[word]/total_count

    return 0

# It is presumed that each chapter starts with its name
# Step 1: Word merging
# Step 1: Filter chapters by name.
# Take advantage of this to find the start of the chapter
def break_text_by_nonalpha(ct):
    groups = []
    current_group = [None, []]
    for c in ct:
        if not c.isalpha():
            if current_group != [None, []]:
                groups.append([current_group[0], "".join(current_group[1])])

            current_group = [c, []]
        else:
            current_group[1].append(c)

    groups.append([current_group[0], "".join(current_group[1])])
    return groups

def group_text_across_nonalpha(ct):
    texts = []
    for p, text in ct:
        if p is not None:
            texts.append(p)
        if text is not None:
            texts.append(text)

    return ''.join(texts)

def is_merge_better(p_text_group, merged):
    if not d.check(merged):
        return False

    non_empty = filter(lambda p_text: p_text[1], p_text_group)
    nonmerged_prob = 1
    for p_text in non_empty:
        nonmerged_prob *= probability(p_text[1].lower().strip())

    merged_prob = probability(merged.lower().strip())
    return nonmerged_prob <= merged_prob


def word_merge_across_nonalpha(ct, max_merge_len=4):
    def is_merge_candidate(p_text_group):
        p_a = len(p_text_group[0][1]) > 0 and len(p_text_group[-1][1]) > 0
        p_b = any(len(text) > 0 for p, text in p_text_group)
        return p_a and p_b

    def merge(group):
        def to_merge_infix(p):
            if p.isspace():
                return ""
            elif p in ("-", "–"):
                return ""
            else:
                return p

        merged = [group[0][1]] # ignore the non-alpha prefix of the first word
        for p, text in group[1:]:
            merged.append(to_merge_infix(p))
            merged.append(text)

        return "".join(merged)

    for merge_len in range(2, max_merge_len + 1):
        groups = (ct[i:i + merge_len] for i in range(len(ct)))
        last_merge = 0
        new_ct = []
        for i, p_text_group in enumerate(groups):
            if i < last_merge:
                continue

            p, text = p_text_group[0]
            merged = merge(p_text_group)
            if is_merge_candidate(p_text_group) and is_merge_better(p_text_group, merged):
                new_ct.append([p, merged])
                last_merge = i + merge_len
            else:
                new_ct.append([p, text])

        ct = new_ct

    return ct


def update_unknown_word_count(ct, unknown_words):
    for (p, word) in ct:
        if word and not d.check(word):
            unknown_words[word] += 1

def filter_chapter_by_name(cn, ct):
    name_match = SequenceMatcher(
            None, 
            cn.lower(), 
            ct.lower(),
            autojunk=False
    ).find_longest_match(
            0,
            len(cn),
            0,
            len(ct)
    )
    if name_match.size > 0.5 * len(cn):
        return ct[name_match.b:]
    else:
        return ct

if __name__ == '__main__':
    raw_csv_file_path = sys.argv[1]
    clean_csv_file_path = sys.argv[2]

    csv.field_size_limit(int(1e10))
    with open(raw_csv_file_path, newline='') as f:
        fin = csv.reader(f, quotechar='"')
        chapters = list(fin)

    unknown_words = defaultdict(int)
    for i, (cn, ct) in enumerate(chapters):
        ct = break_text_by_nonalpha(ct)
        ct = word_merge_across_nonalpha(ct)
        update_unknown_word_count(ct, unknown_words)
        ct = group_text_across_nonalpha(ct)
        ct = filter_chapter_by_name(cn, ct)
        chapters[i] = (cn, ct)
        if (i + 1) % 25 == 0:
            print(f'[Finished Processing up to Chapter {i + 1}]')

    print(unknown_words)
    print(len(unknown_words))
    with open(clean_csv_file_path, 'w') as f:
        fout = csv.writer(f)
        fout.writerows(chapters)
