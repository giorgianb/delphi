import csv
from difflib import SequenceMatcher
import sys
import enchant
import nltk
import re
import string
from collections import defaultdict

d = enchant.DictWithPWL("en_US", "ee-vocab.txt")
#PUNCTUATION = set(string.punctuation) | {"–", "”"}


# It is presumed that each chapter starts with its name
# Step 1: Word merging
# Step 1: Filter chapters by name.
# Take advantage of this to find the start of the chapter






def punctify_text(ct):
    groups = []
    current_group = [None, []]
    for c in ct:
        if not c.isalpha() and not c.isspace():
            if current_group != [None, []]:
                groups.append([current_group[0], "".join(current_group[1]).split()])

            current_group = [c, []]
        else:
            current_group[1].append(c)

    groups.append([current_group[0], "".join(current_group[1]).split()])
    return groups


def word_merge_across_punct(ct):
    global seen_merge_candidates
    def is_merge_candidate(text_prev, text):
        if len(text_prev) == 0 or len(text) == 0:
            return False

        prefix = text_prev[-1]
        suffix = text[0]

        return (not d.check(prefix) or not d.check(suffix))

    for i, ((_, text_prev), (p, text)) in enumerate(zip(ct, ct[1:])):
        if p in ("-", "–") and is_merge_candidate(text_prev, text):
            merged = text_prev[-1] + text[0]
            if merged and d.check(merged):
                ct[i][1][-1] = merged
                ct[i + 1][1] = text[1:]

    return ct

def word_merge_within_punct(ct, max_merge_len=2):
    def is_merge_candidate(word_group):
        return any(not d.check(word) for word in word_group)

    for merge_len in range(1, max_merge_len + 1):
        for i, (p, text) in enumerate(ct):
            groups = (text[i:i + merge_len] for i in range(0, len(text), merge_len))
            new_text = []
            for word_group in groups:
                if is_merge_candidate(word_group):
                    merged = ''.join(word_group)
                    if d.check(merged):
                        new_text.append(merged)
                    new_text.extend(word_group)
                else:
                    new_text.extend(word_group)

            ct[i][1] = new_text

    return ct

unknown_words = defaultdict(int)
def update_unknown_word_count(ct, unknown_words):
    for (p, text) in ct:
        for word in text:
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

    for i, (cn, ct) in enumerate(chapters):
        ct = punctify_text(ct)
        ct = word_merge_across_punct(ct)
        ct = word_merge_within_punct(ct)
        update_unknown_word_count(ct, unknown_words)
        #ct = filter_chapter_by_name(cn, ct)
        chapters[i] = (cn, ct)

    print(unknown_words)
    print(len(unknown_words))
    with open(clean_csv_file_path, 'w') as f:
        fout = csv.writer(f)
        fout.writerows(chapters)
