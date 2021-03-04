import csv
from difflib import SequenceMatcher

csv.field_size_limit(int(1e10))
with open('chapters.csv', newline='') as f:
    fin = csv.reader(f, quotechar='"')
    chapters = list(fin)

for row in chapters:
    if len(row) != 2:
        print(row)

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

for i, (cn, ct) in enumerate(chapters):
    ct = filter_chapter_by_name(cn, ct)
    chapters[i] = (cn, ct.replace("_GREEK_ENGINEERING_COMMA_INTERNAL_USAGE_", ","))

with open('chapters-cleaner.csv', 'w') as f:
    fout = csv.writer(f)
    fout.writerows(chapters)



