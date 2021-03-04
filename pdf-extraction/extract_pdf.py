import PyPDF2

pdf = PyPDF2.PdfFileReader('textbook.pdf')

def print_toc(pdf):
    final_titles = []
    def recurse(current_path, outlines):
        print("Current path:", current_path)
        for i, item in enumerate(outlines):
            if isinstance(item, list):
                current_path.append(outlines[i - 1].title)
                recurse(current_path, item)
                current_path.pop()
            else:
                title = current_path + [item.title]
                num = pdf.getDestinationPageNumber(item)
                
                final_titles.append((title, num))
                if item.top or item.bottom or item.left or item.right or item.typ:
                    print("FOUND")
                    print(title)
                    print((item.top, item.bottom, item.left, item.right, item.typ))

    current_path = []
    recurse(current_path, pdf.outlines)
    return final_titles
