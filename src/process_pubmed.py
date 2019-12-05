"""
    Process PubMed text using package `pubmed_parser` online

    Author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>
"""

import os
import sys

import pubmed_parser as pp

OUTPUT_DIR = "../medline_output/"


def make_dirs(path):
    """recursively make directories"""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def parse_single_doc(f):
    """parse single documents in medline"""
    # set file path
    file_name = "pubmed19n{:04d}.xml.gz".format(f)
    file_name = "../MEDLINE/" + file_name

    # dicts_out is a list of dictionary
    dicts_out = pp.parse_medline_xml(file_name,
                                     year_info_only=False, nlm_category=False,
                                     author_list=False, reference_list=False)

    # load abstracts that are non-empty
    texts = []
    for dict_ in dicts_out:
        abs_text = dict_['abstract']
        if len(abs_text) > 0:
            texts.append(abs_text.strip())

    return texts


def parse_medline(start, end):
    """parse medline documents from start to end"""
    # loop from start to end
    for f in range(start, end + 1):
        out_file = OUTPUT_DIR + "pubmed{:04d}.txt".format(f)
        if os.path.isfile(out_file):
            print("\t\t{} already exists, pass ...".format(out_file))
            continue

        print("\t\tparsing document {}".format(f))
        text = parse_single_doc(f)

        with open(out_file, "w") as fout:
            for text_piece in text:
                print(text_piece, file=fout)
        print("\t\tsaved at {}pubmed{:04d}.txt".format(OUTPUT_DIR, f))


if __name__ == "__main__":
    if len(sys.argv) < 2 + 1:
        print("python {} [start] [end]".format(sys.argv[0]))
        raise ValueError("Invalid input params")

    start = int(sys.argv[1])
    end = int(sys.argv[2])

    make_dirs(OUTPUT_DIR)

    if start < 0 or start > end or end > 972:
        raise ValueError("Both start and end should be in 1 to 972")

    print("\tstarting from {} and ending at {}".format(start, end))
    parse_medline(start=start, end=end)

    print("\tdone!")

