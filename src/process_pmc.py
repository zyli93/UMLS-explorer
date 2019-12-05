"""
    Process PubMed Center (PMC) text using package `pubmed_parser` online

    Author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>

    Notes:
        - note 1
"""

import os
import sys
import string
from collections import defaultdict
from datetime import datetime

import pubmed_parser as pp

INPUT_DIR = "../PMC/"
OUTPUT_DIR = "../pmc_output/"

EMPTY_TITLE = "<Title>"
START_TOKEN = ">>>>>>"

# VAL_INIT = "0-9A-Z", valid initials
VAL_INIT = [str(x) for x in list(range(0, 10))]
VAL_INIT += list(string.ascii_uppercase)


def make_dirs(path):
    """recursively making directories"""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def parse_folders_by_initial(first_char, ini_dir_dict):
    """parse folders by the first character of the folder"""
    all_dirs_char = ini_dir_dict[first_char]

    # time 
    time_string = datetime.now().strftime("%m%d-%H:%M")

    # logging file
    logger = open(OUTPUT_DIR + "logs_{}_{}".format(first_char, time_string), "w")

    # counters = 0
    total, skew = 0, 0

    # loop through all directories
    for subdir in all_dirs_char:
        print("\t\tprocessing directory {} ...".format(subdir))
        print("processing directory {} ...".format(subdir), file=logger)

        with open(OUTPUT_DIR + subdir + ".txt", "w") as fout:
            subdir = INPUT_DIR + subdir + "/"
            subdir_nxml_files = [x for x in os.listdir(subdir) if x[-5:] == ".nxml"] 
            for nxml_file in subdir_nxml_files:
                print("\tprocessing nxml file {} ...".format(nxml_file), file=logger)
                total += 1
                text = parse_single_nxml(subdir + nxml_file, nxml_file)
                if text:
                    print(text, file=fout)
                else:
                    skew += 1
                    print("\tskipping {} due to no abstract".format(nxml_file), file=logger)
    print("first char {}, total {}, skewed {}".format(first_char, total, skew), file=logger)
    logger.close()


def parse_single_nxml(fpath, fname):
    """parse a particular file in the path

    Args:
        fpath - the whole path of the file
        fname - the file name

    Return:
        text - a text formulated in a special format
    """

    fname = fname[:-5]  # remove extension
    parsed_dict = pp.parse_pubmed_xml(fpath)

    abstract = parsed_dict["abstract"]
    if len(abstract) == 0:
        return None

    title = parsed_dict["full_title"] if len(parsed_dict["full_title"]) > 1 else EMPTY_TITLE

    text = START_TOKEN + "{}\n".format(fname)
    text += "{}\n".format(title.strip())
    text += abstract + "\n"

    return text


def build_folders_dict():
    """build dictionary of Initials (key) - Folders (values)"""
    all_folders = [x for x in os.listdir(INPUT_DIR) if os.path.isdir(INPUT_DIR + x)]
    folder_dict = defaultdict(list)
    for folder in all_folders:
        folder_initial = folder[0].upper()
        assert folder_initial in VAL_INIT, "folder initial not in VAL_INIT {}".format(folder_initial)
        folder_dict[folder_initial].append(folder)

    return folder_dict


if __name__ == "__main__":
    if len(sys.argv) < 2 + 1:
        print("python {} [first char start] [first char end]")
        raise ValueError("Invalid input arguments")

    fc_start, fc_end = sys.argv[1], sys.argv[2]
    fc_start, fc_end = fc_start.upper(), fc_end.upper()

    assert len(fc_start) == 1, "invalid start char length"
    assert len(fc_end) == 1, "invalid end char length"
    assert fc_start in VAL_INIT and fc_end in VAL_INIT, "invalid input - not in `0-9 or A-Z`"

    # make output directory:
    make_dirs(OUTPUT_DIR)

    # print out process range
    stard_ind, end_ind  = VAL_INIT.index(fc_start), VAL_INIT.index(fc_end)
    print("processing folders starts with {}"
            .format("".join(VAL_INIT[stard_ind : end_ind + 1])))

    # print initial-folder dictionary generation 
    print("creating initial-folder dictionary ...")
    ini_dir_dict = build_folders_dict()

    for i in range(stard_ind, end_ind + 1):
        print("\tprocessing {} ...".format(VAL_INIT[i]))
        parse_folders_by_initial(VAL_INIT[i], ini_dir_dict)

    print("Done!")
