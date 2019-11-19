#!/bin/python3/

"""
process files
"""

import pandas as pd

# paths
META_SET = "/local2/zyli/umls/2018AB_RRF/META/"

def proc_mrcols():
    print("processing mrcols")
    pass

def proc_mrconso():
    df_a = pd.read_csv(META_SET + "MRCONSO.RFF.aa", sep="|", header=None)
    df_b = pd.read_csv(META_SET + "MRCONSO.RFF.ab", sep="|", header=None)
    df_c = pd.read_csv(META_SET + "MRCONSO.RFF.ac", sep="|", header=None)

    # choose ENG

    df_conso = pd.concat([df_a, df_b, df_c]).reset_index()
    df_conso = df_conso[df_conso[1] == "ENG"]

    # * need to know those columns






if __name__ == "__main__":
