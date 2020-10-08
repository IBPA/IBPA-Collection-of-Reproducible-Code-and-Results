#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import csv
from omics.db import GEOFetcher
if sys.version_info < (3, 0):
    from omics.utils.unicode_handler import UnicodeWriter
    

def filter_homo_sapiens(metadata):
    """MVP usage"""

    return [row for row in metadata if 'Homo sapiens' in row[-2]]

def filter_transcriptomics(metadata):
    """MVP usage"""

    filtered_metadata = []
    for row in metadata:
        is_trans = False
        for exp_type in row[-1]:
            if 'Expression' in exp_type:
                is_trans = True
                break
        if is_trans:
            filtered_metadata.append(row)
    return filtered_metadata

def filter_(metadata):
    """MVP usage"""

    metadata = filter_homo_sapiens(metadata)
    metadata = filter_transcriptomics(metadata)
    return metadata

def generate_table_template(metadata, term, output_file):

    id_ = 1
    for row in metadata:
        row.insert(0, id_)
        id_ += 1

    if sys.version_info < (3, 0):
        with open(output_file, 'w') as f:
            writer = UnicodeWriter(f)
            writer.writerow(['id', 'source', 'accession', 'title', 'description', 'pmid',
                'pub_title', 'pub_date', 'pub_author', 'tech_acc', 'tech_name',
                'sample_acc', 'species', 'exp_type'])
            writer.writerows(metadata)
    else:
        with open(output_file, 'w', encoding='utf-8', newline = '') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'source', 'accession', 'title', 'description', 'pmid',
                'pub_title', 'pub_date', 'pub_author', 'tech_acc', 'tech_name',
                'sample_acc', 'species', 'exp_type'])
            writer.writerows(metadata)
        
    

def main():

    try:
        term = sys.argv[1]
    except:
        raise SyntaxError("must need a string")
        
    try:
        output_file = sys.argv[2]
    except:
        output_file = term + "_output.csv"
        print('Use default output file name:' + term + "_output.csv")

    email      = "fzli@ucdavis.edu"
    api_key    = "f156c113042e581f61325f32b084381ebb07"

    f_geo = GEOFetcher(email, api_key)
    metadata = f_geo.fetch(term)
    metadata = filter_(metadata)
    generate_table_template(metadata, term, output_file)