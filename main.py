# %%
import pandas as pd
import numpy as np
from numpy import random
import os
import os.path
import urllib
import time
import math
import json
import ast
from bs4 import BeautifulSoup
import re
import sys
import nltk
from nltk.corpus import stopwords
#from w3lib.html import remove_tags
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
# import lxml
# from xbrl import XBRLParser, GAAP, GAAPSerializer
# %%
csv_path = "../data/ceo_gender_file.csv"
data_info = pd.read_csv(csv_path)
num_companies = len(data_info)-1
quarters = data_info["Quarter"]
print("Number of companies: ", num_companies)
# %%
all_files = list(data_info["file_path"])
data_folder = "../data/SEC/"
for sec_filing_path in all_files:
    idx = all_files.index(sec_filing_path)
    qtr_id = quarters[idx][-1:]
    sec_filing_path = sec_filing_path.replace("\\", "/")
    sec_filing_id = sec_filing_path[-24:-4]#.replace("2020/QTR\d/", "")
    full_filing_path = os.path.join(data_folder, sec_filing_path)

    temp_html = open(full_filing_path, 'r', encoding = 'utf-8-sig').read()
    try:
        temp_soup = BeautifulSoup(temp_html, 'html.parser') #'lxml')
    except:
        print("File {} Skipped. ERROR IN HTML.".format(sec_filing_id))
        with open("../data/skipped.txt", mode='a+', encoding='utf-8') as f:
            f.write(sec_filing_id)
            f.write("\n")
        continue

    new_folder = data_folder+"2020-readable/QTR"+qtr_id+"/"+sec_filing_id
    # print("new_folder: ", new_folder)
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)


    p_paras = temp_soup.find_all('p')
    span_paras = temp_soup.find_all('span')
    div_paras = temp_soup.find_all('div')

    paras_list = {0:p_paras, 1:span_paras, 2:div_paras}
    best_tag = np.argmax([len(p_paras), len(span_paras), len(div_paras)])
    # print("best_tag: ", best_tag)
    paras = paras_list[best_tag]

    new_readable_file = "{}/{}.txt".format(new_folder,"full-readable")
    contains_TOC = False
    write_to_file = True
    prev_line = None
    with open(new_readable_file, mode='wt', encoding='utf-8') as file:
        for tag in paras:
            new_line = " ".join(tag.text.split()).lower()
            if contains_TOC:
                if set(["part ii", "part 2","part ii.", "part 2."]).intersection({new_line}):
                    # print(set(["part ii", "part 2","part ii.", "part 2."]).intersection({new_line}))
                    write_to_file = True
            else:
                if re.search('table of contents', new_line) is not None \
                or re.search('index', new_line) is not None \
                or re.search('glossary', new_line) is not None:
                    # print("Contains TOC - ", new_line)
                    contains_TOC = True
                    write_to_file = False
            if write_to_file and (prev_line != new_line):
                # print("line: ", new_line)
                prev_line = new_line
                new_line += ' '
                out = new_line.split('. ')
                file.writelines('.\n'.join(out))

    # get "ITEM 1. Business" and "ITEM 1A. Risk Factors"
    business = True
    recording = False
    # your_file = "findalltest.txt"
    b_pattern1 = "^item 1[.:]" #  start of Business Section
    b_pattern2 = '\sitem 1[.:]'
    r_pattern1 = '^item 1a[.:]' # end of Business Section, start of Risk Factors Section
    r_pattern2 = '\sitem 1a[.:]'
    stop_pattern1 = '^item 1b[.:]'# 1([b-z]\W\s+)'' # end of Risk Factors Sections
    stop_pattern2 = '\sitem 1b[.:]'
    business_section = []
    risk_factors_section = []

    for line in open(new_readable_file).readlines():#[180:181]:
        if business is True:
            if recording is False:
                if re.search(b_pattern1, line) is not None or re.search(b_pattern2, line) is not None:
                    # print("Found business", line)
                    recording = True
                    business_section.append(line.strip())
            elif recording is True:
                if re.search(r_pattern1, line) is not None or re.search(r_pattern2, line) is not None:
                    business = False
                    business_section.append(line.strip())
                    risk_factors_section.append(line.strip())
                else:
                    business_section.append(line.strip())
        elif business is False:
            if recording is False:
                if re.search(r_pattern1, line) is not None or re.search(r_pattern2, line) is not None:
                    # print("Found risk factors", line)
                    recording = True
                    risk_factors_section.append(line.strip())
            elif recording is True:
                if re.search(stop_pattern1, line) is not None or re.search(stop_pattern2, line) is not None:
                    # print("hey", line)
                    # recording = False
                    risk_factors_section.append(line.strip())
                    break
                else:
                    risk_factors_section.append(line.strip())

    business_section = " ".join(business_section)
    risk_factors_section = " ".join(risk_factors_section)

    if len(business_section) < 500 or len(risk_factors_section) < 500:
        error_folder = data_folder+"2020-unreadable/QTR"+qtr_id
        # print("error_folder: ", error_folder)
        if not os.path.exists(error_folder):
            os.makedirs(error_folder)
        os.system("mv {}/ {}/".format(new_folder, error_folder))
        new_folder = error_folder+"/"+sec_filing_id

    with open("{}/{}.txt".format(new_folder,"business-section"), 'wt') as filehandle:
        filehandle.write(business_section)
        # for listitem in business_section:#[:-2]:
        #     filehandle.write('%s\n' % listitem)

    with open("{}/{}.txt".format(new_folder,"risk-factors-section"), 'wt') as filehandle:
        filehandle.write(risk_factors_section)
        # for listitem in risk_factors_section:#[:-2]:
        #     filehandle.write('%s\n' % listitem)


    print("{}/{} files completed ({} DONE.)".format(idx+1,num_companies,sec_filing_id))
# %%

# %%

