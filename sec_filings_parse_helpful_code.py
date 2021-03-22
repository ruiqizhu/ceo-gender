import urllib.request
import pandas as pd
import numpy as np
from numpy import random
import os
import os.path
import urllib
import requests
import time
import math
import json
import ast
from bs4 import BeautifulSoup


################# TO HEYLEY #############################################
# Most of this code is probably not relavent to our project. I will highlight the part
# that I think might be helpful.


# create a dictionary to store the cleaned balance sheets
Balance_Sheet_dict = {}

count = 1 # create a count starting from 1
list_save_point = [i*10 for i in range(1,122)] # set the save point alone the progress
for row in list_files_10q10k.index[:20]:
    
    # create the corresponding text file path, which contains SEC 10Q or 10K filing
    file_path = os.path.join(path, list_files_10q10k.loc[row, 'file_path'].replace('\\', '/'))
    
    try: 
    ########################################### USEFUL PART ##########################################
        # read the corresponding text file containing SEC 10Q or 10K filing
        temp_html = open(file_path, 'rb')
    
        # uncode the text file into html by BeautifulSoup for later scrapping
        temp_soup = BeautifulSoup(temp_html, 'html5lib')
        
        # locate the 10Q 10K document in the text file (the .txt contains all the other supporting documents,
        # eg. certification letter, etc. ; we only need 10Q or 10K)
        temp_10q10k_document = temp_soup.find('document')

    ########################################### USEFUL PART ##########################################
        # let the pandas read the html and raw scrap all the table-like blocks
        df = pd.read_html(str(temp_10q10k_document))

    except:
        print ('Warning: Unable to Read Data')
        pass
        
    # locate the html block that contains balance sheet table
    Balance_Sheet = find_balance_sheet(df)

    
    try:
        # cleaning the raw balance sheet found by "find_balance_sheet" function to keep only the values for current period
        Cleaned_Balance_Sheet = clean_balance_sheet(Balance_Sheet)
    
        # store the cleaned balance sheet and convert the Balance Sheet to json format
        Balance_Sheet_dict[list_files_10q10k.loc[row, 'file_path']] = Cleaned_Balance_Sheet.to_json()
        
    except:
        print ('Warning: Unable to Clean!')
        pass
    
    # auto save the file for every 50 files processed
    if count in list_save_point:
        # save the dictionary to a temporary file
        temp_save_txt = Balance_Sheet_dict
        # write the temp file to the system (local drive)
        with open('SEC_balance_sheet_temp.txt', 'w') as f:
            print(temp_save_txt, file=f)
        
        print ('Saved!')
    
    # count the number of files have been processed
    count = count + 1
    print (row)





######################################## Processing Text ##########################################
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize

def get_10k10q_txt(file_path):

    # read the corresponding text file containing SEC 10Q or 10K filing
    temp_html = open(file_path, 'r', encoding = 'utf-8-sig').read()

    # uncode the text file into html by BeautifulSoup for later scrapping
    temp_soup = BeautifulSoup(temp_html, 'html5lib')

    # locate the 10Q 10K document in the text file (the .txt contains all the other supporting documents,
    # eg. certification letter, etc. ; we only need 10Q or 10K)
    temp_10q10k_document = temp_soup.find('document')

    # convert soup html to plain text
    temp_10q10k_document_txt = temp_10q10k_document.text
    
    return temp_10q10k_document_txt

def clean_txt(txt):
    # convert to lower capital
    txt = txt.lower()

    # remove all digits
    txt = remove_digits_html(txt)

    # tokenize the txt and remove double spaces and next line
    txt_tokens = word_tokenize(txt.replace('\xa0', ' ').replace('\n', ' '))
#     print (len(txt_tokens), txt_tokens.count('maryland'))

    # remove stop words
    txt_tokens = [word for word in txt_tokens if word not in stop_words]
#     print (len(txt_tokens), txt_tokens.count('maryland'))

    # remove special characters and punctuations
    txt_tokens = [word for word in txt_tokens if not word in punct]
#     print (len(txt_tokens), txt_tokens.count('maryland'))
    
    return txt_tokens


######################################## Processing Text - End ##########################################