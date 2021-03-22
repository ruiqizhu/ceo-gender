'''
This first baseline is from the "Text-Based Mutual Fund Peer Groups: PRELIMINARY DRAFT" 
research article, written by Simona Abis and Anton Lines of Columbia University.

In this article, they use unsupervised machine learning to categorize US active
equity mutual funds into Strategy Peer Groups (SPGs) based on their strategy 
descriptions in prospectuses.

To implement this strategy, we are going to use this unsupervised machine
learning approach to categorize the companies in our dataset according to
their business and risk factorcs outlined in their annual SEC report (2020).

'''
# %%
# Packages
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering,DBSCAN
from scipy.cluster import hierarchy
from sklearn.metrics import silhouette_score
import matplotlib.patheffects as PathEffects
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
import time
import nltk 
# %%

# "In order to make the sections machine readable we first preprocess them using 
# the bag of words approach. This procedure yields for each section a list of 
# all stemmed words and bi-grams."

# %%
dataset_dir = "../data/SEC/2020-readable/"
quarter_dir = sorted(os.listdir(dataset_dir))
all_filings = {}
for qrt in quarter_dir:
    all_filings[qrt] = os.listdir(dataset_dir+qrt+"/")
# %%
def BOW(dataset_directory, qtr_id, filing_idx):# filing_data):
    filing_data = all_filings[qtr_id][filing_idx]

    fname = os.path.join(dataset_directory, qtr_id, filing_data)
    print(fname)
    print(os.listdir(fname))
    business_section_file = os.path.join(fname, "business-section.txt")
    risk_factors_section_file = os.path.join(fname, "risk-factors-section.txt")

    business_section = open(business_section_file,"r+").read()
    risk_factors_section = open(risk_factors_section_file,"r+").read()

    # print(business_section)
    


BOW(dataset_directory=dataset_dir, qtr_id='QTR1', filing_idx=0)

# %%

# %%
