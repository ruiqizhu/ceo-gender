# %%
import pandas as pd
import os
import re
from wordcloud import WordCloud
# import smart_open
# smart_open.open = smart_open.smart_open
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
import nltk
from nltk.corpus import stopwords
from pprint import pprint
import pyLDAvis.gensim
import pickle 
import pyLDAvis
# %%
csv_path = "../data/ceo_gender_file.csv"
data_info = pd.read_csv(csv_path)
dataset_dir = "../data/SEC/2020-readable/"
# %%
# Female CEO data
female_ceo_df = data_info[data_info['gender'] == 'FEMALE']
female_ceo_file_paths = list(female_ceo_df["file_path"])
female_ceo_qtr = list(female_ceo_df['Quarter'])
female_ceo_list = []
for i in range(len(female_ceo_file_paths)):
    female_ceo_file_paths[i] = female_ceo_file_paths[i].replace("\\", "/")
    female_ceo_list.append(female_ceo_file_paths[i][-24:-4])
# print(female_ceo_file_paths)
# readable_female_ceo_list = []
readable_female_ceo_dict = {'QTR{}'.format(k+1): [] for k in range(4)}
# print(readable_female_ceo_list)
for filing_path in female_ceo_file_paths:
    i = female_ceo_file_paths.index(filing_path)
    if os.path.isdir(os.path.join(dataset_dir,filing_path[5:30])):
        readable_female_ceo_dict[female_ceo_qtr[i]].append(female_ceo_list[i])
print("Total readable female CEO filings: ", sum([len(readable_female_ceo_dict[x]) for x in readable_female_ceo_dict if isinstance(readable_female_ceo_dict[x], list)]), "\n")
# %%
# Male CEO data
male_ceo_df = data_info[data_info['gender'] == 'MALE']
male_ceo_file_paths = list(male_ceo_df["file_path"])
male_ceo_qtr = list(male_ceo_df['Quarter'])
male_ceo_list = []
for i in range(len(male_ceo_file_paths)):
    male_ceo_file_paths[i] = male_ceo_file_paths[i].replace("\\", "/")
    male_ceo_list.append(male_ceo_file_paths[i][-24:-4])
# print(female_ceo_file_paths)
readable_male_ceo_dict = {'QTR{}'.format(k+1): [] for k in range(4)}
for filing_path in male_ceo_file_paths:
    i = male_ceo_file_paths.index(filing_path)
    if os.path.isdir(os.path.join(dataset_dir,filing_path[5:30])):
        readable_male_ceo_dict[male_ceo_qtr[i]].append(male_ceo_list[i])
# print(readable_male_ceo_list)
print("Total readable male CEO filings: ", sum([len(readable_male_ceo_dict[x]) for x in readable_male_ceo_dict if isinstance(readable_male_ceo_dict[x], list)]), "\n")
# %%
def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'part', 'ii'])
    return [[word for word in simple_preprocess(str(doc)) 
             if word not in stop_words and len(word)>1] for doc in texts]

# LDA Topic Modeling algorithm
def lda_func(filings_dict, filings_dict_str, num_topics):
    if os.path.exists(os.path.join(dataset_dir,'.DS_Store')):
        os.system('rm ../data/SEC/2020-readable/.DS_Store')
    for i in range(4):
        if os.path.exists(os.path.join(dataset_dir,'QTR{}'.format(i+1), '.DS_Store')):
            os.system('rm ../data/SEC/2020-readable/QTR{}/.DS_Store'.format(i+1))
    strategy = pd.DataFrame()
    business_data = []
    risk_factors_data = []
    for qrt in filings_dict.keys():
        # print(qrt)
        for filing in filings_dict[qrt]:
            f1 = open(os.path.join(dataset_dir,qrt,filing,"business-section.txt"))#dir + "/" + "full-readable.txt", "r")
            business_data.append(f1.read())
            f2 = open(os.path.join(dataset_dir,qrt,filing,"risk-factors-section.txt"))#dir + "/" + "full-readable.txt", "r")
            risk_factors_data.append(f2.read())
    strategy.insert(0,column = "business_text", value = business_data)
    strategy.insert(1,column = "risk_factors_text", value = risk_factors_data)
    # print(strategy.head())
    # print(len(strategy))
    strategy['business_text'] = strategy['business_text'].map(lambda x: " ".join([re.sub(r"[^a-zA-Z]+", ' ', k) for k in x.split("\n")]))
    strategy['risk_factors_text'] = strategy['risk_factors_text'].map(lambda x: " ".join([re.sub(r"[^a-zA-Z]+", ' ', k) for k in x.split("\n")]))
    long_string1 = ','.join(list(strategy['business_text'].values))
    long_string2 = ','.join(list(strategy['risk_factors_text'].values))
    long_string = long_string1 + long_string2
    
    wordcloud = WordCloud(width=800, height=400,background_color="white", max_words=200, contour_width=3, contour_color='steelblue') # Create a WordCloud object
    wordcloud.generate(long_string) # Generate a word cloud
    wordcloud.to_image() # Visualize the word cloud
    # if not os.path.isfile('./results/{}_wordcloud.png'.format(filings_dict_str)):
    wordcloud.to_file('./results/{}_wordcloud.png'.format(filings_dict_str))

    b_data = strategy.business_text.values.tolist()
    b_data_words = list(sent_to_words(b_data))

    rf_data = strategy.risk_factors_text.values.tolist()
    rf_data_words = list(sent_to_words(rf_data))

    data_words = b_data_words + rf_data_words
    data_words = remove_stopwords(data_words)

    id2word = corpora.Dictionary(data_words) # Create Dictionary
    texts = data_words # Create Corpus
    corpus = [id2word.doc2bow(text) for text in texts] # Term Document Frequency
    # print(corpus[:1][0][:30]) # View
    # print(data_words[:1][0][:30])

    # num_topics = 10# number of topics
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                        id2word=id2word,
                                        num_topics=num_topics) # Build LDA model
    pprint(lda_model.print_topics())# Print the Keyword in the 10 topics
    doc_lda = lda_model[corpus]

    pyLDAvis.enable_notebook()

    LDAvis_data_filepath = os.path.join('./results/ldavis_prepared_'+ filings_dict_str+ '_'+str(num_topics))

    # # this is a bit time consuming - make the if statement True
    # # if you want to execute visualization prep yourself
    if 1 == 1:
        LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
        with open(LDAvis_data_filepath, 'wb') as f:
            pickle.dump(LDAvis_prepared, f)

    # load the pre-prepared pyLDAvis data from disk
    with open(LDAvis_data_filepath, 'rb') as f:
        LDAvis_prepared = pickle.load(f)

    pyLDAvis.save_html(LDAvis_prepared, './results/ldavis_prepared_'+ filings_dict_str+ '_'+str(num_topics) +'.html')
    print("DONE.")
# %%
lda_func(readable_female_ceo_dict, "female_ceo", num_topics=8)
lda_func(readable_male_ceo_dict, "male_ceo", num_topics=8)
quarter_dir = sorted(os.listdir(dataset_dir))
all_filings = {}
for qrt in quarter_dir:
    all_filings[qrt] = os.listdir(dataset_dir+qrt+"/")
lda_func(all_filings, "ALL", num_topics=8)
# %%
filings_dict_str = "female_ceo"
num_topics = 3
LDAvis_data_filepath = os.path.join('./results/ldavis_prepared_'+ filings_dict_str+ '_'+str(num_topics))
with open(LDAvis_data_filepath, 'rb') as f:
        LDAvis_prepared = pickle.load(f)

print(LDAvis_prepared)

# %%
