# %%
import pysentiment2 as ps
import pandas as pd
import re
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
import nltk
from nltk.corpus import stopwords
from statistics import mean
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import unicodedata
import numpy as np
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['DejaVu Sans','Lucida Grande', 'Verdana']

#%%
dataset_dir = "../data/SEC/2020-readable/"
if os.path.exists(os.path.join(dataset_dir,'.DS_Store')):
    os.system('rm ../data/SEC/2020-readable/.DS_Store')
for i in range(4):
    if os.path.exists(os.path.join(dataset_dir,'QTR{}'.format(i+1), '.DS_Store')):
        os.system('rm ../data/SEC/2020-readable/QTR{}/.DS_Store'.format(i+1))
# %%
def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'part', 'ii'])
    return [[word for word in simple_preprocess(str(doc)) 
             if word not in stop_words] for doc in texts]

ADDITIONAL_STOPWORDS = ['covfefe']
nltk.download('wordnet')
def basic_clean(text):
    """
    A simple function to clean up the data. All the words that
    are not designated as a stop word is then lemmatized after
    encoding and basic regex parsing are performed.
    """
    wnl = nltk.stem.WordNetLemmatizer()
    stopwords = stopwords.words('english') + ADDITIONAL_STOPWORDS
    text = (unicodedata.normalize('NFKD', text)
    .encode('ascii', 'ignore')
    .decode('utf-8', 'ignore')
    .lower())
    words = re.sub(r'[^\w\s]', '', text).split()
    return [wnl.lemmatize(word) for word in words if word not in stopwords]

# Positive and Negative are word counts for the words in positive and negative sets.
def get_stats(filing):
    strategy = pd.DataFrame()
    business_data = []
    risk_factors_data = []
    f1 = open(os.path.join(dataset_dir,qrt,filing,"business-section.txt"))#dir + "/" + "full-readable.txt", "r")
    business_data.append(f1.read())
    print(type(f1.read()))
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
    # print(len(long_string))

    b_data = strategy.business_text.values.tolist()
    b_data_words = list(sent_to_words(b_data))

    rf_data = strategy.risk_factors_text.values.tolist()
    rf_data_words = list(sent_to_words(rf_data))

    data_words = b_data_words + rf_data_words
    # print(len(data_words[0]))
    # data_words = remove_stopwords(data_words)

    hiv4 = ps.HIV4()
    tokens = hiv4.tokenize(long_string)
    score = hiv4.get_score(tokens)

    num_words = len(data_words[0])
    num_unique_words = len(set(data_words[0]))
    num_tokens = len(tokens)
    num_pos_words = score['Positive']
    num_neg_words = score['Negative']
    polarity = score['Polarity']
    subjectivity = score['Subjectivity']


    # lm = ps.LM()
    # tokens = lm.tokenize(long_string)
    # score = lm.get_score(tokens)

    # print("score: ", score)
    return num_words,num_unique_words,num_tokens,num_pos_words,\
        num_neg_words,polarity,subjectivity
    
# %%
# Get female CEO stats
csv_path = "../data/ceo_gender_file.csv"
data_info = pd.read_csv(csv_path)
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

fem_stats_dict = {"num_words":[],
                "num_unique_words":[],
                "num_tokens": [],
                "num_pos_words": [],
                "num_neg_words": [],
                "polarity": [],
                "subjectivity": []}
idx = 1
for qrt in readable_female_ceo_dict.keys():
    for filing in readable_female_ceo_dict[qrt]:
        num_words,num_unique_words,num_tokens,num_pos_words,num_neg_words,polarity,subjectivity = get_stats(filing)
        fem_stats_dict["num_words"].append(num_words)
        fem_stats_dict["num_unique_words"].append(num_unique_words)
        fem_stats_dict["num_tokens"].append(num_tokens)
        fem_stats_dict["num_pos_words"].append(num_pos_words)
        fem_stats_dict["num_neg_words"].append(num_neg_words)
        fem_stats_dict["polarity"].append(polarity)
        fem_stats_dict["subjectivity"].append(subjectivity)
        print("{}/{} complete".format(idx,sum([len(readable_female_ceo_dict[x]) for x in readable_female_ceo_dict if isinstance(readable_female_ceo_dict[x], list)])))
        idx+=1
fem_stats_df = pd.DataFrame(fem_stats_dict, columns= fem_stats_dict.keys())

# fem_stats_df.to_csv('./results/fem_ceo_data_stats1.csv', index = False, header=True)

 # %%
# Get male CEO stats
csv_path = "../data/ceo_gender_file.csv"
data_info = pd.read_csv(csv_path)
male_ceo_df = data_info[data_info['gender'] == 'MALE']
male_ceo_file_paths = list(male_ceo_df["file_path"])
male_ceo_qtr = list(male_ceo_df['Quarter'])
male_ceo_list = []
for i in range(len(male_ceo_file_paths)):
    male_ceo_file_paths[i] = male_ceo_file_paths[i].replace("\\", "/")
    male_ceo_list.append(male_ceo_file_paths[i][-24:-4])
# print(male_ceo_file_paths)
# readable_male_ceo_list = []
readable_male_ceo_dict = {'QTR{}'.format(k+1): [] for k in range(4)}
# print(readable_male_ceo_list)
for filing_path in male_ceo_file_paths:
    i = male_ceo_file_paths.index(filing_path)
    if os.path.isdir(os.path.join(dataset_dir,filing_path[5:30])):
        readable_male_ceo_dict[male_ceo_qtr[i]].append(male_ceo_list[i])

male_stats_dict = {"num_words":[],
                "num_unique_words":[],
                "num_tokens": [],
                "num_pos_words": [],
                "num_neg_words": [],
                "polarity": [],
                "subjectivity": []}
idx = 1
for qrt in readable_male_ceo_dict.keys():
    for filing in readable_male_ceo_dict[qrt]:
        num_words,num_unique_words,num_tokens,num_pos_words,num_neg_words,polarity,subjectivity = get_stats(filing)
        male_stats_dict["num_words"].append(num_words)
        male_stats_dict["num_unique_words"].append(num_unique_words)
        male_stats_dict["num_tokens"].append(num_tokens)
        male_stats_dict["num_pos_words"].append(num_pos_words)
        male_stats_dict["num_neg_words"].append(num_neg_words)
        male_stats_dict["polarity"].append(polarity)
        male_stats_dict["subjectivity"].append(subjectivity)
        print("{}/{} complete".format(idx,sum([len(readable_male_ceo_dict[x]) for x in readable_male_ceo_dict if isinstance(readable_male_ceo_dict[x], list)])))
        idx+=1
male_stats_df = pd.DataFrame(male_stats_dict, columns= male_stats_dict.keys())

male_stats_df.to_csv('./results/male_ceo_data_stats.csv', index = False, header=True)

# %%
for col in fem_stats_df.columns:
    print(mean(list(fem_stats_df[col])))
print(" ")
for col in male_stats_df.columns:
    print(mean(list(male_stats_df[col])))
#%%
all_filings_gender = data_info['gender']
all_filings_file_paths = list(data_info["file_path"])
all_filings_qtr = list(data_info['Quarter'])
all_filings_list = []
for i in range(len(all_filings_file_paths)):
    all_filings_file_paths[i] = all_filings_file_paths[i].replace("\\", "/")
    all_filings_list.append(all_filings_file_paths[i][-24:-4])
# print(all_filings_list)
quarter_dir = sorted(os.listdir(dataset_dir))
all_filings = {}
for qrt in quarter_dir:
    all_filings[qrt] = os.listdir(dataset_dir+qrt+"/")
# print(all_filings)
# %%
def clustering(filing_dict,true_k):
    strategy = pd.DataFrame()
    filing_data = []
    for qrt in filing_dict.keys():
        for filing in filing_dict[qrt]:
            f1 = open(os.path.join(dataset_dir,qrt,filing,"business-section.txt"))#dir + "/" + "full-readable.txt", "r")
            f2 = open(os.path.join(dataset_dir,qrt,filing,"risk-factors-section.txt"))#dir + "/" + "full-readable.txt", "r")
            filing_data.append(f1.read()+f2.read())
    print("Total num data points: ", len(filing_data))

    vectorizer = TfidfVectorizer(stop_words={'english'})
    X = vectorizer.fit_transform(filing_data)

    ## UNCOMMENT LINES BELOW TO FIND OPTIMAL K VALUE FOR K-MEANS
    # Sum_of_squared_distances = []
    # sil = []
    # K = range(2,11)
    # for k in K:
    #     km = KMeans(n_clusters=k, max_iter=300, n_init=10)
    #     km = km.fit(X)
    #     Sum_of_squared_distances.append(km.inertia_)
    #     labels = km.labels_
    #     sil.append(silhouette_score(X, labels, metric = 'euclidean'))
    # plt.plot(K, Sum_of_squared_distances, 'bx-')
    # plt.xlabel('k')
    # plt.ylabel('Inertia')
    # plt.title('Elbow Method For Optimal k')
    # plt.savefig("./results/kmeans/elbow-method3.png",dpi=300)
    # plt.show()
    # plt.plot(K, sil, 'bx-')
    # plt.xlabel('k')
    # plt.ylabel('Silhouette Score')
    # plt.title('Silhouette Method For Optimal k')
    # plt.savefig("./results/kmeans/sil-method3.png",dpi=300)
    # plt.show()
    # plt.show()

    
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=500, n_init=10)
    model.fit(X)
    labels=model.labels_
    wiki_cl=pd.DataFrame(list(zip(all_filings_list,all_filings_gender,labels)),columns=['company_id','gender','cluster'])
    wiki_cl.to_csv('./results/kmeans/kmeans_{}.csv'.format(true_k), index = False, header=True)

    # UNCOMMENT LINES BELOW TO PRINT CLUSTER BAR GRAPH
    male_counts = []
    female_counts = []
    for i in range(true_k):
        male_counts.append(sum(wiki_cl[wiki_cl["gender"]=='MALE']['cluster'] == i))
        female_counts.append(sum(wiki_cl[wiki_cl["gender"]=='FEMALE']['cluster'] == i))

    x = np.arange(true_k)  # the label locations
    width = 0.5  # the width of the bars

    fig, ax = plt.subplots(1,1, figsize=(8,8))
    rects1 = ax.bar(x - width/2, male_counts, width, color='b', label='Male CEOs')
    rects2 = ax.bar(x + width/2, female_counts, width, color='r', label='Female CEOs')

    ax.set_ylabel('# of Companies',fontsize=18)
    ax.set_title('K-Means Clustering Result for K={}'.format(true_k),fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    ax.set_xlabel('Clusters',fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend(fontsize=18)
    
    for rects in [rects1,rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',fontsize=18)
    fig.tight_layout()
    plt.savefig("./results/kmeans/{}cluster-bar-graph.png".format(true_k),bbox_inches = "tight",dpi=300)
    plt.show()
    #

    result={'cluster':labels,'wiki':filing_data}
    result=pd.DataFrame(result)
    for k in range(0,true_k):
        s=result[result.cluster==k]
        text=s['wiki'].str.cat(sep=' ')
        text=' '.join([word for word in text.split()])
        wordcloud = WordCloud(width=800, height=400,max_words=200, background_color="white").generate(text)
        print('Cluster: {}'.format(k))
        titles=wiki_cl[wiki_cl.cluster==k][['company_id','gender']]
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        wordcloud.to_file('./results/kmeans/wordcloud_k{}_cluster{}.png'.format(true_k, k))
        plt.axis("off")
        # plt.show()
clustering(all_filings,true_k=5)
# %%
male_counts = []
female_counts
for i in range(8):
    male_counts.append(sum(wiki_cl[wiki_cl["gender"]=='MALE']['cluster'] == i))
    female_counts.append(sum(wiki_cl[wiki_cl["gender"]=='FEMALE']['cluster'] == i))
# sum(wiki_cl[wiki_cl["gender"]=='MALE']['cluster'] == i)
# %%
def get_ngrams(filing_dict, name):
    strategy = pd.DataFrame()
    filing_data = []
    for qrt in filing_dict.keys():
        for filing in filing_dict[qrt]:
            f1 = open(os.path.join(dataset_dir,qrt,filing,"business-section.txt"))#dir + "/" + "full-readable.txt", "r")
            f2 = open(os.path.join(dataset_dir,qrt,filing,"risk-factors-section.txt"))#dir + "/" + "full-readable.txt", "r")
            filing_data.append((f1.read()+f2.read()).split(" "))

    strategy = pd.DataFrame()
    business_data = []
    risk_factors_data = []
    f1 = open(os.path.join(dataset_dir,qrt,filing,"business-section.txt"))#dir + "/" + "full-readable.txt", "r")
    business_data.append(f1.read())

    f2 = open(os.path.join(dataset_dir,qrt,filing,"risk-factors-section.txt"))#dir + "/" + "full-readable.txt", "r")
    risk_factors_data.append(f2.read())
    strategy.insert(0,column = "business_text", value = business_data)
    strategy.insert(1,column = "risk_factors_text", value = risk_factors_data)

    strategy['business_text'] = strategy['business_text'].map(lambda x: " ".join([re.sub(r"[^a-zA-Z]+", ' ', k) for k in x.split("\n")]))
    strategy['risk_factors_text'] = strategy['risk_factors_text'].map(lambda x: " ".join([re.sub(r"[^a-zA-Z]+", ' ', k) for k in x.split("\n")]))
    text1 = list(strategy['business_text'].values)[0].split(" ")
    text2 = list(strategy['risk_factors_text'].values)[0].split(" ")
    text = text1 + text2
    wnl = nltk.stem.WordNetLemmatizer()
    stopwords = nltk.corpus.stopwords.words('english') + ADDITIONAL_STOPWORDS
    words = [wnl.lemmatize(word) for word in text if word not in stopwords and len(word)>1]

    bigram = (pd.Series(nltk.ngrams(words, 2)).value_counts())[:15]
    print(sum(bigram))
    # bigram.sort_values().plot.barh(color='darkgreen', width=.9, figsize=(12, 8),fontsize=18)
    # plt.title('15 Most Frequently Occuring Bigrams in {} Data Set'.format(name),fontsize=20)
    # plt.ylabel('Bigrams',fontsize=18)
    # plt.xlabel('# of Occurances',fontsize=18)
    # plt.savefig("./results/bigram_{}.png".format(name), bbox_inches = "tight", dpi=300)
    # plt.show()

    # trigram = (pd.Series(nltk.ngrams(words, 3)).value_counts())[:15]
    # trigram.sort_values().plot.barh(color='darkgreen', width=.9, figsize=(12, 8),fontsize=18)
    # plt.title('15 Most Frequently Occuring Trigrams in {} Data Set'.format(name),fontsize=20)
    # plt.ylabel('Trigrams',fontsize=18)
    # plt.xlabel('# of Occurances',fontsize=18)
    # plt.savefig("./results/trigram_{}.png".format(name), bbox_inches = "tight", dpi=300)
    # plt.show()
get_ngrams(readable_male_ceo_dict, "Female CEO")
# %%