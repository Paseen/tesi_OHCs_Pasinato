#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer
from nltk import WhitespaceTokenizer, WordPunctTokenizer
import spacy
import string
from nltk.corpus import stopwords

def sistema_tc_parentesi(row):
    if not row.id.startswith("tc"):
        return row
    else:
        if row.text.startswith("["):
            row.text = row.text[1:]
    return row

def create_col_plot(df):
    data = df.groupby(pd.Grouper(key='time', freq='M')).size()
    data = data.reset_index()
    data["month"] = data.time.dt.month
    data["year"] = data.time.dt.year
    data["x_tick"] = ""
    i = 0
    for elem in zip(data["year"], data["month"]):
        data.loc[i, "x_tick"] = f"{elem[0]}, {elem[1]}"
        i+=1
    return data

def statistics(df, title = False):
    if title:
        mean_len_title = round(df.title.str.len().mean(),2)
        std_title = round(df.title.str.len().std(),2)
    mean_len_text = round(df.text.str.len().mean(),2)
    std_text = round(df.text.str.len().std(),2)
    mean_likes = round(df.likes.mean(),2)
    std_likes = round(df.likes.std(),2)
    if title:
        print(f"la lunghezza media del titolo è: {mean_len_title} con deviazione standard {std_title} \nla lunghezza media dei post è: {mean_len_text} con deviazione standard {std_text} \n la media dei likes è {mean_likes} con std di {std_likes}")
    else:
        print(f"La lunghezza media dei commenti è: {mean_len_text} con deviazione standard {std_text} \nla media dei likes è {mean_likes} con std di {std_likes}")
    return

def count_type_comments(df):
    tc = df.id.str.startswith("tc").sum()
    s = df.id.str.startswith("s").sum()
    f = df.id.str.startswith("f").sum()
    print(f"top comments: {tc}       sub comments: {s}        full indent: {f}")
    return tc, s,f 

def max_min_date(df):
    max = df["time"].max()
    min = df["time"].min()
    print(f"The dataframe contains data which range from {min} to {max}")
    return

def n_authors(df):
    n = len(set(df.author))
    print(f"The dataframe contains {n} authors")
    return



## Preprocessing for text analysis


def remove_newlines_tabs(text):
    formatted = text.replace('\\n', ' ').replace('\n', ' ').replace('\t',' ').replace('\\', ' ').replace('. com', '.com')
    return formatted

def remove_whitespaces(text):
    return " ".join(text.split())

def remove_links(text):
    https = re.sub(r'http\S+', ' ', text)
    com = re.sub(r"www\.[A-Za-z]*\.com", " ", https)
    return com

def remove_com(text):
    c = re.sub(r'[A-Za-z]*\.com', ' ', text)
    return c

def remove_stopwords(text):
    null_value = ['ns', 'na']
    new_stop = ['im', 'also', 'ive', 'well', 'hi', 'goodmorning', 'usual', 'give', 'get', 'do', 'hav', 'have', 've', 'be', 'm', 's', 
    'take', 'might', 'end', 'sometim', 'still', 'got', 'realli', 'go', 'even', 'much','bit', 'make', 'alway', 'mani', 'lot', 'ye', 'use', 'one', 'time', 'x']
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.extend(null_value)
    stopwords.extend(new_stop)
    return ' '.join([w for w in text if w not in stopwords])


def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r' ', string)


def remove_punctuation(text):
    punct = list(string.punctuation)
    punct.append(['€','¯', "’"])
    punctuationfree="".join([w for w in text if w not in punct])
    return punctuationfree

def removing_character_repetition(text):
    pattern = re.compile(r"([A-Za-z])\1{1,}", re.DOTALL) #matching for all case alphabets
    formatted = pattern.sub(r"\1\1", text) #limits to two characters
    return formatted

def tokenizer(text):
    tokens = WordPunctTokenizer().tokenize(text)
    return tokens

def lemmatizer(text):
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text

def lemmatize(df):
    info = dict()
    docs = list(nlp.pipe(list(df["text_processed"]), batch_size=1000))
    i = 0
    for doc in docs:
        lemmas = list()
        for token in doc: 
                if ((token.is_stop == False) and (token.is_punct == False)) and (token.pos_ not in ['PRON', 'AUX',  'ADP' ]):
                    lemmas.append(token.lemma_)
        info[i] = lemmas
        i += 1
        
    return pd.DataFrame(info.items(), columns = ["a", "text_lemmatised"]).drop("a", axis = 1)

def get_lemmas(text):
    '''Used to lemmatize the processed tweets'''
    lemmas = []
    nlp = spacy.load('en_core_web_sm', disable = 'ner')
    doc = nlp(text)
    
    # Something goes here :P
    for token in doc: 
        if ((token.is_stop == False) and (token.is_punct == False)) and (token.pos_ not in ['PRON', 'AUX',  'ADP' ]):
            lemmas.append(token.lemma_)
    
    return lemmas

def lemmatize(df):
    info = dict()
    nlp = spacy.load('en_core_web_sm', disable = 'ner')
    docs = list(nlp.pipe(list(df["text_processed"]), batch_size=1000))
    i = 0
    for doc in docs:
        lemmas = list()
        for token in doc: 
                if ((token.is_stop == False) and (token.is_punct == False)) and (token.pos_ not in ['PRON', 'AUX',  'ADP' ]):
                    lemmas.append(token.lemma_)
        info[i] = lemmas
        i += 1
#     pd.DataFrame(info.items(), columns = ["0", "lemmatised"]).drop("0", axis = 1)
    return pd.DataFrame(info.items(), columns = ["a", "text_lemmatised"]).drop("a", axis = 1)

def stemming_spacy(text):
    nlp = spacy.load('en_core_web_sm')
    list=[]
    doc = nlp(text)
    for word in doc:
        l = word.lemma_
        list.append(l)
    return list

def processing(df):
    df["text_processed"] = df.text.str.lower()  
    df["text_processed"] = df["text_processed"].apply(remove_whitespaces)
    df["text_processed"] = df["text_processed"].apply(remove_links)
    df["text_processed"] = df["text_processed"].apply(remove_emoji)
    df["text_processed"] = df["text_processed"].apply(remove_com)
    df["text_processed"] = df["text_processed"].apply(remove_punctuation)
    df["text_processed"] = df["text_processed"].apply(removing_character_repetition)
    df["text_processed"] = df["text_processed"].apply(tokenizer)
    df["text_processed"] = df["text_processed"].apply(remove_stopwords)
    # df = df.join(lemmatize(df))
    df["text_lemmatised"] = lemmatize(df)
    return df


def stats_sentiment(df, type_ = "SU"):
    # n_thread_change = df[(df["change_sentiment"] == True) & (df["author"] == "Hidden")]["thread"].unique().shape[0]
    n_thread_change = df[df["change_sentiment"] == True]["thread"].unique().shape[0]
    tot_threads = df["thread"].unique().shape[0]
    df_only_sentiment = df[df["change_sentiment"] == True]
    mean = round(df_only_sentiment["n_interactions"].mean(), 2)
    std = round(df_only_sentiment["n_interactions"].std() , 2)
    n_no_change = df[df["type_interaction"] == "no_change"]["thread"].unique().shape[0]
    n_no_interaction = df[df["type_interaction"] == "no_interaction"]["thread"].unique().shape[0]
    if type_ == "SU":
        n_no_reply_to_su = df[df["type_interaction"] == "no_reply_to_su"]["thread"].unique().shape[0]
    else:
        n_no_reply_to_su = df[df["type_interaction"] == "no_reply_to_other_user"]["thread"].unique().shape[0]
    n_only_su = df[df["type_interaction"] == "only_su"]["thread"].unique().shape[0]
    # n_positive = df.groupby("thread")["only_positive"].value_counts().reset_index()["only_positive"].value_counts()[True]
    n_positive = df[df["type_interaction"] == "only_positive"]["thread"].unique().shape[0]
    n_negative = df.groupby("thread")["only_positive"].value_counts().reset_index()["only_positive"].value_counts()[False]
    print(f"out of {tot_threads} threads there are: \n   {n_positive} threads in which there is only positive sentiment   {round(n_positive/tot_threads * 100, 2)}%")  #\n   {n_negative} that contains also negative posts")
    print(f"   {n_no_change} in which there's no sentiment change    {round(n_no_change/tot_threads * 100, 2)}%")
    print(f"   {n_no_interaction} in which there's no interaction with super-user    {round(n_no_interaction/tot_threads * 100, 2)}%")
    print(f"   {n_no_reply_to_su} in which there's no reply to the super-user answer    {round(n_no_reply_to_su/tot_threads * 100, 2)}%")
    if type_ == "SU":
        print(f"   {n_only_su} in which there's only super-users    {round(n_only_su/tot_threads * 100, 2)}%")
    print(f"   {n_thread_change} in which there's a sentiment change    {round(n_thread_change/tot_threads * 100, 2)}%")

    print(f"the mean number of interactions for a sentiment change are {mean} with a std of  {std}")
    return


def type_replied_posts(df):
    df = df.value_counts("sentiment").reset_index()
    total = df["count"].sum()
    df["%"] = df["count"].apply(lambda x: round(x/total * 100, 2))
    df['sentiment'] = df['sentiment'].replace({2: 'negative', 1: 'neutral', 0: 'positive'})
    negative_percentage = df[df['sentiment'] == "negative"]["%"].item()
    positive_percentage = df[df['sentiment'] == "positive"]["%"].item()
    neutral_percentage = df[df['sentiment'] == "neutral"]["%"].item()
    print(f"in the df there's {negative_percentage} of negative comments")
    print(f"in the df there's {positive_percentage} of positive comments")
    print(f"in the df there's {neutral_percentage} of neutral comments")
    return


def prepare_df_sentiment_hist_series(df):
    df["year"] = df.time.dt.year
    df["month"] = df.time.dt.month
    df_plot = df.groupby(['year',"month", 'sentiment'], as_index = False).agg({'author':'count'})
    df_plot["date"] = df_plot.year.apply(str) + "-" + df_plot.month.apply(str)
    df_plot['logcount'] = [np.log(x) for x in df_plot.author]
    df_plot["tot_post"] = df_plot.groupby("date")["author"].transform("sum")
    df_plot["%"] = df_plot.author / df_plot.tot_post * 100
    return df_plot