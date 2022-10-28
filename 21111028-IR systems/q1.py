#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import sent_tokenize , word_tokenize
import glob
import re
import os
import numpy as np
import sys
import math
from pathlib import Path
from collections import Counter
import numpy as nap
import pickle


# In[2]:


# Initializing functions 
Stopwords = set(stopwords.words('english'))
ps=PorterStemmer()


# In[4]:


# Declaring global variables
all_words = []
doc_words = [] # doc_words is a list of list for stemmed words for every doc
doc_word_count = [] # doc_word_count list of dict containing word:freq for every doc
dict_global = {} # update the word:frequency value in global dictionary
dict_global_test = {}
doc_length = {} # doc_length stores number of words in each doc
files_with_index = {} # map an index number to each file name


# In[6]:


# Declaring directory where corpora is stored
file_folder = 'english-corpora/*'


# In[7]:


file_folder = 'english-corpora/*'

i = 0
idx = 0
for file_item in glob.glob(file_folder):
    i = i + 1 #index of file as per our convenience and understanding
    print(i,"----",file_item)
    fname = file_item
    files_with_index[idx] = os.path.basename(fname) # map an index number to each file name
    file_opened = open(file_item , "r", encoding='UTF-8')
    text = file_opened.read()
    text = re.sub('[^a-zA-Z\s]',' ',text)
    tokens = word_tokenize(text) #tokenizes the words into tokens
    doc_length[idx] = len(tokens) # doc_length stores number of words in each doc
    curr_words = [] #stores the words in currently opended document
    for word in tokens: #iterate over the words and add their respective stems to the storage.
        if len(word)>1:
            word=ps.stem(word)
            if word not in Stopwords:
                curr_words.append(word)
    
    doc_words.append(curr_words) # doc_words is a list of list for stemmed words for every doc
    counter = dict(Counter(curr_words)) # count the frequency of each word in current doc
    dict_global.update(counter) # update the frequencies in global dictionary
    doc_word_count.append(counter) # doc_word_count list of dict containing word:freq for every doc
    idx = idx + 1
unique_words_all = set(dict_global.keys())


# In[21]:


# doc_word_count
# doc_length


# In[10]:


# tf is a dict containing. 'cold':{'doc1':'freq1','doc2':'freq2',...}
# df is a dict containing. {'cold':0,'hot':0,...}
# df stores number of doc in which word has occured

tf = {x: {} for x in unique_words_all}
df = {x:0 for x in unique_words_all}


# In[13]:


# tf is a dict containing. 'cold':{1:5, 7:20}
# df is a dict containing. 'cold':2

idx =0
for doc in doc_word_count:
    for i in doc.keys():
        df[i] = df[i]+1
        tf[i][idx] = doc[i]   
    idx = idx + 1 
print("Finished")


# In[14]:


Ltot = sum(doc_length.values())
Ltot

# Ld is a list that contains number of token in each doc


# In[15]:


doc_norm={}
idx=0
files_count = len(files_with_index)
for i in doc_word_count:
    l2=0
    for j in i.keys():
        l2+=(i[j]*math.log(files_count/df[j]))**2
    doc_norm[idx]=(math.sqrt(l2))
    idx +=1
    print(idx)


# In[16]:


a_file = open("Pickled_Files/file_idx.pkl", "wb")
pickle.dump(files_with_index, a_file)
a_file.close()
a_file = open("Pickled_Files/unique_words_all.pkl", "wb")
pickle.dump(unique_words_all , a_file)
a_file.close()


# In[17]:


import pickle
with open('Pickled_Files/posting_list.pkl','wb') as file:
    pickle.dump(tf,file)
    file.close()
    
with open('Pickled_Files/df.pkl','wb') as file:
    pickle.dump(df,file)
    file.close()
    
with open('Pickled_Files/doc_len.pkl','wb') as file:
    pickle.dump(doc_length,file)
    file.close()
    
with open('Pickled_Files/doc_words.pkl','wb') as file:
    pickle.dump(doc_words,file)
    file.close()
    
with open('Pickled_Files/doc_norm.pkl','wb') as file:
    pickle.dump(doc_norm,file)
    file.close()


# In[20]:


tf["go"]

