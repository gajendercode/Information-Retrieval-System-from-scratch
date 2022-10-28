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
import numpy as np
import pickle
from collections import defaultdict


# In[2]:


# Initialization of functions
Stopwords = set(stopwords.words('english'))
ps=PorterStemmer()


# In[3]:


def process_text(text):
    regex = re.compile('[^a-zA-Z0-9\s]')
    text_returned = re.sub(regex,'',text)
    return text_returned


# In[4]:


with open('Pickled_Files/posting_list.pkl','rb') as file:
    tf=pickle.load(file)
    file.close()
    
with open('Pickled_Files/df.pkl','rb') as file:
    DF=pickle.load(file)
    file.close()
    
with open('Pickled_Files/file_idx.pkl','rb') as file:
    file_idx=pickle.load(file)
    file.close()
    
with open('Pickled_Files/doc_len.pkl','rb') as file:
    doc_length=pickle.load(file)
    file.close()


# ## Finding average length of all documents

# In[5]:


length = 0
# N=len(file_idx)
for i in doc_length:
    length += doc_length[i]
Length_avg = length/len(file_idx)
Length_avg


# In[6]:


def calculate_IDF(q):
    temp = 0
    if q in DF:
        temp=DF[q]
    else:
        temp = 0
    Numerator = len(file_idx) - temp + 0.5 
    Denominator = temp + 0.5
    idf = math.log(Numerator/Denominator)
    return idf


# In[7]:


doc_length[0]


# In[8]:


def calculate_doc_score(query):
    query = process_text(query)
    query = re.sub(re.compile('\d'),'',query)
    words = word_tokenize(query)
    words = [word.lower() for word in words]
    words = [ps.stem(word) for word in words]
    words=[word for word in words if word not in Stopwords]
    for i in range(len(file_idx)):
        for word in words:
            TF = 0
            if word in tf:
                if i in tf[word]:
                    TF = tf[word][i]
            idf = calculate_IDF(word)
            ans = idf*(k+1)*TF/(TF+k*(1-b+b*(doc_length[i]/Length_avg)))
            BM_score[i] += ans


# In[9]:


# Initializing standard values
k=1.2
b=0.75


# ## Input Query

# In[10]:


# query=input("Enter your query: ")
query = 'this is computer science'


# In[11]:


BM_score={}
for i in range(0,len(file_idx)):
    BM_score[i] = 0
calculate_doc_score(query)
score = sorted(BM_score.items(),key=lambda item: item[1],reverse=True)


# In[12]:


count = 10
for i in score:
    if count == 0:
        break
    print(file_idx[i[0]],i[1])
    count-=1


# In[ ]:




