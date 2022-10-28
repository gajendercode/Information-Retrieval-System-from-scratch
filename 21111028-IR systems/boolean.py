#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


# Initializing functions
Stopwords = set(stopwords.words('english'))
ps = PorterStemmer()


# In[4]:


# Loading Pickled Files
temp = open('Pickled_Files/posting_list.pkl',"rb")
posting_lists = pickle.load(temp)

temp = open('Pickled_Files/file_idx.pkl','rb')
file_index = pickle.load(temp)


# In[5]:


# posting_lists


# In[1]:


# file_index


# In[6]:


unique_words = set(posting_lists.keys())
print(len(unique_words))


# In[7]:


def process_text(text):
    regex = re.compile('[^a-zA-Z\s]')
    text_processed = re.sub(regex,' ',text)
    return text_processed


# In[8]:


# Input query
# query=input("Enter your query: ")
query = 'cold and query'


# In[9]:


# process query
query = process_text(query) # Input the text and remove special characters from it.
query = word_tokenize(query) # tokenize the input query
query


# In[10]:


tokens = query.copy()
tokens = [word.lower() for word in tokens]
x =[ps.stem(word) for word in tokens]
y = []
for word in x:
    if(word not in Stopwords):
        y.append(word)
y
main_words = y.copy()


# In[11]:


n = len(file_index)
word_vector = []
word_vector_matrix = []

for w in main_words:
    word_vector=[0]*n
    if w in unique_words:
        for x in posting_lists[w].keys():
            word_vector[x]=1
    word_vector_matrix.append(word_vector)


# In[12]:


len(word_vector_matrix)


# In[13]:


iterations = len(main_words)-1
for i in range(iterations):
    vector1 = word_vector_matrix[0]
    vector2 = word_vector_matrix[1]
    result = [b1&b2 for b1,b2 in zip(vector1,vector2)]    
    word_vector_matrix.pop(0)
    word_vector_matrix.pop(0) 
    word_vector_matrix.insert(0,result)    


# In[14]:


list(word_vector_matrix[0]).count(1)


# In[15]:


final_word_vector = word_vector_matrix[0]
cnt = 0
files = []
for i in final_word_vector:
    if i==1:
        files.append(file_index[cnt])
    cnt += 1


# In[22]:


print(files)


# In[19]:


files


# In[ ]:




