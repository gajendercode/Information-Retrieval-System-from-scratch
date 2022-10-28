#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
from nltk.corpus import stopwords
from nltk.stem import  PorterStemmer
from nltk.tokenize import word_tokenize
import re
import math
import numpy as np
import pickle


# In[2]:


# Initialization of functions
Stopwords = set(stopwords.words('english'))
ps=PorterStemmer()


# In[3]:


def process_text(text):
    regex = re.compile('[^a-zA-Z0-9\s]')
    text_processed = re.sub(regex,' ',text)
    return text_processed


# In[5]:


with open('Pickled_Files/df.pkl','rb') as file:
    DF=pickle.load(file)
    file.close()
    
with open('Pickled_Files/file_idx.pkl','rb') as file:
    file_idx=pickle.load(file)
    file.close()
    
with open('Pickled_Files/doc_words.pkl','rb') as file:
    doc_words=pickle.load(file)
    file.close()
    
with open('Pickled_Files/doc_norm.pkl','rb') as file:
    doc_norm=pickle.load(file)
    file.close()
    
with open('Pickled_Files/posting_list.pkl','rb') as file:
    tf=pickle.load(file)
    file.close()


# In[6]:


print(tf)


# In[7]:


print(DF)


# ## Tf-idf calculation

# ## Input the query

# In[14]:


query=input("Enter your query:")


# ## Pre-process the query

# In[15]:


text = process_text(query)
text = re.sub(re.compile('\d'),'',text)
words = word_tokenize(text)
words = [word.lower() for word in words]
words = [ps.stem(word) for word in words]
words = [word for word in words if word not in Stopwords]
words = [word for word in words if word in tf.keys()]


# ## Tf-idf, norm calculation for query

# In[16]:


query_tfidf = [] #tf-idf for query
query_norm = 0 # modulus value of tf-idf of query for each document
for word in words:
    tf_idf = (words.count(word)*math.log(len(file_idx)/DF[word]))
    query_tfidf.append(tf_idf)
    query_norm += tf_idf**2
query_norm = math.sqrt(query_norm)
query_tfidf = np.array(query_tfidf)/query_norm


# ## Tf-idf, norm for documents

# In[17]:


cosine_similarity_score = {}
for i in range(len(file_idx)):
    doc_vector = []
    for word in words:
        tf_idf = (doc_words[i].count(word)*math.log(len(file_idx)/DF[word]))
        doc_vector.append(tf_idf)
    doc_vector = np.array(doc_vector)
    cosine_similarity_score[i] = np.dot(query_tfidf,doc_vector)/doc_norm[i]


# ## Sorting the cosine similarity scores

# In[18]:


score=sorted(cosine_similarity_score.items(),key=lambda x:x[1],reverse=True)


# In[19]:


count = 10
for i in score:
    if count == 0:
        break
    print(file_idx[i[0]],i[1])
    count-=1


# In[ ]:




