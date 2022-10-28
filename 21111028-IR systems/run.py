#!/usr/bin/env python
# coding: utf-8

# In[12]:


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
import pandas as pd
import sys


# In[13]:


Stopwords = set(stopwords.words('english'))
ps=PorterStemmer()


# In[14]:


def process_text(text):
    regex = re.compile('[^a-zA-Z\s]')
    text_processed = re.sub(regex,' ',text)
    return text_processed


# In[15]:


def Boolean_Retrieval_System(query, counter = 20):
    temp = open('Pickled_Files/posting_list.pkl',"rb")
    posting_lists = pickle.load(temp)

    temp = open('Pickled_Files/file_idx.pkl','rb')
    file_index = pickle.load(temp)
        
    unique_words = set(posting_lists.keys())
 
    # process query
    query = process_text(query) # Input the text and remove special characters from it.
    query = word_tokenize(query) # tokenize the input query

    tokens = query.copy()
    tokens = [word.lower() for word in tokens]
    x =[ps.stem(word) for word in tokens]
    y = []
    for word in x:
        if(word not in Stopwords):
            y.append(word)
    y
    main_words = y.copy()

    n = len(file_index)
    word_vector = []
    word_vector_matrix = []

    for w in main_words:
        word_vector=[0]*n
        if w in unique_words:
            for x in posting_lists[w].keys():
                word_vector[x]=1
        word_vector_matrix.append(word_vector)
    iterations = len(main_words)-1
    for i in range(iterations):
        vector1 = word_vector_matrix[0]
        vector2 = word_vector_matrix[1]
        result = [b1&b2 for b1,b2 in zip(vector1,vector2)]    
        word_vector_matrix.pop(0)
        word_vector_matrix.pop(0) 
        word_vector_matrix.insert(0,result)    
    final_word_vector = word_vector_matrix[0]
    cnt = 0
    files = []
    for i in final_word_vector:
        if i==1:
            files.append(file_index[cnt])
        cnt += 1
    return files[:10]


# In[16]:


def tfidf(query,count=20):

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


    text = process_text(query)
    text = re.sub(re.compile('\d'),'',text)
    words = word_tokenize(text)
    words = [word.lower() for word in words]
    words = [ps.stem(word) for word in words]
    words = [word for word in words if word not in Stopwords]
    words = [word for word in words if word in tf.keys()]

    query_tfidf = [] #tf-idf for query
    query_norm = 0 # modulus value of tf-idf of query for each document
    for word in words:
        tf_idf = (words.count(word)*math.log(len(file_idx)/DF[word]))
        query_tfidf.append(tf_idf)
        query_norm += tf_idf**2
    query_norm = math.sqrt(query_norm)
    query_tfidf = np.array(query_tfidf)/query_norm

    cosine_similarity_score = {}
    for i in range(len(file_idx)):
        doc_vector = []
        for word in words:
            tf_idf = (doc_words[i].count(word)*math.log(len(file_idx)/DF[word]))
            doc_vector.append(tf_idf)
        doc_vector = np.array(doc_vector)
        cosine_similarity_score[i] = np.dot(query_tfidf,doc_vector)/doc_norm[i]
    score=sorted(cosine_similarity_score.items(),key=lambda x:x[1],reverse=True)

    out = []
    for i in score:
        if count == 0:
            break
        out.append([file_idx[i[0]], i[1]])
        count-=1
    return out


# In[17]:


def bm25(query,count=20):

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


    length = 0
    # N=len(file_idx)
    for i in doc_length:
        length += doc_length[i]
    Length_avg = length/len(file_idx)
    Length_avg


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
    
    # Initializing standard values
    k=1.2
    b=0.75

    BM_score={}
    for i in range(0,len(file_idx)):
        BM_score[i] = 0
    calculate_doc_score(query)
    score = sorted(BM_score.items(),key=lambda item: item[1],reverse=True)

    out = []
    for i in score:
        if count == 0:
            break
        out.append([file_idx[i[0]],i[1]])
        count-=1
    return out


# In[18]:


query_list=pd.read_csv(sys.argv[1],sep='\t',header=None)
# query_list=pd.read_csv('query.txt',sep='\t',header=None)
query_list.columns=['qid','query']


# In[19]:


with open('Pickled_Files/file_idx.pkl',"rb") as temp:
    file_idx=pickle.load(temp)
    temp.close()


# In[20]:


csv=[]
for index, row in query_list.iterrows():
    files=Boolean_Retrieval_System(row['query'])
    for file in files:
        csv.append([row['qid'],1,file,1])
    if len(files)<5:
        remaining=[i for i in file_idx.values() if i not in files]
        remaining=remaining[:5-len(files)]
        for file in remaining:
            csv.append([row['qid'],1,file,0])
pd.DataFrame(csv,columns=['QueryID','Iteration','DocId','Relevance']).to_csv('Output/BRS.csv',index=False)
print("brs done")


# In[21]:


csv=[]
for index, row in query_list.iterrows():
    files=tfidf(row['query'])
    for file in files:
        relevance=0
        if file[1]>0:
            relevance=1
        csv.append([row['qid'],1,file[0],relevance])
pd.DataFrame(csv,columns=['QueryID','Iteration','DocId','Relevance']).to_csv('Output/TFIDF.csv',index=False)
print("tf done")


# In[22]:


csv=[]
for index, row in query_list.iterrows():
    files=bm25(row['query'])
    for file in files:
        relevance=0
        if file[1]>0:
            relevance=1
        csv.append([row['qid'],1,file[0],relevance])
pd.DataFrame(csv,columns=['QueryID','Iteration','DocId','Relevance']).to_csv('Output/BM25.csv',index=False)
print("bm25 done")


# In[ ]:




