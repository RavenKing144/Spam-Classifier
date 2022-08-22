#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[64]:


import re
import nltk
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score


# # Reading Dataset

# In[4]:


path = "Dataset/SMSSpamCollection"
dataset = pd.read_csv(path, sep = "\t", names= ['label', 'message'])


# # Lemitizing

# In[6]:


lematizer = WordNetLemmatizer()


# In[32]:


corpus = []
stopword = set(stopwords.words('english'))
for i in dataset.values:
    sentence = i[1].lower()
    required = re.sub('[^a-z]', ' ', sentence)
    required = required.split()
    required = [lematizer.lemmatize(j) for j in required if j not in stopword]
    required = " ".join(required)
    corpus.append(required)


# # Bag of Words creation

# In[37]:


cv = CountVectorizer(max_features = 3500)
X = cv.fit_transform(corpus).toarray()


# In[43]:


Y = pd.get_dummies(dataset.label)


# In[44]:


Y = Y.iloc[:,1].values


# # Spliting train test data

# In[50]:


x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state = 0)


# # Using Navie Byes for prediction

# In[54]:


span_model = MultinomialNB().fit(x_train, y_train)


# # Accuracy Prediction and confusion matrix visualisation

# In[56]:


y_pred = span_model.predict(x_test)


# In[62]:


result = confusion_matrix(y_test, y_pred)


# In[65]:


score = accuracy_score(y_test, y_pred)


# In[66]:


score


# In[ ]:




