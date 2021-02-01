#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
df = pd.read_csv('simdi.csv')
df.head(5)


# In[2]:


df.info()


# In[3]:


df.describe().round(1)


# In[4]:


null_values=df.isna().sum()
null_values=pd.DataFrame(null_values,columns=['null'])
sum_tot=len(df)
null_values['percent']=null_values['null']/sum_tot*100
round(null_values,3).sort_values('percent',ascending=False)


# In[5]:


df= df.dropna()
df.shape


# In[6]:


import wordcloud
from nltk.corpus import stopwords
import nltk
import string
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
stop = stopwords.words('english')


# In[11]:


df['stopwords'] = df['baslik'].apply(lambda x: len([x for x in x.split() if x in stop]))
df[['baslik','stopwords']].head()


# In[8]:


def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return count

df['punctuation'] = df['baslik'].apply(lambda x: count_punct(x))


# In[9]:


df[['baslik','punctuation']].head()


# In[10]:


df['hastags'] = df['baslik'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
df[['baslik','hastags']].head()


# In[11]:


df.hastags.loc[df.hastags != 0].count()


# In[12]:


df['numerics'] = df['baslik'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
df[['baslik','numerics']].head()


# In[13]:


df['upper'] = df['baslik'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
df[['baslik','upper']].head()


# In[14]:


df['baslik'] = df['baslik'].apply(lambda x: " ".join(x.lower() for x in x.split()))
df['baslik'].head()


# In[15]:


df['baslik'] = df['baslik'].str.replace('[^\w\s]','')
df['baslik'].head()


# In[16]:


from nltk.corpus import stopwords
stop = stopwords.words('english')
df['baslik'] = df['baslik'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
df['baslik'].sample(10)


# In[17]:


from bs4 import BeautifulSoup

def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text

strip_html_tags('sdasdasdasd http://t.co/7AzE4IoGMe Risk Assessmen ')

# I will come back to this later


# In[18]:


from textblob import TextBlob
df['baslik'][:5].apply(lambda x: str(TextBlob(x).correct()))


# In[19]:


import re
import string

def clean_text_round1(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

round1 = lambda x: clean_text_round1(x)


# In[20]:


df['baslik'] = df.baslik.apply(round1)
df.baslik


# In[21]:


def clean_text_round2(text):
    '''Get rid of some additional punctuation and non-sensical text that was missed the first time around.'''
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    return text

round2 = lambda x: clean_text_round2(x)


# In[22]:


df['baslik'] = df.baslik.apply(round2)
df.baslik


# In[23]:


freq = pd.Series(' '.join(df['baslik']).split()).value_counts()[:20]
freq


# In[24]:


df['word_count'] = df['baslik'].apply(lambda x: len(str(x).split(" ")))
df[['baslik','word_count']].head()


# In[25]:


null_values=df.isna().sum()
null_values=pd.DataFrame(null_values,columns=['null'])
sum_tot=len(df)
null_values['percent']=null_values['null']/sum_tot*100
round(null_values,3).sort_values('percent',ascending=False)


# In[26]:


df['char_count'] = df['baslik'].str.len() ## this also includes spaces
df[['baslik','char_count']].head()


# In[27]:


def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/(len(words)+0.000001))


# In[28]:


df['avg_word'] = df['baslik'].apply(lambda x: avg_word(x)).round(1)
df[['baslik','avg_word']].head()


# In[29]:


df.sample(2)


# In[30]:


df['baslik'] = df.baslik.apply(round1)
df.baslik


# In[31]:


df['baslik'] = df.Summary.apply(round2)
df.Summary


# In[32]:


df.to_csv('tertemiz.csv', index=False)


# In[16]:


import pandas as pd
import numpy as np
df = pd.read_csv('simdi.csv')
df.head(5)


# In[18]:


from textblob import TextBlob
df['polarity'] = df[['temiz_haber']].sample(12205)

def detect_subjectivity(temiz_haber):
    return TextBlob(temiz_haber).sentiment.subjectivity
df['subjectivity'] = df.temiz_haber.apply(detect_subjectivity)

def detect_polarity(temiz_haber):
    return TextBlob(temiz_haber).sentiment.polarity
df['polarity'] = df.temiz_haber.apply(detect_polarity)     


def f(df):
    if df['polarity'] > 0:
        val = 1
    elif df['polarity'] == 0:
        val = 0
    else:
        val = -1
    return val
df['sentiment'] = df.apply(f, axis=1)


# In[19]:


df.polarity


# In[20]:


df.head(5)


# In[21]:


df.to_csv('ensonhali.csv', index=False)


# In[ ]:




