#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas
import numpy
import matplotlib.pyplot as pp
get_ipython().run_line_magic('matplotlib', 'inline')
import itertools

stop = set(open('stop-word-list.txt').read().split())
data = pandas.read_csv('VT_tweets_2019Fall_geo.csv')
data


# The "bag" column is better described as a series of bags in list format. Unecessary commas among other things were cleaned from the given text, including lowercasing all letters.
# 
# The datetime data was stored as String instead of any useful numeric type, so I changed everything to Pandas's Timestamp datatype for ease of use later.

# In[2]:


data['bag'] = data.tweet.map(lambda t: t.replace(',', '').lower().split())
data.datetime = pandas.to_datetime(data.datetime)
data.head()


# Stop words are worse than useless in analyzing Term Frequency -- they are downright detrimental to analysis since they pop up so often without contributing anything useful to the analysis.

# In[3]:


data['bag'] = data['bag'].map(lambda x: [word for word in x if word.lower() not in (stop)])
data.head()


# To analyze trends by times, the tweets were all aggregated by the datetime factor. Now all tweets are organized by chronoligical order.
# 
# Each tweet is stored in its posted timmestamp.

# In[4]:


g = data.groupby('datetime')
times = g.bag.aggregate(lambda listofbags: list(itertools.chain.from_iterable(listofbags)))
times


# To create the Term Frequency matrix, all words appearing in a tweet were counted up. Given enough time, this would occur for every tweet. Unfortunately, due to time and computing restraints, only the first 1000 were used.

# In[5]:


TF = times.iloc[:1000].apply(lambda bag: pandas.Series(bag).value_counts())
TF


# The inverse document frequency matrix was created by taking the log of the appearance ratio of each word. Less of a matrix and more like a column really.

# In[6]:


IDF = numpy.log(len(TF)/TF.count())
IDF


# The end TFIDF matrix was made with simple row-wise multiplication.

# In[7]:


TFIDF = TF * IDF
TFIDF


# The first analysis was by appearance count of each keyword. In the end, both "season" and "football" beat any other topic that day.

# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')
#TFIDF.resample('H').sum().plot(figsize = (20, 20))
ct = TFIDF.resample('D').count().sort_values(by = '2019-08-25', axis = 1, ascending = False)
ct


# In[9]:


#ct.plot(subplots = True, figsize = (20, 6) )
pandas.plotting.boxplot(ct.iloc[:, :25], figsize = (35, 5))


# The second trend analysis on the day's TFIDF sums revealed the same results: "season" and "football" were the most important words once again.
# 
# The tilde "~" was a strange top option of all the tweets, beating everything else by a long shot. The thing is, I could not find a single instance of it in anyone's tweets. Both of these facts seem very unlikely from personal experience alone. Unfortunately, even if tildes were a common occurrence, there are too many uses for them on Twitter ranging from being cutesy to passive-aggresive that any trend analysis be futile. Thus, it was removed.

# In[10]:


ds = TFIDF.resample('D').sum().sort_values(by = '2019-08-25', axis = 1, ascending = False)
ds


# In[11]:


pandas.plotting.boxplot(ds.iloc[:, 1:35], figsize = (35, 10))


# Now that we know the upcoming football season is trending, it is useful to find out when it was trending. Sorted by the hour the tweet was posted, both season and football had their TFIDF values summed. 

# In[12]:


fbs = TFIDF[['season', 'football']].resample('H').sum()
fbs


# In[13]:


fbs.plot();


# On the day before the school year's education began, a seminal moment in every undergrad's life, the most popular topic was the upcoming football season of all things. It's honestly not a big surprise -- we are still a football school no matter how bad it seems. 
# 
# The time football spent trending converted to EST are about 7PM to midnight on August 24th, picking back up again at about 7AM on the 25th. This should not be surprising to anyone, as these are pretty normal bedtimes. While college students are sometimes known for having wacky sleep schedules, it seems that sports fans at least have their act together on this one. They even wake up early on Sundays!
# 
# A possible explanation for the upcoming season's popularity on Twitter may be unrelated to Virginia Tech. The last week of the NFL preseason began with multiple games at 7PM on Saturday on 24th. This may explain the immense hourly TFIDF that appeared at 00:00 UTC. 
# 
# A possible next step is checking whether the NFL or Virginia Tech's upcoming football seasons were the source of the Twitter storm. While my first intuitive guess would be the NFL due to the recent coinciding events, the fact that Miami was a trending topic may suggest that VT's upcoming season was the topic of interest (I doubt the Miami Marlins game was trending despite occuring that same day). 
