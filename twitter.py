import tweepy
from textblob import TextBlob
import pandas as pd

#Step 1 - Authentication

ckey = ''
csecret = ''

atoken = ''
asecret = ''

auth = tweepy.OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

api = tweepy.API(auth)

#Step 2 - Query details

#List of possible hashtags and phrases people are likely to tweet with
event = 'Trump'
query_phrases = ['@realdonaldtrump']
screen_name = 'realdonaldtrump'

#Date on which the final is taking place
from_date = '2018-05-13'
to_date = '2018-05-14'

#Getting labels for tweets

def get_label(analysis, threshold = 0):
	if analysis.sentiment[0] > threshold:
		return 'Positive'
	else:
		return 'Negative'

#Step 3 - Get the tweets and store them in a csv along with their labels

tweets = api.user_timeline(screen_name = screen_name,count=100)
#tweets = api.search(q = query_phrases, count = 100, since = from_date, until = to_date)

data = [[tweet.text, get_label(TextBlob(tweet.text))] for tweet in tweets]
dataframe = pd.DataFrame(data, columns = ['tweet','label'])
dataframe.to_csv("%s_tweets.csv" %(event), index = False)