import os
import json
import pandas as pd

#set work directory
os.chdir("/home/candace/Documents/sorted_tweets")

tweets_data_path = '/home/candace/Documents/raw_tweets.csv'
song_list_path = '/home/candace/Documents/song_list.csv'

#Read song list
with open(song_list_path) as f:
    song_list = f.read().splitlines()

#Loads raw tweets
tweets_data = []
tweets_file = open(tweets_data_path, "r")
for line in tweets_file:
    try:
        tweet = json.loads(line)
        tweets_data.append(tweet)
#        print tweet
    except Exception as e:
        print(str(e))
        pass

print len(tweets_data)

tweets = pd.DataFrame()

#tweets['id'] = map(lambda tweet: tweet.get('id', None), tweets_data)
tweets['text'] = map(lambda tweet: tweet.get('text', None),tweets_data)
#tweets['song'] = ''

#Sort through Tweets to see which song they belong
for i in range(len(song_list)):
    tweets_song = pd.Series()
    for j in range((len(tweets.text))):
        try:
            num=tweets.text[j].find(song_list[i])
            if num>=0:
#                print "tweets:",tweets.text[j]
                tweets_song = tweets_song.append(pd.Series(tweets.text[j]))
#                tweets.drop(j)
        except:
            pass
    print i,song_list[i],len(tweets_song)
    
    tweets_song.to_csv(path = str(i),encoding = 'utf-8')
    del tweets_song
