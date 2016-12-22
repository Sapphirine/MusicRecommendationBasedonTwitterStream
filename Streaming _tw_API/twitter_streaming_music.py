#Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import time

#Variables that contains the user credentials to access Twitter API 
access_token = ""
access_token_secret = ""
consumer_key = ""
consumer_secret = ""

song_list_path = '/home/candace/Documents/song_list.csv'
raw_tweets_path = '/home/candace/Documents/raw_tweets.csv'

with open(song_list_path) as f:
    song_list = f.read().splitlines()
#print song_list

#Set up the listener
class StdOutListener(StreamListener):

    def on_data(self, data):
#        print data
        try:
            with open(raw_tweets_path, 'a') as f:
                f.write(data)
                print(data)
                return True
        except BaseException as e:
            print(str(e))
            time.sleep(5)
        return True


    def on_error(self, status):
        print status


if __name__ == '__main__':

    #This handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)

    #This line filter Twitter Streams to capture data by the keywords
    
    stream.filter(track=song_list,languages=['en'],async=True)

