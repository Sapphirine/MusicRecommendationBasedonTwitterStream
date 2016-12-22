import json
import pandas as pd
import os  

#set work directory
os.chdir("/home/candace/Documents/")

tweets_data_path = '/home/candace/Documents/sentiment_analysis_dataset_sample.csv'
mahout_data_path = '/home/candace/Documents/mahout.csv'

whole_data_df = pd.read_csv(tweets_data_path, error_bad_lines=False)
mahout_data_df = whole_data_df[['Sentiment','ItemID','SentimentText']]
mahout_data_df.to_csv(path_or_buf = mahout_data_path,index = False,
                      header=False,sep='\t',encoding = 'utf-8')
