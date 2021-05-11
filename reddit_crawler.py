import praw
import datetime as dt
from psaw import PushshiftAPI
import csv


reddit = praw.Reddit(client_id='cicJG0jkrjc63g',
                     client_secret='85Rd8G3iQL7TVkQ2elKwlH512KVqBQ',
                     user_agent='stocks',
                     username='ilanbeast',
                     password='roeinoob')


api = PushshiftAPI(reddit)
subreddit = 'Stocks'
subreddit = reddit.subreddit(subreddit)
# start_date = int(dt.datetime(2019, 3, 17).timestamp())
# end_date = int(dt.datetime(2020, 9, 17).timestamp())
# filename = 'train.csv'
start_date = int(dt.datetime(2020, 9, 18).timestamp())
end_date = int(dt.datetime(2021, 3, 17).timestamp())
filename = '_.csv'
topics_dict = {"title": [],
               "score": [],
               "id": [],
               "num_comments": [],
               "timestamp": [],
               "body": []}

with open(filename, 'w', encoding='utf8') as f:
    writer = csv.writer(f)
    writer.writerow(['title', 'score', 'id', 'num_comments', 'timestamp', 'body'])
    for submission in api.search_submissions(after=start_date,
                                             before=end_date,
                                             subreddit=subreddit):
        try:
            flair = submission.link_flair_text
        except KeyError:
            flair = "NaN"

        writer.writerow([
                submission.title,
                submission.score,
                submission.id,
                submission.num_comments,
                dt.datetime.fromtimestamp(submission.created),
                submission.selftext
            ])
