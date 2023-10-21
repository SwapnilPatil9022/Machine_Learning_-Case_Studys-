# importing req. Lib.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

##################################################################################
# Project_Name : Social Media sentiment analysics
# Input_CSV_File : Tweets.csv
# Author : Swapnil Ashok Patil
# Date : 17/02/2022
##################################################################################

def Social_Media_Analysis():

    # load our data set
    data = pd.read_csv(r'Tweets.csv')

    print(data.shape)

    # looking into our data
    print(data.head())

    # checking last 5 entries
    print(data.tail())

    # checking columns in our data
    print(data.columns)

    # checking info our data
    print(data.info())

    # checking unique values
    print(data.nunique())

    # checking null values in our data
    print(data.isnull().sum())

    # Preprocessing on data
    # tweet_created column got the date recorts and showing type is object we have to change it of date time format

    data['tweet_created'] = pd.to_datetime(data['tweet_created']).dt.date

    data['tweet_created'] = pd.to_datetime(data['tweet_created'])

    print(data.info())

    print(data.head())

    data['tweet_created'].min()

    data['tweet_created'].max()

    # we have data from 16th feb 2015 to 25 feb 2015 mins we have data of 9 days.

    # checking uniques values in tweet_created columns
    data['tweet_created'].nunique()

    numberoftweets = data.groupby('tweet_created').size()

    print(numberoftweets.dtype)

    print(numberoftweets)

    # here we can see tweets created every day

    # treating with null values

    data.isna().sum()

    print("Percentage null or na values in df")
    ((data.isnull() | data.isna()).sum() * 100 / data.index.size).round(2)

    del data['tweet_coord']
    del data['airline_sentiment_gold']
    del data['negativereason_gold']
    print(data.head())

    freq = data.groupby('negativereason').size()

    counter = data.airline_sentiment.value_counts()
    index = [1, 2, 3]
    plt.figure(1, figsize=(12, 6))
    plt.bar(index, counter, color=['green', 'red', 'blue'])
    plt.xticks(index, ['negative', 'neutral', 'positive'], rotation=0)
    plt.xlabel('Sentiment Type')
    plt.ylabel('Sentiment Count')
    plt.title('Count of Type of Sentiment')

    # checking differtent airlines we have
    data['airline'].unique()

    print("Total number of tweets for each airline \n ",
          data.groupby('airline')['airline_sentiment'].count().sort_values(ascending=False))
    airlines = ['US Airways', 'United', 'American', 'Southwest', 'Delta', 'Virgin America']
    plt.figure(1, figsize=(12, 12))
    for i in airlines:
        indices = airlines.index(i)
        plt.subplot(2, 3, indices + 1)
        new_df = data[data['airline'] == i]
        count = new_df['airline_sentiment'].value_counts()
        Index = [1, 2, 3]
        plt.bar(Index, count, color=['red', 'green', 'blue'])
        plt.xticks(Index, ['negative', 'neutral', 'positive'])
        plt.ylabel('Mood Count')
        plt.xlabel('Mood')
        plt.title('Count of Moods of ' + i)

    neg_tweets = data.groupby(['airline', 'airline_sentiment']).count().iloc[:, 0]
    total_tweets = data.groupby(['airline'])['airline_sentiment'].count()

    my_dict = {'American': neg_tweets[0] / total_tweets[0], 'Delta': neg_tweets[3] / total_tweets[1],
               'Southwest': neg_tweets[6] / total_tweets[2],
               'US Airways': neg_tweets[9] / total_tweets[3], 'United': neg_tweets[12] / total_tweets[4],
               'Virgin': neg_tweets[15] / total_tweets[5]}
    perc = pd.DataFrame.from_dict(my_dict, orient='index')
    perc.columns = ['Percent Negative']
    print(perc)
    ax = perc.plot(kind='bar', rot=0, colormap='Greens_r', figsize=(15, 6))
    ax.set_xlabel('Airlines')
    ax.set_ylabel('Percentage of negative tweets')
    plt.show()

    figure_2 = data.groupby(['airline', 'airline_sentiment']).size()
    figure_2.unstack().plot(kind='bar', stacked=True, figsize=(15, 10))

    print(figure_2)

    negative_reasons = data.groupby('airline')['negativereason'].value_counts(ascending=True)
    negative_reasons.groupby(['airline', 'negativereason']).sum().unstack().plot(kind='bar', figsize=(22, 12))
    plt.xlabel('Airline Company')
    plt.ylabel('Number of Negative reasons')
    plt.title("The number of the count of negative reasons for airlines")
    plt.show()

    # get the number of negative reasons
    data['negativereason'].nunique()

    NR_Count = dict(data['negativereason'].value_counts(sort=False))


    def NR_Count(Airline):
        if Airline == 'All':
            a = data
        else:
            a = data[data['airline'] == Airline]
        count = dict(a['negativereason'].value_counts())
        Unique_reason = list(data['negativereason'].unique())
        Unique_reason = [x for x in Unique_reason if str(x) != 'nan']
        Reason_frame = pd.DataFrame({'Reasons': Unique_reason})
        Reason_frame['count'] = Reason_frame['Reasons'].apply(lambda x: count[x])
        return Reason_frame


    def plot_reason(Airline):
        a = NR_Count(Airline)
        count = a['count']
        Index = range(1, (len(a) + 1))
        plt.bar(Index, count,
                color=['red', 'yellow', 'blue', 'green', 'black', 'brown', 'gray', 'cyan', 'purple', 'orange'])
        plt.xticks(Index, a['Reasons'], rotation=90)
        plt.ylabel('Count')
        plt.xlabel('Reason')
        plt.title('Count of Reasons for ' + Airline)


    plot_reason('All')
    plt.figure(2, figsize=(13, 13))
    for i in airlines:
        indices = airlines.index(i)
        plt.subplot(2, 3, indices + 1)
        plt.subplots_adjust(hspace=0.9)
        plot_reason(i)

    date = data.reset_index()
    # convert the Date column to pandas datetime
    date.tweet_created = pd.to_datetime(date.tweet_created)
    # Reduce the dates in the date column to only the date and no time stamp using the 'dt.date' method
    date.tweet_created = date.tweet_created.dt.date
    date.tweet_created.head()
    df = date
    day_df = df.groupby(['tweet_created', 'airline', 'airline_sentiment']).size()
    # day_df = day_df.reset_index()
    day_df

    day_df = day_df.loc(axis=0)[:, :, 'negative']

    # groupby and plot data
    ax2 = day_df.groupby(['tweet_created', 'airline']).sum().unstack().plot(kind='bar',
                                                                            color=['red', 'green', 'blue', 'yellow',
                                                                                   'purple', 'orange'], figsize=(15, 6),
                                                                            rot=70)
    labels = ['American', 'Delta', 'Southwest', 'US Airways', 'United', 'Virgin America']
    ax2.legend(labels=labels)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Negative Tweets')
    plt.show()

    from wordcloud import WordCloud,STOPWORDS

    new_df=data[data['airline_sentiment']=='positive']
    words = ' '.join(new_df['text'])
    cleaned_word = " ".join([word for word in words.split()
                                if 'http' not in word
                                    and not word.startswith('@')
                                    and word != 'RT'
                                ])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color='black',
                          width=3000,
                          height=2500
                         ).generate(cleaned_word)
    plt.figure(1,figsize=(12, 12))
    plt.imshow(wordcloud)

    new_df=data[data['airline_sentiment']=='negative']
    words = ' '.join(new_df['text'])
    cleaned_word = " ".join([word for word in words.split()
                                if 'http' not in word
                                    and not word.startswith('@')
                                    and word != 'RT'
                                ])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color='black',
                          width=3000,
                          height=2500
                         ).generate(cleaned_word)
    plt.figure(1,figsize=(12, 12))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

    data.drop(data.loc[data['airline_sentiment']=='neutral'].index, inplace=True)

    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    le.fit(data['airline_sentiment'])

    data['airline_sentiment_encoded'] = le.transform(data['airline_sentiment'])
    data.head()

    # Preprocessing the tweet text data

    def tweet_to_words(tweet):
        letters_only = re.sub("[^a-zA-Z]", " ",tweet)
        words = letters_only.lower().split()
        stops = set(stopwords.words("english"))
        meaningful_words = [w for w in words if not w in stops]
        return( " ".join( meaningful_words ))
    plt.axis('off')
    plt.show()

    nltk.download('stopwords')
    data['clean_tweet']=data['text'].apply(lambda x: tweet_to_words(x))

    x = data.clean_tweet
    y = data.airline_sentiment

    print(len(x), len(y))

    # The data is split in the standard 80,20 ratio

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
    print(len(x_train), len(y_train))
    print(len(x_test), len(y_test))

    from sklearn.feature_extraction.text import CountVectorizer

    # instantiate the vectorizer
    vect = CountVectorizer()
    vect.fit(x_train)

    # Use the trained to create a document-term matrix from train and test sets
    x_train_dtm = vect.transform(x_train)
    x_test_dtm = vect.transform(x_test)

    vect_tunned = CountVectorizer(stop_words='english', ngram_range=(1,2), min_df=0.1, max_df=0.7, max_features=100)
    vect_tunned

    #training SVM model with linear kernel
    #Support Vector Classification-wrapper around SVM
    from sklearn.svm import SVC
    model = SVC(kernel='linear', random_state = 10)
    model.fit(x_train_dtm, y_train)
    #predicting output for test data
    pred = model.predict(x_test_dtm)

    #accuracy score
    accuracy_score(y_test,pred)

    #building confusion matrix
    cm = confusion_matrix(y_test, pred)
    print(cm)

    #defining the size of the canvas
    plt.rcParams['figure.figsize'] = [15,8]
    #confusion matrix to DataFrame
    conf_matrix = pd.DataFrame(data = cm,columns = ['Predicted:0','Predicted:1',], index = ['Actual:0','Actual:1',])
    #plotting the confusion matrix
    sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'Paired', cbar = False,linewidths = 0.1, annot_kws = {'size':25})
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.show()

    print(classification_report(y_test,pred))

# Execution start from main function.
def main():

    Social_Media_Analysis()

if __name__ == "__main__":
    main()
