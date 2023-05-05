# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 16:59:42 2023

@author: Akhilesh
"""

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html, State, dash_table
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
stopword = nltk.corpus.stopwords.words('english')

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions=True


df_tweets_filtered = pd.read_csv("data/cleaned_data/df_tweets_cleaned.csv")

def plot_char_length_hist():
    df_hist_char = pd.DataFrame()
    df_hist_char["before_cleaning"] = df_tweets_filtered['tweet'].str.len()
    df_hist_char["after_cleaning"] = df_tweets_filtered["tweets_cleaned"].str.len()
    fig = go.Figure()
    fig.add_histogram(x=df_hist_char["before_cleaning"], name = "Length of character before cleaning", textposition="inside",)
    fig.add_histogram(x=df_hist_char["after_cleaning"], name = "Length of character before cleaning", textposition="inside",)
    # Overlay both histograms
    fig.update_layout(barmode='overlay', width = 500)
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.75)
    return fig

def plot_word_count_hist():
    df_hist_word = pd.DataFrame()
    df_hist_word["before_cleaning"] = df_tweets_filtered['tweet'].str.split().map(lambda x: len(x))
    df_hist_word["after_cleaning"] = df_tweets_filtered["tweets_cleaned"].str.split().map(lambda x: len(x))
    fig = go.Figure()
    fig.add_histogram(x=df_hist_word["before_cleaning"], name = "Length of words before cleaning", textposition="inside",)
    fig.add_histogram(x=df_hist_word["after_cleaning"], name = "Length of words before cleaning", textposition="inside",)
    # Overlay both histograms
    fig.update_layout(barmode='overlay', width = 500)
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.75)
    return fig

def plot_unigram_bar(col_name):
    corpus=[]
    new = df_tweets_filtered[col_name].str.split()
    new = new.values.tolist()
    corpus = [word for i in new for word in i]
    counter=Counter(corpus)
    most=counter.most_common()
    
    x, y= [], []
    for word,count in most[:40]:
        if (word not in stopword):
            x.append(word)
            y.append(count)
    df_bar_unigram = pd.DataFrame(list(zip(x,y)), columns = ["word", "freq"])
    fig = px.bar(df_bar_unigram, x="freq", y="word", orientation='h')
    fig.update_layout(width = 500)
    return fig

def plot_top_ngrams_barchart(text, n=2):
    stop=set(stopword)

    new= text.str.split()
    new=new.values.tolist()
    corpus=[word for i in new for word in i]

    def _get_top_ngram(corpus, n=None):
        vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) 
                      for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:10]

    top_n_bigrams=_get_top_ngram(text,n)[:10]
    x,y=map(list,zip(*top_n_bigrams))
    df_bar_ngram = pd.DataFrame(list(zip(x,y)), columns = ["word", "freq"])
    fig = px.bar(df_bar_ngram, x="freq", y="word", orientation='h')
    fig.update_layout(width = 500)
    return fig
    
    
def plot_histogram(x):    
    fig = go.Figure(data=[go.Histogram(x=x)])
    fig.update_layout(width = 500)
    return fig

def plot_table(table, width_ = 1100):
    return dash_table.DataTable(
        data = table.to_dict('records'),
        columns=[
            {'name': i, 'id': i} for i in table.columns
        ],
        style_table={
            'height': 500,
            'overflowY': 'scroll',
            'width': width_
        },
        export_format="csv",

    )
df_tweets_punct = pd.read_csv("data/cleaned_data/df_tweets_punct.csv")
df_tweets_non_stop = pd.read_csv("data/cleaned_data/df_tweets_non_stop.csv")
df_tweets_tokenized = pd.read_csv("data/cleaned_data/df_tweets_tokenized.csv")
df_tweets_stemmed = pd.read_csv("data/cleaned_data/df_tweets_stemmed.csv")
df_tweets_lemmatized = pd.read_csv("data/cleaned_data/df_tweets_lemmatized.csv")
df_tweets_cv = pd.read_csv("data/cleaned_data/df_tweets_cv.csv")

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "25rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "25rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

fig_stats3 = go.Figure()
fig_stats3.add_trace(go.Indicator(
    mode = "number",
    value = 13,
    number = {'suffix': "cities"},
    title = "Number of Jurisdiction with Soda taxes",
    domain = {'row': 0, 'column': 0},
    ))

fig_stats3.add_trace(go.Indicator(
    mode = "number",
    value = 600000000,
    number = {'prefix': "$"},
    title = "Reveneu Generated from Soda taxes",
    domain = {'row': 0, 'column': 1}))

fig_stats3.add_trace(go.Indicator(
    mode = "number+delta",
    value = 300,
    delta = {'reference': 400, 'relative': True},
    title = "Impact on Soda consumption",
   
    domain = {'row': 1, 'column': 0}))

fig_stats3.add_trace(go.Indicator(
    mode = "number",
    value = 44,
    number = {'suffix': "%"},
    title = "American Adults consuming atleast 1 sugary drink per day",   
    domain = {'row': 1, 'column': 1}))

fig_stats3.add_trace(go.Indicator(
    mode = "number",
    value = 41.7,
    number = {'suffix': "gallons"},
    title = "Sugary drinks consumed between 2009-2010",   
    domain = {'row': 2, 'column': 0}))

fig_stats3.add_trace(go.Indicator(
    mode = "number+delta",
    value = 30,
    number = {'suffix': "gallons"},
    delta = {'reference': 42, 'relative': True, 'valueformat':'.2%'},
    title = "Decreased in the year 2014",   
    domain = {'row': 2, 'column': 1}))


fig_stats3.update_layout(grid = {'rows': 3, 'columns': 2, 'pattern': "independent"})
page_introduction = html.Div([
        html.H5("Should there be taxes imposed on Soda?"),        
        html.Br(),       
        dbc.Row([
            dbc.Toast([
                dcc.Graph(figure = fig_stats3),
                html.P("""
                       Soda taxes, also known as sugar-sweetened beverage taxes, have been implemented in several cities and states in the US with varying success. Here are a few statistics on soda taxes in the US:
                       """),
                html.Ul([
                    html.Li("Number of Jurisdictions with Soda Taxes: As of 2021, there are 13 cities and one state in the US that have implemented soda taxes."),
                    html.Li("Revenue Generated: According to the Tax Foundation, cities with soda taxes generated approximately $600 million in revenue in 2018."),
                    html.Li("Impact on Soda Consumption: A study conducted in Berkeley, California, found that after the city implemented a soda tax, soda consumption decreased by 21%."),
                    html.Li("Health Benefits: A study in Philadelphia, where a soda tax was implemented in 2017, found that the tax led to a reduction in the consumption of sugary drinks and an increase in the consumption of water."),                    
                ]),
                html.P("""
                       Controversy: Soda taxes have been met with resistance from the beverage industry and some consumers who see it as a regressive tax that disproportionately impacts low-income communities.
                       It's important to note that the impact of soda taxes can vary greatly depending on the specific tax rate, jurisdiction, and socioeconomic factors, and additional research is needed to fully understand their effectiveness in reducing sugar-sweetened beverage consumption and improving public health.
                       """)
            ], header = "Here are some stats on soda consumption in the US", style = {"width":"100%"})    
        ]),
        html.Br(),
        html.Br(),                       
        dbc.Row([
            dbc.Col([
                 dbc.Toast([
                     html.Ul([
                        html.Li("Soda contains high amounts of added sugar, which contributes to a variety of health problems, including obesity, type 2 diabetes, and heart disease"),
                        html.Li("By taxing soda, governments can discourage its consumption and encourage people to choose healthier options."),
                        html.Li("Revenues generated from the tax can be used to fund public health initiatives, such as education campaigns about healthy eating, building more bike lanes, and funding research into preventative measures for health problems."),
                        html.Li("The tax would provide an additional financial incentive for soda companies to reduce the sugar content in their products and invest in developing healthier alternatives.")
                    ])
                 ], header = "Why Should Soda Taxes be imposed?", style = {"width":"100%"})   
            ]),
            
            dbc.Col([
                dbc.Toast([
                    html.Ul([
                        html.Li("The tax would unfairly burden lower-income individuals who disproportionately consume soda and may not have the means to switch to more expensive, healthier alternatives.")    ,
                        html.Li("There is limited evidence that taxes on soda actually reduce its consumption. In some cases, people may simply switch to other sugary drinks that are not taxed."),
                        html.Li("The tax would place an additional burden on the beverage industry, which could result in job losses and decreased economic activity. The industry may also pass the cost of the tax onto consumers in the form of higher prices, which would affect their purchasing power."),
                        html.Li("Ultimately, the decision to impose a tax on soda is a complex issue that requires careful consideration of both the potential benefits and drawbacks.")                        
                    ])    
                ], header = "Why shouldn't taxes be imposed",style = {"width":"100%"})    
            ])
        ]),     
        
    ])
                       
text_data_mining = html.Div([
        dbc.Accordion([
            dbc.AccordionItem([
                dbc.Row([
                    dbc.Col([
                        dbc.Toast([
                            html.P("""
                                   Data was collected from the newsapi.org website based on 4 different queries.
                                   For each query, around 100 articles were extracted. 
                                   """),
                            html.P(html.A("Link to code", href= "https:github.com/Akhilesh199797"))
                        ], header = "Extracting Data from News API", style = {"width":"100%"})    
                    ]),
                    dbc.Col([
                        dbc.Toast([
                            html.Img(src = "static/images/news_api_folder.png", style = {"width":"100%"}),
                            html.Br(),
                            dbc.Button("View Image", id="news-api-img-button", n_clicks=0),
                            dbc.Modal(
                                [
                                    dbc.ModalHeader(dbc.ModalTitle("Twitter API Data")),
                                    dbc.ModalBody([
                                        html.Img(src = "static/images/news_api_folder.png", style = {"width":"100%"}),
                                    ]),
                                ],
                                id="news-api-img-modal",
                                size="xl",
                                is_open=False,
                            ),
                        ], header = "Screenshot of the News API data", style = {"width":"100%"})
                    ])
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Toast([
                            html.P("""
                                   A twitter developer account was created and the bearer token was used to 
                                   extract tweets based on the location - United States of America. 
                                   The twitter API was queried for 4 different labels. 
                                   A single csv file containing tweets filtered according to a relevance count
                                   and the label was created. 
                                   """),
                            html.P(html.A("Link to code", href= "https:github.com/Akhilesh199797"))
                        ], header = "Extracting Data from Twitter API", style = {"width":"100%"})    
                    ]),
                    dbc.Col([
                        dbc.Toast([
                            html.Img(src = "static/images/twitter_api_data.png", style = {"width":"100%"}),
                            html.Br(),
                            dbc.Button("View Image", id="twitter-api-img-button", n_clicks=0),
                            dbc.Modal(
                                [
                                    dbc.ModalHeader(dbc.ModalTitle("Twitter API Data")),
                                    dbc.ModalBody([
                                        html.Img(src = "static/images/twitter_api_data.png", style = {"width":"100%"}),
                                    ]),
                                ],
                                id="twitter-api-img-modal",
                                size="xl",
                                is_open=False,
                            ),
                        ], header = "Screenshot of the Twitter API data", style = {"width":"100%"})
                    ])
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Toast([
                            html.P("""
                                   The data returned from news API has a limit of 200 characters for the
                                   Description key and the Content key in the json object returned. Hence, 
                                   it was required to webscrape each URL returned from the news api json. Selenium 
                                   was used and every <p> tag on each website was extracted to get the whole sites text data.
                                   """),
                            html.P(html.A("Link to code", href= "https:github.com/Akhilesh199797"))
                        ], header = "Webscraping URL's from newsapi json", style = {"width":"100%"})    
                    ]),
                                   
                    dbc.Col([
                        dbc.Toast([
                            html.Img(src = "static/images/webscrape_data.png", style = {"width":"100%"}),
                            html.Br(),                            
                            dbc.Button("View Image", id="web-img-button", n_clicks=0),
                            dbc.Modal(
                                [
                                    dbc.ModalHeader(dbc.ModalTitle("Webscraped Data")),
                                    dbc.ModalBody([
                                        html.Img(src = "static/images/webscrape_data.png", style = {"width":"100%"}),
                                    ]),
                                ],
                                id="web-img-modal",
                                size="xl",
                                is_open=False,
                            ),
                        ], header = "Screenshot of the Webscraped data", style = {"width":"100%"})
                    ])
                ])
            ], title = "Extracting Raw Data"),

            dbc.AccordionItem([
                dbc.Row([
                    dbc.Col([
                        dbc.Toast([
                            html.P("""
                                   Removing punctuation from text is a common preprocessing step in Natural Language Processing (NLP) tasks. 
                                   This helps to standardize the text and can also be useful for some NLP models that do not handle punctuation well. 
                                   There are various ways to remove punctuation from text, but one common method is to use string manipulation techniques such as string replace or regex.
                                   """),
                            html.A("View Code", href = "https://github.com"),
                            html.Br(),
                            html.Br(),
                            dbc.Button("View Data", id="puncts-button", n_clicks=0),
                            dbc.Modal([
                                    dbc.ModalHeader(dbc.ModalTitle("Data After removing punctions")),
                                    dbc.ModalBody(plot_table(df_tweets_punct))
                                ],
                              id="puncts-df-modal",
                              size="xl",
                              is_open=False,
                            )
                        ], header = "Removing Punctuations", style = {"width":"100%"})
                    ]),
                    dbc.Col([
                        dbc.Toast([
                            html.P("""
                                   Tokenization is the process of breaking down a larger piece of text into smaller units called tokens, which can be either words or subwords. 
                                   In NLP, tokenization is a crucial step for many tasks such as text classification, information retrieval, and text generation.
                                   In NLP libraries such as NLTK and spaCy, tokenization is implemented as a built-in function.
                                   For example, in NLTK you can use the word_tokenize() function to perform word-level tokenization:
                                   """),
                            html.A("View Code", href = "https://github.com"),
                            html.Br(),
                            html.Br(),
                            dbc.Button("View Data", id="token-button", n_clicks=0),                        
                            dbc.Modal([
                                    dbc.ModalHeader(dbc.ModalTitle("Data After Tokenization")),
                                    dbc.ModalBody(plot_table(df_tweets_tokenized))
                                ],
                              id="tokenized-df-modal",
                              size="xl",
                              is_open=False,
                            )
                        ], header = "Tokenisation", style = {"width":"100%"})
                    ])
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Toast([
                            html.P("""
                                   Stopword removal is a common preprocessing step in NLP, where stopwords are a list of commonly used words that carry little meaning, such as "the", "and", "of", etc. 
                                   The idea behind stopword removal is to remove these words from the text as they do not add significant value to the meaning of the text and can also increase the dimensionality of the data, making NLP models slower and less accurate.
                                   In NLP libraries such as NLTK and spaCy, stopword removal is implemented as a built-in function. For example, in NLTK, you can use the stopwords corpus and the remove_stopwords() function to remove stopwords from a list of tokens:
                                   """),
                            html.A("View Code", href = "https://github.com"),
                            html.Br(),
                            html.Br(),
                            dbc.Button("View Data", id="stops-button", n_clicks=0),
                            dbc.Modal([
                                    dbc.ModalHeader(dbc.ModalTitle("Data After Removing Stop Words")),
                                    dbc.ModalBody(plot_table(df_tweets_non_stop))
                                ],
                              id="non-stop-df-modal",
                              size="xl",
                              is_open=False,
                            )
                        ], header = "Removing Stop Words", style = {"width":"100%"})
                    ]),
                    dbc.Col([
                        dbc.Toast([
                            html.P("""
                                   Stemming is the process of reducing words to their base or root form, usually by removing suffixes (e.g. "running" -> "run"). In NLP, stemming is used to standardize words to their basic form so that words with the same meaning but with different endings (such as "run" and "running") can be treated as the same word.
                                   There are several popular stemming algorithms, including the Porter Stemming Algorithm, Snowball Stemming Algorithm (which is an extension of the Porter Stemming Algorithm), and the Lancaster Stemming Algorithm.
                                   In NLP libraries such as NLTK and spaCy, stemming is implemented as a built-in function. For example, in NLTK, you can use the PorterStemmer class from the nltk.stem module to perform stemming:
                                   """),
                            html.A("View Code", href = "https://github.com"),
                            html.Br(),
                            html.Br(),
                            dbc.Button("View Data", id="stemming-button", n_clicks=0),
                            dbc.Modal([
                                    dbc.ModalHeader(dbc.ModalTitle("Data After Stemming")),
                                    dbc.ModalBody(plot_table(df_tweets_stemmed))
                                ],
                              id="stemmed-df-modal",
                              size="xl",
                              is_open=False,
                            )
                        ], header = "Stemming", style = {"width":"100%"})
                    ])
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Toast([
                            html.P("""
                                   Lemmatization is the process of reducing words to their base or dictionary form, known as the lemma. Unlike stemming, which just removes suffixes, lemmatization considers the context of the word and its morphological structure to determine its lemma. This makes lemmatization more accurate than stemming, especially for irregular words, and results in a valid word that is semantically equivalent to the original word.
                                   In NLP libraries such as NLTK and spaCy, lemmatization is implemented as a built-in function. For example, in spaCy, you can use the .lemma_ property of a token to get its lemma:
                                   """),
                            html.A("View Code", href = "https://github.com"),
                            html.Br(),
                            html.Br(),
                            dbc.Button("View Data", id="lemmatization-button", n_clicks=0),
                            dbc.Modal([
                                    dbc.ModalHeader(dbc.ModalTitle("Data After Lemmatization")),
                                    dbc.ModalBody(plot_table(df_tweets_lemmatized))
                                ],
                              id="lemmatized-df-modal",
                              size="xl",
                              is_open=False,
                            )
                        ], header = "Stemming", style = {"width":"100%"})
                    ]),
                    dbc.Col([
                        dbc.Toast([
                            html.P("""
                                   CountVectorization is a method used in NLP to represent text data as numerical data. It's a way of encoding text data into a numerical format, such as a matrix, where each row represents a document, and each column represents a word, and each cell contains the count of the word in the document.
                                   The most basic form of CountVectorization is called the Bag of Words (BOW) model. The BOW model simply counts the number of times each word appears in the document, ignoring grammar and word order. The result is a sparse matrix where most cells contain a zero, since most words do not appear in most documents.
                                   """),
                            html.A("View Code", href = "https://github.com"),
                            html.Br(),
                            html.Br(),
                            dbc.Button("View Data", id="cv-button", n_clicks=0),                            
                            dbc.Modal([
                                    dbc.ModalHeader(dbc.ModalTitle("Data After Countvectorization")),
                                    dbc.ModalBody(plot_table(df_tweets_cv))
                                ],
                              id="cv-df-modal",
                              size="xl",
                              is_open=False,
                            )
                        ], header = "Count Vectorisation", style = {"width":"100%"})
                    ]),

                ])
            ], title = "Preprocessing Text data"),
            
            dbc.AccordionItem([
                dbc.Row([
                    dbc.Toast([
                        dcc.Graph(figure = plot_char_length_hist())
                    ], header = "Character Count", style = {"width":"100%"}
                    )
                ]),
                dbc.Row([
                    dbc.Toast([
                        dcc.Graph(figure = plot_word_count_hist())    
                    ], header = "Word Count",  style = {"width":"100%"})
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Toast([
                            dcc.Graph(figure = plot_unigram_bar("tweet"))
                        ], header = "Unigram Bar for Raw Tweets", style = {"width":"100%"})    
                    ]),
                    dbc.Col([
                        dbc.Toast([
                            dcc.Graph(figure = plot_unigram_bar("tweets_cleaned"))
                        ], header = "Unigram Bar for Cleaned Tweets", style = {"width":"100%"})    
                    ])
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Toast([
                            dcc.Graph(figure = plot_top_ngrams_barchart(df_tweets_filtered["tweet"],2))
                        ], header = "Bi Gram Bar for Raw Tweets", style = {"width":"100%"})    
                    ]),
                    dbc.Col([
                        dbc.Toast([
                            dcc.Graph(figure = plot_top_ngrams_barchart(df_tweets_filtered["tweets_cleaned"],2))
                        ], header = "Bi Gram Bar for Cleaned Tweets", style = {"width":"100%"})    
                    ])
                ])
            ], title = "Visualizing cleaned data")
        ], always_open = True)    
    ])     

raw_df_clustering = pd.read_csv("data/cleaned_data/df_tweets_cleaned.csv")
processed_df_clustering = pd.read_csv("data/cleaned_data/df_tweets_cv.csv")   
processed_df_arm = pd.read_csv("data/cleaned_data/tweets_trans.csv", error_bad_lines=False)
tab_text_analytics = html.Div([
        dbc.Tabs([
            dbc.Tab([
                html.Br(),
                html.H4("Overview"),
                dbc.Row([
                    dbc.Col([
                        dbc.Toast([
                            html.P("""
                                   On the data extraction and preparation tabs it was evident that tweets across multiple topics such as
                                   'Soda Tax', 'Sugar Tax', 'Sweetened Beverage Tax', 'Obesity' were extracted. But, the question arises regarding 
                                   the need to cluster the data when there is already data extracted for pre-defined topics.
                                   """),
                        ], style = {"width":"100%"}),
                        html.Br(),
                        html.Hr(),
                        dbc.Toast([
                            html.P("""
                                   Clustering text data is a powerful tool that can reveal more nuanced insights and relationships between topics and sentiments within large volumes of textual data. By utilizing clustering algorithms, it is possible to identify specific groups of words used in tweets that may be used in different topics. For example, if tweets were gathered under the topics of 'Sugar Tax' and 'Sweetened Beverage Tax', clustering could identify similar words and phrases used by individuals tweeting about these topics. These insights can help understand the underlying patterns and relationships between topics and subtopics within the data.
                                   """),
                            html.P("""
                                   Moreover, clustering can be used to categorize text documents into different groups or topics, making it easier to organize large collections of documents based on their underlying themes. This categorization can help identify common themes and patterns within a specific domain, making it easier to extract valuable insights from the data. Clustering can also be used for tasks such as text summarization, search engine optimization, and text anomaly detection, making it a versatile tool for analyzing text data in various contexts.
                                   """),
                            html.P("""
                                   Moreover, clustering can be used to categorize text documents into different groups or topics, making it easier to organize large collections of documents based on their underlying themes. This categorization can help identify common themes and patterns within a specific domain, making it easier to extract valuable insights from the data. Clustering can also be used for tasks such as text summarization, search engine optimization, and text anomaly detection, making it a versatile tool for analyzing text data in various contexts.
                                   """)
                        ], header = "Why Clustering?", style = {"width":"100%"})
                    ]),
                    dbc.Col([
                        html.Img(src = "static/images/clustering_text.png")
                    ])
                    
                ]),
                html.Br(),
                html.Hr(),
                html.H4("Data Prep"),
                dbc.Row([
                    dbc.Toast([
                        html.P("""
                               Before performing clustering on raw text data, it is important to properly prepare and preprocess the data to ensure that it is in a suitable format for analysis. The following are the common steps for data preparation for clustering on raw text data:
                               """),
                        html.Ul([
                            html.Li("Text cleaning: The raw text data often contains irrelevant information such as stop words, punctuation marks, numbers, and special characters that can negatively affect the performance of the clustering algorithm. Therefore, it is necessary to remove such unwanted elements through techniques such as tokenization and stop-word removal.")    ,
                            html.Li("Text normalization: Normalization is the process of converting all text data to a uniform format by removing unnecessary variations in text data. Common normalization techniques include stemming and lemmatization, which reduce words to their root form."),
                            html.Li("Text representation: Text data needs to be transformed into numerical form for clustering algorithms to process it. This can be done using techniques such as bag-of-words, term frequency-inverse document frequency (TF-IDF), or word embeddings."),
                            html.Li("Feature selection: Feature selection involves selecting the most relevant features or characteristics that will be used to compare and group the data. This step can help reduce the dimensionality of the data and improve the performance of the clustering algorithm."),
                            html.Li("Data scaling: Some clustering algorithms, such as k-means, are sensitive to the scale of the data. Therefore, it may be necessary to scale or normalize the data to ensure that all features have equal importance in the clustering process."),
                            html.Li("Data sampling: In cases where the text data is too large or there is a class imbalance in the data, it may be necessary to sample the data to improve the performance of the clustering algorithm.")
                        ]),
                        html.P("""
                               Overall, data preparation for clustering on raw text data involves several important steps such as text cleaning, normalization, representation, feature selection, data scaling, and sampling. These steps are critical for ensuring that the data is in a suitable format for analysis and can improve the performance and accuracy of the clustering algorithm.
                               """)
                    ], header = "Steps involved in preparing data", style = {"width":"100%"})
                ]),
                html.Br(),                
                dbc.Row([
                    dbc.Col([
                        dbc.Button("View Raw Data", id="raw-clustering-button", n_clicks=0),
                        dbc.Modal([
                                dbc.ModalHeader(dbc.ModalTitle("Raw Twitter Data")),
                                dbc.ModalBody(plot_table(raw_df_clustering))
                            ],
                          id="raw-clustering-df-modal",
                          size="xl",
                          is_open=False,
                        )
                    ]),
                    dbc.Col([
                        dbc.Button("View Processed Data", id="processed-clustering-button", n_clicks=0),
                        dbc.Modal([
                                dbc.ModalHeader(dbc.ModalTitle("Processed Data")),
                                dbc.ModalBody(plot_table(processed_df_clustering))
                            ],
                          id="processed-clustering-df-modal",
                          size="xl",
                          is_open=False,
                        )
                    ])
                ]),
                html.Br(),
                html.Hr(),
                html.H4("Code"),
                dbc.Row([
                    dbc.Col([
                        html.P("Python Code")    
                    ]),
                    dbc.Col([
                        html.P("R Code")    
                    ])
                ]),
                html.Br(),
                html.Hr(),
                html.H4("Results"),
                dbc.Row([
                    dbc.Accordion([
                        dbc.AccordionItem([
                            dbc.Row([
                                dbc.Toast([
                                    html.Img(src = "static/images/elbow_plot.png")
                                ], header = "Elbow Plot for determing clusters", style = {"width":"100%"})    
                            ]),
                            html.Br(),
                            html.Hr(),
                            html.H6("Sillouhete score comparision"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Toast([
                                        html.Img(src = "static/images/sil_score_2.png")
                                    ],header = "n-Clusters = 2", style = {"width":"100%"})    
                                ]),
                                dbc.Col([
                                    dbc.Toast([
                                        html.Img(src = "static/images/sil_score_3.png")
                                    ],header = "n-Clusters = 3", style = {"width":"100%"})
                                ])
                            ]),
                            html.Br(),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Toast([
                                        html.Img(src = "static/images/sil_score_4.png")
                                    ],header = "n-Clusters = 4", style = {"width":"100%"})    
                                ]),
                                dbc.Col([
                                    dbc.Toast([
                                        html.Img(src = "static/images/sil_score_5.png")
                                    ],header = "n-Clusters = 5", style = {"width":"100%"})
                                ])
                            ]),
                            html.Br(),
                            html.Hr(),
                            html.H6("Looking words in each cluster"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Toast([
                                        html.Img(src = "static/images/cluster1.png")
                                    ],header = "Cluster 1", style = {"width":"100%"})    
                                ]),
                                dbc.Col([
                                    dbc.Toast([
                                        html.Img(src = "static/images/cluster2.png")
                                    ],header = "Cluster 2", style = {"width":"100%"})
                                ])
                            ]),
                            html.Br(),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Toast([
                                        html.Img(src = "static/images/cluster3.png")
                                    ],header = "Cluster 3", style = {"width":"100%"})    
                                ]),
                                dbc.Col([
                                    dbc.Toast([
                                        html.Img(src = "static/images/cluster4.png")
                                    ],header = "Cluster 4", style = {"width":"100%"})
                                ])
                            ]),
                        ], title = "K-Means Clustering"),
                        dbc.AccordionItem([
                            html.Img(src = "static/images/dendro.png", style = {"width":"100%"})
                        ], title = "Hierarchical Clustering")
                    ], always_open = True)
                ]),
                html.Br(),
                html.Hr(),
                html.H4("Conclusion"),
                dbc.Row([
                    
                ])
            ], label = "Clustering"),
            
            dbc.Tab([
                html.Br(),
                html.H4("Overview"),
                dbc.Row([            
                    dbc.Toast([
                        html.P("""
                               Association rule mining is a popular technique in data mining that is used to discover relationships between variables in large datasets. When applied to textual data, it can help identify patterns and relationships between words or phrases that may be difficult to discern through other methods.

                                 Here's an overview of the use of association rule mining on textual data:
                               """),
                               
                        html.Ul([
                          html.Li("Preprocessing: Textual data needs to be preprocessed before applying association rule mining. This may include tasks such as removing stop words, stemming, and tokenization.")  ,
                          html.Li("Frequent itemset mining: Association rule mining requires identifying frequent itemsets, which are sets of words or phrases that frequently appear together in the text data. To identify frequent itemsets, algorithms such as Apriori or FP-Growth can be used."),
                          html.Li("Generating association rules: Once frequent itemsets are identified, association rules can be generated. Association rules are statements of the form “if X, then Y.” For example, “if people buy bread, they are likely to buy milk.” The support and confidence measures are used to determine the strength of the association between X and Y."),
                          html.Li("Interpretation: The results of association rule mining can be interpreted to gain insights into the relationships between words or phrases in the text data. This can help identify patterns, trends, and relationships that may not be apparent from a simple analysis of the text data.")
                        ]),
                        html.P("""
                               Applications of association rule mining on textual data include market basket analysis, where the technique can be used to identify products that are frequently purchased together, and text mining, where it can be used to identify relationships between words or phrases in large collections of text data.
                               Overall, association rule mining is a valuable tool for analyzing textual data, providing insights into relationships between words or phrases that may be difficult to uncover using other methods.
                               """),
                                          
                    ], header = "How can ARM be used in our Context?",style = {"width":"100%"})    
                  
                ]),
                html.Br(),
                dbc.Row([
                    dbc.Toast([
                        html.Img(src = "static/images/supp_conf_lift.png", style = {"width":"60%"}),
                        html.P("""
                               Support: Support is a measure of the frequency of occurrence of an itemset in the dataset. It is calculated as the proportion of transactions in the dataset that contain the itemset. For example, if we have a dataset of 1,000 transactions, and the itemset {milk, bread} appears in 200 transactions, then the support of {milk, bread} is 200/1,000 = 0.2.
                               """),
                        html.P("""
                               Confidence: Confidence is a measure of the strength of the association between the antecedent and the consequent in a rule. It is calculated as the proportion of transactions that contain both the antecedent and the consequent, out of the transactions that contain the antecedent. For example, if we have a rule {milk} → {bread} with 100 transactions containing both milk and bread, and 200 transactions containing milk, then the confidence of the rule is 100/200 = 0.5.
                               """),
                        html.P("""
                               Lift: Lift is a measure of the strength of the association between the antecedent and the consequent in a rule, compared to what would be expected if they were independent. A lift value of 1 indicates that the antecedent and consequent are independent, while a lift value greater than 1 indicates a positive association between them. Lift is calculated as the ratio of the support of the itemset containing both the antecedent and the consequent to the product of the supports of the antecedent and the consequent. For example, if we have a rule {milk} → {bread} with a support of 0.2, and the support of milk is 0.4 and the support of bread is 0.3, then the lift of the rule is (0.2) / (0.4 * 0.3) = 1.67. This indicates that the presence of milk is associated with 67% higher likelihood of the presence of bread, compared to what would be expected if milk and bread were independent.
                               """),
                        
                    ], header = ["Support, Confidence and Lift"],style = {"width":"100%"})    
                ]),
                html.Br(),
                html.Hr(),
                html.H4("Data Prep"),
                dbc.Row([
                   dbc.Toast([
                       html.P("""
                              Text data processing is an essential step in preparing textual data for association rule mining. Here are some common steps in text data processing for association rule mining:
                              """),
                       html.Ul([
                            html.Li("""
                                    Tokenization: Tokenization involves breaking up the text data into individual tokens or words. This step helps in identifying patterns and relationships between words or phrases in the text data.
                                    """),
                            html.Li("""
                                    Stop word removal: Stop words are common words such as "the," "is," and "a" that do not add much meaning to the text data. Removing stop words helps reduce noise in the text data and improves the accuracy of association rule mining.
                                    """),
                            html.Li("""
                                   Stemming: Stemming involves reducing words to their root or base form. For example, the words "running," "runs," and "ran" can be stemmed to "run." Stemming helps reduce the number of unique words in the text data and makes it easier to identify patterns and relationships between words.
                                   """),
                            html.Li("""
                                    Filtering: Filtering involves removing words or phrases that are not relevant to the analysis. For example, if the analysis is focused on a particular topic, irrelevant words or phrases can be removed from the text data.
                                    """),
                            html.Li("""
                                    Transformation: Transformation involves converting the text data into a format that can be used for association rule mining. For example, the text data can be converted into a transaction database format, where each transaction represents a document or a set of documents.
                                    """),
                            html.Li("""
                                    Frequent itemset mining: Frequent itemset mining involves identifying sets of words or phrases that frequently appear together in the text data. This step is essential for generating association rules.
                                    """),
                            html.Li("""
                                    Generating association rules: Once frequent itemsets are identified, association rules can be generated. Association rules are statements of the form "if X, then Y," which represent the relationships between words or phrases in the text data.
                                    """)
                       ]),
                       html.P("""
                              Overall, text data processing is a critical step in preparing textual data for association rule mining. By following these steps, it is possible to identify patterns and relationships between words or phrases in the text data and gain insights into the underlying structure of the data.
                              """)
                   ], header = "Steps involved in Preparing and Transforming the Raw Data", style = {"width":"100%"}) 
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Button("View Raw Data", id="raw-arm-button", n_clicks=0),
                        dbc.Modal([
                                dbc.ModalHeader(dbc.ModalTitle("Raw Twitter Data")),
                                dbc.ModalBody(plot_table(raw_df_clustering))
                            ],
                          id="raw-arm-df-modal",
                          size="xl",
                          is_open=False,
                        )
                    ]),
                    dbc.Col([
                        dbc.Button("View Processed Data", id="processed-arm-button", n_clicks=0),
                        dbc.Modal([
                                dbc.ModalHeader(dbc.ModalTitle("Processed Data")),
                                dbc.ModalBody(plot_table(processed_df_arm))
                            ],
                          id="processed-arm-df-modal",
                          size="xl",
                          is_open=False,
                        )
                    ])
                ]),
                html.Br(),
                html.Hr(),
                html.H4("Code"),
                dbc.Row([
                    
                ]),
                html.Br(),
                html.Hr(),
                html.H4("Results"),
                dbc.Row([
                    dbc.Col([
                        dbc.Toast([
                            html.Img(src = "static/images/arm_rules_sup.png", style = {"width":"100%"})
                        ], header = "Top 15 Rules Sorted by Support", style = {"width":"100%"})    
                    ]),
                    dbc.Col([
                        dbc.Toast([
                            html.Img(src = "static/images/arm_net_supp.png", style = {"width":"100%"})
                        ], header = "Top 15 Rules Sorted by Support", style = {"width":"100%"})
                    ])
                ]),
                html.Br(),
                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        dbc.Toast([
                            html.Img(src = "static/images/arm_rules_conf.png", style = {"width":"100%"})
                        ], header = "Top 15 Rules Sorted by Confidence", style = {"width":"100%"})    
                    ]),
                    dbc.Col([
                        dbc.Toast([
                            html.Img(src = "static/images/arm_net_conf.png", style = {"width":"100%"})
                        ], header = "Top 15 Rules Sorted by Confidence", style = {"width":"100%"})
                    ])
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Toast([
                            html.Img(src = "static/images/arm_rules_lift.png", style = {"width":"100%"})
                        ], header = "Top 15 Rules Sorted by Lift", style = {"width":"100%"})    
                    ]),
                    dbc.Col([
                        dbc.Toast([
                            html.Img(src = "static/images/arm_net_lift.png",style = {"width":"100%"})
                        ], header = "Top 15 Rules Sorted by Lift", style = {"width":"100%"})
                    ])
                ]),
                html.H4("Conclusion"),
                dbc.Row([
                    
                ])
                
            ], label = "Association Rule Mining"),
            dbc.Tab([
                html.Br(),
                html.H4("Overview"),
                dbc.Row([
                    html.P("""
                           Latent Dirichlet Allocation (LDA) is a statistical model used in machine learning and natural language processing for topic modeling. The goal of LDA is to identify the topics that are present in a large collection of text data.
                           """),
                    html.P("""
                            The basic idea behind LDA is that each document in the collection is assumed to be a mixture of several topics, and each topic is characterized by a distribution of words. LDA attempts to discover these latent topics by analyzing the patterns of word co-occurrences in the documents.                           
                           """),
                    html.P("""
                            The Dirichlet distribution is used in LDA to model the distribution of topics in the documents and the distribution of words in the topics. The Dirichlet distribution is a probability distribution over the simplex, meaning that it assigns probabilities to all possible combinations of values that sum to 1. In the case of LDA, the Dirichlet distribution is used to model the distribution of topics and words in the corpus.                           
                           """),
                    html.P("""
                           Overall, LDA is a powerful tool for analyzing large collections of text data and extracting meaningful insights from them. It has applications in a variety of fields, including information retrieval, recommendation systems, and social media analysis.
                           """)
                ]),
                html.Br(),
                html.Hr(),
                html.H4("Data Prep"),
                dbc.Row([
                    html.P("""
                           Corpus Creation: First, we need to gather a collection of documents that we want to analyze. This collection of documents is called a "corpus." The corpus can be created by web scraping, document collection or dataset from any source.
                           """),
                           
                    html.P("""
                            Text Preprocessing: Once we have the corpus, we need to preprocess the text to remove any noise or irrelevant information. This includes removing punctuation, stop words (common words such as "a," "an," and "the"), and words that are too common or too rare. We can also perform stemming or lemmatization to reduce words to their root form.                           
                           """),
                    html.P("""
                           Text Representation: After preprocessing, we need to represent the text in a numerical format that LDA can understand. This is typically done using a bag-of-words model, where each document is represented as a vector of word frequencies. Other text representations, such as term frequency-inverse document frequency (TF-IDF), can also be used.
                           """),
                    html.P("""
                           Model Training: Once the data is preprocessed and represented numerically, we can train an LDA model on the corpus. This involves specifying the number of topics we want to identify and running an algorithm to infer the topic distributions for each document and the word distributions for each topic.
                           """),
                    html.P("""
                           Model Evaluation: Finally, we need to evaluate the performance of the LDA model by assessing the coherence and interpretability of the topics. This involves examining the top words in each topic and determining if they make intuitive sense and are coherent. We may need to adjust the number of topics or the preprocessing parameters to improve the quality of the topics.
                           """)                           
                ]),
                html.Br(),
                html.Hr(),
                html.H4("Code"),
                dbc.Row([
                    
                ]),
                html.Br(),
                html.Hr(),
                html.H4("Results"),
                dbc.Row([
                    html.Img(src = "static/images/lda_plot1.png", style = {"width":"100%"})
                ]),
                dbc.Row([
                    html.Img(src = "static/images/lda_plot2.png", style = {"width":"100%"})
                ]),
                html.Br(),
                html.Hr(),
                html.H4("Conclusion"),
                dbc.Row([
                    
                ])
            ], label = "Latent Direchelet Allocation")
        ])        
    ])                                   
sidebar = html.Div(
    [
        html.H2("Should soda be taxed?", className="display-4"),
        html.Hr(),
        html.P(
            "Using text analytics to debate on a soda taxes", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Introduction", href="/", active="exact"),
                dbc.NavLink("Text Data Mining", href="/page-1", active="exact"),
                dbc.NavLink("Text Analytics -Unsupervised", href="/page-2", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return page_introduction
    elif pathname == "/page-1":
        return text_data_mining
    elif pathname == "/page-2":
        return tab_text_analytics
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )

def toggle_modal(n1, is_open):
    if n1:
        return not is_open
    return is_open

app.callback(
    Output("twitter-api-img-modal", "is_open"),
    Input("twitter-api-img-button", "n_clicks"),
    State("twitter-api-img-modal", "is_open"),
)(toggle_modal)

app.callback(
    Output("news-api-img-modal", "is_open"),
    Input("news-api-img-button", "n_clicks"),
    State("news-api-img-modal", "is_open"),
)(toggle_modal)

app.callback(
    Output("web-img-modal", "is_open"),
    Input("web-img-button", "n_clicks"),
    State("web-img-modal", "is_open"),
)(toggle_modal)

app.callback(
    Output("puncts-df-modal", "is_open"),
    Input("puncts-button", "n_clicks"),
    State("puncts-df-modal", "is_open"),
)(toggle_modal)

app.callback(
    Output("tokenized-df-modal", "is_open"),
    Input("token-button", "n_clicks"),
    State("tokenized-df-modal", "is_open"),
)(toggle_modal)

app.callback(
    Output("non-stop-df-modal", "is_open"),
    Input("stops-button", "n_clicks"),
    State("non-stop-df-modal", "is_open"),
)(toggle_modal)

app.callback(
    Output("stemmed-df-modal", "is_open"),
    Input("stemming-button", "n_clicks"),
    State("stemmed-df-modal", "is_open"),
)(toggle_modal)

app.callback(
    Output("lemmatized-df-modal", "is_open"),
    Input("lemmatization-button", "n_clicks"),
    State("lemmatized-df-modal", "is_open"),
)(toggle_modal)

app.callback(
    Output("cv-df-modal", "is_open"),
    Input("cv-button", "n_clicks"),
    State("cv-df-modal", "is_open"),
)(toggle_modal)

app.callback(
    Output("raw-clustering-df-modal", "is_open"),
    Input("raw-clustering-button", "n_clicks"),
    State("raw-clustering-df-modal", "is_open"),
)(toggle_modal)

app.callback(
    Output("processed-clustering-df-modal", "is_open"),
    Input("processed-clustering-button", "n_clicks"),
    State("processed-clustering-df-modal", "is_open"),
)(toggle_modal)

app.callback(
    Output("raw-arm-df-modal", "is_open"),
    Input("raw-arm-button", "n_clicks"),
    State("raw-arm-df-modal", "is_open"),
)(toggle_modal)

app.callback(
    Output("processed-arm-df-modal", "is_open"),
    Input("processed-arm-button", "n_clicks"),
    State("processed-arm-df-modal", "is_open"),
)(toggle_modal)

if __name__ == "__main__":
    app.run_server(debug = True)