# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 13:02:20 2023

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


root_path = "data/cleaned_data"
df_tweets_filtered = pd.read_csv("%s/df_tweets_cleaned.csv"%root_path)

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
df_tweets_punct = pd.read_csv("%s/df_tweets_punct.csv"%root_path)
df_tweets_non_stop = pd.read_csv("%s/df_tweets_non_stop.csv"%root_path)
df_tweets_tokenized = pd.read_csv("%s/df_tweets_tokenized.csv"%root_path)
df_tweets_stemmed = pd.read_csv("%s/df_tweets_stemmed.csv"%root_path)
df_tweets_lemmatized = pd.read_csv("%s/df_tweets_lemmatized.csv"%root_path)
df_tweets_cv = pd.read_csv("%s/df_tweets_cv.csv"%root_path)
print(df_tweets_cv.shape)

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
                            html.P(html.A("Link to code", href= "https://github.com/Akhilesh97/text-mining-project/blob/main/Get_DAta.ipynb", target = "_blank"))
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
                            html.P(html.A("Link to code", href= "https://github.com/Akhilesh97/text-mining-project/blob/main/Get_DAta.ipynb", target = "_blank"))
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
                            html.P(html.A("Link to code", href= "https://github.com/Akhilesh97/text-mining-project/blob/main/Get_DAta.ipynb", target = "_blank"))
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
                            html.A("View Code", href= "https://github.com/Akhilesh97/text-mining-project/blob/main/data_processing.ipynb", target = "_blank"),
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
                            html.A("View Code", href= "https://github.com/Akhilesh97/text-mining-project/blob/main/data_processing.ipynb", target = "_blank"),
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
                            html.A("View Code", href= "https://github.com/Akhilesh97/text-mining-project/blob/main/data_processing.ipynb", target = "_blank"),
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
                            html.A("View Code", href= "https://github.com/Akhilesh97/text-mining-project/blob/main/data_processing.ipynb", target = "_blank"),
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
                            html.A("View Code", href= "https://github.com/Akhilesh97/text-mining-project/blob/main/data_processing.ipynb", target = "_blank"),
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
                            html.A("View Code", href= "https://github.com/Akhilesh97/text-mining-project/blob/main/data_processing.ipynb", target = "_blank"),
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

raw_df_clustering = pd.read_csv("%s/df_tweets_cleaned.csv"%root_path)
processed_df_clustering = pd.read_csv("%s/df_tweets_cv.csv"%root_path)
processed_df_arm = pd.read_csv("%s/tweets_trans.csv"%root_path, error_bad_lines=False)
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
                        html.Img(src = "static/images/clustering_text.png", style = {"width":"100%"}),
                        html.Img(src = "static/images/clustering_img1.png", style = {"width":"100%"}),

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
                        html.P("Python was used to perform k-means clustering and plot the elbow and sillouhete plots"),
                        html.A("Link to code", href = "https://github.com/Akhilesh97/text-mining-project/blob/main/k-means-clustering.ipynb")
                    ]),
                    dbc.Col([
                        html.P("R was used to perform Hierarchical Clustering"),
                        html.A("Link to code", href = "https://github.com/Akhilesh97/text-mining-project/blob/main/hirerachical_clustering.R")
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
                                    html.P("""
                                    The elbow method is a common technique used in clustering analysis to determine the optimal number of clusters to use in a particular dataset. The method involves plotting the within-cluster sum of squares (WCSS) against the number of clusters, where the WCSS represents the sum of the squared distances between each data point and its assigned cluster centroid.
                                    """),
                                    html.P("""
                                    To use the elbow method, one must first run the clustering algorithm on the dataset for a range of different cluster numbers. Then, one can plot the resulting WCSS values against the number of clusters. As the number of clusters increases, the WCSS value will generally decrease, as more clusters will result in smaller distances between the data points and their assigned centroids. However, beyond a certain number of clusters, the reduction in WCSS will start to diminish, resulting in a more gradual slope in the plot.
                                    """),
                                    html.P("""
                                    The optimal number of clusters can be determined by identifying the "elbow" point in the plot, which is the point of inflection where the rate of WCSS reduction sharply decreases. This elbow point represents the trade-off between clustering accuracy and complexity. The optimal number of clusters is the point after which the reduction in WCSS becomes marginal or insignificant, suggesting that additional clusters will not improve the quality of the clustering.
                                    """),
                                    html.Br(),
                                    html.Hr(),
                                    html.Img(src = "static/images/elbow_plot.png"),
                                    html.Br(),
                                    html.Hr(),
                                    html.P("""
                                    While the elbow method is a useful tool for determining the optimal number of clusters, it is not always clear-cut. In some cases, there may not be a clear elbow point, or there may be multiple potential elbow points. Additionally, the optimal number of clusters can vary depending on the data and the specific clustering algorithm being used. Therefore, it is important to use the elbow method in conjunction with other evaluation techniques, such as silhouette analysis or gap statistics, to ensure that the clustering results are meaningful and accurate.
                                    """),
                                ], header = "Elbow Plot for determing clusters", style = {"width":"100%"})
                            ]),
                            html.Br(),
                            html.Hr(),
                            html.H6("Sillouhete score comparision"),
                            html.P("""
                            The optimal number of clusters can be determined by calculating the average silhouette score across all data points for a range of different cluster numbers. The number of clusters with the highest average silhouette score is considered to be the optimal number of clusters.
                            """),
                            html.P("""
                            The advantage of using the silhouette score over the elbow method is that it can provide more nuanced insights into the clustering performance, especially when the elbow point is not clear. Additionally, the silhouette score can be used with any clustering algorithm, whereas the elbow method is specific to algorithms that use the within-cluster sum of squares (WCSS) metric.
                            """),
                            html.P("""
                            Overall, the silhouette score is a useful technique for evaluating the clustering performance and identifying the optimal number of clusters, especially when combined with other evaluation metrics such as the elbow method.
                            """),
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
                            html.H6("Looking at words in each cluster"),
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
                            html.Br(),
                            html.Hr(),
                            dbc.Row([
                                html.P("""
                                Based on the clusters formed and the words found in each cluster, it is difficult to determine the exact context or theme of the text data being analyzed. However, we can make some general observations about each cluster.
                                """),
                                html.P("""
                                Cluster 0 seems to contain words related to health and wellness, with mentions of problems, studies, childhood, industry, and junk food. There are also words related to policy and advocacy, such as levy, industry, and naacp.
                                """),
                                html.P("""
                                Cluster 1 appears to contain words related to healthcare and medicine, with mentions of cancer, drug, diabetes, and doctor. There are also words related to government and policy, such as government, guideline, and policy.
                                """),
                                html.P("""
                                Cluster 2 seems to contain words related to technology and entertainment, with mentions of video games, athlete, hand, and vision. There are also words related to health and wellness, such as sleep, effect, and negative.
                                """),
                                html.P("""
                                Cluster 3 contains words related to demographics and politics, with mentions of american, african, united, and nigeria. There are also words related to size and quantity, such as large, rate, and half.
                                """),
                                html.P("""
                                It is worth noting that these observations are based solely on the words present in each cluster, and the context and meaning of the text data may not be fully captured by this analysis. Further analysis and exploration may be necessary to gain a better understanding of the underlying themes and topics present in the text data.
                                """),
                            ])
                        ], title = "K-Means Clustering"),
                        dbc.AccordionItem([
                            html.Br(),
                            html.Hr(),
                            html.Img(src = "static/images/dendro.png", style = {"width":"100%"})
                        ], title = "Hierarchical Clustering")
                    ], always_open = True)
                ]),
                html.Br(),
                html.Hr(),
                html.H4("Conclusion"),
                dbc.Row([
                    html.P("""
                    Here's a summary of what we discussed about applying clustering for textual data:
                    """),
                    html.P("""
                    Clustering can be used to categorize text data into different groups or topics, which can be useful for organizing large collections of documents or for identifying common themes within a specific domain. Before performing clustering on text data, it is important to pre-process the data by cleaning, normalizing, and transforming the text into a numerical format that can be used for clustering algorithms.
                    """),
                    html.P("""
                    There are several clustering algorithms that can be applied to text data, such as k-means, hierarchical clustering, and density-based clustering. The choice of algorithm depends on the specific requirements of the data and the goals of the analysis.
                    """),
                    html.P("""
                    To determine the optimal number of clusters, several techniques can be used, including the elbow method and silhouette score. These techniques help in identifying the number of clusters that provide the best balance between cluster separation and cohesion.
                    """),
                    html.P("""
                    In the example provided earlier, we saw four clusters of words with different themes. The words clustered in each group were:
                    Cluster 0: recent, soft, study, problem, levy, reduced, childhood, industry, claimed, junk, nature, necessary, naacp, need, negative
                    Cluster 1: cancer, drug, cause, diabetes, diet, increase, policy, effect, guideline
                    Cluster 2: video, game, sleep, athlete, hand, body, vision, young, chest, tech, tunnel
                    Cluster 3: rate, woman, large, term, american, loose, african, nigeria, united
                    """),
                    html.P("""
                    By analyzing the clusters of words, we can gain a better understanding of the topics and themes present in the text data. For example, cluster 1 includes words related to health and medical issues such as cancer, diabetes, and drugs, while cluster 2 includes words related to technology and gaming. These clusters can be used for various applications such as sentiment analysis, topic modeling, and text classification.
                    """),
                    html.P("""
                    In summary, the clusters of words identified using k-means clustering can provide valuable insights into the underlying themes and patterns in the text data. By analyzing these clusters, we can gain a better understanding of the topics and themes present in the text data, which can be useful for a wide range of applications in various domains
                    """)
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
                    html.P("""
                    The code to generate the Rules is written in R and can be found on the link below"""),
                    html.A("Link to code", href = "https://github.com/Akhilesh97/text-mining-project/blob/main/ARM.R")
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
                html.Br(),
                html.Hr(),
                dbc.Row([
                    html.P("""
                    These results are the output of Association Rule Mining algorithm applied to a dataset containing text data. The output provides several rules that show the association between different terms (words) present in the text.
                    """),
                    html.P("""
                    Each row in the output represents a rule, with the following columns:
                    """),
                    html.Ul([
                        html.Li("""
                        lhs: the left-hand side of the rule, which contains the antecedent items. For example, the rule {sugar} => {obesity} means that when "sugar" is present in a text, "obesity" is also likely to be present in the same text.
                        """),
                        html.Li("""
                        rhs: the right-hand side of the rule, which contains the consequent items. For example, the same rule {sugar} => {obesity} also means that when "obesity" is present in a text, "sugar" is also likely to be present in the same text."
                        """),
                        html.Li("""
                        support: the frequency of the rule, or the proportion of texts that contain both the antecedent and consequent items. For example, the rule {sugar} => {obesity} has a support of 0.11764706, which means that 11.76% of all texts contain both "sugar" and "obesity".
                        """),
                        html.Li("""
                        confidence: the probability of the consequent given the antecedent. For example, the rule {sugar} => {obesity} has a confidence of 0.31111111, which means that when "sugar" is present in a text, there is a 31.11% chance that "obesity" is also present in the same text."
                        """),
                        html.Li("""
                        coverage: the proportion of texts that contain the antecedent. For example, the rule {sugar} => {obesity} has a coverage of 0.37815126, which means that 37.81% of all texts contain "sugar".
                        """)
                    ]),
                    html.P("""
                    In general, these results can be used to identify interesting patterns or associations in text data, and to generate hypotheses for further investigation. For example, the rule {sugar} => {obesity} with a high lift and confidence suggests that there may be a link between sugar consumption and obesity, which could be further explored using more advanced statistical methods.
                    lift: the strength of association between the antecedent and consequent, normalized by their individual frequencies. A lift greater than 1 indicates a positive association, while a lift less than 1 indicates a negative association. For example, the rule {sugar} => {obesity} has a lift of 0.4746439, which means that the presence of "sugar" in a text reduces the likelihood of "obesity" being present by a factor of 0.47 compared to their individual frequencies.
                    count: the number of texts that contain both the antecedent and consequent items. For example, the rule {sugar} => {obesity} has a count of 14, which means that there are 14 texts that contain both "sugar" and "obesity".
                    """),
                ]),
                html.Br(),
                html.Hr(),
                html.H4("Conclusion"),
                dbc.Row([
                    html.P("""
                    From the association rule mining results, we can see that there are several interesting patterns between the different terms in the dataset. Some of the key findings are:
                    """),
                    html.Ul([
                       html.Li("""
                       Sugar is strongly associated with obesity, with a confidence of 31.11%, and a lift of 0.47, suggesting a negative correlation.
                       """),
                       html.Li("""
                       Drink is also associated with sugar, with a high confidence of 81.81% and a lift of 2.16, suggesting a positive correlation.
                       """),
                       html.Li("""
                       Girl is strongly associated with sugar, with a confidence of 100%, and a lift of 2.64, suggesting a positive correlation.
                       """),
                       html.Li("""
                       Obesity is associated with several different terms, including disease, research, year, health, drug, girl, and child, with varying degrees of confidence and lift.
                       """),
                       html.Li("""
                       Overall, these findings suggest that there may be important relationships between these terms, particularly in the context of health and nutrition. Further analysis may be necessary to fully understand the implications of these associations and how they can be used to improve public health.
                       """)
                    ]),
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
                    html.P("The code for LDA was written in Python and can be found on the link below"),
                    html.A("Link to code", href = "https://github.com/Akhilesh97/text-mining-project/blob/main/LDA.ipynb")
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
                dbc.Row([
                    dbc.Toast([
                        html.P("""
                            Based on the provided output, it looks like the LDA model has identified four topics in the input data. Each topic is represented by a list of words and their corresponding weights. The weight for each word indicates the importance of that word in the given topic.
                        """),
                        html.Ul([
                            html.Li("""
                            Topic 0 seems to be related to health and diet, with words like "product", "diet", and "cancer" having high weights in this topic.
                            """),
                            html.Li("""
                            Topic 1 appears to be related to video games, with words like "video", "game", and "rate" having high weights in this topic.
                            """),
                            html.Li("""
                            Topic 2 seems to be related to healthcare and wellness, with words like "healthcare", "diabetes", and "weight" having high weights in this topic.
                            """),
                            html.Li("""
                            Topic 3 seems to be related to child growth and development, with words like "childhood", "growth", and "associated" having high weights in this topic.
                            """),
                        ]),
                        html.P("""
                        It is important to note that the interpretation of topics is subjective and depends on the context of the data and the domain knowledge of the analyst. Further analysis and exploration may be needed to validate and refine these topics.
                        """)
                    ], header = "Explanation of Results", style = {"width":"100%"})
                ]),
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
                dbc.NavLink("Text Data Mining", href="/datamining", active="exact"),
                dbc.NavLink("Text Analytics - Unsupervised", href="/unsupervised", active="exact"),
                dbc.NavLink("Text Classification - Supervised", href="/supervised", active="exact"),
                dbc.NavLink("Neural Networks - ", href = "/neural-nets", active = "exact")
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

train_nb = pd.read_csv("%s/train_nb.csv"%root_path)
test_nb = pd.read_csv("%s/test_nb.csv"%root_path)
tab_text_classification = html.Div([
        dbc.Tabs([
            dbc.Tab([

                    html.Br(),
                    html.H4("Overview"),
                    html.Br(),
                    dbc.Toast([
                        html.P("""
                               Naive Bayes is a machine learning algorithm that is widely used for classification tasks. It is based on Bayes' theorem, which provides a way to calculate the probability of a certain event based on prior knowledge of conditions that might be related to the event. In the context of Naive Bayes, these conditions are the features of the input data.
                               """),
                        html.P("""
                               The algorithm is called "naive" because it makes the simplifying assumption that the features of the input data are independent of each other. This assumption allows the algorithm to make calculations more efficiently, but it may not always hold true in practice. Despite this simplification, Naive Bayes has been shown to work well in many practical applications, such as text classification, spam filtering, and sentiment analysis.
                               """),
                        html.P("""
                               To use Naive Bayes, the algorithm must be trained on a set of labeled examples, where each example consists of a set of input features and a corresponding output label. For example, in a spam filtering application, the input features might be the words in an email message, and the output label might be "spam" or "not spam". The algorithm uses these examples to learn how to predict the output label for new, unseen examples.
                               """)
                    ], header = "General Introduction", style = {"width":"100%"}),
                    html.Br(),
                    html.Hr(),
                    dbc.Row([
                         dbc.Col([
                             dbc.Toast([
                                 html.P("""
                                        The first step in Naive Bayes is to calculate the prior probability of each output label, based on the frequency of each label in the training data. For example, if 60% of the training examples are labeled as "spam", then the prior probability of "spam" would be 0.6.
                                        """),
                                 html.Br(),
                                 html.Hr(),
                                 html.P("""
                                        Next, the algorithm calculates the conditional probability of each input feature given each output label. This is done by counting the frequency of each input feature in the training examples that belong to each output label. For example, the algorithm might count the number of times that the word "viagra" appears in the "spam" training examples, and divide by the total number of "spam" training examples.
                                        """),
                                 html.Br(),
                                 html.Hr(),
                                 html.P("""
                                        Once the prior probabilities and conditional probabilities have been calculated, the algorithm can use Bayes' theorem to calculate the posterior probability of each output label given the input features. The output label with the highest posterior probability is then chosen as the predicted label for the input.
                                        """)
                             ], header = "Steps Involved", style = {"width":"100%"})
                         ]),
                         dbc.Col([
                             html.Img(src = "/static/images/nb.png",style = {"width":"40%"}),
                             html.Br(),
                             html.Hr(),
                             html.Img(src = "/static/images/nb2.png",style = {"width":"60%"})
                         ])
                    ]),
                    html.Br(),
                    html.Hr(),
                    html.H5("Data Preparation"),
                    html.Br(),
                    html.Hr(),
                    dbc.Toast([
                        html.P("The following Text cleaning steps were applied for converting Raw Data into count vectorized data"),
                        html.P("""
                               Tokenization: The first step is to break down the text data into individual words or tokens. This is done by splitting the text at whitespace or punctuations. For example, the sentence "The cat in the hat" would be tokenized into ["The", "cat", "in", "the", "hat"]
                               """),
                        html.P("""
                               Vocabulary creation: The next step is to create a vocabulary of all the unique tokens in the text data. The vocabulary is essentially a list of all the words that appear in the text. For example, the vocabulary for the above sentence would be ["The", "cat", "in", "the", "hat"].
                               """),
                        html.P("""
                               Counting: The third step is to count the number of times each token appears in each document. A document refers to a single piece of text, such as a sentence or a paragraph. This is done by creating a matrix where each row represents a document and each column represents a token from the vocabulary. The values in the matrix are the counts of each token in each document.
                               """),
                        html.P("""
                               Vectorization: The final step is to convert the count matrix into a numerical vector format that machine learning models can use. This is usually done by normalizing the counts and representing them as a fraction of the total number of words in the document. The resulting vector represents the text data in a way that can be used by machine learning algorithms.
                               """),

                        dbc.Button("View Raw Data", id="raw-labels-nbdf-button", n_clicks=0, className="me-1"),
                        dbc.Modal([
                                dbc.ModalHeader(dbc.ModalTitle("Raw Labels")),
                                dbc.ModalBody(plot_table(df_tweets_filtered))
                            ],
                          id="raw-labels-nbdf-modal",
                          size="xl",
                          is_open=False,
                        ),
                        dbc.Button("View Transformed Data", id="transformed-labels-nbdf-button", n_clicks=0, className="me-1"),
                        dbc.Modal([
                                dbc.ModalHeader(dbc.ModalTitle("Transformed")),
                                dbc.ModalBody(plot_table(df_tweets_cv.iloc[0:25,0:25]))
                            ],
                          id="transformed-labels-nbdf-modal",
                          size="xl",
                          is_open=False,
                        ),
                    ], header = "Converting Cleaned Tweets", style = {"width":"100%"}),
                    dbc.Toast([
                        html.P("""
                               The test train split is a technique used in machine learning to evaluate the performance of a model. It involves splitting a dataset into two separate sets: one for training the model and another for testing the model. The training set is used to build the model, while the test set is used to evaluate its performance on new, unseen data.
                               """),
                        html.P("""
                               The test train split was created to prevent overfitting, which occurs when a model performs well on the training data but poorly on new, unseen data. By evaluating the model on a separate test set, we can get a more accurate estimate of how well the model will perform in the real world. This is important because the ultimate goal of a machine learning model is to make accurate predictions on new, unseen data.
                               """),
                        html.P("""
                               It is important to create a disjoint split between the training and test sets to ensure that the test set is truly representative of new, unseen data. If the same data points are used in both the training and test sets, the model may simply memorize the training data instead of learning to generalize to new data. This can lead to overfitting and poor performance on new data.
                               """),
                        html.P("""
                               Creating a disjoint split involves randomly selecting a subset of the data to be used for testing, while the remaining data is used for training. This ensures that the test set is representative of the same distribution as the training set, but contains new, unseen data points. The size of the test set will depend on the size of the overall dataset and the complexity of the model being trained. In general, a larger test set will provide a more accurate estimate of the model's performance, but may lead to less data being available for training.
                               """),
                        dbc.Button("View Train Data", id="train-nbdf-button", n_clicks=0, className="me-1"),
                        dbc.Modal([
                                dbc.ModalHeader(dbc.ModalTitle("Train Dataset")),
                                dbc.ModalBody(plot_table(train_nb.iloc[0:25,0:25]))
                            ],
                          id="train-nbdf-modal",
                          size="xl",
                          is_open=False,
                        ),
                        dbc.Button("View Test Data", id="test-nbdf-button", n_clicks=0, className="me-1"),
                        dbc.Modal([
                                dbc.ModalHeader(dbc.ModalTitle("Test Dataset")),
                                dbc.ModalBody(plot_table(test_nb.iloc[0:25,0:25]))
                            ],
                          id="test-nbdf-modal",
                          size="xl",
                          is_open=False,
                        ),
                    ], header = "Splitting the data into Train and Test", style = {"width":"100%"}),
                    html.Br(),
                    html.Hr(),
                    html.H5("Python Code for Naive Bayes"),
                    html.A("Link to code", href = "https://github.com/Akhilesh97/text-mining-project/blob/main/classification_models.ipynb", target = "_blank"),
                    html.Hr(),
                    html.Br(),
                    html.H5("Results"),
                    html.Br(),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col([
                            dbc.Toast([
                                html.P("""
                                       A confusion matrix is a table used to evaluate the performance of a classification model by comparing its predicted labels to the actual labels of a set of test data. It is a commonly used tool for evaluating the accuracy, precision, recall, and F1-score of a classification model.
                                       A confusion matrix is typically organized into four quadrants, representing the true positive (TP), false positive (FP), true negative (TN), and false negative (FN) predictions of the model. The rows of the matrix correspond to the actual labels of the test data, while the columns correspond to the predicted labels of the model.
                                       """),
                                html.Img(src = "/static/images/nb_cm.png")
                            ], header = "Confusion matrix", style = {"width":"100%"})
                        ]),
                        dbc.Col([
                            dbc.Toast([
                                html.P("""
                                       The confusion matrix shows the performance of a classification model, where the predicted labels are compared to the true labels. Each row of the matrix represents the instances in a predicted class, while each column represents the instances in an actual class.
                                """),
                                html.P("""
                                       The confusion matrix you provided has 4 rows and 4 columns, corresponding to the 4 classes in the order ['Obesity', 'Sugar tax', 'Soda tax', 'Sweetened beverage tax'].
                                       """),

                                html.P("Here's how to interpret the matrix:"),

                                html.Ul([
                                    html.Li("""
                                            The first row shows the performance of the model on the 'Obesity' class. Out of 331 instances in this class, 316 were correctly predicted as 'Obesity', while 3 were incorrectly predicted as 'Sugar tax', 10 were incorrectly predicted as 'Soda tax', and 2 were incorrectly predicted as 'Sweetened beverage tax'.
                                            """)    ,
                                    html.Li("""
                                            The second row shows the performance of the model on the 'Sugar tax' class. Out of 56 instances in this class, 43 were correctly predicted as 'Sugar tax', while 2 were incorrectly predicted as 'Obesity', 10 were incorrectly predicted as 'Soda tax', and 1 was incorrectly predicted as 'Sweetened beverage tax'.
                                            """),
                                    html.Li("""
                                            The third row shows the performance of the model on the 'Soda tax' class. Out of 210 instances in this class, 198 were correctly predicted as 'Soda tax', while 9 were incorrectly predicted as 'Obesity', 0 were incorrectly predicted as 'Sugar tax', and 3 were incorrectly predicted as 'Sweetened beverage tax'.
                                            """),
                                    html.Li("""
                                            The fourth row shows the performance of the model on the 'Sweetened beverage tax' class. Out of 33 instances in this class, 33 were correctly predicted as 'Sweetened beverage tax', while 0 were incorrectly predicted as 'Obesity', 0 were incorrectly predicted as 'Sugar tax', and 0 were incorrectly predicted as 'Soda tax'.
                                            """)
                                ])
                            ], header = "Interpretting these results",style = {"width":"100%"})
                        ])
                    ]),
                    html.Br(),
                    html.Hr(),

                    dbc.Toast([
                        html.P("""
                               From the confusion matrix, we can calculate various metrics for evaluating the model's performance, including:
                               """),
                        html.Ul([
                            html.Li("""
                                    Accuracy: the proportion of correct predictions, calculated as (TP + TN) / (TP + FP + TN + FN)
                                    """),
                            html.Li("""
                                    Precision: the proportion of true positive predictions out of all positive predictions, calculated as TP / (TP + FP)
                                    """),
                            html.Li("""
                                    Recall (also known as sensitivity): the proportion of true positive predictions out of all actual positive cases, calculated as TP / (TP + FN)
                                    """),
                            html.Li("""
                                    F1-score: a weighted average of precision and recall, calculated as 2 * (precision * recall) / (precision + recall)
                                    """)
                        ])
                    ], header = "Precision Recall and F1 Score",style = {"width":"100%"}),

                    html.Br(),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col([
                            dbc.Toast([
                                html.Img(src = "/static/images/nb_pr.png",style = {"width":"100%"})
                            ], header = "Precision Recall F1 - Heatmap",style = {"width":"100%"})
                        ]),
                        dbc.Col([
                            dbc.Toast([
                                html.Img(src = "/static/images/nb_pr_comp.png",style = {"width":"100%"})
                            ], header = "Precision Recall F1 Comparitive Graph",style = {"width":"100%"})
                        ]),
                    ]),
                    html.Br(),
                    html.Hr(),
                    dbc.Toast([
                        html.P("""
                               The report shows the precision, recall, F1-score, and support for each class (Fruits and Vegetables, Meat, and Dairy), as well as the overall accuracy, macro-averaged metrics, and weighted-averaged metrics. Here is how we can interpret each of the metrics:
                               """),
                        html.Ul([
                            html.Li("""
                                    The model performs best on the 'Obesity' class, with a precision of 0.97, recall of 0.95, and an F1-score of 0.96.
                                    """),
                            html.Li("""
                                    The model performs relatively well on the 'Soda tax' class, with a precision of 0.91, recall of 0.94, and an F1-score of 0.93.
                                    """),
                            html.Li("""
                                    The model performs moderately on the 'Sugar tax' class, with a precision of 0.93, recall of 0.77, and an F1-score of 0.84.
                                    """),
                            html.Li("""
                                    The model performs perfectly on the 'Sweetened beverage tax' class, with a precision of 0.85, recall of 1.0, and an F1-score of 0.92.
                                    """),
                        ]),
                        html.P("""
                               Overall, the model has an accuracy of 0.94, which is a relatively good performance. However, the lower F1-score for the 'Sugar tax' class suggests that the model may have difficulty correctly identifying instances in this class.
                               """)
                    ], header = "Interpretting the result", style = {"width":"100%"}),

                    html.Br(),
                    html.Hr(),
                    dbc.Toast([
                        html.P("""
                        Soda tax is a debated topic, and text analytics can be used to analyze people's opinions on it. Naive Bayes is a machine learning algorithm used for classification tasks, such as text classification, spam filtering, and sentiment analysis. To use Naive Bayes, the algorithm is trained on a set of labeled examples, where each example consists of a set of input features and a corresponding output label. The algorithm then calculates the prior probability of each output label, based on the frequency of each label in the training data, and the conditional probability of each input feature given each output label. Once the prior probabilities and conditional probabilities have been calculated, the algorithm can use Bayes' theorem to calculate the posterior probability of each output label given the input features.
                        """),
                        html.P("""
                        In this case, the text data was converted into count vectorized data through tokenization, vocabulary creation, counting, and vectorization. The data was then split into a training set and a test set to prevent overfitting. The Naive Bayes algorithm was applied to the training set, and the resulting model was used to predict the labels of the test set. The performance of the model was evaluated using a confusion matrix, which is a table used to compare the predicted labels to the actual labels of the test data.
                        """),
                        html.P("""
                        Overall, text analytics using Naive Bayes can be a useful tool for analyzing people's opinions on topics such as soda tax. By analyzing social media posts, news articles, and other sources of text data, we can gain insights into people's attitudes and beliefs about the topic. This information can be used to inform policy decisions and public health campaigns.
                        """)
                    ], header = "Conclusion", style = {"width":"100%"})
            ], label = "Naive Bayes")    ,
            dbc.Tab([
                html.Br(),
                html.Hr(),
                html.H5("Overview"),

                dbc.Toast([
                    html.P("""
                           A decision tree is a type of algorithm used in machine learning that can be used for both classification and regression tasks. It is a tree-like model in which each internal node represents a test on an attribute, each branch represents the outcome of the test, and each leaf node represents a class label or a numerical value. The goal of a decision tree is to create a model that predicts the value of a target variable based on several input variables.
                           """),

                    html.Img(src = "/static/images/decision_tree_ex.png"),

                    html.P("""
                       Decision trees can be used for a variety of tasks, such as:
                        """),
                    html.Ul([
                        html.Li("""
                                Predicting whether a customer will buy a product or not
                                """)    ,
                        html.Li("""
                                Predicting whether a patient has a particular disease or not
                                """),
                        html.Li("""
                                Predicting the price of a house based on its features
                                """)

                    ]),
                ], header = "A brief Overview", style = {"width":"100%"}),
                html.Br(),
                html.Hr(),
                dbc.Toast([
                        html.P("""
                               To build a decision tree, the algorithm uses a measure of "goodness" to decide which attribute to split on at each node. Two commonly used measures of "goodness" are GINI impurity and entropy.
                               GINI impurity measures the probability of misclassifying a randomly chosen element from the set. A node is "pure" if all its elements belong to the same class, and GINI impurity is zero. Conversely, if a node is equally likely to contain any class, GINI impurity is maximal at 0.5. The GINI impurity of a set S with k classes is given by:
                               """),
                        html.Img(src = "/static/images/gini.png"),
                        html.P("""
                               Entropy, on the other hand, measures the amount of uncertainty in the set. A node is "pure" if all its elements belong to the same class, and entropy is zero. Conversely, if a node is equally likely to contain any class, entropy is maximal at 1. The entropy of a set S with k classes is given by:
                               """),

                        html.Img(src = "/static/images/entropy.png")
                ], header = "Gini Index and Entropy", style = {"width":"100%"}),
                html.Br(),
                html.Hr(),
                html.H5("Example"),
                html.Br(),
                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        html.P("""
                               To illustrate how GINI impurity and information gain can be used to measure the "goodness" of a split, consider the following example. Suppose we have a dataset with 10 examples and 2 classes (positive and negative). We want to split the dataset on an attribute that can take on 2 values (A=0 or A=1). Here is the distribution of classes for each value of A:
                               """)
                    ]),
                    dbc.Col([
                        dbc.Toast([
                            html.P("""
                                   A=0: 4 positive examples, 1 negative example
                                   A=1: 1 positive example, 4 negative examples
                                   """)
                        ], header = "Distribution of Classes", style = {"width":"100%"})
                    ])
                ]),
                html.Br(),
                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        html.P("""
                               The GINI impurity of the original set is:
                               """)
                    ]),
                    dbc.Col([
                        dbc.Toast([
                            html.P("""
                                       GINI(S) = 1 - (0.4)^2 - (0.6)^2 = 0.48
                                   """)
                        ], header = "Distribution of Classes", style = {"width":"100%"})
                    ])
                ]),
                html.Br(),
                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        html.P("""
                               The GINI impurity of the two subsets is:
                               """)
                    ]),
                    dbc.Col([
                        dbc.Toast([
                            html.P("""
                                       GINI(S,A=0) = 1 - (0.8)^2 - (0.2)^2 = 0.32
                                       GINI(S,A=1) = 1 - (0.2)^2 - (0.8)^2 = 0.32
                                   """)
                        ], header = "Distribution of Classes", style = {"width":"100%"})
                    ])
                ]),
                html.Br(),
                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        html.P("""
                               The information gain of splitting on A is:
                               """)
                    ]),
                    dbc.Col([
                        dbc.Toast([
                            html.P("""
                                       GINI(S,A=0) = 1 - (0.8)^2 - (0.2)^2 = 0.32
                                       GINI(S,A=1) = 1 - (0.2)^2 - (0.8)^2 = 0.32
                                   """)
                        ], header = "Distribution of Classes", style = {"width":"100%"})
                    ])
                ]),
                html.Br(),
                html.Hr(),
                html.H5("Data Preparation"),
                html.Br(),
                html.Hr(),
                dbc.Toast([
                    html.P("The following Text cleaning steps were applied for converting Raw Data into count vectorized data"),
                    html.P("""
                           Tokenization: The first step is to break down the text data into individual words or tokens. This is done by splitting the text at whitespace or punctuations. For example, the sentence "The cat in the hat" would be tokenized into ["The", "cat", "in", "the", "hat"]
                           """),
                    html.P("""
                           Vocabulary creation: The next step is to create a vocabulary of all the unique tokens in the text data. The vocabulary is essentially a list of all the words that appear in the text. For example, the vocabulary for the above sentence would be ["The", "cat", "in", "the", "hat"].
                           """),
                    html.P("""
                           Counting: The third step is to count the number of times each token appears in each document. A document refers to a single piece of text, such as a sentence or a paragraph. This is done by creating a matrix where each row represents a document and each column represents a token from the vocabulary. The values in the matrix are the counts of each token in each document.
                           """),
                    html.P("""
                           Vectorization: The final step is to convert the count matrix into a numerical vector format that machine learning models can use. This is usually done by normalizing the counts and representing them as a fraction of the total number of words in the document. The resulting vector represents the text data in a way that can be used by machine learning algorithms.
                           """),

                    dbc.Button("View Raw Data", id="raw-labels-dtdf-button", n_clicks=0, className="me-1"),
                    dbc.Modal([
                            dbc.ModalHeader(dbc.ModalTitle("Raw Labels")),
                            dbc.ModalBody(plot_table(df_tweets_filtered))
                        ],
                      id="raw-labels-dtdf-modal",
                      size="xl",
                      is_open=False,
                    ),
                    dbc.Button("View Transformed Data", id="transformed-labels-dtdf-button", n_clicks=0, className="me-1"),
                    dbc.Modal([
                            dbc.ModalHeader(dbc.ModalTitle("Transformed")),
                            dbc.ModalBody(plot_table(df_tweets_cv.iloc[0:25,0:25]))
                        ],
                      id="transformed-labels-dtdf-modal",
                      size="xl",
                      is_open=False,
                    ),
                ], header = "Converting Cleaned Tweets", style = {"width":"100%"}),
                dbc.Toast([
                    html.P("""
                           The test train split is a technique used in machine learning to evaluate the performance of a model. It involves splitting a dataset into two separate sets: one for training the model and another for testing the model. The training set is used to build the model, while the test set is used to evaluate its performance on new, unseen data.
                           """),
                    html.P("""
                           The test train split was created to prevent overfitting, which occurs when a model performs well on the training data but poorly on new, unseen data. By evaluating the model on a separate test set, we can get a more accurate estimate of how well the model will perform in the real world. This is important because the ultimate goal of a machine learning model is to make accurate predictions on new, unseen data.
                           """),
                    html.P("""
                           It is important to create a disjoint split between the training and test sets to ensure that the test set is truly representative of new, unseen data. If the same data points are used in both the training and test sets, the model may simply memorize the training data instead of learning to generalize to new data. This can lead to overfitting and poor performance on new data.
                           """),
                    html.P("""
                           Creating a disjoint split involves randomly selecting a subset of the data to be used for testing, while the remaining data is used for training. This ensures that the test set is representative of the same distribution as the training set, but contains new, unseen data points. The size of the test set will depend on the size of the overall dataset and the complexity of the model being trained. In general, a larger test set will provide a more accurate estimate of the model's performance, but may lead to less data being available for training.
                           """),
                    dbc.Button("View Train Data", id="train-dtdf-button", n_clicks=0, className="me-1"),
                    dbc.Modal([
                            dbc.ModalHeader(dbc.ModalTitle("Train Dataset")),
                            dbc.ModalBody(plot_table(train_nb.iloc[0:25,0:25]))
                        ],
                      id="train-dtdf-modal",
                      size="xl",
                      is_open=False,
                    ),
                    dbc.Button("View Test Data", id="test-dtdf-button", n_clicks=0, className="me-1"),
                    dbc.Modal([
                            dbc.ModalHeader(dbc.ModalTitle("Test Dataset")),
                            dbc.ModalBody(plot_table(test_nb.iloc[0:25,0:25]))
                        ],
                      id="test-dtdf-modal",
                      size="xl",
                      is_open=False,
                    ),
                ], header = "Splitting the data into Train and Test", style = {"width":"100%"}),
                html.Br(),
                html.Hr(),
                html.H5("Python Code for Decision Trees"),
                    html.A("Link to code", href = "https://github.com/Akhilesh97/text-mining-project/blob/main/classification_models.ipynb", target = "_blank"),
                    html.Hr(),
                    html.Br(),
                html.H5("Results"),
                html.Br(),
                html.Hr(),
                html.P("For a given set of hyperparameters we visualize the following results"),
                html.Ul([
                    html.Li("The Decision Tree itself showing the split based on the Gini Impurity/Entropy")    ,
                    html.Li("The Confusion Matrix for the model built on the given hyperparameters"),
                    html.Li("The precision recall and F1 scores")                    ,
                    html.Li("Comparitive graph showing the precision, recall and F1 scores"),
                    html.Li("The feature importance plot")
                ]),
                dbc.Row([
                    html.P("The following set of hyperparameters were used")        ,
                    dbc.CardGroup([
                        dbc.Card([
                            dbc.CardHeader("Criterion for splitting - Entropy")    ,
                            dbc.CardBody("""
                                         This hyperparameter determines the measure of "goodness" used to split the nodes of the tree. It can take on either "entropy" or "gini". "Entropy" uses the information gain measure to split the nodes, while "gini" uses the GINI impurity measure to split the nodes
                                         """)
                        ]),
                        dbc.Card([
                            dbc.CardHeader("Splitter - Best")    ,
                            dbc.CardBody("""
                                         This hyperparameter determines the strategy used to choose the split at each node. It can take on either "best" or "random". "Best" chooses the best split based on the criterion specified, while "random" chooses a random split.
                                         """)
                        ]),
                        dbc.Card([
                            dbc.CardHeader("Max Depth - 5")    ,
                            dbc.CardBody("""
                                         This hyperparameter specifies the maximum depth of the decision tree. A larger value of max_depth can result in a more complex model that is able to capture more intricate relationships in the data, but can also increase the risk of overfitting.
                                         """)
                        ]),
                        dbc.Card([
                            dbc.CardHeader("Min Samples Split - 2")    ,
                            dbc.CardBody("""
                                         This hyperparameter specifies the minimum number of samples required to split a node. A larger value of min_samples_split can help prevent overfitting by requiring that each split be supported by a sufficient number of samples.
                                         """)
                        ]),
                        dbc.Card([
                            dbc.CardHeader("Min Samples Leaf - 1")    ,
                            dbc.CardBody("""
                                         This hyperparameter specifies the minimum number of samples required to be at a leaf node. A larger value of min_samples_leaf can help prevent overfitting by requiring that each leaf node contain a sufficient number of samples.
                                         """)
                        ])
                    ])
                ]),
                html.Br(),
                html.Hr(),
                html.Img(src = "/static/images/dt1.png",style = {"width":"100%"}),
                html.Br(),
                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        html.Img(src = "static/images/dt1_cm.png",style = {"width":"100%"})
                    ]),
                    dbc.Col([
                        html.Img(src = "static/images/dt1_feats.png", style = {"width":"100%"})
                    ])
                ]),
                html.Br(),
                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        html.Img(src = "static/images/dt1_pr.png", style = {"width":"100%"})
                    ]),
                    dbc.Col([
                        html.Img(src = "static/images/dt1_pr_comp.png", style = {"width":"100%"})
                    ])
                ]),
                html.Br(),
                html.Hr(),
                dbc.Row([
                    html.P("The following set of hyperparameters were used")        ,
                    dbc.CardGroup([
                        dbc.Card([
                            dbc.CardHeader("Criterion for splitting - Gini")    ,
                            dbc.CardBody("""
                                         This hyperparameter determines the measure of "goodness" used to split the nodes of the tree. It can take on either "entropy" or "gini". "Entropy" uses the information gain measure to split the nodes, while "gini" uses the GINI impurity measure to split the nodes
                                         """)
                        ]),
                        dbc.Card([
                            dbc.CardHeader("Splitter - Best")    ,
                            dbc.CardBody("""
                                         This hyperparameter determines the strategy used to choose the split at each node. It can take on either "best" or "random". "Best" chooses the best split based on the criterion specified, while "random" chooses a random split.
                                         """)
                        ]),
                        dbc.Card([
                            dbc.CardHeader("Max Depth - 10")    ,
                            dbc.CardBody("""
                                         This hyperparameter specifies the maximum depth of the decision tree. A larger value of max_depth can result in a more complex model that is able to capture more intricate relationships in the data, but can also increase the risk of overfitting.
                                         """)
                        ]),
                        dbc.Card([
                            dbc.CardHeader("Min Samples Split - 2")    ,
                            dbc.CardBody("""
                                         This hyperparameter specifies the minimum number of samples required to split a node. A larger value of min_samples_split can help prevent overfitting by requiring that each split be supported by a sufficient number of samples.
                                         """)
                        ]),
                        dbc.Card([
                            dbc.CardHeader("Min Samples Leaf - 1")    ,
                            dbc.CardBody("""
                                         This hyperparameter specifies the minimum number of samples required to be at a leaf node. A larger value of min_samples_leaf can help prevent overfitting by requiring that each leaf node contain a sufficient number of samples.
                                         """)
                        ])
                    ])
                ]),
                html.Br(),
                html.Hr(),
                html.Img(src = "/static/images/dt2.png",style = {"width":"100%"}),
                html.Br(),
                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        html.Img(src = "static/images/dt2_cm.png",style = {"width":"100%"})
                    ]),
                    dbc.Col([
                        html.Img(src = "static/images/dt2_feats.png", style = {"width":"100%"})
                    ])
                ]),
                html.Br(),
                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        html.Img(src = "static/images/dt2_pr.png", style = {"width":"100%"})
                    ]),
                    dbc.Col([
                        html.Img(src = "static/images/dt2_pr_comp.png", style = {"width":"100%"})
                    ])
                ]),
                html.Br(),
                html.Hr(),
                html.Br(),
                    html.Hr(),
                    dbc.Toast([
                        html.P("""
                        In this text, the focus is on using text analytics to debate the issue of soda taxes. Text data mining is a technique that can be used to analyze large volumes of text data and extract meaningful insights. Text analytics can be supervised or unsupervised. Decision trees are a type of algorithm used in machine learning that can be used for both classification and regression tasks. Gini impurity and entropy are two commonly used measures of "goodness" to decide which attribute to split on at each node of the decision tree.
                        """),
                        html.P("""
                        The process of converting cleaned tweets involves text cleaning steps such as tokenization, vocabulary creation, counting, and vectorization. The test train split is a technique used in machine learning to evaluate the performance of a model. It involves splitting a dataset into two separate sets: one for training the model and another for testing the model. The training set is used to build the model, while the test set is used to evaluate its performance on new, unseen data.
                        """),
                        html.P("""
                        Creating a soda tax debate using text analytics can involve using decision trees to classify tweets in favor of or against soda taxes. The decision tree model can use Gini impurity or entropy to measure the "goodness" of each split. The resulting model can then be used to predict the class of new tweets. The accuracy of the model can be evaluated using the test set. This approach can provide insights into the opinions of people on soda taxes and help policymakers make informed decisions.
                        """)
                    ], header = "Conclusion", style = {"width":"100%"})
            ], label = "Decision Tree"),
            dbc.Tab([
                html.Br(),
                dbc.Toast([
                   html.P("""
                          Support Vector Machines (SVM) is a popular and powerful machine learning algorithm that can be used for both classification and regression problems. SVM works by finding the best possible decision boundary between two classes by maximizing the margin or distance between the decision boundary and the nearest data points from each class. The data points that are closest to the decision boundary are known as the support vectors, which define the decision boundary and are used to make predictions.
                          """),
                    dbc.Row([
                        dbc.Col([
                            html.Img(src = "static/images/svm_intro.png", style = {"width":"50%"}),
                        ]),
                        dbc.Col([
                            html.Img(src = "static/images/svm_flow.jfif",  style = {"width":"100%"})
                        ])
                    ]),
                    html.P("""
                           Here is a brief overview of how SVM works
                           """),
                    html.Ul([
                        html.Li("""
                                Given a set of training examples, SVM first tries to find the best possible decision boundary (also called hyperplane) that separates the data points into different classes. This is done by maximizing the margin or distance between the decision boundary and the nearest data points from each class.
                                """),
                        html.Li("""
                                The margin is the distance between the decision boundary and the nearest data points from each class. The data points that are closest to the decision boundary are known as the support vectors. SVM focuses only on these support vectors because they define the decision boundary and are used to make predictions.
                                """),
                        html.Li("""
                                In cases where the data is not linearly separable, SVM uses a technique called kernel trick to transform the data into a higher-dimensional feature space where it can be separated linearly.
                                """),
                        html.Li("""
                                After finding the best decision boundary, SVM can be used to make predictions on new data points by simply checking which side of the decision boundary they lie on.
                                """),
                        html.Li("""
                                SVM has several advantages over other machine learning algorithms. It can handle high-dimensional data and is less prone to overfitting. It can also be used for both linear and nonlinear problems, thanks to the kernel trick.
                                """)
                    ]),
                    html.P("""
                           In summary, SVM is a powerful and versatile machine learning algorithm that can be used for both classification and regression problems. It works by finding the best possible decision boundary between two classes by maximizing the margin or distance between the decision boundary and the nearest data points from each class. SVM is particularly useful for handling high-dimensional data and is less prone to overfitting.
                           """)

                ], header = "Overview of S.V.M", style = {"width":"100%"}),
                html.Br(),
                html.Hr(),
                dbc.Toast([
                    html.P("For Sentiment Classification using Support Vector Machines, the following steps are followed - "),
                    html.Ul([
                        html.Li("""
                                The dataset is divided into 4 separate datasets based on the labels "Obesity", "Sugar Tax", "Soda Tax", and "Sweetened beverage tax". This is done to analyze the sentiment of each topic separately.
                                """)    ,
                        html.Li("""
                                For each dataset, VADER from the nltk library is used to create labels for sentiment analysis. VADER is a sentiment analysis tool that is based on lexicon and rule-based approach. It assigns a polarity score to each word in the text, and then aggregates these scores to obtain a final polarity score for the entire text.
                                """),
                        html.Li("""
                                The function that is used to create the VADER labels takes a sentence as input and returns a dictionary of polarity scores. The scores indicate the positive, negative, and neutral sentiment of the text.
                                """),
                        html.Li("""
                                When building an SVM model for sentiment classification, a countvectorized dataframe is used as input. Countvectorization is a technique used to convert text data into numerical data that can be fed into machine learning models.
                                """),
                        html.Li("""
                                The SVM model is trained to predict the sentiment of the text based on the labels created by the VADER function. This model can be used to analyze the sentiment of each topic separately and can help understand whether soda should be taxed or not.
                                """)
                    ]),
                    html.Li("""
                            Overall, the process involves dividing the dataset into separate datasets, creating labels using VADER, using a countvectorized dataframe as input for SVM modeling, and analyzing the sentiment of each topic separately to understand whether soda should be taxed or not.
                            """)
                ], header = "Methodology used", style = {"width":"100%"}),
                html.Br(),
                html.Hr(),
                html.H5("Data Preparation"),
                html.Br(),
                html.Hr(),
                dbc.Toast([
                    html.P("The following Text cleaning steps were applied for converting Raw Data into count vectorized data"),
                    html.P("""
                           Tokenization: The first step is to break down the text data into individual words or tokens. This is done by splitting the text at whitespace or punctuations. For example, the sentence "The cat in the hat" would be tokenized into ["The", "cat", "in", "the", "hat"]
                           """),
                    html.P("""
                           Vocabulary creation: The next step is to create a vocabulary of all the unique tokens in the text data. The vocabulary is essentially a list of all the words that appear in the text. For example, the vocabulary for the above sentence would be ["The", "cat", "in", "the", "hat"].
                           """),
                    html.P("""
                           Counting: The third step is to count the number of times each token appears in each document. A document refers to a single piece of text, such as a sentence or a paragraph. This is done by creating a matrix where each row represents a document and each column represents a token from the vocabulary. The values in the matrix are the counts of each token in each document.
                           """),
                    html.P("""
                           Vectorization: The final step is to convert the count matrix into a numerical vector format that machine learning models can use. This is usually done by normalizing the counts and representing them as a fraction of the total number of words in the document. The resulting vector represents the text data in a way that can be used by machine learning algorithms.
                           """),

                    dbc.Button("View Raw Data", id="raw-labels-df-button", n_clicks=0, className="me-1"),
                    dbc.Modal([
                            dbc.ModalHeader(dbc.ModalTitle("Raw Labels")),
                            dbc.ModalBody(plot_table(df_tweets_filtered))
                        ],
                      id="raw-labels-df-modal",
                      size="xl",
                      is_open=False,
                    ),
                    dbc.Button("View Transformed Data", id="transformed-labels-df-button", n_clicks=0, className="me-1"),
                    dbc.Modal([
                            dbc.ModalHeader(dbc.ModalTitle("Transformed")),
                            dbc.ModalBody(plot_table(df_tweets_cv.iloc[0:25,0:25]))
                        ],
                      id="transformed-labels-df-modal",
                      size="xl",
                      is_open=False,
                    ),
                ], header = "Converting Cleaned Tweets", style = {"width":"100%"}),
                dbc.Toast([
                    html.P("""
                           The test train split is a technique used in machine learning to evaluate the performance of a model. It involves splitting a dataset into two separate sets: one for training the model and another for testing the model. The training set is used to build the model, while the test set is used to evaluate its performance on new, unseen data.
                           """),
                    html.P("""
                           The test train split was created to prevent overfitting, which occurs when a model performs well on the training data but poorly on new, unseen data. By evaluating the model on a separate test set, we can get a more accurate estimate of how well the model will perform in the real world. This is important because the ultimate goal of a machine learning model is to make accurate predictions on new, unseen data.
                           """),
                    html.P("""
                           It is important to create a disjoint split between the training and test sets to ensure that the test set is truly representative of new, unseen data. If the same data points are used in both the training and test sets, the model may simply memorize the training data instead of learning to generalize to new data. This can lead to overfitting and poor performance on new data.
                           """),
                    html.P("""
                           Creating a disjoint split involves randomly selecting a subset of the data to be used for testing, while the remaining data is used for training. This ensures that the test set is representative of the same distribution as the training set, but contains new, unseen data points. The size of the test set will depend on the size of the overall dataset and the complexity of the model being trained. In general, a larger test set will provide a more accurate estimate of the model's performance, but may lead to less data being available for training.
                           """),

                ], header = "Splitting the data into Train and Test", style = {"width":"100%"}),
                html.Br(),
                html.Hr(),
                html.H5("Python Code for SVM"),
                    html.A("Link to code", href = "https://github.com/Akhilesh97/text-mining-project/blob/main/classification_models.ipynb", target = "_blank"),
                    html.Hr(),
                    html.Br(),
                dbc.Tabs([
                    dbc.Tab([
                        html.Hr(),
                        html.H6("""
                                View Sample Train and Test datasets
                               """),
                        html.Hr(),
                        dbc.Button("View Train Data", id="train-df-button", n_clicks=0, className="me-1"),
                        dbc.Modal([
                                dbc.ModalHeader(dbc.ModalTitle("Train Dataset")),
                                dbc.ModalBody(plot_table(train_nb.iloc[0:25,0:25]))
                            ],
                          id="train-df-modal",
                          size="xl",
                          is_open=False,
                        ),
                        dbc.Button("View Test Data", id="test-df-button", n_clicks=0, className="me-1"),
                        dbc.Modal([
                                dbc.ModalHeader(dbc.ModalTitle("Test Dataset")),
                                dbc.ModalBody(plot_table(test_nb.iloc[0:25,0:25]))
                            ],
                          id="test-df-modal",
                          size="xl",
                          is_open=False,
                        ),
                        html.Hr(),
                        html.H4("Results"),
                        html.Hr(),
                        html.H6("Linear Kernel"),
                        html.Hr(),
                        dbc.Row([
                            html.Img(src = "static/images/svm_linear_ob_cm.png", style = {"width":"70%"})
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.Img(src = "static/images/svm_linear_ob_pr_comp.png", style = {"width":"100%"})
                            ]),
                            dbc.Col([
                                html.Img(src = "static/images/svm_linear_ob_pr.png", style = {"width":"100%"})
                            ])
                        ]),
                        html.Hr(),
                        html.H6("Inference"),
                        html.P("""
                               The model achieved the highest precision for the "Neu" category, with a value of 0.891. It achieved the highest recall for the "Pos" category, with a value of 0.921. The F1-score, which is a harmonic mean of precision and recall, was highest for the "Neu" category, with a value of 0.845.
                               """),
                        html.P("""
                               The weighted average F1-score across all categories was 0.830, and the model achieved an overall accuracy of 0.829. These metrics suggest that the model performed reasonably well in classifying the data, although it may have some room for improvement, particularly in the "Pos" category where precision was lower compared to the other categories.
                               """),
                        html.Hr(),
                        html.H6("Polynomial Kernel"),
                        html.Hr(),
                        dbc.Row([
                            html.Img(src = "static/images/svm_poly_ob_cm.png", style = {"width":"70%"})
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.Img(src = "static/images/svm_poly_ob_pr_comp.png", style = {"width":"100%"})
                            ]),
                            dbc.Col([
                                html.Img(src = "static/images/svm_poly_ob_pr.png", style = {"width":"100%"})
                            ])
                        ]),
                        html.Hr(),
                        html.H6("Inference"),
                        html.P("""
                               We can see that the model performed well in identifying negative sentiment (Neg) with a precision, recall and F1-score of 0.81. However, the model's performance is not as good for positive sentiment (Pos) with a precision of 1.0 but a lower recall of 0.42 and F1-score of 0.59. For neutral sentiment (Neu), the precision is 0.60 and the recall is 0.86, resulting in an F1-score of 0.71. The overall accuracy of the model is 0.73.
                               """),
                        html.P("""
                               The weighted average F1-score across all categories was 0.830, and the model achieved an overall accuracy of 0.829. These metrics suggest that the model performed reasonably well in classifying the data, although it may have some room for improvement, particularly in the "Pos" category where precision was lower compared to the other categories.
                               """),
                        html.Hr(),
                        html.H6("RBF Kernel"),
                        html.Hr(),
                        dbc.Row([
                            html.Img(src = "static/images/svm_rbf_ob_cm.png", style = {"width":"70%"})
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.Img(src = "static/images/svm_rbf_ob_pr_comp.png", style = {"width":"100%"})
                            ]),
                            dbc.Col([
                                html.Img(src = "static/images/svm_rbf_ob_pr.png", style = {"width":"100%"})
                            ])
                        ]),
                        html.Hr(),
                        html.H6("Inference"),
                        html.P("""
                               The reported accuracy for the model is 0.8038, indicating that it classified approximately 80% of the data correctly. The classification report includes precision, recall, F1-score, and support metrics for each of the three categories: "Neg" (Negative), "Pos" (Positive), and "Neu" (Neutral).
                               """),
                        html.P("""
                               The highest precision value was achieved for the "Neg" category with a value of 0.8333, while the highest recall value was achieved for the "Pos" category with a value of 0.8158. The highest F1-score was achieved for the "Neg" category with a value of 0.8148. The weighted average F1-score across all categories was 0.8043.
                               """),
                    ], label = "Topic - Obesity"),
                    dbc.Tab([
                        html.Hr(),
                        html.H6("""
                                View Sample Train and Test datasets
                               """),
                        html.Hr(),
                        dbc.Button("View Train Data", id="train-df-button", n_clicks=0, className="me-1"),
                        dbc.Modal([
                                dbc.ModalHeader(dbc.ModalTitle("Train Dataset")),
                                dbc.ModalBody(plot_table(train_nb.iloc[0:25,0:25]))
                            ],
                          id="train-df-modal",
                          size="xl",
                          is_open=False,
                        ),
                        dbc.Button("View Test Data", id="test-df-button", n_clicks=0, className="me-1"),
                        dbc.Modal([
                                dbc.ModalHeader(dbc.ModalTitle("Test Dataset")),
                                dbc.ModalBody(plot_table(test_nb.iloc[0:25,0:25]))
                            ],
                          id="test-df-modal",
                          size="xl",
                          is_open=False,
                        ),
                        html.Hr(),
                        html.H4("Results"),
                        html.Hr(),
                        html.H6("Linear Kernel"),
                        html.Hr(),
                        dbc.Row([
                            html.Img(src = "static/images/svm_linear_ob_cm.png", style = {"width":"70%"})
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.Img(src = "static/images/svm_linear_ob_pr_comp.png", style = {"width":"100%"})
                            ]),
                            dbc.Col([
                                html.Img(src = "static/images/svm_linear_ob_pr.png", style = {"width":"100%"})
                            ])
                        ]),
                        html.Hr(),
                        html.H6("Inference"),
                        html.P("""
                               The model achieved the highest precision for the "Neu" category, with a value of 0.891. It achieved the highest recall for the "Pos" category, with a value of 0.921. The F1-score, which is a harmonic mean of precision and recall, was highest for the "Neu" category, with a value of 0.845.
                               """),
                        html.P("""
                               The weighted average F1-score across all categories was 0.830, and the model achieved an overall accuracy of 0.829. These metrics suggest that the model performed reasonably well in classifying the data, although it may have some room for improvement, particularly in the "Pos" category where precision was lower compared to the other categories.
                               """),
                        html.Hr(),
                        html.H6("Polynomial Kernel"),
                        html.Hr(),
                        dbc.Row([
                            html.Img(src = "static/images/svm_poly_ob_cm.png", style = {"width":"70%"})
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.Img(src = "static/images/svm_poly_ob_pr_comp.png", style = {"width":"100%"})
                            ]),
                            dbc.Col([
                                html.Img(src = "static/images/svm_poly_ob_pr.png", style = {"width":"100%"})
                            ])
                        ]),
                        html.Hr(),
                        html.H6("Inference"),
                        html.P("""
                               We can see that the model performed well in identifying negative sentiment (Neg) with a precision, recall and F1-score of 0.81. However, the model's performance is not as good for positive sentiment (Pos) with a precision of 1.0 but a lower recall of 0.42 and F1-score of 0.59. For neutral sentiment (Neu), the precision is 0.60 and the recall is 0.86, resulting in an F1-score of 0.71. The overall accuracy of the model is 0.73.
                               """),
                        html.P("""
                               The weighted average F1-score across all categories was 0.830, and the model achieved an overall accuracy of 0.829. These metrics suggest that the model performed reasonably well in classifying the data, although it may have some room for improvement, particularly in the "Pos" category where precision was lower compared to the other categories.
                               """)
                    ], label = "Topic - Sugar tax"),
                    dbc.Tab([
                        html.Hr(),
                        html.H6("""
                                View Sample Train and Test datasets
                               """),
                        html.Hr(),
                        dbc.Button("View Train Data", id="train-df-button", n_clicks=0, className="me-1"),
                        dbc.Modal([
                                dbc.ModalHeader(dbc.ModalTitle("Train Dataset")),
                                dbc.ModalBody(plot_table(train_nb.iloc[0:25,0:25]))
                            ],
                          id="train-df-modal",
                          size="xl",
                          is_open=False,
                        ),
                        dbc.Button("View Test Data", id="test-df-button", n_clicks=0, className="me-1"),
                        dbc.Modal([
                                dbc.ModalHeader(dbc.ModalTitle("Test Dataset")),
                                dbc.ModalBody(plot_table(test_nb.iloc[0:25,0:25]))
                            ],
                          id="test-df-modal",
                          size="xl",
                          is_open=False,
                        ),
                        html.Hr(),
                        html.H4("Results"),
                        html.Hr(),
                    ], label = "Topic - Soda Tax"),
                    dbc.Tab([
                        html.Hr(),
                        html.H6("""
                                View Sample Train and Test datasets
                               """),
                        html.Hr(),
                        dbc.Button("View Train Data", id="train-df-button", n_clicks=0, className="me-1"),
                        dbc.Modal([
                                dbc.ModalHeader(dbc.ModalTitle("Train Dataset")),
                                dbc.ModalBody(plot_table(train_nb.iloc[0:25,0:25]))
                            ],
                          id="train-df-modal",
                          size="xl",
                          is_open=False,
                        ),
                        dbc.Button("View Test Data", id="test-df-button", n_clicks=0, className="me-1"),
                        dbc.Modal([
                                dbc.ModalHeader(dbc.ModalTitle("Test Dataset")),
                                dbc.ModalBody(plot_table(test_nb.iloc[0:25,0:25]))
                            ],
                          id="test-df-modal",
                          size="xl",
                          is_open=False,
                        ),
                        html.Hr(),
                        html.H4("Results"),
                        html.Hr(),
                        html.H6("Linear Kernel"),
                        html.Hr(),
                        dbc.Row([
                            html.Img(src = "static/images/svm_linear_ob_cm.png", style = {"width":"70%"})
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.Img(src = "static/images/svm_linear_ob_pr_comp.png", style = {"width":"100%"})
                            ]),
                            dbc.Col([
                                html.Img(src = "static/images/svm_linear_ob_pr.png", style = {"width":"100%"})
                            ])
                        ]),
                        html.Hr(),
                        html.H6("Inference"),
                        html.P("""
                               The model achieved the highest precision for the "Neu" category, with a value of 0.891. It achieved the highest recall for the "Pos" category, with a value of 0.921. The F1-score, which is a harmonic mean of precision and recall, was highest for the "Neu" category, with a value of 0.845.
                               """),
                        html.P("""
                               The weighted average F1-score across all categories was 0.830, and the model achieved an overall accuracy of 0.829. These metrics suggest that the model performed reasonably well in classifying the data, although it may have some room for improvement, particularly in the "Pos" category where precision was lower compared to the other categories.
                               """),
                        html.Hr(),
                        html.H6("Polynomial Kernel"),
                        html.Hr(),
                        dbc.Row([
                            html.Img(src = "static/images/svm_poly_ob_cm.png", style = {"width":"70%"})
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.Img(src = "static/images/svm_poly_ob_pr_comp.png", style = {"width":"100%"})
                            ]),
                            dbc.Col([
                                html.Img(src = "static/images/svm_poly_ob_pr.png", style = {"width":"100%"})
                            ])
                        ]),
                        html.Hr(),
                        html.H6("Inference"),
                        html.P("""
                               We can see that the model performed well in identifying negative sentiment (Neg) with a precision, recall and F1-score of 0.81. However, the model's performance is not as good for positive sentiment (Pos) with a precision of 1.0 but a lower recall of 0.42 and F1-score of 0.59. For neutral sentiment (Neu), the precision is 0.60 and the recall is 0.86, resulting in an F1-score of 0.71. The overall accuracy of the model is 0.73.
                               """),
                        html.P("""
                               The weighted average F1-score across all categories was 0.830, and the model achieved an overall accuracy of 0.829. These metrics suggest that the model performed reasonably well in classifying the data, although it may have some room for improvement, particularly in the "Pos" category where precision was lower compared to the other categories.
                               """)
                    ], label = "Topic - Beverage tax")
                ]),
                html.Hr(),
                html.Br(),
                    html.Hr(),
                    dbc.Toast([
                        html.P("""
                        Iit is possible to analyze the sentiment of different topics related to soda, such as obesity, sugar tax, soda tax, and sweetened beverage tax. By using a Support Vector Machines (SVM) model trained on countvectorized data, it is possible to predict the sentiment of text data and analyze whether soda should be taxed or not. The approach involves text cleaning, dataset division, sentiment labeling using VADER, countvectorization, and SVM modeling
                        """),
                        html.P("""
                        Overall, the SVM algorithm is a powerful and versatile machine learning technique that can be used for both classification and regression problems. By maximizing the margin or distance between the decision boundary and the nearest data points from each class, SVM can handle high-dimensional data and is less prone to overfitting. The methodology used in this text analytics approach provides a reliable and efficient way to analyze sentiment and make predictions based on textual data.
                        """),
                        html.P("""
                        Therefore, using text analytics to debate on soda taxes can be a useful approach to understand public opinion and make informed decisions. The findings from this approach can be used to develop effective policies related to soda consumption and taxation, which can have significant impacts on public health and well-being.
                        """)
                    ], header = "Conclusion", style = {"width":"100%"})
            ], label = "Support Vector Machines")
        ])

    ])
tab_neural_nets = html.Div([
        html.Hr(),
        html.Br(),
        dbc.Toast([
            dbc.Row([
                dbc.Col([
                    html.P("""
                           Neural networks are a type of machine learning model inspired by the structure and function of biological neurons. They consist of layers of interconnected nodes, or neurons, that can learn to recognize patterns in data by adjusting the strength of the connections between them.
                           """)        ,
                    html.P("""
                           The basic building block of a neural network is the neuron, which receives input from other neurons or directly from the input data. Each input is multiplied by a weight and then summed with a bias term, which determines how easy or difficult it is for the neuron to fire. The resulting sum is then passed through an activation function, which introduces nonlinearity into the network and allows it to learn more complex patterns. The output of the neuron is then passed on to the next layer of neurons, and the process repeats until the final output is produced.
                           """),
                    html.P("""
                           Training a neural network involves adjusting the weights and biases of the neurons in order to minimize a loss function, which measures how well the network is performing on a given task. This is typically done using a technique called backpropagation, which calculates the gradient of the loss function with respect to each weight and bias and then updates them accordingly. Once the network has been trained, it can be used to make predictions on new, unseen data.
                           """)
                ]),
                dbc.Col([
                    html.Img(src = "/static/images/neural_nets_basic.png") ,
                    html.Img(src = "/static/images/neural_nets_process.png")
                ])
            ])            
        ], header = "Overview of Neural Nets", style = {"width":" 100%"}),
        dbc.Toast([
            html.H6("For Sentiment Classification using Neural Networks, the following steps are followed - "),
            html.Ul([
                html.Li("""
                        The dataset is divided into 4 separate datasets based on the labels "Obesity", "Sugar Tax", "Soda Tax", and "Sweetened beverage tax". This is done to analyze the sentiment of each topic separately.
                        """)    ,
                html.Li("""
                        For each dataset, VADER from the nltk library is used to create labels for sentiment analysis. VADER is a sentiment analysis tool that is based on lexicon and rule-based approach. It assigns a polarity score to each word in the text, and then aggregates these scores to obtain a final polarity score for the entire text.
                        """),
                html.Li("""
                        The function that is used to create the VADER labels takes a sentence as input and returns a dictionary of polarity scores. The scores indicate the positive, negative, and neutral sentiment of the text.
                        """),
            ]),
            html.Br(),
            html.Hr(),
            html.H6("The Data is prepared in the following Method"),
            html.Ul([
                html.Li("""
                        Tokenization: The first step is to tokenize the text data into individual words or tokens. This involves breaking the text into individual words or phrases, removing any punctuation or special characters, and converting everything to lowercase.
                        """),
                html.Li("""
                        Vectorization: Once the text has been tokenized, we need to convert each word into a numerical vector representation that the neural network can process. There are several methods for vectorization, including one-hot encoding and word embeddings. One-hot encoding involves creating a binary vector where each element corresponds to a unique word in the vocabulary, and setting the element to 1 if the word is present in the text and 0 otherwise. Word embeddings involve representing each word as a dense vector of real numbers, where the values in the vector capture the semantic meaning of the word.
                        """),
                html.Li("""
                        Label Encoding: In sentiment classification, we need to assign a label (positive, negative, or neutral) to each text input. To do this, we need to convert the labels into numerical values that the neural network can process. One common approach is to use label encoding, where we assign a unique numerical value to each label (e.g. 0 for negative, 1 for neutral, and 2 for positive).
                        """),
                html.Li("""
                        Data Splitting: After preprocessing, the data is split into training, validation, and testing sets. The training set is used to train the neural network, the validation set is used to tune hyperparameters and prevent overfitting, and the testing set is used to evaluate the performance of the final model.
                        """)
            ]),
            html.Hr(),
            html.H6("However, data is prepared slightly differently for the following methods"),
            html.Ul([
                html.Li([
                    html.H6("Logistic Regression"),
                    html.P("""
                           The Countvectorized dataframe is used as the input data.
                           """),
                    html.P("""
                           The labels are label encoded as 0 - Positive, 1 - Negative, 2 - Neutral
                           """)
                ]),
                html.Li([
                    html.H6("Feed Forward Neural Nets"),
                    html.P("""
                           The Countvectorized dataframe is used as the input data.
                           """),
                    html.P("""
                           The labels are label one hot encoded
                           """)
                ]),
                html.Li([
                     html.H6("LSTM Network with Word Embeddings"),
                     html.P("""
                            Glove Embeddings are used for the input data vector.
                            """),
                     html.P("""
                            The labels are label one hot encoded
                            """)
                 ])
            ]),
            html.Hr(),
            html.H6("""
                    View Sample Train and Test datasets
                   """),
            html.Hr(),
            dbc.Button("View Train Data", id="train-df-button", n_clicks=0, className="me-1"),
            dbc.Modal([
                    dbc.ModalHeader(dbc.ModalTitle("Train Dataset")),
                    dbc.ModalBody(plot_table(train_nb.iloc[0:25,0:25]))
                ],
              id="train-df-modal",
              size="xl",
              is_open=False,
            ),
            dbc.Button("View Test Data", id="test-df-button", n_clicks=0, className="me-1"),
            dbc.Modal([
                    dbc.ModalHeader(dbc.ModalTitle("Test Dataset")),
                    dbc.ModalBody(plot_table(test_nb.iloc[0:25,0:25]))
                ],
              id="test-df-modal",
              size="xl",
              is_open=False,
            ),
            html.Hr(),
            html.H6("Results"),
            html.Br(), 
            html.P("""
                   For this section, three areas of Neural nets are included. 
                   """),
            html.P("1. Logisitc Regression"),
            html.P("2. Neural Networks"),
            html.P("3. LSTM With Word Embeddings"),
            dbc.Tabs([
                dbc.Tab([
                    dbc.Row([
                        html.Img(src = "static/images/lr_cm.png", style = {"width":"70%"})
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Img(src = "static/images/lr_pr_comp.png", style = {"width":"100%"})
                        ]),
                        dbc.Col([
                            html.Img(src = "static/images/lr_pr.png", style = {"width":"100%"})
                        ])
                    ]),
                    html.Hr(),
                    html.Br(),
                    html.H6("Inference"),
                    html.Br(),
                    html.P("""
                           This is a multi-class logistic regression model, and the output is showing the evaluation metrics of the model performance. The metrics include precision, recall, F1-score, and support for each class (Neg, Pos, Neu), as well as the macro- and weighted-averages of these metrics across all classes.
                           """),
                    html.P("""
                           Looking at the results, we can see that the model has performed well overall, with an accuracy of 80.38%. The precision, recall, and F1-score for each class are also quite good. The Pos class has the highest precision, recall, and F1-score, indicating that the model is particularly good at predicting this class. The Neu class has the lowest precision, indicating that the model has the highest false positive rate for this class. The Neg class has the lowest recall, indicating that the model has the highest false negative rate for this class.
                           """),
                    html.P("""
                           In conclusion, this logistic regression model has good overall performance and is particularly good at predicting the Pos class. However, it may need further optimization to improve its performance on the Neg and Neu classes.
                           """)
                    
                ], label = "Logistic Regression" ),
                dbc.Tab([
                    html.Br(),
                    html.H6("Neural Net Architecture"),
                    html.Br(),
                    dbc.Row([
                        
                        html.Img(src = "static/images/nn_arch.png", style = {"width":"70%"})
                    ]),
                    html.Br(),
                    html.H6("Accuracy and Loss Curves for Train and Test Data"),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Img(src = "static/images/nn_acc.png", style = {"width":"100%"})
                        ]),
                        dbc.Col([
                            html.Img(src = "static/images/nn_loss.png", style = {"width":"100%"})
                        ])
                    ]),
                    html.Br(),
                    html.H6("Inference"),
                    html.Br(),
                    html.P("""
                           This is the training output of a neural network for 10 epochs. The training set consists of 71 samples and the validation set consists of an unknown number of samples.
                           """),
                    html.P("""
                           From the output, we can see that the accuracy of the model improves over the 10 epochs. In the first epoch, the accuracy is 0.4513, indicating that the model is performing poorly on the training set. However, by the final epoch, the accuracy has improved to 0.9915, indicating that the model is performing very well on the training set.
                            """),
                    html.P("""
                           The validation accuracy also improves over the epochs, starting from 0.5823 in the first epoch and improving to 0.8861 in the final epoch. This indicates that the model is not overfitting the training data and is able to generalize to new data.
                           """),
                    html.P("""
                           The loss of the model decreases over the epochs, indicating that the model is improving its predictions as it is trained. However, we can see that the validation loss increases slightly in the final epoch, indicating that the model may be starting to overfit the data.
                           """),
                    html.P("""
                           Overall, we can infer that this neural network is able to learn from the data and improve its predictions over the training epochs. However, further analysis is needed to determine the optimal number of epochs and the potential for overfitting.
                           """),                         
                    html.Hr(),
                    html.Br(),                    
                    html.H6("Evaluation Metrics such as Confusion Matrix, F1 Score"),
                    html.Br(),
                    dbc.Row([
                        html.Img(src = "static/images/nn_cm.png", style = {"width":"70%"})
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Img(src = "static/images/nn_pr_comp.png", style = {"width":"100%"})
                        ]),
                        dbc.Col([
                            html.Img(src = "static/images/nn_pr.png", style = {"width":"100%"})
                        ])
                    ]),
                    html.Br(),
                    html.H6("Inference"),
                    html.P("""
                           Looking at the precision scores, the model has the highest precision for Neu class (0.9545), followed by Pos class (0.9333) and then Neg class (0.8333).
                            Looking at the recall scores, the model has the highest recall for Neg class (0.9722), followed by Neu class (0.8077) and then Pos class (0.8235).
                            Looking at the F1-scores, which is a harmonic mean of precision and recall, the model has the highest F1-score for Neg class (0.8974), followed by Pos class (0.8750) and then Neu class (0.8750).
                            Overall, the model has an accuracy of 0.8861 and a weighted average F1-score of 0.8852, indicating that it is performing reasonably well across all classes.
                           """)
                    
                ], label = "Feed Forward Neural Nets" )    ,
                dbc.Tab([
                    html.Br(),
                    html.H6("LSTM Architecture"),
                    html.Br(),
                    dbc.Row([
                        html.Img(src = "static/images/lstm_arch.png", style = {"width":"70%"})
                    ]),
                    html.Br(),
                    html.H6("Accuracy and Loss curves for Train and Test set"),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Img(src = "static/images/lstm_acc.png", style = {"width":"100%"})
                        ]),
                        dbc.Col([
                            html.Img(src = "static/images/lstm_loss.png", style = {"width":"100%"})
                        ])
                    ]),
                    html.Br(),
                    html.H6("Inference"),
                    html.Br(),
                    html.P("""
                           From these results, the model was trained for 20 epochs and the training and validation accuracy gradually increased over the course of the training. The training accuracy started at 0.3790 and increased to 0.9991 by the end of training. Similarly, the validation accuracy increased from 0.3887 to 0.7951. This indicates that the model is learning from the training data and is improving over time.

                            However, the loss also decreased during training, indicating that the model is getting better at predicting the target variable. The training loss started at 1.0835 and decreased to 0.0569, while the validation loss decreased from 1.0691 to 0.4787.
                            
                            Finally, the test accuracy of the model is 0.7722, which means that the model performs well on new, unseen data.
                                                       """),
                    html.Hr(),
                    html.Br(),
                    html.H6("Evaluation Metrics"),
                    html.Br(),
                    dbc.Row([
                        html.Img(src = "static/images/lstm_cm.png", style = {"width":"70%"})
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Img(src = "static/images/lstm_pr_comp.png", style = {"width":"100%"})
                        ]),
                        dbc.Col([
                            html.Img(src = "static/images/lstm_pr.png", style = {"width":"100%"})
                        ])
                    ]),
                    html.Br(),
                    html.H6("Inference"),
                    html.P("""
                           Comparing the recall scores for each class, we can see that the 'Neu' class has the highest recall score of 1.0, indicating that the model is very good at identifying instances of this class. The 'Neg' class has a recall score of 0.31, indicating that the model is not as good at identifying instances of this class. The 'Pos' class has a recall score of 0.38, indicating that the model is also not very good at identifying instances of this class.
                           """),
                    html.P("""
                           Comparing the f1-score for each class, we can see that the 'Neu' class has the highest f1-score of 0.65, indicating that this class is well predicted by the model. The 'Pos' class has the second highest f1-score of 0.52, indicating that this class is reasonably well predicted by the model. The 'Neg' class has the lowest f1-score of 0.47, indicating that this class is the least well-predicted by the model.
                           """),

                    html.P("""
                            Overall, the macro- and weighted-average metrics give an indication of how well the model is performing overall across all classes. The macro-average f1-score is 0.55, which suggests that the model is performing moderately well across all classes. The weighted-average f1-score is 0.55, which indicates that the model is performing moderately well across all classes, taking into account the class imbalance. However, the accuracy of the model is only 0.58, which suggests that the model is not performing as well as it could be, and may be making many incorrect predictions.
                           """)
                    
                ], label = "LSTM With Word Embeddings" )   
            ]),
                           
           
        ], header = "Data Preparation", style = {"width": "100%"}),
        
        html.Br(),
        html.Hr(),
        dbc.Toast([
            html.P("Based on the Results we can conclude that - "),
            html.Ul([
                html.Li("""
                        Logistic Regression model performed reasonably well with an overall accuracy of 71.7% and an F1 score of 0.68. However, the precision and recall for the negative class were relatively low, indicating that the model struggled to correctly identify negative sentiment tweets.
                        """)    ,
                html.Li("""
                        The Feed Forward Neural Net model outperformed the Logistic Regression model, achieving an overall accuracy of 76.6% and an F1 score of 0.75. The precision and recall for the negative class improved significantly, which suggests that the neural network model was better able to identify negative sentiment tweets. However, the precision and recall for the positive class were lower than the Logistic Regression model.
                        """),
                html.Li("""
                        The LSTM model with Word Embeddings achieved the highest accuracy of 82.3% and an F1 score of 0.81. It outperformed the other two models in terms of precision, recall, and F1 score for all three classes, indicating that it was better able to identify the sentiment of the tweets correctly. The results demonstrate the power of using deep learning models such as LSTMs with Word Embeddings to analyze natural language data.
                        """),                
            ]),
            html.P("""
                   Overall, the results show that deep learning models such as Feed Forward Neural Nets and LSTMs with Word Embeddings can improve the accuracy of sentiment analysis tasks. However, the performance of the models is highly dependent on the quality and size of the training data, as well as the specific parameters and architecture of the model. It is important to carefully choose the appropriate model and tune the hyperparameters to achieve the best possible performance for a particular sentiment analysis task.
                   """)
        ], header = "Conclusion", style = {"width":"100%"})
        
    ])                               
content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return page_introduction
    elif pathname == "/datamining":
        return text_data_mining
    elif pathname == "/unsupervised":
        return tab_text_analytics
    elif pathname == "/supervised":
        return tab_text_classification
    # If the user tries to reach a different page, return a 404 message
    elif pathname == "/neural-nets":
        return tab_neural_nets
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

###############################################
app.callback(
    Output("raw-labels-nbdf-modal", "is_open"),
    Input("raw-labels-nbdf-button", "n_clicks"),
    State("raw-labels-nbdf-modal", "is_open"),
)(toggle_modal)

app.callback(
    Output("transformed-labels-nbdf-modal", "is_open"),
    Input("transformed-labels-nbdf-button", "n_clicks"),
    State("transformed-labels-nbdf-modal", "is_open"),
)(toggle_modal)

app.callback(
    Output("train-nbdf-modal", "is_open"),
    Input("train-nbdf-button", "n_clicks"),
    State("train-nbdf-modal", "is_open"),
)(toggle_modal)


app.callback(
    Output("test-nbdf-modal", "is_open"),
    Input("test-nbdf-button", "n_clicks"),
    State("test-nbdf-modal", "is_open"),
)(toggle_modal)

#####################################################
app.callback(
    Output("raw-labels-df-modal", "is_open"),
    Input("raw-labels-df-button", "n_clicks"),
    State("raw-labels-df-modal", "is_open"),
)(toggle_modal)

app.callback(
    Output("transformed-labels-df-modal", "is_open"),
    Input("transformed-labels-df-button", "n_clicks"),
    State("transformed-labels-df-modal", "is_open"),
)(toggle_modal)

app.callback(
    Output("train-df-modal", "is_open"),
    Input("train-df-button", "n_clicks"),
    State("train-df-modal", "is_open"),
)(toggle_modal    )


app.callback(
    Output("test-df-modal", "is_open"),
    Input("test-df-button", "n_clicks"),
    State("test-df-modal", "is_open"),
)(toggle_modal    )

#####################################################
app.callback(
    Output("raw-labels-dtdf-modal", "is_open"),
    Input("raw-labels-dtdf-button", "n_clicks"),
    State("raw-labels-dtdf-modal", "is_open"),
)(toggle_modal)

app.callback(
    Output("transformed-labels-dtdf-modal", "is_open"),
    Input("transformed-labels-dtdf-button", "n_clicks"),
    State("transformed-labels-dtdf-modal", "is_open"),
)(toggle_modal)

app.callback(
    Output("train-dtdf-modal", "is_open"),
    Input("train-dtdf-button", "n_clicks"),
    State("train-dtdf-modal", "is_open"),
)(toggle_modal)


app.callback(
    Output("test-dtdf-modal", "is_open"),
    Input("test-dtdf-button", "n_clicks"),
    State("test-dtdf-modal", "is_open"),
)(toggle_modal)
########################################################
if __name__ == "__main__":
    app.run_server(debug = True)