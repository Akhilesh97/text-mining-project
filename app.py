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
                dbc.NavLink("Text Analytics", href="/page-2", active="exact"),
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
        return html.P("Oh cool, this is page 2!")
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

if __name__ == "__main__":
    app.run_server(debug = True)