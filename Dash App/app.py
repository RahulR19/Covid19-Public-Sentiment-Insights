import pandas as pd
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

import GetOldTweets3 as got

from statsmodels.tsa.arima_model import ARIMA
from textblob import TextBlob
from datetime import datetime as dt, timedelta
import time as ti
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english')) # Set of stopwords (Words that doesn't give meaningful information)

lemmatizer = WordNetLemmatizer()  # Used for converting words with similar meaning to a single word.

def text_process(tweet):

    processed_tweet = [] # To store processed text

    tweet = tweet.lower() # Convert to lower case

    tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', 'URL', tweet) # Replaces any URLs with the word URL

    tweet = re.sub(r'@[\S]+', 'USER_MENTION', tweet) # Replace @handle with the word USER_MENTION

    tweet = re.sub(r'#(\S+)', r' \1 ', tweet) # Removes # from hashtag

    tweet = re.sub(r'\brt\b', '', tweet) # Remove RT (retweet)

    tweet = re.sub(r'\.{2,}', ' ', tweet) # Replace 2+ dots with space

    tweet = tweet.strip(' "\'') # Strip space, " and ' from tweet

    tweet = re.sub(r'\s+', ' ', tweet) # Replace multiple spaces with a single space

    words = tweet.split()

    for word in words:

        word = word.strip('\'"?!,.():;') # Remove Punctuations

        word = re.sub(r'(.)\1+', r'\1\1', word) # Convert more than 2 letter repetitions to 2 letter (happppy -> happy)

        word = re.sub(r'(-|\')', '', word) # Remove - & '

        if (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None): # Check if the word starts with an english letter

            if(word not in stop_words):                                 # Check if the word is a stopword.

                word = str(lemmatizer.lemmatize(word))                  # Lemmatize the word

                processed_tweet.append(word)

    return ' '.join(processed_tweet)

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'Covid19 Sentiment Analysis'
server = app.server

date_range = {1:['2020-03-25','2020-04-14'],
              2:['2020-04-15','2020-05-03'],
              3:['2020-05-04','2020-05-17'],
              4:['2020-05-18','2020-05-31'],
              5:['2020-06-01','2020-06-14']}

phase_range = {1:'LD1',2:'LD2',3:'LD3',4:'LD4',5:'Unlock1'}

image_range = {0:'General cloud.png',
               1:'Lockdown1 cloud.png',
               2:'Lockdown2 cloud.png',
               3:'Lockdown3 cloud.png',
               4:'Lockdown4 cloud.png',
               5:'Unlock1 cloud.png'}

image_location = '/assets/'

# ----------------------------------------------------------------------------------------------------------------------------------------
#Import and clean data (importing csv into pandas)
df = pd.read_csv("Hashtag Data.csv")
df2 = pd.read_csv("Sentiment.csv")
df2['Date'] = pd.to_datetime(df2['Date'])
df3 = pd.read_csv("Tone Data.csv")
df4 = pd.read_csv("Graph Prediction.csv")


# ----------------------------------------------------------------------------------------------------------------------------------------
# App layout
app.layout = html.Div([

    html.Div(
    dbc.Row(dbc.Col(html.H1("Covid-19 Sentiment Analysis",className = "jumbotron")),style={'padding-left':'25px'}),
    style={'padding-bottom':'20px'}),


    dbc.Row(dbc.Col(html.Div(
    dcc.Slider(id="slct_period",
                min=0,
                max=5,
                marks={
                    0:{'label': 'General', 'style': {'color': '#FFFFFF','font-size':'22px','padding-left':'15px'}},
                    1:{'label': 'LD1', 'style': {'color': '#FFFFFF','font-size':'22px'}},
                    2:{'label': 'LD2', 'style': {'color': '#FFFFFF','font-size':'22px'}},
                    3:{'label': 'LD3', 'style': {'color': '#FFFFFF','font-size':'22px'}},
                    4:{'label': 'LD4', 'style': {'color': '#FFFFFF','font-size':'22px'}},
                    5:{'label': 'Unlock1', 'style': {'color': '#FFFFFF','font-size':'22px','padding-right':'20px'}}
                },
                value=0
                ),style={'margin-left':'15px','margin-right':'-5px'}))),

    dbc.Row(dbc.Col(html.Div(
    dcc.Graph(id = 'sentiment_analysis'),style={'padding-top':'20px','padding-bottom':'40px','padding-left':'30px'}))),

    dbc.Row(dbc.Col(dcc.Graph(id = 'sentiment_analysis1')),style={'padding-bottom':'40px','padding-left':'30px'}),

    dbc.Row(
    children = [
    dbc.Col(
    dcc.Graph(id = 'sentiment_analysis3'),width=6,style={'padding-left':'45px'}),

    dbc.Col(html.Div(html.Img(id = 'word_cloud',style={'height':'450px','width':'730px'})),width=6)]),

    dbc.Row(dbc.Col(html.Div(dcc.Graph(id='sentiment_graph'),style={'padding-top':'40px','padding-bottom':'30px','padding-left':'30px'}))),

    html.Div("Sentiment prediction end date: ",style={'width': '50%', 'display': 'inline-block', 'font-size':'25px', 'text-align': 'right','padding-right':'10px'}),

    html.Div(
    dcc.DatePickerSingle(
        id='sentiment_date',
        min_date_allowed=dt(2020, 6, 2),
        max_date_allowed=dt(2020, 9, 1),
        initial_visible_month=dt(2020,6,1),
        date='2020-6-2'),
        style={'margin-left':'-5px','width': '40%', 'display': 'inline-block'}),

    html.Div("Find the sentiment of text: ",style={'width': '50%', 'display': 'inline-block', 'font-size':'25px', 'text-align': 'right', 'margin-bottom':'25px', 'margin-top':'100px', 'font-family': 'Helvetica'}),

    html.Div(
    dcc.Input(
            id="sentiment_text",
            type = 'text',
            size="25",
            value='',
            placeholder="Input the text",
            style = {"height":"30px"}),
            className='input',
            style={'width': '45%', 'display': 'inline-block', 'text-align': 'left','padding-left':'5px'}),

    html.Div("Sentiment: ",style={'text-align': 'center','padding':'15px','font-size':'25px','font-family':'Helvetica','padding-right':'135px'}),
    html.Div(id="sentiment_prediction")
 ],style={'background': '#84CEEB','z-index':'0px','height':'360vh'}
)



# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components


@app.callback(
     Output('sentiment_analysis', 'figure'),
    [Input('slct_period', 'value')]
)

def update_graph(slct_period):
    if slct_period == 0:
        df_plot = df2.copy()
    else:
        start,end=map(str,date_range[slct_period])
        df_plot = df2[(df2['Date'] >= start) & (df2['Date'] <= end)]

    df_plot = df_plot.pivot_table('Value',['Date'],'Sentiment')
    df_plot.reset_index(drop=False, inplace=True)

    start = min(df_plot['Date'])-timedelta(hours=12)
    end = max(df_plot['Date'])+timedelta(hours=12)

    fig = go.Figure()

    fig.add_scatter(x=df_plot['Date'],y=df_plot['Positive'],mode='lines+markers',marker_color = '#03ac13',name="Positve",hoverinfo="x+y")
    fig.add_scatter(x=df_plot['Date'],y=df_plot['Neutral'],mode='lines+markers',marker_color = '#4169e1',name="Neutral",hoverinfo="x+y")
    fig.add_scatter(x=df_plot['Date'],y=df_plot['Negative'],mode='lines+markers',marker_color = '#e3242b',name="Negative",hoverinfo="x+y")

    fig.update_layout(title={'text': "Sentiment of People",'y':0.95,'x':0.48,'xanchor': 'center','yanchor': 'top'}, paper_bgcolor='#97CAEF', plot_bgcolor='#97CAEF',margin_pad=15)
    fig.update_xaxes(showgrid=False,title=dict(text="Date"),range=[start,end])
    fig.update_yaxes(showgrid=False,title=dict(text="Percentage"))
    #fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    #fig.update_yaxes(showline=True, linewidth=2, linecolor='black')

    return fig
# ------------------------------------------------------------------------------

@app.callback(
     Output('sentiment_analysis1', 'figure'),
    [Input('slct_period', 'value')]
)

def update_graph(slct_period):
    if slct_period == 0:
        df_plot = df.copy().groupby('hashtag').sum().reset_index().sort_values(by='value',ascending=False)
    else:
        df_plot = df[df['Phase']==phase_range[slct_period]]

    fig = px.bar(df_plot,x='hashtag',y='value',color='value',labels={'hashtag':'Hashtags','value':"Values"},height=500)
    fig.update_layout(title={'text': "Trending Hashtags",'y':0.95,'x':0.47,'xanchor': 'center','yanchor': 'top'}, paper_bgcolor='#97CAEF', plot_bgcolor='#97CAEF')
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)

    return fig
# ------------------------------------------------------------------------------

@app.callback(
     Output('sentiment_analysis3', 'figure'),
    [Input('slct_period', 'value')]
)

def update_graph(slct_period):
    if slct_period == 0:
        df_plot = df3.copy().groupby('Tone').sum().reset_index().sort_values(by='Value',ascending=False)
    else:
        start,end=map(str,date_range[slct_period])
        df_plot = df3[df3['Phase']==phase_range[slct_period]]
    fig = px.pie(df_plot,names="Tone",values="Value")
    fig.update_layout(paper_bgcolor='#97CAEF', plot_bgcolor='#97CAEF')

    if slct_period == 0:
        fig.update_layout(title={'text': "Tone of 2500 Sample tweets",'y':0.95,'x':0.48,'xanchor': 'center','yanchor': 'top'})
    else:
        fig.update_layout(title={'text': "Tone of 500 Sample tweets",'y':0.95,'x':0.475,'xanchor': 'center','yanchor': 'top'})

    return fig
# ------------------------------------------------------------------------------

@app.callback(
     Output('word_cloud', 'src'),
    [Input('slct_period', 'value')]
)

def update_image(slct_period):
    return image_location+image_range[slct_period]

@app.callback(
    Output('sentiment_graph', 'figure'),
    [Input('sentiment_date', 'date')])

def update_output(date):

    start = dt(2020,6,1)
    end = date
    year, month, day = map(int, end.split('-'))
    end = dt(year, month, day)

    result = df4
    train = result.iloc[:68]
    test = result.iloc[68:]
    model = ARIMA(train.Positive, order=(1,1,1))
    model_fit = model.fit(disp=-1)

    delta = end - start
    test_dates = []
    for i in range(delta.days + 1):
        test_dates.append(pd.to_datetime(str(start + timedelta(days=i)).split()[0]))

    forecast = model_fit.forecast(steps=delta.days+1)

    predicted_data = pd.DataFrame(forecast[0],index=test_dates,columns=['Positive'])

    #Neutral
    model = ARIMA(train.Neutral, order=(1,1,1))
    model_fit = model.fit(disp=-1)
    forecast = model_fit.forecast(steps=delta.days+1)
    predicted_data['Neutral'] = forecast[0]

    #Negative
    model = ARIMA(train.Negative, order=(1,1,1))
    model_fit = model.fit(disp=-1)
    forecast = model_fit.forecast(steps=delta.days+1)
    predicted_data['Negative'] = forecast[0]

    #Plot the data
    if(end < dt(2020,6,4)):
        start = min(test_dates)-timedelta(minutes = 30)
        end = max(test_dates)+timedelta(minutes = 30)
    elif(end < dt(2020,6,8)):
        start = min(test_dates)-timedelta(minutes = 90)
        end = max(test_dates)+timedelta(minutes = 90)
    elif(end < dt(2020,6,12)):
        start = min(test_dates)-timedelta(hours = 3)
        end = max(test_dates)+timedelta(hours = 3)
    elif(end < dt(2020,6,16)):
        start = min(test_dates)-timedelta(hours = 6)
        end = max(test_dates)+timedelta(hours = 6)
    else:
        start = min(test_dates)-timedelta(hours = 12)
        end = max(test_dates)+timedelta(hours = 12)

    fig = go.Figure()

    fig.add_scatter(x=test_dates, y=predicted_data['Positive'],mode='lines+markers',marker_color = '#03ac13',name="Positve",hoverinfo="x+y")
    fig.add_scatter(x=test_dates, y=predicted_data['Neutral'],mode='lines+markers',marker_color = '#4169e1',name="Neutral",hoverinfo="x+y")
    fig.add_scatter(x=test_dates, y=predicted_data['Negative'],mode='lines+markers',marker_color = '#e3242b',name="Negative",hoverinfo="x+y")

    fig.update_layout(title={'text': "Predicted Sentiment Graph",'y':0.95,'x':0.48,'xanchor': 'center','yanchor': 'top'}, paper_bgcolor='#97CAEF', plot_bgcolor='#97CAEF',margin_pad=15)
    fig.update_xaxes(showgrid=False,title=dict(text="Date"),range=[start,end])
    fig.update_yaxes(showgrid=False,title=dict(text="Percentage"))

    #fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    #fig.update_yaxes(showline=True, linewidth=2, linecolor='black')

    return fig
# ------------------------------------------------------------------------------

@app.callback([
     Output('sentiment_prediction', 'children'),
     Output('sentiment_prediction','style')],
    [Input('sentiment_text', 'value')]
)

def update_output_div(input_text):

    predict = TextBlob(input_text).sentiment.polarity

    if(predict>0):
        result='Positive'
        style={'text-align':'center', 'margin-top':'-52px', 'margin-left':'90px', 'font-size':'25px', 'font-family':'Geneva', 'color': 'green'}

    elif(predict<0):
        result='Negative'
        style={'text-align':'center', 'margin-top':'-52px', 'margin-left':'100px', 'font-size':'25px', 'font-family':'Geneva', 'color':'red'}

    else:
        result='Neutral'
        style = {'text-align':'center', 'margin-top':'-52px', 'margin-left':'85px', 'font-size':'25px', 'font-family':'Geneva', 'color':'black'}

    return result,style

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    app.run_server(debug=True)
