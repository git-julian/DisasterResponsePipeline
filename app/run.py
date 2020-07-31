import json

import joblib
import plotly
import pandas as pd


import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet', 'stopwords'])

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine



app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///data/DisasterResponseData.db')
df = pd.read_sql_table('Disasters', engine)

# load model
model = joblib.load("models/Classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    cat_counts = df.drop(axis=1, labels=['id', 'message', 'original', 'genre']).sum().sort_values(ascending=False)
    cat_names = list(cat_counts.index)

    # Findig the 5 most frequnet categories / classes
    df_top = list(df.drop(axis=1, labels=['id', 'message', 'original', 'genre']).sum().iloc[:10].index)
    # Visualisation of the fife most frequent topics
    # Aggregate on  genre
    df_agg = df.groupby('genre').sum()
    # find top 10 classes
    top_categ = df_agg.drop(axis=1, labels=['id']).sum().sort_values(ascending=False).index[0:10].tolist()
    # prepere visualisation
    df_genre_agg = df_agg[top_categ]
    df_genre_agg = df_genre_agg.transpose().reset_index()
    df_genre_agg.columns = ['genre', 'direct', 'news', 'social']



    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [

# Distribution of all categories
        {
            'data': [
                Bar(
                    x=cat_names,
                    y=cat_counts
                )
            ],

            'layout': {
                'title': 'Display categories by occurrence: ',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
# Distribution of all genres

        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },


# Top Categoeries by genre

        {
            'data': [

                Bar(
                    name="direct",
                    x=df_genre_agg["genre"],
                    y=df_genre_agg["direct"]
                    # offsetgroup=0,
                ),
                Bar(
                    name="news",
                    x=df_genre_agg["genre"],
                    y=df_genre_agg["news"]
                    # offsetgroup=1,
                ),
                Bar(
                    name="social",
                    x=df_genre_agg["genre"],
                    y=df_genre_agg["social"]
                    # offsetgroup=2,
                )

            ],
            'layout': {
                'title': "Top categories by genre:",
                'yaxis': {
                    'title': "Number of messages"
                },
                'xaxis': {
                    'title': "Genre distribution of to occurring classes "
                }
            }
        }







    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()