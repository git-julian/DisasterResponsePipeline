import pickle
import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download(['punkt', 'wordnet', 'stopwords'])
import re

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score,f1_score


def load_data(database_filepath):

    '''
    Import Data from a given Database into a Pandas Data Frame and provide devide in the Data into a Set of Features and Target

    :param database_filepath: String that holds the Database Filepath
    :return: X DataFrame that holds the pred Features
    :return: y Is a list of the Target Text
    :retrun: categroy_names: Is a list of feature names


    '''

    engine = create_engine('sqlite:///' + database_filepath)

    df = pd.read_sql_table('Disasters', con=engine)

    X = df['message']

    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)

    category_names = Y.columns

    return X, Y, category_names


def tokenize(text, stop_word_corpus='english'):
    '''
    Filter and tokenize the Text corpus with common methods for NLP

    :param text: List of Strings - Text to be tokanized
    :param stop_word_corpus: String that gives infoarmion on which stopword corpus to use / Default is set to English
    :return:
    '''
    # 1) Normalize
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())  # 2) Lower Case
    # 3) Tokenize as words
    tokens = word_tokenize(text)
    # 4) Lemmatize
    stop_words = stopwords.words(stop_word_corpus)
    clean_tokens = [WordNetLemmatizer().lemmatize(w) for w in tokens if not w in stop_words]

    return clean_tokens


def build_model(do_grid_search = True):
    '''
  Pipeline that builds the model unsing countVectorizer, TFIDF and a MultiOutput Random Forrest Calssifier
    :return:
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=10)))
    ])
    if (do_grid_search == True):

        # define parameters for Grid search
        grid_search_parameters = {
            'vect__ngram_range': ((1, 1), (1, 2)),
            'tfidf__use_idf': [True],
            'clf__estimator__min_samples_split': [2, 4]

        }

        # create gridsearch object and return as final model pipeline
        model_pipeline = GridSearchCV(estimator=pipeline, param_grid=grid_search_parameters, cv=10)

        return model_pipeline
    else:
        return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    The is method outputs the following evaluation metrics for a multiclassification model
        1)How man cases are sorted into the respective class
        2)The Accuracy
        3)The Precision
        4)The Recall
        5 The f1 Score

        :param- model
        :param- df that holds the testing data for the features
        :param- df that contains the testing data for the target
        :pram- lsit  that holds the category names for each class
        :return: df- that holds the metrics for each of the classes
    '''

    # Get model predictions
    Y_prediction = model.predict(X_test)
    # Prepare Classification Report DataFrame
    df_eval = pd.DataFrame(columns=['emergency_case', 'accuracy', 'precision', 'recall', 'f1-score', 'number_of_cases'])
    # count the number of cases for each class
    count_cases = Y_test.sum().to_dict()

    # iterate over all classification columns, add model metrics
    for i, column in enumerate(Y_test):
        df_eval.loc[len(df_eval)] = [column, \
                                     accuracy_score(Y_test[column], Y_prediction[:, i]), \
                                     precision_score(Y_test[column], Y_prediction[:, i], average='weighted'), \
                                     recall_score(Y_test[column], Y_prediction[:, i], average='weighted'), \
                                     f1_score(Y_test[column], Y_prediction[:, i], average='weighted'), \
                                     count_cases[column]
                                     ]

    df_eval.set_index('emergency_case', inplace=True)
    print(df_eval)
    return df_eval


def save_model(model, model_filepath):
    '''

    Save the Model as a file

    :param model: Classification model
    :param model_filepath: String
    :return:
    '''
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
