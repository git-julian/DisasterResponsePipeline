# libary imports

import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    This function loads in the text data and combines it into a Pandas Dataframe

    :param messages_filepath: String- that provides the filepath to the Disaster Messages CSV File
    :param categories_filepath: String - that provides the filepath to the Disaster Categories CSV File

    :return df: DataFrame- that containes the merged Datasets


    '''
    #load message csv file
    messages = pd.read_csv(messages_filepath)

    #load categories csv file
    categories = pd.read_csv(categories_filepath)

    #combine the two datasets
    df = messages.merge(categories, on=['id'], how='inner')

    return df


def clean_data(df):
    '''

    :param df: Dataframe- containing the merged dataset
    :return: df:
    '''

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand = True)

    # select the first row of the categories dataframe
    row = categories.loc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2]).values.tolist()

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]

        # convert column from string to numeric
    categories[column] = categories[column].astype(int)

     # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, sort=False)

    # Remove duplicates
    df.drop_duplicates(inplace=True)


    return df




def save_data(df, database_filename):

    '''
    Saves a given Dataframe into a SQL Lite Database - The Function overrides any old versions

    :param df: The Dataframe that should be stored into the database
    :param database_filename: String of the Database Filename
    :return:
    '''

    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql("Disasters", engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()