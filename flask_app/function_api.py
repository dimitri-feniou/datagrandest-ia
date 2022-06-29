from pydoc import doc
import requests
import math
import pandas as pd
import nltk
import string
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
import numpy as np
from torch import cosine_similarity
from gensim.utils import simple_preprocess
import gensim

'''
Function for load data from API DATAGRANDEST -- GET all data from the api
'''
def load_dataset():
    global df_dataset
    # Request on api datagrandest
    url_api = "https://www.datagrandest.fr/data4citizen/d4c/api/datasets/2.0/search/"
    url_count = url_api + "start=1&rows=1"
    count = requests.get(url_count).json()['result']['count']
    print(count)
    rows = 1000
    start = 0
    pages = int(math.ceil(count/rows))
    api_result = []
    print(pages)
    for page in range(pages):
        start = rows * page or 1
        if start + rows > count:
            rows = count - start
        url = str(url_api) + "start=" + str(start) + "&rows=" + str(rows)
        print(url)
        api_result = api_result + requests.get(url).json()['result']['results']
    # Create a dataframe pandas with data from api
    df_dataset = pd.DataFrame(api_result)
    # Create columns in dataset transform each row 
    df_dataset['name_file'] = [[d.get('name') for d in x]
                               for x in df_dataset['resources']]
    df_dataset['name_file'] = df_dataset['name_file'].map(tuple)
    df_dataset['tags_name'] = [[d.get('name')
                                for d in x] for x in df_dataset['tags']]
    df_dataset['tags_name'] = df_dataset['tags_name'].apply(' '.join)
    return df_dataset

def load_dataset_similarity():
    global df_dataset
    global df_dataset_merge
    url_api = "https://www.datagrandest.fr/data4citizen/d4c/api/datasets/2.0/search/"
    url_count = url_api + "start=1&rows=1"
    # normalement vérifier "status_code" == 200 et si ['success'] == True
    count = requests.get(url_count).json()['result']['count']
    print(count)
    rows = 1000
    start = 0
    pages = int(math.ceil(count/rows))
    api_result = []
    print(pages)
    for page in range(pages):
        start = rows * page or 1
        if start + rows > count:
            rows = count - start
        url = str(url_api) + "start=" + str(start) + "&rows=" + str(rows)
        print(url)
        # api_result = requests.get(url_api).json()['result']['results']
        api_result = api_result + requests.get(url).json()['result']['results']
    df_dataset = pd.DataFrame(api_result)
    df_dataset['name_file'] = [[d.get('name') for d in x]
                               for x in df_dataset['resources']]
    df_dataset['name_file'] = df_dataset['name_file'].map(tuple)
    df_dataset['tags_name'] = [[d.get('name')
                                for d in x] for x in df_dataset['tags']]
    df_dataset['tags_name'] = df_dataset['tags_name'].apply(' '.join)
    # Create a new dataframe -- ADD title and description on the same column call documents
    df_dataset_merge = pd.DataFrame(df_dataset['title'].astype(
        str) + " " + df_dataset['notes'].astype(str), columns=['documents'])
    # Create columns from another dataframe
    df_dataset_merge['url'] = df_dataset['url']
    df_dataset_merge['id'] = df_dataset_merge.index
    # Clean html tag
    df_dataset_merge['documents'] = df_dataset_merge['documents'].str.replace(
        r'<[^<>]*>', ' ', regex=True)
    df_dataset_merge['documents'] = df_dataset_merge['documents'].str.replace(
        '(<br/>|\d+\.)', ' ').str.split().agg(" ".join)
    return df_dataset_merge

'''
Function for Theme extraction (LDA MODEL)
'''
# Function for clear sentences from gensim library
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


french_stopwords = nltk.corpus.stopwords.words('french')
new_words = ("données")
for i in new_words:
    french_stopwords.append(i)
mots = set(line.strip() for line in open(
    '/home/dimitri/Documents/code/python/projet_ia_datagrand/dictionnaire.txt'))
lemmatizer = FrenchLefffLemmatizer()


def Preprocess_similarity(listofSentence):
 preprocess_list = []
 for sentence in listofSentence :
# Remove punctuation of sentence and lowercase it
  sentence_w_punct = "".join([i.lower() for i in sentence if i not in string.punctuation])
# Remove digit on list of sentence 
  sentence_w_num = ''.join(i for i in sentence_w_punct if not i.isdigit())
# Tokenize word, Create a string for each word in the list of sentence
  tokenize_sentence = nltk.tokenize.word_tokenize(sentence_w_num)
# Remove french stop words
  words_w_stopwords = [i for i in tokenize_sentence if i not in french_stopwords]
# Lemmatization is the process of grouping together the different inflected forms of a word so they can be analyzed as a single item
  words_lemmatize = (lemmatizer.lemmatize(w) for w in words_w_stopwords)

  sentence_clean = ' '.join(w for w in words_w_stopwords if w.lower() in mots or not w.isalpha())

# Append to the empty list preprocess_list
  preprocess_list.append(sentence_clean)
 return preprocess_list
# Function for load LDA model 
def load_lda_model(corpus,id2word,number_topics):
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       random_state=100,
                                       chunksize=100,
                                       passes=10,
                                       num_topics=number_topics)
    return lda_model
'''
Function used for similarity 
'''
# Basic function used for test the similarity of 1 document beetween the other in dataframe pandas 
def most_similar(dataframe, doc_id, similarity_matrix, matrix):
    # Choose in the dataframe a document with this ID 
    print(f'Document {doc_id}: {dataframe.iloc[doc_id]["documents"]}')
    print('\n')
    print(f'Similar Documents using {matrix}:')
    if matrix == 'Cosine Similarity':
        similar_ix = np.argsort(similarity_matrix[doc_id])[::-1]
    elif matrix == 'Euclidean Distance':
        similar_ix = np.argsort(similarity_matrix[doc_id])
    for ix in similar_ix:
        if ix == doc_id:
            continue
        print('\n')
        # Return Document 
        print(f'Document: {dataframe.iloc[ix]["documents"]}')
        print(f'{matrix} : {similarity_matrix[doc_id][ix]}')

# Function use in route /resultat for search in dataframe the row corresponding to the URL
def document_enter(dataframe, url_data: str):
    contain_url = dataframe[dataframe['url'].str.contains(
        url_data, regex=False, na=False)]
    id_value = int(contain_url.index.values)
    document_select = (
        f'Document {id_value}: {dataframe.iloc[id_value]["documents"]}')
    return document_select
# Function use in route /resultat for get cosine value for the document selected 
def most_similar_url(dataframe, url_data: str, similarity_matrix, matrix):
    results = []
    contain_url = dataframe[dataframe['url'].str.contains(
        url_data, regex=False, na=False)]
    print(contain_url)
    id_value = int(contain_url.index.values)
    print(id_value)
    if matrix == 'Cosine Similarity':
        similar_ix = np.argsort(similarity_matrix[id_value])[::-1]
    for ix in similar_ix:
        if ix == id_value:
            continue
        elif similarity_matrix[id_value][ix]>= 0.50:
            dataframe_value = dict(
                dataframe.iloc[ix][['Documentid', 'documents', 'url']],cosine_similary = similarity_matrix[id_value][ix])
            results.append(dataframe_value)
    return results
# Function for Have all Cosine value for each documents    
def most_similar_all(data,similarity_matrix,matrix,cosine_value):
    global df_merge_similar
    # Create two list for append informations about documents we want to compare similarity and another list to append the value of similarity
    results_fiche_select = []
    results_fiche_similar = []
    for i in range(len(data)):
            print(i)
        # Create a dictionnary with information about the document we want to compare similarity
            dict_fiche_select = dict(id_fiche_select=data.iloc[i]['id'],url_fiche_select=data.iloc[i]['url'])
            # Append the dictionnary to the list
            results_fiche_select.append(dict_fiche_select)
            # Test the similarity of documents 
            if matrix=="Cosine Similarity":
                # Order results of cosine similarity in descending order
                similar_ix=np.argsort(similarity_matrix[i])[::-1]
            for ix in similar_ix:
                if ix == i:
                    continue 
                elif similarity_matrix[i][ix]>= cosine_value:
                    # df_documents_similar_columns = df_document_similarity.columns[i]
                    dict_fiche_similar = dict(id_fiche_select=data.iloc[i]['id'],id_fiche_similar=data.iloc[ix]['id'],url_document_similair=data.iloc[ix]['url'],cosine_similarity=similarity_matrix[i][ix])
                    results_fiche_similar.append(dict_fiche_similar)
        # Create two dataframe for export 2 csv with the result  
    df_fiche_select = pd.DataFrame(results_fiche_select)
    df_fiche_similar = pd.DataFrame(results_fiche_similar)
    df_fiche_select.to_csv('fiche_select.csv',index=False)
    df_fiche_similar.to_csv('fiche_similar.csv',index=False)




