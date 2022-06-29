from flask import Flask, render_template, request, current_app, Blueprint, jsonify
from .function_api import *
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import sqlite3
import gensim.corpora as corpora
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from flask_sqlalchemy import SQLAlchemy
from flask_login import login_required, current_user
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import json
import time

app = Blueprint('app', __name__)

"""
route Auth Login /Disable 
if you want to activate it , please comment out the line below and change route (/) with def home() in (/home)
uncomment the @login_requirement for route you login required
"""

# @app.route('/')
# def index():
#     return render_template('index.html')


@app.route('/profile')
# @login_required
def profile():
    return render_template('profile.html')


"""
route Homepage Application 
"""


@app.route('/')
# @login_required
def home():
    # Display the number of document in the database on homepage
    connection_obj = sqlite3.connect(
        '/home/dimitri/Documents/code/python/projet_ia_datagrand/flask_app/database_datagrandest_new.db')
    # cursor object
    cursor_obj = connection_obj.cursor()
    cursor_obj.execute("SELECT * FROM document")
    count_document = (len(cursor_obj.fetchall()))
    return render_template('home.html', count_document=count_document)


"""
route for cosine analyse with url parameter
SEE cosine similarity to select document and compare it to other 
"""


@app.route('/cosinus_analyse', methods=['GET', 'POST'])
# @login_required
def cosinus_render():
    return render_template('cosine_analyse.html')


@app.route("/resultat", methods=["GET", "POST"])
# @login_required
def resultat():
    errors = []
    document_return = ''
    results = {}
    if request.method == "POST":
        # get url that the person has entered
        try:
            url = request.form.get('url')
        except:
            errors.append(
                "Unable to get URL. Please make sure it's valid and try again."
            )
            return render_template('index.html', errors=errors)

        if url:
            # Read sqlite query results into a pandas DataFrame
            con = sqlite3.connect('./flask_app/database_datagrandest_new.db')
            # Load dataset
            df_dataset_merge = pd.read_sql_query("SELECT * FROM document", con)
            df_dataset_merge.rename(
                columns={df_dataset_merge.columns[0]: 'documents'}, inplace=True)
            documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(
                df_dataset_merge.documents)]
            model_w2d = gensim.models.doc2vec.Doc2Vec(
                vector_size=150, window=10, min_count=1, workers=10)
            model_w2d.build_vocab(documents)
            model_w2d.train(
                documents, total_examples=model_w2d.corpus_count, epochs=10)
            document_embeddings = np.zeros((df_dataset_merge.shape[0], 150))
            for i in range(len(document_embeddings)):
                document_embeddings[i] = model_w2d.docvecs[i]
            pairwise_similarities = cosine_similarity(document_embeddings)
            results = most_similar_url(
                df_dataset_merge, url, pairwise_similarities, 'Cosine Similarity')
            document_return = document_enter(df_dataset_merge, url)

        return render_template('cosine_analyse.html', results=results)


"""
route for page to topics modelling 
"""


@app.route('/topics_modelling_input', methods=['GET', 'POST'])
def topics_input():
    return render_template('topics_modelling_input.html')


@app.route('/topics_modelling', methods=['GET', 'POST'])
def topic_render():
    if request.method == 'POST':
        my_range = request.form.get('my_range')
        my_range = (int(my_range))
        print(my_range)
        if my_range:
            df_dataset = load_dataset()
            documents_df = pd.DataFrame((df_dataset['title'].astype(
                str) + df_dataset['notes'].astype(str)+df_dataset['tags_name']), columns=['documents'])
            documents_df['documents'] = documents_df['documents'].str.replace(
                '<[^<>]*>', ' ', regex=True)
            data = documents_df.values.tolist()
            document_df_cleaned = Preprocess_similarity(data)
            data_words = list(sent_to_words(document_df_cleaned))
            id2word = corpora.Dictionary(data_words)  # Create Corpus
            texts = data_words  # Term Document Frequency
            corpus = [id2word.doc2bow(text) for text in texts]
            lda_model = load_lda_model(corpus, id2word, number_topics=my_range)
            visualization = gensimvis.prepare(lda_model, corpus, id2word)
            render = pyLDAvis.save_html(
                visualization, './flask_app/templates/LDAModel.html')

        return render_template('lda_render.html', render=render)


"""
route for see table with all cosine similarity for each document 
You can load algorithm for search the cosine similarity of all document(you can change the cosine parameter with interface)
"""


@app.route('/table_input_test', methods=['GET', 'POST'])
def table_input_test():
    return render_template('table_input_test.html')


@app.route('/table_algorithm_input', methods=['GET', 'POST'])
def table_input():
    if request.method == 'GET':
        return render_template('table_algorithm_input.html')
    
    if request.method == 'POST':
        my_range = request.form.get('my_range')
        my_range = (float(my_range))
        print(my_range)
        if my_range:
            df_documents = load_dataset_similarity()
            model = SentenceTransformer('all-MiniLM-L6-v2')
            sentences = df_documents['documents'].tolist()
            document_embeddings = model.encode(
                sentences, show_progress_bar=True)
            pairwise_similarities = cosine_similarity(document_embeddings)
            df_similar = most_similar_all(
                df_documents, pairwise_similarities, 'Cosine Similarity', my_range)
            msg = 'New record created successfully'
        return jsonify(msg)
    


@app.route('/table_ajax', methods=['GET'])
def table_ajax():
    return render_template('table_ajax.html')


@app.route('/api/data')
def data():
    df_fiche_select = pd.read_csv('fiche_select.csv')
    df_fiche_similar = pd.read_csv('fiche_similar.csv')
    df_merge_similar = pd.merge(
        df_fiche_select, df_fiche_similar, on='id_fiche_select')
    json_df = df_merge_similar.to_json(orient='records')
    return (json_df)


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.9', port=4455)
