import sqlite3
import pandas as pd

# Connect to SQLite database
conn = sqlite3.connect('database_datagrandest_new.db')
# Load CSV data into Pandas DataFrame
database_document = pd.read_csv('/home/dimitri/Documents/code/python/projet_ia_datagrand/flask_app/database_document_id.csv')
database_tags = pd.read_csv('/home/dimitri/Documents/code/python/projet_ia_datagrand/flask_app/database_creation/database_tags.csv')
database_metadata = pd.read_csv('/home/dimitri/Documents/code/python/projet_ia_datagrand/flask_app/database_creation/database_metadata.csv')
database_type = pd.read_csv('/home/dimitri/Documents/code/python/projet_ia_datagrand/flask_app/database_creation/database_type.csv')
# Write the data to a sqlite table
database_document.to_sql('document', conn, if_exists='append', index = False)
database_tags.to_sql('tags', conn, if_exists='append', index = False)
database_metadata.to_sql('metadata', conn, if_exists='append', index = False)
database_type.to_sql('type', conn, if_exists='append', index = False)
# Create a cursor object
cur = conn.cursor()
# Close connection to SQLite database
conn.close()
# Fetch and display result
for row in cur.execute('SELECT * FROM document'):
    print(row)

