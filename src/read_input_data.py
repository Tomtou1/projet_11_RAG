import json
import pandas as pd
from langchain_core.documents import Document
from bs4 import BeautifulSoup

def read_and_process_inputdata(file_path, debug=False):
    # Read the json file in data
    with open(file_path) as f:
        raw_data = json.load(f)

    #Understand the Data and view it if asked
    if debug:
        print(f"type of raw_data: {type(raw_data)}")
        print(f"type of raw_data[0]: {type(raw_data[0])}") 
        print(f"First event: {raw_data[0]}")
        print(f"Number of events: {len(raw_data)}")

    #Create dataframe and clean it
    df = pd.json_normalize(raw_data) 
    if debug: print(f"DataFrame shape: {df.shape}")
    # Select Date
    df['firstdate_begin'] = pd.to_datetime(df['firstdate_begin'], errors='coerce',utc=True)
    if debug: print(df['firstdate_begin'].dtype)
    df = df.dropna(subset=['firstdate_begin'])
    df = df[df['firstdate_begin'].dt.year == 2025]
    if debug: print(f"DataFrame shape after year selection: {df.shape}")
    #Select Location
    df = df[df['location_city'] == 'Lille']
    if debug: print(f"DataFrame shape after location selection: {df.shape}")
    #Drop NA in long description
    df = df.dropna(subset=['longdescription_fr'])
    if debug: print(f"DataFrame shape after dropping NA: {df.shape}")

    #Clean HTML function
    def clean_html(text):
        return BeautifulSoup(str(text), "html.parser").get_text(separator=" ")
    df['longdescription_fr_clean'] = df['longdescription_fr'].apply(clean_html)

    # Feature Engineering for RAG
    df['combined_text'] = (
    "Title: " + df['title_fr'] + "\n" +
    "Description: " + df['longdescription_fr_clean'] + "\n" +
    "Localisation: " + df['location_address'] + "\n" +
    "Ville: " + df['location_city'] + "\n" +
    "Date de début: " + df['firstdate_begin'].astype(str) + "\n" +
    "Date de fin: " + df['lastdate_end'].astype(str) + "\n" +
    "Age Min: " + df['age_min'].fillna('').astype(str) + "\n" +
    "Age Max: " + df['age_max'].fillna('').astype(str)
    )
    # Text + metadata selection
    metadata_cols = ['uid','canonicalurl','firstdate_begin','location_city','registration']
    df_final = df[['combined_text'] + metadata_cols]
    if debug: print(f"DataFrame shape: {df_final.shape}")

    processed_docs = []
    for _, row in df_final.iterrows():
        doc = Document(
            page_content=row['combined_text'],
            metadata={col: row[col] for col in metadata_cols}
        )
        processed_docs.append(doc)

    return processed_docs