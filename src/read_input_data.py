import json
import pandas as pd
from langchain_core.documents import Document

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

    #Clean the Data
    df = pd.json_normalize(raw_data) 
    if debug: print(f"DataFrame shape: {df.shape}")
    df['firstdate_begin'] = pd.to_datetime(df['firstdate_begin'], errors='coerce',utc=True)
    print(df['firstdate_begin'].dtype)
    df = df.dropna(subset=['firstdate_begin'])
    df = df[df['firstdate_begin'].dt.year == 2025]
    if debug: print(f"DataFrame shape after year selection: {df.shape}")
    df = df[df['location_city'] == 'Lille']
    if debug: print(f"DataFrame shape after location selection: {df.shape}")
    df = df.dropna(subset=['description_fr'])
    if debug: print(f"DataFrame shape after dropping NA: {df.shape}")

    # Feature Engineering for RAG
    df['combined_text'] = (
        "Title: " + df['title_fr'] + "\n" +
        "Description: " + df['description_fr'] + "\n" +
        "Location: " + df['location_address'].fillna('') + 
        "city" + df['location_city'].fillna('') + "\n" +
        "Date_Start: " + str(df['firstdate_begin'].fillna('')) +
        "Date_End: " + str(df['lastdate_end'].fillna(''))
    )

    # Text + metadata selection
    metadata_cols = ['uid','location_city','firstdate_begin']
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