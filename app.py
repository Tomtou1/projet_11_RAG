from src.read_input_data import read_and_process_inputdata

if __name__ == "__main__":
    docs = read_and_process_inputdata('data/evenements-publics-openagenda.json', debug=True)
    print(f"Total processed documents: {len(docs)}")
    