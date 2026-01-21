def test_text_splitting(chunks, processed_docs):
    assert len(chunks) > len(processed_docs), "Cela doit créer au moins un chunk par document"

def test_chunk_metadata(chunks, processed_docs):
    for chunk in chunks:
        assert 'uid' in chunk.metadata, "Chunks doit avoir les metadonnées uid"
        assert chunk.metadata['uid'] is not None, "uid ne doit pas être None"

def test_all_data_in_vector_db(chunks, processed_docs, vector_store):
    assert vector_store.index.ntotal == len(chunks), f"Vector store doit avoir {len(chunks)} vecteurs mais a {vector_store.index.ntotal}"