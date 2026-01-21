from dotenv import load_dotenv

def test_query_returns_lille_2025_events_only(vector_store):    
    query = "événements culturels"
    
    results = vector_store.similarity_search(query, k=5)
    
    for result in results:
        assert "lille" in result.metadata['location_city'].lower(), "Les résultats doivent être localisés à Lille"
        assert 2025 == result.metadata['firstdate_begin'].year, "Les résultats doivent être datés de 2025"
