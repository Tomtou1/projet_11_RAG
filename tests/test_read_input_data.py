
def test_data_filtering_by_year_2025(processed_docs):
    assert all('2025' in doc.page_content 
                for doc in processed_docs), "Tous les documents doivent être de 2025"

def test_data_filtering_by_city_lille(processed_docs):
    assert all('Lille' in doc.page_content 
                for doc in processed_docs), "Tous les documents doivent être de Lille"

def test_combined_text(processed_docs):   
    assert len(processed_docs) > 0, "Should have at least one processed document"
    
    for doc in processed_docs:
        content = doc.page_content
        
        # Check that all expected fields are present
        assert 'Title:' in content, "combined_text should contain 'Title:'"
        assert 'Description:' in content, "combined_text should contain 'Description:'"
        assert 'Location:' in content, "combined_text should contain 'Location:'"
        assert 'Date_Start:' in content, "combined_text should contain 'Date_Start:'"
        assert 'Date_End:' in content, "combined_text should contain 'Date_End:'"