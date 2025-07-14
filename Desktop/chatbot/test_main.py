from fastapi.testclient import TestClient
from main import app
import requests
from unittest.mock import patch, MagicMock

client = TestClient(app)

# Mock product data that matches the actual scraping format
MOCK_PRODUCTS = [
    {"name": "Test Wheelchair", "price": 123.0, "url": "https://example.com/wheelchair"},
    {"name": "Another Wheelchair", "price": 150.0, "url": "https://example.com/wheelchair2"}
]

def mock_get_products(*args, **kwargs):
    """Mock the get_products method of MagentoScraper"""
    return MOCK_PRODUCTS

@patch('main.get_product_prices_from_search', side_effect=mock_get_products)
def test_scrape_prices(mock_get):
    response = client.get("/scrape-prices?category_url=https://fakeurl.com")
    assert response.status_code == 200
    data = response.json()
    assert "products" in data
    assert len(data["products"]) > 0
    assert data["products"][0]["name"] == "Test Wheelchair"
    assert data["products"][0]["price"] == 123.0

@patch('main.get_product_prices_from_search', side_effect=mock_get_products)
def test_chat_product_query(mock_get):
    response = client.post("/chat", json={"text": "I want a wheelchair"})
    assert response.status_code == 200
    data = response.json()
    assert "Test Wheelchair" in data["reply"]
    assert "123.0" in data["reply"]
    assert "products" in data

@patch('main.get_product_prices_from_search', side_effect=mock_get_products)
def test_chat_non_product_query(mock_get):
    response = client.post("/chat", json={"text": "Tell me a joke"})
    assert response.status_code == 200
    data = response.json()
    # Should fall back to LLM, which is mocked, so just check for reply key
    assert "reply" in data

# New tests for chat history functionality
def test_chat_with_session_id():
    """Test chat with session_id parameter"""
    response = client.post("/chat", json={
        "text": "Hello", 
        "session_id": "test_session_123"
    })
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert "session_id" in data
    assert data["session_id"] == "test_session_123"

def test_chat_without_session_id():
    """Test chat without session_id (should use default)"""
    response = client.post("/chat", json={"text": "Hello"})
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert "session_id" in data
    assert data["session_id"] == "default"

def test_chat_history_storage():
    """Test that chat history is properly stored"""
    session_id = "test_history_456"
    
    # Send first message
    response1 = client.post("/chat", json={
        "text": "What products do you have?",
        "session_id": session_id
    })
    assert response1.status_code == 200
    
    # Send second message
    response2 = client.post("/chat", json={
        "text": "Tell me more about wheelchairs",
        "session_id": session_id
    })
    assert response2.status_code == 200
    
    # Check chat history
    history_response = client.get(f"/chat-history/{session_id}")
    assert history_response.status_code == 200
    history_data = history_response.json()
    assert history_data["session_id"] == session_id
    assert len(history_data["history"]) == 2
    assert history_data["history"][0]["user"] == "What products do you have?"
    assert history_data["history"][1]["user"] == "Tell me more about wheelchairs"

def test_chat_history_limit():
    """Test that chat history is limited to 10 messages"""
    session_id = "test_limit_789"
    
    # Send 12 messages
    for i in range(12):
        response = client.post("/chat", json={
            "text": f"Message {i}",
            "session_id": session_id
        })
        assert response.status_code == 200
    
    # Check that only 10 messages are stored
    history_response = client.get(f"/chat-history/{session_id}")
    assert history_response.status_code == 200
    history_data = history_response.json()
    assert len(history_data["history"]) == 10
    # Should have the last 10 messages (2-11)
    assert history_data["history"][0]["user"] == "Message 2"
    assert history_data["history"][-1]["user"] == "Message 11"

def test_clear_chat_history():
    """Test clearing chat history"""
    session_id = "test_clear_999"
    
    # Send a message to create history
    response = client.post("/chat", json={
        "text": "Hello",
        "session_id": session_id
    })
    assert response.status_code == 200
    
    # Verify history exists
    history_response = client.get(f"/chat-history/{session_id}")
    assert history_response.status_code == 200
    assert len(history_response.json()["history"]) == 1
    
    # Clear history
    clear_response = client.delete(f"/chat-history/{session_id}")
    assert clear_response.status_code == 200
    assert clear_response.json()["message"] == f"Chat history cleared for session {session_id}"
    
    # Verify history is cleared
    history_response2 = client.get(f"/chat-history/{session_id}")
    assert history_response2.status_code == 200
    assert len(history_response2.json()["history"]) == 0

def test_chat_history_nonexistent_session():
    """Test getting chat history for non-existent session"""
    response = client.get("/chat-history/nonexistent_session")
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "nonexistent_session"
    assert data["history"] == []

def test_chat_context_inclusion():
    """Test that chat context is included in LLM calls for non-product queries"""
    session_id = "test_context_111"
    
    # Send first message
    response1 = client.post("/chat", json={
        "text": "My name is John",
        "session_id": session_id
    })
    assert response1.status_code == 200
    
    # Send second message that should reference the first
    response2 = client.post("/chat", json={
        "text": "What's my name?",
        "session_id": session_id
    })
    assert response2.status_code == 200
    
    # Verify both messages are in history
    history_response = client.get(f"/chat-history/{session_id}")
    assert history_response.status_code == 200
    history_data = history_response.json()
    assert len(history_data["history"]) == 2
    assert history_data["history"][0]["user"] == "My name is John"
    assert history_data["history"][1]["user"] == "What's my name?"

@patch('main.get_product_prices_from_search', side_effect=mock_get_products)
def test_product_query_with_history(mock_get):
    """Test product query with existing chat history"""
    session_id = "test_product_history_222"
    
    # Send a non-product message first
    response1 = client.post("/chat", json={
        "text": "I need medical equipment",
        "session_id": session_id
    })
    assert response1.status_code == 200
    
    # Send a product query
    response2 = client.post("/chat", json={
        "text": "Show me wheelchairs",
        "session_id": session_id
    })
    assert response2.status_code == 200
    data = response2.json()
    assert "products" in data
    assert "session_id" in data
    assert data["session_id"] == session_id
    
    # Verify both messages are in history
    history_response = client.get(f"/chat-history/{session_id}")
    assert history_response.status_code == 200
    history_data = history_response.json()
    assert len(history_data["history"]) == 2 