import pytest
from index import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_home(client):
    """Test the home page"""
    response = client.get('/')
    assert response.status_code == 200
    assert response.data == b'Hello World'

def test_predict(client):
    # Define the payload to send in the POST request
    payload = {
        'avg_income': 50000,
        'house_age': 20,
        'num_rooms': 3,
        'num_bedrooms': 2,
        'population': 1000
    }

    # Send a POST request to the /predict endpoint
    response = client.post('/predict', 
        data=json.dumps(payload), 
        content_type='application/json')

    # Assert the response status code
    assert response.status_code == 200

    # Assert the response content
    response_json = json.loads(response.data)
    assert 'prediction' in response_json