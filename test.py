import pytest
from flaskRoute import app  # Import your Flask app instance

@pytest.fixture
def client():
    # Create a test client using the Flask application configured for testing
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_predict_route(client):
    # Define the input data to be sent to the predict route
    input_data = {
        'avg_income': 100000.0,a
        'house_age': 10.0,
        'num_rooms': 3.0,
        'num_bedrooms': 3.0,
        'population': 5000.0

    }

    # Send a POST request to the /predict route with JSON data
    response = client.post('/predict', json=input_data)

    # Ensure the request was successful
    assert response.status_code == 200

    # Expected output (replace this with the expected output from your model)
    expected_output = {
        'prediction': 1  # Replace with your actual expected prediction value
    }

    # Compare the actual output with the expected output
    assert response.json == expected_output
