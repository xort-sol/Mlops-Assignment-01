import json
from api import index

def test_index_route(app, client):
    res = client.get('/')
    assert res.status_code == 200
    expected = '<h2 class="text-center">flask example ci/cd</h2>'
    assert expected in res.get_data(as_text=True)