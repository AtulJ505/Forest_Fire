import os
from unittest.mock import Mock, patch
import pytest
import application

@pytest.fixture
def client():
    application.app.config.update(TESTING=True)
    return application.app.test_client()

@pytest.mark.parametrize('path', ['/', '/predict'])
def test_forms_load_without_loading_model_artifacts(client, path):
    with patch.object(application, 'load_models') as loader:
        assert client.get(path).status_code == 200
        loader.assert_not_called()

@pytest.mark.parametrize('value', ['', 'not-a-number', 'NaN', 'Infinity', '-Infinity'])
def test_invalid_input_returns_400_before_prediction(client, value):
    payload = {field: '1' for field in application.FIELDS}
    payload['Temperature'] = value
    with patch.object(application, 'load_models') as loader:
        assert client.post('/predict', data=payload).status_code == 400
        loader.assert_not_called()

def test_missing_input_returns_400(client):
    assert client.post('/predict', data={}).status_code == 400

def test_prediction_preserves_training_feature_order(client):
    payload = {field: str(i) for i, field in enumerate(application.FIELDS)}
    model, scaler = Mock(), Mock()
    scaler.transform.return_value = [[0]]
    model.predict.return_value = [3.5]
    with patch.object(application, 'load_models', return_value=(model, scaler)):
        response = client.post('/predict', data=payload)
    assert response.status_code == 200
    assert b'3.5' in response.data
    scaler.transform.assert_called_once_with([[float(i) for i in range(9)]])
    model.predict.assert_called_once_with([[0]])

def test_artifact_paths_are_independent_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    application.load_models.cache_clear()
    with patch.object(application.pickle, 'load', side_effect=['model', 'scaler']) as loader:
        assert application.load_models() == ('model', 'scaler')
    assert loader.call_count == 2
    application.load_models.cache_clear()
