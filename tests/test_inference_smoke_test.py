from api.model_loader import predict_clause
from ml.model_registry import resolve_model_dir


def test_resolve_model_dir_exists():
    model_dir = resolve_model_dir()
    assert model_dir.exists()
    assert model_dir.is_dir()


def test_predict_clause_smoke():
    result = predict_clause("This agreement shall terminate upon thirty days written notice.")

    assert isinstance(result, dict)
    assert "label" in result
    assert "confidence" in result

    assert isinstance(result["label"], str)
    assert result["label"] != ""

    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0