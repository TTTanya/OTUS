import pytest

from model import PapClassification, load_model


@pytest.fixture(scope="function")
def model():
    return load_model()

# тест проверяет, что модель верно классифицирует запрос, и для каждого параметра (text, expected_label) проверяет соответствующее значение предсказания модели. Если значение не соответствует ожидаемому результату, тест выдает ошибку.
@pytest.mark.parametrize(
    "text, expected_label",
    [
        ("договор аренды офисных помещений", "pap"),
        ("система управления охраной труда в организации", "not_pap")
    ],
)
def test_pap(model, text: str, expected_label: str):
    model_pred = model(text)
    assert isinstance(model_pred, PapClassification)
    assert model_pred.label == expected_label
