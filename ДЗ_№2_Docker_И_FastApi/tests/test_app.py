import pytest
import requests


@pytest.mark.parametrize(
    "input_text, expected_label",
    [
        ("договор аренды офисных помещений", "pap"),
        ("система управления охраной труда в организации", "not_pap")
    ],
)
def test_pap(input_text: str, expected_label: str):
    response = requests.get("http://0.0.0.0/predict/", params={"text": input_text})
    assert response.json()["text"] == input_text
    assert response.json()["pap_label"] == expected_label
