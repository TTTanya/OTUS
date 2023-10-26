from dataclasses import dataclass
from pathlib import Path

import yaml
from transformers import pipeline

# загружаем config
config_path = Path(__file__).parent / "config.yaml"
with open(config_path, "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# моя модель классификации юридических запросов на форму, которую я использую в рабочих задачах
@dataclass
class PapClassification:

    label: str
    score: float


def load_model():

    model_hf = pipeline(config["task"], model=config["model"], device=-1, framework="tf")

    def model(text: str) -> PapClassification:
        pred = model_hf(text)
        pred_best_class = pred[0]
        return PapClassification(
            label=pred_best_class["label"],
            score=pred_best_class["score"],
        )

    return model
