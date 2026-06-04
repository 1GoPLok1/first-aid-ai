import logging
from typing import Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from prompts import SYSTEM_PROMPTS, OUT_OF_SCOPE_RESPONSE, EMERGENCY_DISCLAIMER

logger = logging.getLogger(__name__)

LABELS = ["EMERGENCY", "LIFESTYLE", "OUT_OF_SCOPE", "CHITCHAT"]

ROUTING_MAP = {
    "EMERGENCY": {
        "collection": "first_aid_protocols",
        "prompt": "emergency_prompt",
        "add_disclaimer": True,
    },
    "LIFESTYLE": {
        "collection": "healthy_lifestyle",
        "prompt": "lifestyle_prompt",
        "add_disclaimer": False,
    },
    "OUT_OF_SCOPE": {
        "collection": None,
        "prompt": None,
        "add_disclaimer": False,
        "static_response": OUT_OF_SCOPE_RESPONSE,
    },
    "CHITCHAT": {
        "collection": None,
        "prompt": "chitchat_prompt",
        "add_disclaimer": False,
    },
}

class QueryRouter:
    def __init__(
        self,
        model_name: str = "cointegrated/rubert-tiny2",
        device: Optional[str] = None,
    ):
        self.model_name = model_name

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info("Загрузка классификатора %s на %s...", model_name, self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(LABELS),
            ignore_mismatched_sizes=True,
        )
        self.model.to(self.device)
        self.model.eval()

        logger.info("Классификатор загружен. Классы: %s", LABELS)

    def classify(self, query: str) -> str:
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_class_id = torch.argmax(logits, dim=-1).item()

        label = LABELS[predicted_class_id]
        logger.info("Запрос классифицирован как '%s': %.100s", label, query)
        return label

    def classify_with_scores(self, query: str) -> dict:
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)[0]

        scores = {
            label: round(prob.item(), 4)
            for label, prob in zip(LABELS, probabilities)
        }
        predicted_label = max(scores, key=scores.get)

        return {
            "label": predicted_label,
            "scores": scores,
        }

    def get_collection_and_prompt(
        self, label: str
    ) -> Tuple[Optional[str], Optional[str]]:
        route = ROUTING_MAP.get(label, ROUTING_MAP["CHITCHAT"])
        collection = route.get("collection")
        prompt = route.get("prompt")
        return collection, prompt

    def get_static_response(self, label: str) -> Optional[str]:
        route = ROUTING_MAP.get(label, {})
        return route.get("static_response")

    def needs_retrieval(self, label: str) -> bool:
        route = ROUTING_MAP.get(label, {})
        return route.get("collection") is not None

    def get_emergency_disclaimer(self) -> str:
        return EMERGENCY_DISCLAIMER

    def should_add_disclaimer(self, label: str) -> bool:
        route = ROUTING_MAP.get(label, {})
        return route.get("add_disclaimer", False)

    def process_query(self, query: str) -> dict:
        label = self.classify(query)
        collection, prompt = self.get_collection_and_prompt(label)
        static_response = self.get_static_response(label)
        needs_retrieval = self.needs_retrieval(label)
        add_disclaimer = self.should_add_disclaimer(label)
        disclaimer = self.get_emergency_disclaimer() if add_disclaimer else None

        result = {
            "query": query,
            "label": label,
            "collection": collection,
            "prompt": prompt,
            "needs_retrieval": needs_retrieval,
            "static_response": static_response,
            "disclaimer": disclaimer,
        }

        logger.info(
            "Маршрутизация: label=%s, collection=%s, prompt=%s, retrieval=%s, disclaimer=%s",
            label,
            collection,
            prompt,
            needs_retrieval,
            bool(disclaimer),
        )

        return result