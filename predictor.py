from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import List, Sequence
from urllib.parse import urlencode, urlparse, urlunparse

import numpy as np
import pandas as pd
import requests


class PredictorConfigurationError(Exception):
    """Raised when the application cannot configure any predictor."""


class AzureMLPredictor:
    """
    Invoke an Azure ML managed online endpoint.

    Authentication supports either an endpoint key (preferred) or a bearer
    token that was retrieved via `az account get-access-token`.
    """

    def __init__(
        self,
        endpoint_url: str,
        access_secret: str,
        deployment_name: str | None = None,
        timeout: int = 60,
    ) -> None:
        if not endpoint_url:
            raise PredictorConfigurationError("AZUREML_ENDPOINT_URL is not set.")
        if not access_secret:
            raise PredictorConfigurationError(
                "Set AZUREML_ENDPOINT_KEY or AZUREML_ENDPOINT_TOKEN."
            )

        self.endpoint_url = self._normalize_endpoint_url(endpoint_url, deployment_name)
        self.timeout = timeout
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_secret}",
        }

    def predict(self, features: pd.DataFrame) -> List[float]:
        logging.info(
            "Posting payload to Azure ML endpoint: n_cols=%s columns=%s",
            features.shape[1],
            features.columns.tolist(),
        )

        payload = self._build_payload(features)
        response = requests.post(
            self.endpoint_url,
            headers=self.headers,
            json=payload,
            timeout=self.timeout,
        )

        if not response.ok:
            raise RuntimeError(
                f"Azure ML endpoint returned {response.status_code}: {response.text}"
            )

        raw = response.json()
        predictions = self._extract_predictions(raw)
        return [float(value) for value in predictions]

    @staticmethod
    def _build_payload(features: pd.DataFrame) -> dict:
        # AzureML's inference-schema expects a dict-of-lists representation.
        return {
            "Inputs": {
                "data": {
                    column: features[column].tolist() for column in features.columns
                }
            },
            "GlobalParameters": {"method": "predict"},
        }

    @staticmethod
    def _extract_predictions(payload) -> List[float]:
        if isinstance(payload, dict):
            for key in ("Results", "results", "predictions", "output", "value"):
                if key in payload:
                    return AzureMLPredictor._flatten(payload[key])
        elif isinstance(payload, list):
            return AzureMLPredictor._flatten(payload)

        raise RuntimeError(f"Unexpected Azure ML response payload: {payload}")

    @staticmethod
    def _flatten(values: Sequence) -> List[float]:
        flattened: List[float] = []
        for value in values:
            if isinstance(value, list):
                flattened.extend(AzureMLPredictor._flatten(value))
            else:
                flattened.append(value)
        return flattened

    @staticmethod
    def _normalize_endpoint_url(endpoint_url: str, deployment_name: str | None) -> str:
        parsed = urlparse(endpoint_url)
        path = parsed.path.rstrip("/")
        if not path.endswith("/score"):
            path = f"{path}/score"

        query = parsed.query
        params = {}
        if query:
            for item in query.split("&"):
                if not item:
                    continue
                key, value = item.split("=", 1) if "=" in item else (item, "")
                params[key] = value
        if deployment_name:
            params["deployment"] = deployment_name

        encoded_query = urlencode(params) if params else ""
        normalized = urlunparse(
            (parsed.scheme, parsed.netloc, path, parsed.params, encoded_query, parsed.fragment)
        )
        return normalized


class LocalModelPredictor:
    """Fallback predictor that loads model.pkl from disk for offline testing."""

    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise PredictorConfigurationError(f"モデルファイルが見つかりません: {model_path}")

        with model_path.open("rb") as buffer:
            self.model = pickle.load(buffer)

    def predict(self, features: pd.DataFrame) -> List[float]:
        raw = self.model.predict(features)
        flattened = np.asarray(raw).reshape(-1)
        return flattened.astype(float).tolist()


def build_predictor(model_path: Path) -> object:
    """
    Create the best available predictor.

    Priority:
        1. Azure ML endpoint (AZUREML_ENDPOINT_URL + KEY/TOKEN)
        2. Local model.pkl fallback
    """

    endpoint_url = os.getenv("AZUREML_ENDPOINT_URL")
    api_key = os.getenv("AZUREML_ENDPOINT_KEY")
    aad_token = os.getenv("AZUREML_ENDPOINT_TOKEN")
    deployment_name = os.getenv("AZUREML_DEPLOYMENT_NAME")

    access_secret = api_key or aad_token
    if endpoint_url and access_secret:
        logging.info("Using Azure ML endpoint predictor.")
        return AzureMLPredictor(endpoint_url, access_secret, deployment_name=deployment_name)

    if model_path.exists():
        logging.info("Falling back to local model predictor.")
        return LocalModelPredictor(model_path)

    raise PredictorConfigurationError(
        "AzureML endpoint is not configured and model.pkl is missing. "
        "Set AZUREML_ENDPOINT_URL/AZUREML_ENDPOINT_KEY or place a local model."
    )
