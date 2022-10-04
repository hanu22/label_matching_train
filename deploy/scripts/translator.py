import logging
import os
import uuid
from typing import Union

import requests
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%d/%m/%y %H:%M:%S",
    handlers=[RichHandler(markup=True)],
)

TRANSLATOR_API_KEY = os.getenv("TRANSLATOR_API_KEY")
endpoint = r"https://api.cognitive.microsofttranslator.com"
location = "northeurope"

headers = {
    "Ocp-Apim-Subscription-Key": TRANSLATOR_API_KEY,
    "Ocp-Apim-Subscription-Region": location,
    "Content-type": "application/json",
    "X-ClientTraceId": str(uuid.uuid4()),
}

params = {
    "api-version": "3.0",
    "to": "en",
}


def translate(
    texts: list[str],
    timeout: float,
) -> Union[list[dict], None]:
    """Posts a request to Azure Translator API.

    Args:
        texts (list): a list of texts to translate
        timeout (float): request timeout

    Returns:
        Union[list[dict], None]: list of translation dictionaries:

        {
            "detectedLanguage": {
                "language": str,
                "score": float,
            },
            "translations": [
                {
                    "text": str,
                    "to": str,
                }
            ],
        }

        or None if an error is raised
    """
    # API expects a list of dictionaries, one for each text
    body = [{"text": text} for text in texts]

    logging.info("Posting translation request...")

    try:
        response = requests.post(
            endpoint + "/translate",
            params=params,
            headers=headers,
            json=body,
            timeout=timeout,
        )

    except requests.exceptions.Timeout:
        logging.error(f"TimeoutError after {timeout}")
        return None

    else:
        try:
            response.raise_for_status()

        except requests.exceptions.HTTPError as e:
            logging.error(f"HTTPError: {e}")
            return None

        else:
            response = response.json()
            logging.info("Returning translation dictionary...")
            return response
