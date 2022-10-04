import logging
import os
import uuid
from typing import Union

import pandas as pd
import requests
from preprocess import markets_non_english
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


def translate_non_english_markets(df: pd.DataFrame) -> pd.DataFrame:
    """Translates Non-English markets text into English.

    Translation results are saved into new columns
    English markets are kept as is in "text" which spacy expects for training

    Args:
        df (pd.DataFrame): dataframe

    Returns:
        pd.DataFrame: dataframe with translated text
    """
    # Create empty columns
    df["language"] = None
    df["score"] = None
    df["text_translated"] = None

    for ind, row in df.iterrows():

        if row["market"] in markets_non_english:

            print(f"translating {ind=}")

            response = translate(
                texts=[row["text"]],
                timeout=5,
            )

            language = response[0]["detectedLanguage"]["language"]
            score = response[0]["detectedLanguage"]["score"]
            text_translated = response[0]["translations"][0]["text"]

            df.loc[ind, "language"] = language
            df.loc[ind, "score"] = score
            df.loc[ind, "text_translated"] = text_translated

    # Copy "text_translated" into "text" only for Non-English markets
    df.loc[df["market"].isin(markets_non_english), "text"] = df.loc[
        df["market"].isin(markets_non_english), "text_translated"
    ]

    return df
