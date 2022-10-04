import json
import logging

import azure.functions as func
import spacy
from pydantic import ValidationError
from rich.logging import RichHandler
from scripts.translator import translate

from .validate import RequestModel, ResponseModel

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%d/%m/%y %H:%M:%S",
    handlers=[RichHandler(markup=True)],
)

# Create list of target NERs and lowercase them
ents_target = []
ents_target = list(map(str.lower, ents_target))

model = "en_recipe_usage_shredding"

logging.info("Loading model...")
nlp = spacy.load(model)


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("HTTP trigger function processed a request.")

    try:
        req_body = req.get_json()

    except ValueError as e:
        return func.HttpResponse(
            body=json.dumps({"valueError": f"{e}"}),
            status_code=500,
        )

    # Validate request
    else:

        try:
            RequestModel(**req_body)

        except ValidationError as e:
            return func.HttpResponse(
                body=json.dumps({"validationError": e.errors()}),
                status_code=500,
            )

        else:
            texts = req_body.get("texts")

    # Translate all texts
    # Response is Union[list[dict], None]
    translation = translate(
        texts=texts,
        timeout=10,
    )

    if translation is None:
        return func.HttpResponse(
            body=json.dumps({"translationError"}),
            status_code=500,
        )

    # Create a list to save each text results as one dictionary
    # of 3 sub-dictionaries: text, categories, and entities
    texts_list = list()

    # Loop over all texts
    for ind, text in enumerate(texts):

        # Create 3 sub-dictionaries: text, categories, and entities
        text_dict = {
            "original": text,
            "language": translation[ind]["detectedLanguage"]["language"],
            "score": translation[ind]["detectedLanguage"]["score"],
            "translation": translation[ind]["translations"][0]["text"],
        }
        categories_dict = {"multiclass": None, "multilabel": list()}
        entities_dict = {ent: list() for ent in ents_target}

        # Update categories sub-dictionary
        if "textcat" in nlp.pipe_names:

            # If detected language is not English -> make doc from translation
            # Otherwise -> make doc from original
            if text_dict["language"] != "en":
                doc = nlp(text_dict["translation"])
            else:
                doc = nlp(text_dict["original"])

            categories_dict["multiclass"] = max(
                doc.cats,
                key=lambda key: doc.cats[key],
            ).lower()

            categories_dict["multilabel"] = [
                k.lower() for k, v in doc.cats.items() if round(v, 2) > 0.50
            ]

        # Update entities sub-dictionary
        if "ner" in nlp.pipe_names:

            # If detected language is English or Not -> make doc from original
            # as we cannot relate translation to original at token level
            # TODO: relate translation to original at token level
            doc = nlp(text_dict["original"])

            for ent in doc.ents:
                if ent.label_.lower() in ents_target:
                    entities_dict[ent.label_.lower()].append(
                        {
                            ent.text: {
                                "start_char": ent.start_char + 1,
                                "end_char": ent.end_char,
                            }
                        }
                    )

        texts_list.append(
            {
                "text": text_dict,
                "categories": categories_dict,
                "entities": entities_dict,
            }
        )

    # Add results to a output dict
    response_body = {
        "texts": texts_list,
        "model": nlp.meta,
    }

    # Validate response
    try:
        ResponseModel(**response_body)

    except ValidationError as e:
        return func.HttpResponse(
            body=json.dumps({"validationError": e.errors()}),
            status_code=500,
        )

    else:

        return func.HttpResponse(
            body=json.dumps(
                response_body,
                ensure_ascii=False,
            ),
            status_code=200,
        )
