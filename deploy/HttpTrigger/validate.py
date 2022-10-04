from pydantic import BaseModel, confloat, conlist, constr


class RequestModel(BaseModel):
    texts: conlist(item_type=constr(min_length=1), min_items=1)


class categoriesModel(BaseModel):
    multiclass: constr(min_length=1)
    multilabel: conlist(item_type=constr(min_length=1), min_items=1)


class TextModel(BaseModel):
    original: constr(min_length=1)
    language: constr(min_length=1)
    score: confloat(ge=0, le=1)
    translation: constr(min_length=1)


class TextsModel(BaseModel):
    text: TextModel
    categories: categoriesModel
    entities: dict


class ResponseModel(BaseModel):
    texts: conlist(item_type=TextsModel, min_items=1)
    model: dict
