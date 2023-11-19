from typing import List

# import pandas as pd
import yaml
from fastapi import APIRouter, FastAPI, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from service.config.responses import responses

from service.api.exceptions import UserNotFoundError, NotAuthorizedError, \
    ModelNotFoundError
from service.log import app_logger
from service.api.percents import get_percentage
from service.api.recofind import find_reco, calculate_similarity

config_file = "config/config.yaml"
with open(config_file) as f:
    config = yaml.load(f, Loader=yaml.Loader)

with open('service/pretrained_models/cold_users.txt', "r",
          encoding="utf-8") as file: cold_users = [int(line.strip()) for line
                                                   in file.readlines()]


class ExplainResponse(BaseModel):
    p: int
    explanation: str


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


router = APIRouter()

auth = HTTPBearer()


@router.get(
    path="/health",
    tags=["Health"],
)
async def health() -> str:
    return "I am alive"


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
    responses=responses
)
async def get_reco(
    request: Request,
    model_name: str,
    user_id: int,
    token: HTTPAuthorizationCredentials = Depends(auth)
) -> RecoResponse:
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")
    # Write your code here

    if token.credentials != config['auth']['token']:
        raise NotAuthorizedError(error_message=f"Token: {token} not found",
                                 status_code=401)

    if model_name not in config['service']['models']:
        raise ModelNotFoundError(error_message=f"Model: "
                                               f"{model_name} not found")

    if user_id > 10 ** 9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")
    elif model_name == "test_model":
        k_recs = request.app.state.k_recs
        reco = list(range(k_recs))
    else:
        reco = find_reco(model_name, user_id)

    return RecoResponse(user_id=user_id, items=reco)


@router.get(
    path="/explain/{model_name}/{user_id}/{item_id}",
    tags=["Explanations"],
    response_model=ExplainResponse,
)
async def explain(
    request: Request,
    model_name: str,
    user_id: int,
    item_id: int,
    token: HTTPAuthorizationCredentials = Depends(auth)
) -> ExplainResponse:
    """
    Пользователь переходит на карточку контента, на которой нужно показать
    процент релевантности этого контента зашедшему пользователю,
    а также текстовое объяснение почему ему может понравиться этот контент.

    :param request: запрос.
    :param model_name: название модели, для которой нужно получить объяснения.
    :param user_id: id пользователя, для которого нужны объяснения.
    :param item_id: id контента, для которого нужны объяснения.
    :param token: токен для аутентификации
    :return: Response со значением процента релевантности и текстовым
             объяснением, понятным пользователю.
    - "p": "процент релевантности контента item_id для пользователя user_id"
    - "explanation": "текстовое объяснение почему рекомендован item_id"
    """

    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")
    reco = find_reco(model_name, user_id)
    if token.credentials != config['auth']['token']:
        raise NotAuthorizedError(error_message=f"Token: {token} not found",
                                 status_code=401)

    if model_name not in config['service']['models']:
        raise ModelNotFoundError(error_message=f"Model: "
                                               f"{model_name} not found")

    if user_id > 10 ** 9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    elif item_id not in reco:
        return ExplainResponse(p=0, explanation='Item not found for this user')
    elif user_id in cold_users:
        return ExplainResponse(p=85, explanation='Popular')
    elif model_name == "userknn_model":
        percent, simuser_id = calculate_similarity(item_id, user_id)
        if percent > 0:
            explanation = f'The movies from recommendation match with the ' \
                          f'user {simuser_id} by {percent}%.'
        else:
            explanation = "No match found."
    elif model_name == "test_model":
        explanation = "Test explanation."
        percent = 0
    else:
        percent = get_percentage(user_id, reco)
        if percent < 50:
            percent += 50
        explanation = f'The movies you have watched match the ' \
                      f'recommendations by {percent}%'
    return ExplainResponse(p=percent, explanation=explanation)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
