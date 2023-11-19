from http import HTTPStatus

from starlette.testclient import TestClient

from service.settings import ServiceConfig

GET_RECO_PATH = "/reco/{model_name}/{user_id}"
GET_EXPLAIN_PATH = "/explain/{model_name}/{user_id}/{item_id}"


def test_health(
    client: TestClient,
) -> None:
    with client:
        response = client.get("/health")
    assert response.status_code == HTTPStatus.OK


def test_get_reco_success(
    client: TestClient,
    service_config: ServiceConfig,
) -> None:
    user_id = 123
    path = GET_RECO_PATH.format(model_name="test_model", user_id=user_id)
    with client:
        response = client.get(path)
    assert response.status_code == HTTPStatus.OK
    response_json = response.json()
    assert response_json["user_id"] == user_id
    assert len(response_json["items"]) == service_config.k_recs
    assert all(isinstance(item_id, int) for item_id in response_json["items"])


def test_get_reco_for_unknown_user(
    client: TestClient,
) -> None:
    user_id = 10 ** 10
    path = GET_RECO_PATH.format(model_name="test_model", user_id=user_id)
    with client:
        response = client.get(path)
    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.json()["errors"][0]["error_key"] == "user_not_found"


def test_get_reco_for_unauthorized_user(
    client: TestClient,
) -> None:
    user_id = 4
    token_false = "557757567"
    path = GET_RECO_PATH.format(model_name="test_model", user_id=user_id)
    with client:
        response = client.get(
            path, headers={'Authorization': f'Bearer {token_false}'})
    assert response.status_code == HTTPStatus.UNAUTHORIZED
    assert response.json()["errors"][0]["error_key"] == "unauthorized"


def test_get_reco_for_unknown_models(
    client: TestClient,
) -> None:
    user_id = 4
    false_model = "abc"
    token = "12345678"
    path = GET_RECO_PATH.format(model_name=false_model, user_id=user_id)
    with client:
        response = client.get(
            path, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.json()["errors"][0]["error_key"] == "model not found"


def test_explain_unknown_model(
    client: TestClient
):
    user_id = 2
    item_id = 13865
    token = "12345678"
    path = GET_EXPLAIN_PATH.format(
        model_name="Abmodel",
        user_id=user_id,
        item_id=item_id,
    )
    with client:
        response = client.get(
            path, headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.json()["errors"][0]["error_key"] == "model not found"


def test_explain_for_unknown_user(
    client: TestClient,
) -> None:
    user_id = 10 ** 10
    item_id = 123456
    token = "12345678"
    path = GET_EXPLAIN_PATH.format(model_name="userknn_model", user_id=user_id,
                                   item_id=item_id)
    with client:
        response = client.get(
            path, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.json()["errors"][0]["error_key"] == "user_not_found"


def test_explain_item(
    client: TestClient,
):
    user_id = 1097534
    item_id = 14
    token = "12345678"
    path = GET_EXPLAIN_PATH.format(model_name="userknn_model", user_id=user_id,
                                   item_id=item_id)
    with client:
        response = client.get(
            path, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == HTTPStatus.OK
    assert response.json()["explanation"] == "The movies from recommendation " \
                                             "match with the user 313966 by 58%"
    assert response.json()["p"] == 58
