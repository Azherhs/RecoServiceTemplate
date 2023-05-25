import pandas as pd

userknn_recos_off = pd.read_csv('service/pretrained_models/my_datas.csv')

popular_model_recs = [15297, 10440, 4151, 13865, 9728, 3734, 12192, 142, 2657,
                      4880]

with open('service/pretrained_models/cold_users.txt', "r",
          encoding="utf-8") as file: cold_users = [int(line.strip()) for line
                                                   in file.readlines()]

light_fm_recos_off = \
    pd.read_csv('service/pretrained_models/lightfm_recses.csv')

similarity_df = pd.read_csv("service/pretrained_models/userknn_similarity")


def find_reco(model_name, user_id):
    global reco
    if model_name == "lightfm_model":
        reco = eval(light_fm_recos_off.loc[user_id, "item_id"])
    elif model_name == "userknn_model":
        if user_id > 962000:
            reco = popular_model_recs
        elif len(eval(userknn_recos_off.loc[user_id, "item_id"])) != 10:
            reco = popular_model_recs
        else:
            reco = eval(userknn_recos_off.loc[user_id, "item_id"])

    elif model_name == "popular_model":
        reco = popular_model_recs
    return reco


def calculate_similarity(item_id: int, user_id: int):
    matched_users = similarity_df[(similarity_df['item_id'] == item_id) &
                                  (similarity_df['user_id'] == user_id)]
    if matched_users.empty:
        return 0, None  # No match found

    percent_similarity = int(matched_users['similarity'].values[0] * 100)
    most_similar_user_id = matched_users['similar_user_id'].values[0]
    return percent_similarity, most_similar_user_id
