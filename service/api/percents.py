import pandas as pd


def get_percentage(user_id: int, reco: list):
    item_df = pd.read_csv("service/pretrained_models/sourcedata/items.csv")
    interaction_df = pd.read_csv("service/pretrained_models/sourcedata/interactions.csv")

    user_items = interaction_df[interaction_df['user_id'] == user_id]['item_id']
    user_genres = item_df[item_df['item_id'].isin(user_items)]['genres']

    reco_genres = item_df[item_df['item_id'].isin(reco)]['genres']

    user_genre_counts = pd.Series(','.join(user_genres).replace(" ", "").split(',')).value_counts()
    reco_genre_counts = pd.Series(','.join(reco_genres).replace(" ", "").split(',')).value_counts()

    intersections = user_genre_counts.index.intersection(reco_genre_counts.index[:10])

    percentage = int(len(intersections) / len(reco_genre_counts) * 100)

    return percentage
