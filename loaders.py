# third parties imports
import pandas as pd
import os
from datetime import datetime
from surprise import Reader
from surprise import Dataset

# local imports
from constants import Constant as C

def load_ratings(surprise_format=True):
    """Load ratings from a CSV file into a Pandas DataFrame or a Surprise Dataset."""
    df_ratings = pd.read_csv(C.EVIDENCE_PATH / C.RATINGS_FILENAME)
    if surprise_format:
        # Convert the DataFrame to a Surprise Dataset
        reader = Reader(rating_scale=C.RATINGS_SCALE)  # Utiliser l'échelle de notation définie
        surprise_dataset = Dataset.load_from_df(df_ratings[[C.USER_ID_COL, C.ITEM_ID_COL, C.RATING_COL]], reader)
        return surprise_dataset  
    else:
        return df_ratings



def load_items():
    df_items = pd.read_csv(C.CONTENT_PATH / C.ITEMS_FILENAME)
    df_items = df_items.set_index(C.ITEM_ID_COL)
    # Ré-indexer le DataFrame pour avoir une séquence continue de valeurs à partir de zéro
    df_items = df_items.reset_index(drop=True)
    
    return df_items


def export_evaluation_report(df):
    """ Export the report to the evaluation folder.
    The name of the report is versioned using today's date.
    Parameters:
    df (DataFrame): The evaluation report DataFrame to export.
    """
    today = datetime.today().strftime("%Y_%m_%d")
    filename = f"evaluation_report_{today}.csv"
    filepath = os.path.join(C.EVALUATION_PATH, filename)
    df.to_csv(filepath, index=False)