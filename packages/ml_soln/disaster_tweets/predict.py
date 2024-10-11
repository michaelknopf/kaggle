import numpy as np
import pandas as pd
from keras import Sequential
from ml_soln.common.predict import save_prediction, get_model_and_paths
from ml_soln.disaster_tweets import ctx

def predict_and_save(model: Sequential = None,
                     job_name: str = None):

    model, paths = get_model_and_paths(ctx, model, job_name)

    X = ctx().data_preparer.test_data()
    y_probs = model.predict(X)
    prediction_df = format_predictions(y_probs)

    prediction_file = save_prediction(submission_df=prediction_df,
                                      predictions_dir=paths.predictions_dir)

    return prediction_file, prediction_df

def format_predictions(y_probs: pd.Series) -> pd.DataFrame:
    X = ctx().data_preparer._load_raw_test_data()
    y = np.argmax(y_probs, axis=1)
    return pd.concat([
        X['id'],
        pd.Series(y, name='target')
    ], axis=1)
