from ml_soln.common.predict import save_prediction, get_model_and_paths
from ml_soln.housing_prices import ctx
from sklearn.pipeline import Pipeline


def predict_and_save(model: Pipeline = None,
                     job_name: str = None):

    model, paths = get_model_and_paths(ctx, model, job_name)

    X_test = ctx().data_preparer.test_data()
    predictions = model.predict(X_test)
    X_test['SalePrice'] = predictions

    return save_prediction(submission_df=X_test,
                           predictions_dir=paths.predictions_dir,
                           columns=['Id', 'SalePrice'])
