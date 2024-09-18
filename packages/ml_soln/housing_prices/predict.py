from ml_soln.common.predict import save_prediction
from ml_soln.housing_prices import ctx
from sklearn.pipeline import Pipeline


def predict_and_save(model: Pipeline = None,
                     job_name: str = None):

    if not model and not job_name:
        raise ValueError("model or job_name must be provided")

    if not model:
        model = ctx().model_persistence.load_model(job_name)

    paths = ctx().paths
    if job_name:
        paths = paths.clone(job_name)

    X_test = ctx().data_preparer.test_data()
    predictions = model.predict(X_test)
    X_test['SalePrice'] = predictions

    return save_prediction(submission_df=X_test,
                           predictions_dir=paths.predictions_dir,
                           columns=['Id', 'SalePrice'])
