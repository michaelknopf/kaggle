from ml_soln.housing_prices import ctx
from ml_soln.common.predict import save_prediction


def train_and_test():

    X, y = ctx().data_preparer.train_data()
    ctx().model.pipeline.fit(X, y)

    X_test = ctx().data_preparer.test_data()
    predictions = ctx().model.pipeline.predict(X_test)
    X_test['SalePrice'] = predictions

    return save_prediction(X_test, ctx().paths.predictions_dir, columns=['Id', 'SalePrice'])


if __name__ == '__main__':
    train_and_test()
