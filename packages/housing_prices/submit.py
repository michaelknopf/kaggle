from housing_prices.predict import train_and_test
from housing_prices.paths import paths
from common.kaggle_facade import KaggleFacade, HOUSE_PRICES_COMPETITION_NAME

def predict_and_submit():
    submission_file = train_and_test()
    kaggle = KaggleFacade(HOUSE_PRICES_COMPETITION_NAME, paths)
    kaggle.submit_predictions(file_path=submission_file)

if __name__ == '__main__':
    predict_and_submit()
