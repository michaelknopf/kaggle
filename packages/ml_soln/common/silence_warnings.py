import warnings

def silence_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, message=".*Found unknown categories in columns.*")
