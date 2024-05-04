from download.import_data_module import import_dataframe_from_csv
from data_preprocess.data_cleaner_module import clean_Data
from data_preprocess.data_preparation_module import enrich_rf_features, prepare_cleaned_data
from autogluon.tabular import TabularDataset, TabularPredictor

crabs = import_dataframe_from_csv("model_data\CrabAgePrediction.csv")
crabs = clean_Data(crabs)
crabs.reset_index(drop=True, inplace=True)
crabs = enrich_rf_features(crabs)
crabs = prepare_cleaned_data(crabs)

train = crabs.sample(frac=0.8, random_state=42)
test = crabs.drop(train.index)

train_data = TabularDataset(train)
test_data = TabularDataset(test)

predictor = TabularPredictor(label="Age", problem_type='regression', eval_metric='r2').fit(train_data, presets="best_quality")

print(predictor.evaluate(test_data, silent=True))

leaderboard = predictor.leaderboard(test_data)

print(leaderboard.head())