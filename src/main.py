from download.import_data_module import import_dataframe_from_csv 
from data_preprocess.data_cleaner_module import clean_Data
from data_preprocess.data_preparation_module import prepare_cleaned_data, enrich_rf_features, split_data
from ml.model_creator_module import create_rf_model, save_model, load_model, get_model_prediction
from evaluation.model_metrics_module import get_model_metrics, get_cross_validation_metrics

if __name__ == '__main__':
    crabs = import_dataframe_from_csv(path="model_data\CrabAgePrediction.csv")
    crabs = clean_Data(crabs)
    crabs.reset_index(drop=True, inplace=True)
    crabs = enrich_rf_features(crabs)
    crabs = prepare_cleaned_data(crabs)
    print(crabs.head())
    X_train, X_test, y_train, y_test = split_data(crabs, target_name="Age")
    rf_model = create_rf_model(X_train, y_train, in_max_depth=7, in_max_features='sqrt', in_min_samples_leaf=2, in_min_samples_split=6, in_n_estimators=215)
    save_model(rf_model)
    #model_predict, mae, mse, rmse, r2 = get_model_metrics(rf_model, X_test, y_test)
    
    cv_scores, cv_scores_mean, cv_scores_std = get_cross_validation_metrics(rf_model, X_train, y_train)
    print(f"Scores: {cv_scores}\n Mean: {cv_scores_mean}\n Devation: {cv_scores_std}")


    test = load_model()

    model_predict, mae, mse, rmse, r2 = get_model_metrics(test, X_test, y_test)
    print(f"predict: {model_predict} Mae: {mae}\n Mse: {mse} \n RMSE: {rmse} \n R2: {r2}")
    pred = get_model_prediction([[2,1,1,1,1,1,1,1,1,1,1,1,1,1]])
    print(X_test.columns)
    print(pred)
    