from data_model_master import DataModel


def _main():

    #   read data
    data_file_name = "..\\resources\\physiological_data_01.csv"

    #   initialize the model object
    model = DataModel()
    #   read the csv file and generate the data
    model.read_file_and_generate_data(data_file_name,
                                      cols=[6, -2],
                                      test_size=0.3,
                                      r_state=None,
                                      stratify=True)

    #   perform random forest classification on raw data
    raw_rf_pred = model.train_rf_model()
    print("Accuracy of raw RF classification: ", model.get_model_accuracy(raw_rf_pred))

    #   perform Gaussian Naive Bayes classification on raw data
    raw_gnb_pred = model.train_gnb_model()
    print("Accuracy of raw GNB classification: ", model.get_model_accuracy(raw_gnb_pred))

    #   train PCA model to reduce dimensionality of features
    model.standardize_features()
    model.train_pca_model()

    #   perform random forest classification on PCA-reduced data data
    pca_rf_pred = model.train_rf_model(with_pca=True)
    print("Accuracy of PCA-reduced RF classification: ", model.get_model_accuracy(pca_rf_pred))

    #   perform Gaussian Naive Bayes classification on PCA-reduced data
    pca_gnb_pred = model.train_gnb_model(with_pca=True)
    print("Accuracy of PCA-reduced GNB classification: ", model.get_model_accuracy(pca_gnb_pred))

    #   perform balancing of the data classes to see if we can get a better classification
    model.balance_data(num_neighbors=5, with_pca=True)
    #   now perform the classifications with balanced, PCA-reduced data
    smote_rf_pred = model.train_rf_model(with_pca=True, smote=True)
    print("Accuracy of smote, PCA-reduced RF classification: ", model.get_model_accuracy(smote_rf_pred))
    smote_gnb_pred = model.train_gnb_model(with_pca=True, smote=True)
    print("Accuracy of smote, PCA-reduced GNB classification: ", model.get_model_accuracy(smote_gnb_pred))

    #   tune classifier hyperparameters
    tuned_pred, cnf = model.tune_model_hyperparameters()
    print()
    print("Accuracy of smote, PCA-reduced RF classification after tuning: ", model.get_model_accuracy(tuned_pred))
    print("And the confusion matrix: \n", cnf)

    return


if __name__ == '__main__':
    _main()
