"""
The master file to generate data and model for the classification.
"""

import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


class DataModel(object):

    def __init__(self):
        self._X = None

        self._X_train = None
        self._X_test = None
        self._y_train = None
        self._y_test = None

        self._dataset = None

    def _initialize_standardize_parameters(self):
        self._X_std_train = None
        self._X_std_test = None

        return

    def _initialize_pca_parameters(self):
        self._X_pca_train = None
        self._X_pca_test = None

    def _initialize_balanced_parameters(self):
        self._X_balanced = None
        self._y_balanced = None

    def get_training(self):
        return self._X_test, self._y_train

    def get_test(self):
        return self._X_test, self._y_test

    def get_dataset(self):
        return self._dataset

    def read_file_and_generate_data(self, fname, cols=[None, None], test_size=None, r_state=None, stratify=False):
        self._dataset = pd.read_csv(fname)

        if cols == [None, None]:
            cols = [0, -2]
        elif cols[0] is None:
            cols = [0, cols[1]]
        elif cols[1] is None:
            cols = [cols[0], -2]

        self._X = self._dataset.iloc[:, cols[0]:cols[1]]
        y = self._dataset['Label']

        if test_size is None:
            test_size = 0.2
        if r_state is None:
            r_state = 42
        if stratify:
            stratify = y

        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(self._X, y,
                                                                                    test_size=test_size,
                                                                                    random_state=r_state,
                                                                                    stratify=stratify)

        return None

    def balance_data(self, num_neighbors=5, with_pca=False):
        """
        Perform over-sampling using SMOTE (Synthetic Minority
        Over-sampling) when training classes are imbalanced.

        :param num_neighbors:
        :param with_pca:
        :return:
        """
        # SMOTE number of neighbors
        k = num_neighbors

        sm = SMOTE(sampling_strategy='auto', k_neighbors=k)
        if with_pca:
            self._X_balanced, self._y_balanced = sm.fit_resample(self._X_pca_train, self._y_train)
        else:
            self._X_balanced, self._y_balanced = sm.fit_resample(self._X_train, self._y_train)

        return None

    def standardize_features(self):
        """
        Standardize features by removing the mean and scaling to unit variance

        The standard score of a sample `x` is calculated as:

            z = (x - u) / s

        where `u` is the mean of the training samples or zero if `with_mean=False`,
        and `s` is the standard deviation of the training samples or one if
        `with_std=False`.

        :return:
        """
        self._initialize_standardize_parameters()

        scaler = StandardScaler()
        scaler.fit(self._X_train)
        self._X_std_train = scaler.transform(self._X_train)
        self._X_std_test = scaler.transform(self._X_test)

        self._X_std_train = pd.DataFrame(self._X_std_train)
        self._X_std_test = pd.DataFrame(self._X_std_test)

        self._X_std_train.columns = list(self._X.columns)
        self._X_std_test.columns = list(self._X.columns)

        return None

    def get_model_accuracy(self, prediction_obj, normalize=True):
        """
        Accuracy classification score.

        :param prediction_obj: array-like of shape (n_samples,) or (n_samples, n_outputs) The predicted classes.
        :param normalize: bool, optional (default=True)
        If ``False``, return the number of correctly classified samples.
        Otherwise, return the fraction of correctly classified samples.
        :return: float
        If ``normalize == True``, return the fraction of correctly
        classified samples (float), else returns the number of correctly
        classified samples (int).

        The best performance is 1 with ``normalize == True`` and the number
        of samples with ``normalize == False``.
        """

        return accuracy_score(self._y_test, prediction_obj, normalize)

    def train_rf_model(self, with_pca=False, smote=False):
        """
        create and train a random forest classifier using already generated training and test data.

        :return:
        """

        if with_pca:
            X_train, X_test = self._X_pca_train, self._X_pca_test
        else:
            X_train, X_test = self._X_train, self._X_test

        if smote:
            X_train, y_train = self._X_balanced, self._y_balanced
            classifier = RandomForestClassifier(n_estimators=50, criterion='gini', bootstrap=True)
            classifier.fit(X_train, y_train)
            prediction = classifier.predict(X_test)
            return prediction

        classifier = RandomForestClassifier(n_estimators=50, criterion='gini', bootstrap=True)
        classifier.fit(X_train, self._y_train)
        prediction = classifier.predict(X_test)

        return prediction

    def train_gnb_model(self, with_pca=False, smote=False):
        """
        create and train a Gaussian Naive Bayes classifier using already generated training and test data.

        :return:
        """

        if with_pca:
            X_train, X_test = self._X_pca_train, self._X_pca_test
        else:
            X_train, X_test = self._X_train, self._X_test

        if smote:
            X_train, y_train = self._X_balanced, self._y_balanced
            gnb = GaussianNB()
            gnb.fit(X_train, y_train)
            prediction = gnb.predict(X_test)
            return prediction

        gnb = GaussianNB()
        gnb.fit(X_train, self._y_train)
        prediction = gnb.predict(X_test)

        return prediction

    def train_pca_model(self):
        pca = PCA(.95).fit(self._X_std_train)
        print("Number of principal components that explain 95% variability in data:", pca.n_components_)
        self._X_pca_train = pca.transform(self._X_std_train)
        self._X_pca_test = pca.transform(self._X_std_test)

    def tune_model_hyperparameters(self):
        # this can be used to tune classifier hyperparameters
        pipe = Pipeline(
            [
                ('resample', SMOTE()),
                ('model', RandomForestClassifier())]
        )

        kf = StratifiedKFold(n_splits=10, shuffle=True)

        p_grid = dict(model__n_estimators=[50, 100, 200])
        grid_search = GridSearchCV(
            estimator=pipe, param_grid=p_grid, cv=kf, refit=True
        )
        grid_search.fit(self._X_pca_train, self._y_train)

        #   Adding below in as could be helpful to know how to get fitted scaler if used
        # best = grid_search.best_estimator_
        # print(best)
        prediction = grid_search.predict(self._X_pca_test)
        cnf_matrix = confusion_matrix(self._y_test, prediction)

        return prediction, cnf_matrix
