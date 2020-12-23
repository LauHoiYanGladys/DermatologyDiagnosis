import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler


# references
# https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/

# Function to create sequential model, required for KerasClassifier

def create_ANN_model_batch():
    model = keras.Sequential(
        [
            layers.Dense(250, input_dim=33, activation="relu", name="layer1"),
            layers.Dense(250, activation="relu", name="layer2"),
            layers.Dense(250, activation="relu", name="layer3"),
            layers.Dense(6, activation='softmax', name="layer4"),
        ]
    )
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def main():
    # Read dataset
    filename = r".\dermatology.csv"
    df = pd.read_csv(filename)

    # rename the columns
    df.columns = ['erythema', 'scaling', 'definite_borders', 'itching', 'koebner_phenomenon', 'polygonal_papules',
                  'follicular_papules', 'oral_mucosal_involvement', 'knee_and_elbow_involvement', 'scalp_involvement',
                  'family_history', 'melanin_incontinence', 'eosinophils_in_the_infiltrate', 'PNL_infiltrate',
                  'fibrosis_of_the_papillary_dermis', 'exocytosis', 'acanthosis', 'hyperkeratosis', 'parakeratosis',
                  'clubbing_of_the_rete_ridges', 'elongation_of_the_rete_ridges', 'thinning_of_the_suprapapillary_epidermis',
                  'spongiform pustule', 'munro_microabcess', 'focal_hypergranulosis', 'disappearance_of the granular layer',
                  'vacuolisation_and_damage_of_basal_layer', 'spongiosis', 'saw-tooth_appearance_of_retes',
                  'follicular_horn_plug', 'perifollicular_parakeratosis', 'inflammatory_monoluclear_inflitrate',
                  'band-like_infiltrate', 'age', 'diagnosis']

    # remove the age feature as there are missing values
    df.drop('age', axis=1, inplace=True)
    print(df.head(10)) #show the first x rows of the data (e.g. 10 rows)
    # count the number of instances of each class
    instances = []
    numClass = 6
    for i in range(1,numClass+1):
        # Get a bool series representing which row satisfies the condition i.e. True for
        # row in which value of 'Age' column is more than 30
        seriesObj = df.apply(lambda x: True if x['diagnosis'] == i else False, axis=1)
        # Count number of True in series
        numOfRows = len(seriesObj[seriesObj == True].index)
        instances.append(numOfRows)

    # make the diagnosis label run from 0-5 instead of 1-6 to facilitate later steps
    labenc = LabelEncoder()
    df['diagnosis'] = labenc.fit_transform(df['diagnosis'])
    print(df.head(10))  # show the first x rows of the data (e.g. 10 rows)

    # Split dataset into attributes and labels
    X = df.iloc[:, :-1].values  # all row, 1st to 2nd-to-last col
    y = df.iloc[:, -1]  # last col

    # Split data into training (80%) and test (20%) sets, "stratify=y" ensures the split follows the original y distribution
    # Set random_state for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0, stratify=y)  # get test set

    # Feature scaling
    scaler = StandardScaler()  # normalization: zero mean, unit variance
    scaler.fit(X_train)  # scaling factor determined from the training set
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)  # apply the same scaling to the test set

    # second implementation of ANN
    ros = RandomOverSampler(random_state=88)
    X_resampled, y_resampled = ros.fit_sample(X_train, y_train)

    yvals, counts = np.unique(y_train, return_counts=True)
    yvals_ros, counts_ros = np.unique(y_resampled, return_counts=True)
    print('Classes in training set:', dict(zip(yvals, counts)), '\n',
          'Classes in rebalanced training set:', dict(zip(yvals_ros, counts_ros)))

    # use GridSearchCV to search for the optimal epoch number and batch size of the model
    nn_model = KerasClassifier(build_fn=create_ANN_model_batch, verbose=0)
    batch_size = range(10, 101, 10)
    epochs = range(100, 1001, 100)
    param_grid = dict(batch_size=batch_size, nb_epoch=epochs)
    grid = GridSearchCV(estimator=nn_model, param_grid=param_grid, n_jobs=-1, scoring='accuracy', cv=5)
    grid_result = grid.fit(X_resampled, y_resampled)

    print("Best parameters set found on training set:")
    print()
    print(grid.best_params_)
    print()
    print("Grid scores on training set:")
    print()
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    y_pred = grid_result.predict(X_test)

    print('Confusion matrix (test set) of ANN:')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


    # # SVM (support vector classifier SVC)

    # find the best parameter by grid search
    param_grid = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}]


    print("# Tuning hyper-parameters for accuracy")
    print()

    clf = GridSearchCV(
        SVC(class_weight='balanced'), param_grid, scoring='accuracy'
    )
    clf.fit(X_train, y_train)

    print("Best parameters set found on training set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print('Confusion matrix (test set) of SVM:')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_true, y_pred))
    print()

    plt.show()

if __name__ == "__main__":
    main()
