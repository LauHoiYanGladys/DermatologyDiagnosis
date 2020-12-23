import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from keras.utils import np_utils
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from imblearn.over_sampling import RandomOverSampler
from matplotlib.colors import Normalize

# references
# https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/

# Function to create sequential model, required for KerasClassifier
def create_ANN_model(nodeNum, withDropOut):

    if withDropOut: # create model with dropout
        model = keras.Sequential(
            [
                layers.Dropout(0.5, input_dim=33), # 50% of input are randomly ignored
                layers.Dense(nodeNum, activation="relu", name="layer1"),
                layers.Dense(nodeNum, activation="relu", name="layer2"),
                layers.Dense(nodeNum, activation="relu", name="layer3"),
                layers.Dense(6, activation='softmax', name="layer4"),
            ]
        )

    else: # create model without dropout
        model = keras.Sequential(
            [
                layers.Dense(nodeNum, input_dim=33, activation="relu", name="layer1"),
                layers.Dense(nodeNum, activation="relu", name="layer2"),
                layers.Dense(nodeNum, activation="relu", name="layer3"),
                layers.Dense(6, activation='softmax', name="layer4"),
            ]
        )
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

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

# Utility function to move the midpoint of a colormap to be around
# the values of interest.


def main():
    # Read dataset
    filename = r"D:\ML\datasets\dermatology.csv"
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

    # # compute class weights for dealing with imbalance dataset
    # maxInstance = max(instances)
    # weights = []
    # for i in range(numClass):
    #     weight = maxInstance/instances[i]
    #     weights.append(weight)
    #
    # # make a dictionary of class weights
    #
    # class_weights = {0: weights[0],
    #                 1: weights[1],
    #                 2: weights[2],
    #                 3: weights[3],
    #                 4: weights[4],
    #                 5: weights[5]}

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

    # Implementation 1 of ANN
    # convert integers to dummy variables (i.e. one hot encoded)
    # dummy_y_train = np_utils.to_categorical(y_train)
    # dummy_y_test = np_utils.to_categorical(y_test)

    # # Fine tune the number of nodes
    # node_nums = range(50, 301, 50)
    # accuracy_node_num = []
    # accuracy_node_num_dropout = []
    # dropOutStatus = [True, False]
    #
    # for withDropOut in dropOutStatus:
    #     for node_num in node_nums:
    #         # stratified K fold
    #         skf = StratifiedKFold(n_splits=5, random_state=1)
    #         fold_num = 1
    #
    #         # Define per-fold score containers
    #         acc_per_fold = []
    #         loss_per_fold = []
    #
    #         for train_index, test_index in skf.split(X_train, y_train):
    #             X_train_CV, X_test_CV = X_train[train_index], X_train[test_index]
    #             dummy_y_train_CV, dummy_y_test_CV = dummy_y_train[train_index], dummy_y_train[test_index]
    #
    #             # create the model
    #             model = create_ANN_model(node_num, withDropOut)
    #
    #             # fit the keras model on the dataset
    #             model.fit(X_train_CV, dummy_y_train_CV, class_weight=class_weights)
    #             #
    #             # # make prediction on test set
    #             # dummy_y_pred_CV = model.predict(X_test_CV)
    #
    #             # evaluate keras model on test set
    #             loss, accuracy = model.evaluate(X_test_CV, dummy_y_test_CV)
    #             acc_per_fold.append(accuracy)
    #             loss_per_fold.append(loss)
    #
    #             # print per fold result
    #             # print('Fold ', fold_num, 'accuracy = ', accuracy, ', loss = ', loss)
    #
    #             # Increase fold number
    #             fold_num = fold_num + 1
    #
    #         average_acc = np.mean(acc_per_fold)
    #         if withDropOut:
    #             accuracy_node_num_dropout.append(average_acc)
    #         else:
    #             accuracy_node_num.append(average_acc)
    # print('Accuracies without dropout: ', accuracy_node_num)
    # print('Accuracies with dropout: ', accuracy_node_num_dropout)
    #     # print('Accuracy: %.2f' % (accuracy*100))
    #
    # # plot the accuracies
    # plt.figure()
    # plt.plot(node_nums, accuracy_node_num_dropout, label="With 50% dropout")
    # plt.plot(node_nums, accuracy_node_num, label="No dropout")
    # plt.xlabel('Number of nodes')
    # plt.ylabel('Accuracy')
    # plt.title('5-fold CV accuracy with different number of nodes with and without dropout')
    # plt.legend()
    #
    #
    # # # Evaluate the model on the test set with the best parameters determined by CV
    # #  (use confusion matrix and other evaluation metrics)
    # model = create_ANN_model(250, False)
    #
    # # fit the keras model on the dataset
    # model.fit(X_train, dummy_y_train, class_weight=class_weights)
    #
    # # make prediction on test set
    # dummy_y_pred = model.predict(X_test)
    #
    # print('Confusion matrix (test set) of ANN:')
    # y_pred = dummy_y_pred.argmax(axis=1)
    # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred))

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
    # # class_weights='balanced' adjusts C according to class frequency
    # clf = SVC(kernel='linear', class_weight='balanced')  # default C=1,

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

    # #create class weights
    # class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    # class_weight_dict = dict(enumerate(class_weights))

    # # estimator = KerasClassifier(build_fn=create_ANN_model, nodeNum=100, epochs=150, batch_size=10, verbose=0)
    # estimator = KerasClassifier(build_fn=create_ANN_model, nodeNum=100, epochs=150, batch_size=10, verbose=0)
    #
    # kfold = KFold(n_splits=5, shuffle=True, random_state=0) # StratifiedKFold because imbalance data
    # results = cross_val_score(estimator, X_train, dummy_y_train, cv=kfold)
    # print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    #
    # # fit the model on training set
    # estimator.fit(X_train, dummy_y_train)
    #
    # # get predictions
    # y_pred = estimator.predict(X_test)

    # # 10-fold CV to try different numbers of nodes in each layer
    # nodeMin = 100
    # nodeMax = 300
    # node_range = range(nodeMin, nodeMax+1, 50)
    # # kfold = 10
    # accuracies = []
    #
    # for currNodeNum in node_range:
    #     # create model
    #     model = KerasClassifier(build_fn=create_ANN_model, nodeNum=currNodeNum, epochs=150, batch_size=10, verbose=0)
    #     # evaluate using 10-fold cross validation
    #     kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    #     scores = cross_val_score(model, X_train, y_train, cv=kfold)
    #     accuracies.append(scores.mean())
    #
    # plt.plot(node_range, accuracies, 'o')
    # plt.xlabel('Node number for ANN')
    # plt.ylabel('Cross-validation accuracy')
    # plt.show()

    #
    # # Create multilayer perceptron (MLP) object
    # # Set regularization factor 'alpha' to 1e-4 (This is NOT the learning rate.)
    # mlp = MLPClassifier(hidden_layer_sizes=(5, 3), alpha=1e-4, max_iter=300)
    #
    # # Train the model using the training sets
    # mlp.fit(X_train, y_train)
    #
    # # Make predictions (probability and class) using the test set
    # prob = mlp.predict_proba(X_test)
    # y_pred = mlp.predict(X_test)
    # print('Predicted probability of the first 5 test examples:\n', prob[0:5], '\n')
    # print('Predicted class of the first 5 test examples:\n', y_pred[0:5], '\n')
    #
    # # Prediction accuracy on test data
    # print('Score =', mlp.score(X_test, y_test), '\n')

    # # Train the model using the training sets
    # clf.fit(X_train, y_train)
    #
    # # Make predictions using the test set
    # y_pred = clf.predict(X_test)
    #
    # # Prediction accuracy on training data
    # print('Training accuracy =', clf.score(X_train, y_train), '\n')
    #
    # # Evaluate the model (confusion matrix and other evaluation metrics)
    # print('Confusion matrix (test set) of SVM:')
    # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred))