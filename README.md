# Machine Learning in the Diagnosis of Erythemato-squamous dermatological diseases

## Introduction
The erythemato-squamous dermatological (ESD) diseases consist of many distinct disease entities characterized by redness and scaling. They present a diagnostic challenge because of overlapping clinical and histopathological features. It is important to differentiate them as they have different management options. For instance, more severe cases of psoriasis require treatment by methotrexate and phototherapy, which are not applicable to other diseases. Developing a machine learning algorithm that helps in diagnosing such diseases ensures that suitable treatments are commenced in a timely manner. 

The UCI Dermatology Data Set consists of 34 clinical and histopathological attributes of 366 patients with one of six ESDs and is a suitable dataset for developing such an algorithm.

This implemented an artificial neural network (ANN) and a support vector machine (SVM) using the sklearn and Keras libraries in Python to diagnose dermatology diseases using the above Data Set. 
The current dataset is imbalanace. It is dominated by the Psoriasis class (111 data); the other classes have 72 data points or less. To address this issue in ANN, oversampling of the under-represented class was done. For SVM, the regularization parameter C of each class was multiplied by class weights, the latter of which were inversely proportional to class frequencies.  

## First-run Result

For the current application, SVM performs better than ANN, with slightly higher overall accuracy, as well as precision, recall and F1-score for class 1 and 3. 

### ANN result
#### Best parameter in cross-validation
![image](https://drive.google.com/uc?export=view&id=1a_hHMtZQS5BxgdPC4tnY7Qa99CKorDlQ)

#### Classification report
![image](https://drive.google.com/uc?export=view&id=1drchi5hmbKWfRb-u69uUP923oIPSgjYa)

#### Confusion matrix
![image](https://drive.google.com/uc?export=view&id=1tnLibjWasWqW3-ZeVv5uLUjOvZC7ODe5)

### SVM result
#### Best parameter in cross-validation
![image](https://drive.google.com/uc?export=view&id=1SIbXd7iosy5eqNmmZNAnJnpqTx2uWYv7)

#### Classification report
![image](https://drive.google.com/uc?export=view&id=184dvRyz3TjufmgsilBdOIK-QF479Hc5d)

#### Confusion matrix
![image](https://drive.google.com/uc?export=view&id=1A1n7lr1_-fdFG04sW4wG8zo91yJOv_Nw)



## Re-run Result

However, on re-run of the analysis, ANN gave an accuracy of 92% only, with a larger proportion of class 3 misclassified as class 1. The best parameters from CV has also changed, with the best epoch number changed to 900 instead of 100. Notably, the CV accuracies of 900 epochs and 100 epochs with batch size 10 are very close (0.987 and 0.985 respectively). This slight difference, causing a change in epoch number chosen, may lead to the large difference in eventual model performance. In contrast, the rerun of SVM gave the same results as the first run. SVM may therefore be a more reliable classifier here.

### ANN result
#### Best parameter in cross-validation
![image](https://drive.google.com/uc?export=view&id=1mPAE93O9-IAWjhCrTQOGI_J02Ro6HhWE)

#### Classification report and confusion matrix
![image](https://drive.google.com/uc?export=view&id=1drchi5hmbKWfRb-u69uUP923oIPSgjYa)

### SVM result
#### Best parameter in cross-validation
![image](https://drive.google.com/uc?export=view&id=11J9Wkvk_dE4dPcpaJqrKUzlrPd_785PD)

#### Classification report and confusion matrix
![image](https://drive.google.com/uc?export=view&id=1MW0QSl4NaAq3b9NMQQgh4LCxEUWUth8m)
