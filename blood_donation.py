import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

def load_data():
    dataset = pd.read_csv("D:\\bbms\\blood-bank-management-system-main\\blood-train.csv")
    dataset['target'] = dataset['Made Donation in March 2007']
    return dataset.iloc[:, 0:5], dataset.iloc[:, 5]

def fit_model(model, xtrain, ytrain):
    model.fit(xtrain, ytrain)
    return model

def evaluate_model(model, xtest, ytest):
    predicted_type = model.predict(xtest)
    accuracy = accuracy_score(ytest, predicted_type)
    cm = confusion_matrix(ytest, predicted_type)
    return accuracy, cm

def plot_confusion_matrices(models, xtrain, xtest, ytrain, ytest):
    num_models = len(models)
    num_cols = 3
    num_rows = (num_models + num_cols - 1) // num_cols
    plt.figure(figsize=(15, 5*num_rows))
    for i, (name, model) in enumerate(models.items(), start=1):
        model = fit_model(model, xtrain, ytrain)
        accuracy, cm = evaluate_model(model, xtest, ytest)
        plt.subplot(num_rows, num_cols, i)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix for " + name + " Model (Accuracy: {:.2f}%)".format(accuracy * 100))
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

def plot_roc_curves(models, xtest, ytest):
    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        model = fit_model(model, xtrain, ytrain)
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(xtest)[:,1]
        else:
            y_score = model.decision_function(xtest)
        fpr, tpr, _ = roc_curve(ytest, y_score)
        auc = roc_auc_score(ytest, y_score)
        plt.plot(fpr, tpr, label=name + ' (AUC = {:.2f})'.format(auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def plot_accuracy_bar(models, xtrain, xtest, ytrain, ytest):
    accuracy_scores = []
    for name, model in models.items():
        model = fit_model(model, xtrain, ytrain)
        accuracy, _ = evaluate_model(model, xtest, ytest)
        accuracy_scores.append(accuracy)
    plt.figure(figsize=(10, 6))
    plt.bar(models.keys(), accuracy_scores, color='skyblue')
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Different Models')
    plt.ylim([0, 1])
    plt.show()

def plot_training_accuracy_bar(models, xtrain, ytrain):
    train_accuracy_scores = []
    for name, model in models.items():
        model = fit_model(model, xtrain, ytrain)
        train_accuracy = model.score(xtrain, ytrain)
        train_accuracy_scores.append(train_accuracy)
    plt.figure(figsize=(10, 6))
    plt.bar(models.keys(), train_accuracy_scores, color='orange')
    plt.xlabel('Models')
    plt.ylabel('Training Accuracy')
    plt.title('Training Accuracy of Different Models')
    plt.ylim([0, 1])
    plt.show()

# Define the data
x, y = load_data()
xtrain, xtest, ytrain, ytest = train_test_split(x, y, stratify=y)

# Dictionary to store models
models = {
    "KNN": KNeighborsClassifier(n_neighbors=6, metric='minkowski', p=2),
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "XGBoost": xgb.XGBClassifier(),
    "SVM (RBF)": SVC(kernel='rbf', probability=True),
    "ANN (RBF)": MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', solver='adam', max_iter=2000)
}

# Plot confusion matrices
plot_confusion_matrices(models, xtrain, xtest, ytrain, ytest)

# Plot ROC curves
plot_roc_curves(models, xtest, ytest)

# Plot accuracy bar graph
plot_accuracy_bar(models, xtrain, xtest, ytrain, ytest)

# Plot training accuracy bar graph
plot_training_accuracy_bar(models, xtrain, ytrain)
