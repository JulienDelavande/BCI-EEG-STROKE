"""
Classif module providing functions for evaluating a DL/ML classification 
model. It provides functions for computing diverse performance metrics 
and plotting them.

List of classes & functions:
"""

import numpy as np
import sklearn.metrics as skm
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve


def compute_prediction_scores(y_test, y_pred, verbose=False):
    """
    Function to compute the prediction scores of a model.

    :param y_test: The true labels.
    :param y_pred: The predicted labels.

    :return: The prediction scores as a dictionary.
    """
    # Compute the confusion matrix
    tn, fp, fn, tp = skm.confusion_matrix(y_test, y_pred).ravel()

    # Compute the prediction scores
    acc = skm.accuracy_score(y_test, y_pred)
    rec = skm.recall_score(y_test, y_pred)
    pre = skm.precision_score(y_test, y_pred)
    f1 = skm.f1_score(y_test, y_pred)

    if verbose:
        print(f"TN: {tn},\nFP: {fp},\nFN: {fn},\nTP: {tp}")
        print(f"Accuracy : {acc}")
        print(f"Recall   : {rec}")
        print(f"Precision: {pre}")
        print(f"F1: {f1}")

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "acc": acc,
        "rec": rec,
        "pre": pre,
        "f1": f1}


def ROC_curve(
        y_test,
        y_pred_proba,
        verbose=False,
        display=False,
        savefig=False
):
    """
    Function to compute the ROC AUC of a model.

    :param y_test      : The true labels.
    :param y_pred_proba: The predicted probabilities.

    :return: The ROC AUC .
    """
    fpr, tpr, thresholds = skm.roc_curve(y_test, y_pred_proba)
    auc = skm.auc(fpr, tpr)

    if verbose:
        print(f"ROC AUC: {auc}")

    if display:
        # Plot the ROC curve
        fig = plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, label=f"ROC curve (area = {auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curve")
        plt.legend(loc="lower right")
        plt.show()

        # Plot the ROC curve with thresholds
        fig = plt.figure(figsize=(8, 8))
        plt.step(fpr, tpr, "bo", where="post", alpha=0.2)
        plt.fill_between(fpr, tpr, step="post", alpha=0.2, color="b")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curve")

        for tp, fp, th in zip(tpr, fpr, thresholds):
            plt.annotate(
                np.round(th, 2),
                xy=(fp, tp),
                xytext=(fp + 0.15, tp - 0.15),
                arrowprops=dict(facecolor="black",
                                shrink=0.05,
                                arrowstyle="->",
                                connectionstyle="arc3,rad=0.3")
            )

        if savefig:
            plt.savefig("ROC_curve.png")
        plt.show()

    return auc


def PR_curve(
        y_test,
        y_pred_proba,
        verbose=False,
        display=False,
        savefig=False
):
    """
    Function to compute the PR AUC of a model.

    :param y_test      : The true labels.
    :param y_pred_proba: The predicted probabilities.

    :return: The PR AUC .
    """
    pre, rec, thresholds = skm.precision_recall_curve(
        y_test, y_pred_proba, pos_label=1)
    auc = skm.auc(rec, pre)
    ap = skm.average_precision_score(y_test, y_pred_proba)

    if verbose:
        print(f"PR AUC: {auc}")
        print(f"AP    : {ap}")

    if display:
        # Plot the PR curve
        fig = plt.figure(figsize=(8, 8))
        plt.plot(rec, pre,
                 label=f"PR curve (area = {auc:.2f}, AP = {ap:.2f})")
        plt.plot([0, 1], [1, 0], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("PR curve")
        plt.legend(loc="lower right")
        plt.show()

        # Plot the PR curve with thresholds
        fig = plt.figure(figsize=(8, 8))
        plt.step(rec, pre, "bo", where="post", alpha=0.2)
        plt.fill_between(rec, pre, step="post", alpha=0.2, color="b")
        plt.plot([0, 1], [1, 0], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("PR curve")

        for tp, fp, th in zip(pre, rec, thresholds):
            plt.annotate(
                np.round(th, 2),
                xy=(fp, tp),
                xytext=(fp + 0.15, tp - 0.15),
                arrowprops=dict(facecolor="black",
                                shrink=0.05,
                                arrowstyle="->",
                                connectionstyle="arc3,rad=0.3")
            )

        if savefig:
            plt.savefig("PR_curve.png")
        plt.show()

    return auc
