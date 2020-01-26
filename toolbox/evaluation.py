from sklearn.metrics import f1_score, hamming_loss, recall_score, precision_score
import numpy as np


def binarize_model_output(y, threshold=0.5):
    return y > threshold

def evaluate_multiLabel(l_true, l_pred, printOutput):

    f1_micro = f1_score(l_true, l_pred, average='micro')
    f1_macro = f1_score(l_true, l_pred, average='macro')
    recall_micro = recall_score(l_true, l_pred, average='micro')
    recall_macro = recall_score(l_true, l_pred, average='macro')
    precision_micro = precision_score(l_true, l_pred, average='micro')
    precision_macro = precision_score(l_true, l_pred, average='macro')
    if printOutput:
        print(f" Macro Evaluation: f1_Score= {f1_macro} , Recall = {recall_macro} , Precision = {precision_macro}")
        print(f" Micro Evaluation: f1_Score= {f1_micro} , Recall = {recall_micro} , Precision = {precision_micro}")

    return f1_micro


def output_evaluation(model, sample_size, max_question_words, n_top_labels, l_true, predictions, normalize_embeddings, learning_rate, vocab_size, n_epochs, thres=None):
    """

    :param model:
    :param sample_size:
    :param max_question_words:
    :param n_top_labels:
    :param l_true:
    :param predictions:
    :param normalize_embeddings:
    :param learning_rate:
    :param vocab_size:
    :param n_epochs:
    :return:
    """

    print(f"Model Evaluation\n")
    print(f"normalize_embeddings = {normalize_embeddings}, learning_rate = {learning_rate}, vocab_size = {vocab_size}, epochs={n_epochs}")
    print(f"Parameter Settings:\n Sample size = {sample_size}, Max. number of words per question = {max_question_words}, Number of Top Labels used = {n_top_labels}\n")
    print(model.summary())
    if thres is None:
        max_f1, max_thres = optimize_thres(predictions, l_true)
        print(f"\nMetrics with optimized threshold of {max_thres}")
        l_pred = binarize_model_output(predictions, max_thres)
        evaluate_multiLabel(l_true, l_pred, True)

    else:
        print(f"\nMetrics without optimized thres of {thres}")
        l_pred=binarize_model_output(predictions, thres)
        evaluate_multiLabel(l_true, l_pred, True)


def optimize_thres(predictions, true_binary):
    max_f1 = -1
    max_thres = -1
    for i in np.arange(0.0,1.0,0.01):
        print(f"threshold is {i}")
        pred_bin = binarize_model_output(predictions, i)
        f1 = evaluate_multiLabel(true_binary, pred_bin, False)
        if f1 is not None and f1>max_f1:
            max_f1=f1
            max_thres=i
    # print(f"Optimal parameters: f1_micro_score= {max_f1}, threshold = {max_thres}")
    return max_f1, max_thres
