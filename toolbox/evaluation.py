from sklearn.metrics import f1_score, hamming_loss, recall_score, precision_score


def binarize_model_output(y, threshold=0.5):
    return y > threshold

def evaluate_multiLabel(l_true, l_pred):

    f1_micro = f1_score(l_true, l_pred, average='micro')
    f1_macro = f1_score(l_true, l_pred, average='macro')
    recall_micro = recall_score(l_true, l_pred, average='micro')
    recall_macro = recall_score(l_true, l_pred, average='macro')
    precision_micro = precision_score(l_true, l_pred, average='micro')
    precision_macro = precision_score(l_true, l_pred, average='macro')

    print(f" Macro Evaluation: f1_Score= {f1_macro} , Recall = {recall_macro} , Precision = {precision_macro}")
    print(f" Micro Evaluation: f1_Score= {f1_micro} , Recall = {recall_micro} , Precision = {precision_micro}")

