import tqdm
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from bert_score import BERTScorer

import numpy as np
COLORS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442",
            "#56B4E9", "#E69F00", "#000000", "#0072B2", "#009E73",
            "#D55E00", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"]

def get_bert_metrics(real_pred, sample_pred):
    bert_scorer = BERTScorer(lang='en')
    bert_scores_pair=bert_scorer.score([sample_pred], [real_pred])
    return bert_scores_pair[0].mean().item(),\
           bert_scores_pair[1].mean().item(),\
           bert_scores_pair[2].mean().item()

def get_rouge_scorer(real_preds, sample_preds):
    from rouge_score import rouge_scorer
    rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = rouge_scorer.score(real_preds, sample_preds)
    return rouge_scores['rouge1'].fmeasure,\
            rouge_scores['rouge2'].fmeasure,\
            rouge_scores['rougeL'].fmeasure

def rouge_bert_scorer(real_preds, sample_preds):
    rouges_1=list()
    rouges_2=list()
    rouges_L=list()
    berts_p=list()
    berts_r=list()
    berts_f1=list()
    for real_pred, sample_pred in tqdm.tqdm(zip(real_preds, sample_preds)):
        rouge_1,rouge_2,rouge_L=get_rouge_scorer(real_pred, sample_pred)
        bert_p,bert_r,bert_f1=get_bert_metrics(real_pred, sample_pred)
        rouges_1.append(rouge_1)
        rouges_2.append(rouge_2)
        rouges_L.append(rouge_L)
        berts_p.append(bert_p)
        berts_r.append(bert_r)
        berts_f1.append(bert_f1)
    return rouges_1, rouges_2, rouges_L, berts_p, berts_r, berts_f1



def get_roberta_roc_metrics(data, preds):
    labels = [1] * len(data['real']) + [0] * len(data['sampled'])
    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)

def get_roberta_precision_recall_metrics(data, preds):
    labels = [1] * len(data['real']) + [0] * len(data['sampled'])
    precision, recall, _ = precision_recall_curve(labels,preds)
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)


def get_roc_metrics(real_preds, sample_preds):
    fpr, tpr, _ = roc_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)


def get_precision_recall_metrics(real_preds, sample_preds):
    precision, recall, _ = precision_recall_curve([0] * len(real_preds) + [1] * len(sample_preds),
                                                  real_preds + sample_preds)
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)

def get_tpr_fpr_threshold(real_preds, sample_preds, fpr_percentage):
    fpr, tpr, threshold = roc_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    closest_index = np.argmin(np.abs(fpr - fpr_percentage))
    tpr_at_percentage=tpr[closest_index]
    threshold_at_percentage = threshold[closest_index]
    return tpr_at_percentage, threshold_at_percentage

