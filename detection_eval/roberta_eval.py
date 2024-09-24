import os.path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import json
from utils.metrics import  get_roberta_roc_metrics, get_roberta_precision_recall_metrics
from detection_eval.eval_utils import eval_data_prepare
from dpo_builders.detectors_utils.roberta_lg import Commercial_Detectors
from utils.load_save_file_utils import setup_seed

chat=True
use_LABEL=True


def detect_eval(eval_data, eval_preds):
    fpr, tpr, roc_auc = get_roberta_roc_metrics(eval_data, eval_preds)
    p, r, pr_auc = get_roberta_precision_recall_metrics(eval_data, eval_preds)
    pred_results = {'name': f'evaluation by {args.detection_method} on {args.dataset}',
                    'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
                    'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r}
                    }
    result_file=f"{args.results_file}/{args.detection_method}"

    if args.use_mix:
        results_path = f"{result_file}/mixft_{args.base_model_name}_eval_mix_{args.mix_ratio}_beta_{args.dpo_beta}_ep_{args.dpo_epoch}.json"
    else:
        results_path = f"{result_file}/{args.base_model_name}_eval.json"
    with open(results_path, 'w') as fout:
        json.dump(pred_results, fout)
    return results_path

def detect_texts(args):

    detector=Commercial_Detectors(args)
    eval_data=eval_data_prepare(args)
    eval_preds=detector(eval_data)
    return eval_data, eval_preds

def open_results(results_path):
    with open(results_path, 'r') as fin:
        res = json.load(fin)
        return res


def get_results(results_path):
    res=open_results(results_path)

    def get_auroc_fpr_tpr(res):
        return res['metrics']['roc_auc'], res['metrics']['fpr'], res['metrics']['tpr']

    def get_pr_metrics(res):
        return res['pr_metrics']['pr_auc'], res['pr_metrics']['precision'], res['pr_metrics']['recall']

    from statistics import mean
    cols=get_auroc_fpr_tpr(res)
    print("ROC_AUC: {:.4f}, FPR: {:.4f}, TPR: {:.4f}".format(cols[0], mean(cols[1]), mean(cols[2])))

    cols=get_pr_metrics(res)
    print("PR_AUC: {:.4f}, Precision: {:.4f}, Recall: {:.4f}".format(cols[0],  mean(cols[1]),  mean(cols[2])))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--detection_method', type=str, default="roberta-base")
    parser.add_argument('--ft_detection_method', type=str, default="roberta-large")
    parser.add_argument('--dataset', type=str, default="openwebtext")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--base_model_name', type=str, default="llama2-13b")#mistral-46b,llama2-13b,llama3-70b
    parser.add_argument('--cache_dir', type=str,
                        default="../../../detectionfiles/cache")
    parser.add_argument('--dataset_file', type=str,
                        default="../../../detectionfiles/openwebtext")
    parser.add_argument('--results_file', type=str,
                        default="../../../detectionfiles/results/openwebtext")
    parser.add_argument('--dpo_dir', type=str,
                        default="../../../detectionfiles/dpo_checkpoint")
    parser.add_argument('--ft_reference_model_name', type=str, default="llama2-7b")  # mistral-7b, gpt-j-6B, llama2-7b
    parser.add_argument('--ft_scoring_model_name', type=str, default="llama2-7b")  # mistral-7b, gpt-neo-2.7B, llama2-7b

    parser.add_argument('--detect_batch_size', type=int, default=100)
    parser.add_argument('--m_length', type=int, default=200)
    parser.add_argument('--mix_ratio', type=float, default=0.5)
    parser.add_argument('--dpo_beta', type=float, default=0.1)
    parser.add_argument('--dpo_epoch', type=int, default=5)
    parser.add_argument('--discrepancy_analytic', action='store_true')
    parser.add_argument('--use_for_score', action='store_true')
    parser.add_argument('--use_ft', action='store_true')
    parser.add_argument('--use_mix', action='store_true')
    parser.add_argument('--detect_num', type=int, default=500)
    args = parser.parse_args()
    setup_seed(args.seed)
    eval_data, eval_preds=detect_texts(args)
    results_path=detect_eval(eval_data, eval_preds)
    get_results(results_path)