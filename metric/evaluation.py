"""
This module computes evaluation metrics for MSMARCO dataset on the ranking task.
Command line:
python msmarco_eval_ranking.py <path_to_reference_file> <path_to_candidate_file>
Creation Date : 06/12/2018
Last Modified : 1/21/2019
Authors : Daniel Campos <dacamp@microsoft.com>, Rutger van Haasteren <ruvanh@microsoft.com>
"""
import sys
import json
from collections import Counter

MaxMRRRank = 10

def load_reference_from_stream(f):
    # 读取标准参考文件
    qids_to_relevant_passageids = {}
    for line in f:
        try:
            sample = json.loads(line.strip())
            qid = sample["question_id"]
            if qid not in qids_to_relevant_passageids:
                qids_to_relevant_passageids[qid] = []
            for answer_paragraph in sample["answer_paragraphs"]:
                qids_to_relevant_passageids[qid].append(answer_paragraph["paragraph_id"])
        except:
            raise IOError('\"%s\" is not valid format' % line)
    return qids_to_relevant_passageids


def load_reference(path_to_reference):
    """Load Reference reference relevant passages
    Args:path_to_reference (str): path to a file to load.
    Returns:qids_to_relevant_passageids (dict): dictionary mapping from query_id (int) to relevant passages (list of ints).
    """
    with open(path_to_reference, 'r') as f:
        qids_to_relevant_passageids = load_reference_from_stream(f)
    return qids_to_relevant_passageids


def load_candidate_from_stream(f):
    qid_to_ranked_candidate_passages = {}
    try:
        preds = json.load(f)
        for qid in preds.keys():
            tmp = [0] * 1000
            qid_to_ranked_candidate_passages[qid] = tmp
            for rank, pid in enumerate(preds[qid][:1000]):
                qid_to_ranked_candidate_passages[qid][rank] = pid
    except:
        raise IOError('Submitted file is not valid format')
    return qid_to_ranked_candidate_passages


def load_candidate(path_to_candidate):
    """Load candidate data from a file.
    Args:path_to_candidate (str): path to file to load.
    Returns:qid_to_ranked_candidate_passages (dict): dictionary mapping from query_id (int) to a list of 1000 passage ids(int) ranked by relevance and importance
    """

    with open(path_to_candidate, 'r') as f:
        qid_to_ranked_candidate_passages = load_candidate_from_stream(f)
    return qid_to_ranked_candidate_passages


def quality_checks_qids(qids_to_relevant_passageids, qids_to_ranked_candidate_passages):
    """Perform quality checks on the dictionaries
    Args:
    p_qids_to_relevant_passageids (dict): dictionary of query-passage mapping
        Dict as read in with load_reference or load_reference_from_stream
    p_qids_to_ranked_candidate_passages (dict): dictionary of query-passage candidates
    Returns:
        bool,str: Boolean whether allowed, message to be shown in case of a problem
    """
    message = ''
    allowed = True

    # Create sets of the QIDs for the submitted and reference queries
    candidate_set = set(qids_to_ranked_candidate_passages.keys())
    ref_set = set(qids_to_relevant_passageids.keys())

    # Check that we do not have multiple passages per query
    for qid in qids_to_ranked_candidate_passages:
        # Remove all zeros from the candidates
        duplicate_pids = set(
            [item for item, count in Counter(qids_to_ranked_candidate_passages[qid]).items() if count > 1])

        if len(duplicate_pids - set([0])) > 0:
            message = "Cannot rank a passage multiple times for a single query. QID={qid}, PID={pid}".format(
                qid=qid, pid=list(duplicate_pids)[0])
            allowed = False

    return allowed, message

def compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages):
    """Compute MRR metric 计算mrr分数
    把标准答案在被评价系统给出结果中的排序取倒数作为它的准确度，再对所有的问题取平均
    Args:
    p_qids_to_relevant_passageids (dict): dictionary of query-passage mapping
        Dict as read in with load_reference or load_reference_from_stream
    p_qids_to_ranked_candidate_passages (dict): dictionary of query-passage candidates
    Returns:
        dict: dictionary of metrics {'MRR': <MRR Score>}
    """
    all_scores = {}
    MRR = 0
    qids_with_relevant_passages = 0
    
    ranking = []
    ranking_qid_2_idx_dict = {}
    ranking_qid_2_idx_dict_top_10 = {}
    not_find_qid = []
    recall_q_top1 = set()
    recall_q_top50 = set()
    recall_q_all = set() # 有有效召回的query的数量
    print('待评测的query数量：', len(qids_to_ranked_candidate_passages))
    for qid in qids_to_ranked_candidate_passages: # 枚举所有query
        if qid in qids_to_relevant_passageids: # qid存在
            ranking.append(0)
            target_pid = qids_to_relevant_passageids[qid] # 标准的pid list
            candidate_pid = qids_to_ranked_candidate_passages[qid] # 模型的pid list
            for i in range(0, MaxMRRRank): # mmr@10
                if candidate_pid[i] in target_pid: # 如果在标准答案中
                    ranking_qid_2_idx_dict_top_10[qid] = i
                    MRR += 1.0 / (i + 1) # 在参考答案中的位置倒数
                    ranking.pop()
                    ranking.append(i + 1) # 只记录在模型预测的pid中排在最前面的在标准答案中的位置
                    break
            for i, pid in enumerate(candidate_pid): # 枚举模型pid list
                if pid == 0:
                    break
                if pid in target_pid: # 如果在标准答案中
                    recall_q_all.add(qid)
                    if i < 50:
                        ranking_qid_2_idx_dict[qid] = i
                        recall_q_top50.add(qid)
                    if i == 0:
                        recall_q_top1.add(qid)
                    break # 只记录一个pid
            if qid not in ranking_qid_2_idx_dict:
                not_find_qid.append(qid)
    if len(ranking) == 0:
        raise IOError("No matching QIDs found. Are you sure you are scoring the evaluation set?")

    print('在前10找到答案的query数量：', len(ranking_qid_2_idx_dict_top_10))
    print('在前50找到答案的query数量：', len(ranking_qid_2_idx_dict))
    print('用前10找到答案的做分母的mrr：', MRR / len(ranking_qid_2_idx_dict_top_10))

    MRR = MRR / len(qids_to_relevant_passageids)  # 除以query的个数
    recall_top1 = len(recall_q_top1) * 1.0 / len(qids_to_relevant_passageids) # 在前1召回正确的qid num/所有qid num
    recall_top50 = len(recall_q_top50) * 1.0 / len(qids_to_relevant_passageids)
    recall_all = len(recall_q_all) * 1.0 / len(qids_to_relevant_passageids)
    all_scores['MRR@10'] = MRR
    all_scores["recall@1"] = recall_top1
    all_scores["recall@50"] = recall_top50
    all_scores["recall@all"] = recall_all
    all_scores['QueriesRanked'] = len(qids_to_ranked_candidate_passages)
    return all_scores, ranking_qid_2_idx_dict, ranking_qid_2_idx_dict_top_10, not_find_qid


def compute_metrics_from_files(path_to_reference, path_to_candidate, perform_checks=True):
    qids_to_relevant_passageids = load_reference(path_to_reference)  # {qid1:[pid1,pid2,...], qid2:[..]...}
    qids_to_ranked_candidate_passages = load_candidate(path_to_candidate)
    if perform_checks:
        # 检查每个qid中的pids是否有重复的文章
        allowed, message = quality_checks_qids(qids_to_relevant_passageids, qids_to_ranked_candidate_passages)
        if message != '': print(message)

    return compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages)


def main():
    """Command line:
    python result_eval.py <path_to_reference_file> <path_to_candidate_file>
    """

    if len(sys.argv) == 3:
        path_to_reference = sys.argv[1] # 标准答案
        path_to_candidate = sys.argv[2] # 模型预测答案

    else:
        print('Usage: result_eval.py <reference ranking> <candidate ranking>')
        exit()

    metrics, ranking_qid_2_idx_dict, ranking_qid_2_idx_dict_top_10, not_find_qid = compute_metrics_from_files(path_to_reference, path_to_candidate)

    result = dict()
    for metric in sorted(metrics):
        result[metric] = metrics[metric]
    result_json = json.dumps(result)
    

    with open('output/eval_q_rank.json', 'w', encoding='utf-8') as f_out:
        json.dump(ranking_qid_2_idx_dict, f_out, ensure_ascii=False, indent=2)
    
    with open('output/eval_q_rank_mrr10.json', 'w', encoding='utf-8') as f_out:
        json.dump(ranking_qid_2_idx_dict_top_10, f_out, ensure_ascii=False, indent=2)
    
    with open('output/eval_not_find_in50_qid.json', 'w', encoding='utf-8') as f_out:
        json.dump(not_find_qid, f_out, ensure_ascii=False, indent=2)
    
    print(result_json)
    return result['MRR@10']


if __name__ == '__main__':
    main()
