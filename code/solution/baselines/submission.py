"""
原基线：答案最多票，获胜组内用 embedding 选择代表候选

方法说明:
1. 多数投票: 选择最终答案出现最多的候选
2. 嵌入一致性: 当有多于一个候选有相同答案时，使用嵌入向量选择最"中心"的候选
3. 无有效答案时: 使用嵌入一致性选择

表现:
- 验证集: 86.2%
- 测试集: 69.2%
"""
import numpy as np

from common import parse_answer


def predict(bank, arrays, checkpoint):
    """基于嵌入和多数投票的选答方法"""
    embeddings = arrays['embeddings']  # shape: [N, 4, 1024]
    n = len(bank)
    
    predictions = []
    
    for i in range(n):
        candidates = bank[i]['candidates']
        
        # Step 1: 解析每个候选的最终答案
        answers = []
        for j in range(4):
            ans = parse_answer(candidates[j]['text'])
            answers.append(ans)
        
        # Step 2: 多数投票
        # 统计每个答案出现的次数（忽略None）
        answer_counts = {}
        for j, ans in enumerate(answers):
            if ans is not None:
                answer_counts[ans] = answer_counts.get(ans, []) + [j]
        
        if answer_counts:
            # 找到出现次数最多的答案
            best_answer = max(answer_counts.keys(), 
                             key=lambda x: len(answer_counts[x]))
            candidates_with_best = answer_counts[best_answer]
            
            # 如果只有一个候选有这个答案，直接选择
            if len(candidates_with_best) == 1:
                predictions.append(candidates_with_best[0])
                continue
            
            # 如果有多个候选有相同答案，使用嵌入来区分
            if len(candidates_with_best) > 1:
                selected = _embedding_conflict_resolution(
                    embeddings[i], candidates_with_best
                )
                predictions.append(selected)
                continue
        else:
            # Step 3: 没有有效答案，使用嵌入一致性选择
            predictions.append(_embedding_consensus_selection(embeddings[i]))
    
    return predictions


def _embedding_conflict_resolution(emb_q, candidate_indices):
    """
    当多个候选有相同答案时，使用嵌入向量选择最"中心"的候选
    
    假设: 最可靠的候选在语义上与其他候选更加一致
    """
    # 提取这些候选的嵌入
    candidates_emb = emb_q[candidate_indices]
    
    # 计算质心
    centroid = candidates_emb.mean(axis=0)
    
    # 选择离质心最近的
    distances = np.linalg.norm(candidates_emb - centroid, axis=1)
    best_idx = np.argmin(distances)
    
    return candidate_indices[best_idx]


def _embedding_consensus_selection(emb_q):
    """
    当没有有效答案时，使用嵌入一致性选择
    
    策略: 选择嵌入向量在空间中位于"中心位置"的候选
    """
    # L2归一化
    norms = np.linalg.norm(emb_q, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    emb_normalized = emb_q / norms
    
    # 余弦相似度矩阵
    cos_sim = np.dot(emb_normalized, emb_normalized.T)
    
    # 每个候选与其他候选的平均相似度
    avg_similarity = cos_sim.sum(axis=1) - 1.0
    
    return int(np.argmax(avg_similarity))


if __name__ == "__main__":
    from common import load_inputs
    
    print("Testing embedding-based selection method...")
    bank, arrays = load_inputs('validation')
    choices = predict(bank, arrays, None)
    print(f"Generated {len(choices)} predictions")
