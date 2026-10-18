"""基线消融：纯答案投票，不使用 embedding。"""
from common import parse_answer

def predict(bank, arrays, checkpoint):
    n = len(bank)
    predictions = []
    for i in range(n):
        candidates = bank[i]['candidates']
        answers = [parse_answer(c['text']) for c in candidates]
        answer_indices = {}
        for j, ans in enumerate(answers):
            if ans is not None:
                if ans not in answer_indices:
                    answer_indices[ans] = []
                answer_indices[ans].append(j)
        if answer_indices:
            best_answer = max(answer_indices.keys(),
                             key=lambda x: len(answer_indices[x]))
            predictions.append(answer_indices[best_answer][0])
        else:
            predictions.append(0)
    return predictions
