"""Reuse the established math features; optionally fill the 16 remaining slots."""
from fractions import Fraction
import re
import numpy as np

from solution.group_ranker.features import EMBED_DIM, FEATURE_DIM, answer_groups, best_group, standardize
from solution.handcrafted_ranker.features import build_features as math_features
from solution.handcrafted_ranker.arithmetic import (
    TOLERANCE, body_without_final, checked_equations, numbers_of,
)
from common import parse_answer

SCHEMAS = ("math", "expanded")
EXTRA_NAMES = (
    "log_words", "log_distinct_numbers", "question_token_recall", "focus_final_jaccard",
    "equation_check_coverage", "log_invalid_equations", "valid_trace_match_fraction",
    "question_number_coverage",
)
STOP = frozenset("a an the and or to of in on for with by is are was were be been it its this that these those he she they his her their how what who which many much does do did has have had will would can could please find calculate answer".split())


def tokens(text):
    words = re.findall(r"[a-z]+", str(text).lower())
    return {word[:-1] if len(word)>4 and word.endswith("s") else word
            for word in words if word not in STOP}


def candidate_extras(question, candidates):
    result = np.zeros((len(candidates), 8), dtype=np.float64)
    question_tokens = tokens(question)
    focus = re.split(r"(?<=[.!?])\s+(?=[A-Z])", question.strip())[-1]
    focus_tokens = tokens(focus)
    question_numbers = set(numbers_of(question))
    for i,candidate in enumerate(candidates):
        text=candidate["text"]
        answer=parse_answer(text)
        if answer is None:
            continue
        body=body_without_final(text)
        body_numbers=set(numbers_of(body))
        body_tokens=tokens(body)
        lines=[line for line in body.splitlines() if line.strip()]
        final_tokens=tokens(" ".join(lines[-3:]))
        union=focus_tokens | final_tokens
        steps=checked_equations(text)
        valid=[step for step in steps if step.valid]
        value=Fraction(answer)
        matches=sum(abs(step.rhs-value)<TOLERANCE for step in valid)
        result[i]=(
            np.log1p(len(text.split())),
            np.log1p(len(body_numbers)),
            len(question_tokens & body_tokens)/len(question_tokens) if question_tokens else 0,
            len(focus_tokens & final_tokens)/len(union) if union else 0,
            len(steps)/max(text.count("="),1),
            np.log1p(sum(not step.valid for step in steps)),
            matches/len(valid) if valid else 0,
            len(question_numbers & body_numbers)/len(question_numbers) if question_numbers else 0,
        )
    return result


def build_features(candidates, embeddings, groups, *, question, schema="math"):
    if schema not in SCHEMAS:
        raise ValueError("Unknown residual feature schema")
    features=math_features(candidates,embeddings,groups,question=question,feature_set="math")
    if schema=="expanded" and groups:
        e=np.asarray(embeddings,dtype=np.float64)
        order=sorted(range(len(candidates)),key=lambda j:(candidates[j]["text"],e[j].tobytes()))
        extra=candidate_extras(question,candidates)
        for g,indices in enumerate(groups.values()):
            members=[j for j in order if j in indices]
            values=extra[members]
            features[g,2072:2080]=values.mean(axis=0)
            features[g,2080:2088]=values.max(axis=0)
    return features
