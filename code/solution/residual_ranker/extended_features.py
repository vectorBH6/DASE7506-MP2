"""Additional offline text/trace evidence, appended to the frozen expanded schema."""
from fractions import Fraction
import math
import re
import numpy as np
from common import parse_answer
from solution.residual_ranker.features import build_features as previous_features, tokens
from solution.group_ranker.features import answer_groups, best_group, EMBED_DIM
from solution.handcrafted_ranker.arithmetic import (
    TOLERANCE, body_without_final, checked_equations, numbers_of,
)

BASE_DIM = 2088
SCHEMAS = ("lexical", "trace", "combined", "contextual")
CANDIDATE_NAMES = (
    "focus_body_recall", "focus_tail_recall", "focus_lastline_recall",
    "question_lastline_jaccard", "focus_unit_lastline_recall",
    "log_nonempty_lines", "log_raw_equals", "log_answer_mentions",
    "answer_nonnegative_integer", "signed_log_answer", "answer_question_log_ratio",
    "novel_body_number_fraction",
    "log_valid_steps", "all_checked_steps_valid", "last_checked_rhs_matches",
    "last_valid_rhs_matches", "all_steps_match_fraction", "invalid_step_reaches",
    "adjacent_valid_link_fraction", "valid_result_reused_fraction",
    "dangling_valid_result_fraction", "valid_lhs_match_fraction",
    "last_valid_match_position", "last_valid_result_disagreement",
)
INTERACTION_INDICES = (0, 2, 11, 12, 15, 17, 18, 20)
UNITS = frozenset("dollar cent money mile kilometer metre meter foot feet inch yard gallon liter litre minute hour day week month year apple orange book ticket student person people worker egg pound ounce kilogram gram kg percent percentage".split())


def recall(target, observed):
    return len(target & observed) / len(target) if target else 0.0


def signed_log(value):
    bounded = min(abs(value), Fraction(10**100))
    return math.copysign(math.log1p(float(bounded)), 1.0 if value > 0 else -1.0) if value else 0.0


def candidate_extensions(question, candidates):
    """24 float64 signals per candidate; an unparseable candidate has an all-zero row."""
    result = np.zeros((len(candidates), len(CANDIDATE_NAMES)), dtype=np.float64)
    question_tokens = tokens(question)
    focus = re.split(r"(?<=[.!?])\s+(?=[A-Z])", question.strip())[-1]
    focus_tokens = tokens(focus)
    focus_units = focus_tokens & UNITS
    question_numbers = set(numbers_of(question))
    max_question = max(question_numbers, default=Fraction(0))
    for i, candidate in enumerate(candidates):
        text = candidate["text"]
        answer = parse_answer(text)
        if answer is None:
            continue
        value = Fraction(answer)
        body = body_without_final(text)
        lines = [line.strip() for line in body.splitlines() if line.strip()]
        last_tokens = tokens(lines[-1] if lines else "")
        tail_tokens = tokens(" ".join(lines[-3:]))
        body_values = numbers_of(body)
        body_numbers = set(body_values)
        steps = checked_equations(text)
        valid = [step for step in steps if step.valid]
        n, nv = len(steps), len(valid)
        matches = [abs(step.rhs-value) < TOLERANCE for step in steps]
        valid_matches = [abs(step.rhs-value) < TOLERANCE for step in valid]
        # Literal-number reuse is evidence of continuity, not symbolic correctness.
        operands = [set(numbers_of(step.line.split("=", 1)[0])) for step in valid]
        linked = sum(valid[j-1].rhs in operands[j] for j in range(1, nv))
        reused = 0
        dangling = 0
        later_operands = set()
        for j in range(nv-1, -1, -1):
            reused += valid[j].rhs in later_operands
            dangling += not valid_matches[j] and valid[j].rhs not in later_operands
            later_operands.update(operands[j])
        union = question_tokens | last_tokens
        last_value = valid[-1].rhs if valid else value
        disagreement = float((value-last_value) / max(Fraction(1), abs(value), abs(last_value)))
        match_positions = [j+1 for j, flag in enumerate(valid_matches) if flag]
        result[i] = (
            recall(focus_tokens, tokens(body)),
            recall(focus_tokens, tail_tokens),
            recall(focus_tokens, last_tokens),
            len(question_tokens & last_tokens)/len(union) if union else 0,
            recall(focus_units, last_tokens),
            math.log1p(len(lines)), math.log1p(text.count("=")),
            math.log1p(sum(number == value for number in body_values)),
            value >= 0 and value.denominator == 1,
            signed_log(value),
            abs(signed_log(value))-math.log1p(float(min(max_question, Fraction(10**100)))),
            len(body_numbers-question_numbers)/len(body_numbers) if body_numbers else 0,
            math.log1p(nv), bool(n) and nv == n,
            bool(matches) and matches[-1],
            bool(valid_matches) and valid_matches[-1],
            sum(matches)/n if n else 0,
            any(match and not step.valid for match, step in zip(matches, steps)),
            linked/max(nv-1, 1),
            reused/nv if nv else 0,
            dangling/nv if nv else 0,
            sum(abs(step.lhs-value) < TOLERANCE for step in valid)/nv if nv else 0,
            max(match_positions)/nv if match_positions else 0,
            disagreement if valid else 0,
        )
    return result


def feature_names(schema):
    if schema not in SCHEMAS:
        raise ValueError("Unknown extended schema")
    names = list(CANDIDATE_NAMES[:12] if schema=="lexical" else
                 CANDIDATE_NAMES[12:] if schema=="trace" else CANDIDATE_NAMES)
    extra = [f"mean_{name}" for name in names] + [f"max_{name}" for name in names]
    if schema == "contextual":
        extra += [f"group_z_{name}" for name in CANDIDATE_NAMES]
        extra += [f"vote_x_{CANDIDATE_NAMES[j]}" for j in INTERACTION_INDICES]
    return extra


def feature_dim(schema):
    return BASE_DIM + len(feature_names(schema))


def columns(schema):
    """Select a schema from the full contextual matrix without recomputing features."""
    if schema == "lexical":
        added = list(range(BASE_DIM, BASE_DIM+12))+list(range(BASE_DIM+24, BASE_DIM+36))
    elif schema == "trace":
        added = list(range(BASE_DIM+12, BASE_DIM+24))+list(range(BASE_DIM+36, BASE_DIM+48))
    else:
        added = list(range(BASE_DIM, feature_dim(schema)))
    return np.array(list(range(BASE_DIM))+added, dtype=np.int32)


def build_features(candidates, embeddings, groups, *, question, schema="contextual"):
    feature_dim(schema)
    old = previous_features(candidates, embeddings, groups, question=question, schema="expanded")
    extra = candidate_extensions(question, candidates)
    e = np.asarray(embeddings, dtype=np.float64)
    order = sorted(range(len(candidates)), key=lambda j:(candidates[j]["text"], e[j].tobytes()))
    means = np.zeros((len(groups), 24), np.float64)
    maxima = np.zeros_like(means)
    counts = np.zeros((len(groups), 1), np.float64)
    for g, indices in enumerate(groups.values()):
        members = [j for j in order if j in indices]
        means[g] = extra[members].mean(axis=0)
        maxima[g] = extra[members].max(axis=0)
        counts[g, 0] = len(members)/max(len(candidates), 1)
    z = (means-means.mean(axis=0))/(means.std(axis=0)+1e-6) if len(groups) else means.copy()
    interactions = means[:, INTERACTION_INDICES]*counts
    full = np.concatenate((old, means, maxima, z, interactions), axis=1).astype(np.float32)
    return full[:, columns(schema)]
