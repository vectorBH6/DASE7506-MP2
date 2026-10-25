"""Deterministic local arithmetic evidence; no references, files, network or eval."""
from fractions import Fraction
import re
from typing import NamedTuple

import numpy as np

from common import parse_answer

FEATURE_SETS = {
    "basic": (),
    "math": (1, 2, 3, 4, 6, 7),
    "math_question": (1, 2, 3, 4, 5, 6, 7),
    "full": tuple(range(8)),
    "math_interactions": (1, 2, 3, 4, 6, 7, 8, 9),
}
GROUP_FEATURE_NAMES = (
    "mean_log_words", "mean_log_checked_equations", "mean_invalid_equation_fraction",
    "mean_last_equation_matches", "mean_trace_reaches", "mean_question_number_coverage",
    "any_last_equation_matches", "any_trace_reaches",
    "vote_share_x_invalid_fraction", "vote_share_x_last_equation_matches",
)
TOLERANCE = Fraction(1, 1000)
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_TOKEN = re.compile(r"\d+(?:\.\d+)?|[()+*/-]")
_RUN = re.compile(r"[0-9.()+*/=\s-]+")
_BOXED = re.compile(r"\\boxed\{([^{}]+)\}")


def feature_version(feature_set):
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    if feature_set == "basic":
        return "answer-group-basic-v1"
    return "answer-group-handcrafted-v3" if feature_set == "math_interactions" else "answer-group-handcrafted-v2"


def scalar_indices(feature_set):
    feature_version(feature_set)
    return list(range(16)) + [16 + i for i in FEATURE_SETS[feature_set]]


def _bounded(value):
    if value.numerator.bit_length() > 4096 or value.denominator.bit_length() > 4096:
        raise ValueError("Arithmetic size limit")
    return value


class _Parser:
    """Complete recursive descent with explicit operators and a(b) multiplication."""
    def __init__(self, text):
        if len(text) > 2048:
            raise ValueError("Expression length limit")
        compact = re.sub(r"\s+", "", text)
        self.tokens = _TOKEN.findall(text)
        if "".join(self.tokens) != compact or not self.tokens or len(self.tokens) > 256:
            raise ValueError("Unsupported or truncated expression")
        self.position = 0
        self.operations = 0
        self.depth = 0

    def peek(self):
        return self.tokens[self.position] if self.position < len(self.tokens) else None

    def expression(self):
        value = self.term()
        while self.peek() in ("+", "-"):
            op = self.peek()
            self.position += 1
            other = self.term()
            value = _bounded(value + other if op == "+" else value - other)
            self.operations += 1
        return value

    def term(self):
        value = self.factor()
        while self.peek() in ("*", "/", "("):
            op = self.peek()
            if op != "(":
                self.position += 1
            other = self.factor()
            value = _bounded(value / other if op == "/" else value * other)
            self.operations += 1
        return value

    def factor(self):
        self.depth += 1
        if self.depth > 32:
            raise ValueError("Nesting limit")
        sign = 1
        while self.peek() in ("+", "-"):
            if self.peek() == "-":
                sign = -sign
            self.position += 1
        token = self.peek()
        if token == "(":
            self.position += 1
            value = self.expression()
            if self.peek() != ")":
                raise ValueError("Unbalanced parentheses")
            self.position += 1
        elif token is not None and _NUMBER.fullmatch(token):
            if len(token) > 64:
                raise ValueError("Number length limit")
            value = Fraction(token)
            self.position += 1
        else:
            raise ValueError("Missing operand")
        self.depth -= 1
        return _bounded(sign * value)

    def parse(self):
        value = self.expression()
        if self.position != len(self.tokens):
            raise ValueError("Adjacent numbers or trailing tokens")
        return value, self.operations


def evaluate_expression(text):
    return _Parser(text).parse()


def _brace(text, start):
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text) or text[start] != "{":
        raise ValueError("Expected LaTeX braced argument")
    depth = 1
    for i in range(start + 1, len(text)):
        depth += (text[i] == "{") - (text[i] == "}")
        if depth == 0:
            return text[start + 1:i], i + 1
    raise ValueError("Unbalanced LaTeX")


def _latex(text, depth=0):
    if depth > 24:
        raise ValueError("LaTeX nesting limit")
    output, i = [], 0
    commands = re.compile(r"\\(frac|dfrac|tfrac|text|mathrm|boxed)\b")
    while i < len(text):
        match = commands.match(text, i)
        if match:
            command = match.group(1)
            first, end = _brace(text, match.end())
            if command in ("frac", "dfrac", "tfrac"):
                second, end = _brace(text, end)
                output.append("((" + _latex(first, depth + 1) + ")/(" + _latex(second, depth + 1) + "))")
            elif command in ("text", "mathrm"):
                output.append(" @ ")
            else:
                output.append("(" + _latex(first, depth + 1) + ")")
            i = end
        else:
            output.append(text[i])
            i += 1
    result = "".join(output)
    result = re.sub(r"\*\*(.*?)\*\*", r"\1", result)
    result = result.replace("×", "*").replace("÷", "/").replace("−", "-")
    result = result.replace(r"\%", "%")
    result = re.sub(r"\\(?:times|cdot)\b", "*", result)
    result = re.sub(r"\\div\b", "/", result)
    result = re.sub(r"\\(?:left|right)\b", "", result)
    result = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", result)
    result = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"((\1)/100)", result)
    return result.replace("{", "(").replace("}", ")").replace("$", "").replace("#", "")


class Equation(NamedTuple):
    lhs: Fraction
    rhs: Fraction
    valid: bool
    line: str


def checked_equations(text):
    steps = []
    for original in str(text).splitlines():
        if "=" not in original or len(original) > 8192:
            continue
        try:
            line = _latex(re.sub(r"^\s*[-*]\s+", "", original))
        except ValueError:
            continue
        # Letters are retained as boundaries; never delete unknown variables.
        for match in _RUN.finditer(line):
            run = match.group().strip()
            if "=" not in run:
                continue
            start = match.start() + len(match.group()) - len(match.group().lstrip())
            end = match.end() - len(match.group()) + len(match.group().rstrip())
            before = line[start - 1] if start else ""
            after = line[end] if end < len(line) else ""
            if (before and (before.isalnum() or before in "^_\\.")) or (after and (after.isalnum() or after in "^_\\.")):
                continue
            prefix = line[:start].rstrip()
            if prefix and prefix[-1] in "^_\\":
                continue
            # A suffix beginning with an operator may follow an omitted variable.
            if run[:1] in ("+", "-", "*", "/") and prefix and (prefix[-1].isalnum() or prefix[-1] in "^_)"):
                continue
            parts = run.split("=")
            for step_index, (left, right) in enumerate(zip(parts, parts[1:])):
                if step_index == 0 and re.search(r"(?:@|\bof)$", prefix):
                    continue
                try:
                    lhs, lhs_ops = evaluate_expression(left)
                    rhs, rhs_ops = evaluate_expression(right)
                    if lhs_ops + rhs_ops == 0:
                        continue
                except (ValueError, ZeroDivisionError, OverflowError, RecursionError):
                    continue
                steps.append(Equation(lhs, rhs, abs(lhs - rhs) < TOLERANCE, original))
    return steps


def numbers_of(text):
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", str(text))
    return [Fraction(m.group()) for m in _NUMBER.finditer(text) if len(m.group()) <= 64]


def body_without_final(text):
    if "####" in text:
        return text.rsplit("####", 1)[0]
    matches = list(_BOXED.finditer(text))
    return text[:matches[-1].start()] if matches else text


def candidate_features(question, candidates):
    """[k,6] float64; invalid answers have exactly zero rows, arbitrary k supported."""
    question_numbers = set(numbers_of(question))
    features = np.zeros((len(candidates), 6), dtype=np.float64)
    for i, candidate in enumerate(candidates):
        text = candidate["text"]
        answer = parse_answer(text)
        if answer is None:
            continue
        value = Fraction(answer)
        steps = checked_equations(text)
        matches = [step.valid and abs(step.rhs - value) < TOLERANCE for step in steps]
        body_numbers = set(numbers_of(body_without_final(text)))
        features[i] = (
            np.log1p(len(text.split())),
            np.log1p(len(steps)),
            sum(not step.valid for step in steps) / len(steps) if steps else 0,
            float(matches[-1]) if matches else 0,
            float(any(matches)),
            len(question_numbers & body_numbers) / len(question_numbers) if question_numbers else 0,
        )
    return features


def aggregate_features(question, candidates, members):
    candidate = candidate_features(question, candidates)
    result = np.zeros((len(members), 8), dtype=np.float64)
    for g, indices in enumerate(members):
        if not indices:
            raise ValueError("Empty answer group")
        values = candidate[indices]
        result[g, :6] = values.mean(axis=0)
        result[g, 6:] = values[:, 3:5].max(axis=0)
    return result
