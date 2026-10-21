"""P2 弱证据：文本/题目关系与有界算术；不读取标签，不执行候选代码。"""
import ast
from fractions import Fraction
import math
import re
import time

import numpy as np

WORDS = re.compile(r"[a-z]+|\d+(?:\.\d+)?")
LETTERS = re.compile(r"[a-z]+")
NUMBERS = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
BOXED = re.compile(r"\\boxed\{([^{}]+)\}")
REGIONS = (re.compile(r"<<([\s\S]*?)>>"),
           re.compile(r"(?<!\\)(\$\$|\$)(.*?)(?<!\\)\1", re.DOTALL))
NUMERIC_LINE = re.compile(r"[0-9.+*/()= \t×÷\-]+")


def body_of(text):
    if "####" in text:
        return text.rsplit("####", 1)[0]
    matches = list(BOXED.finditer(text))
    if matches:
        match = matches[-1]
        return text[:match.start()] + text[match.end():]
    return text


def numbers_in(text):
    result = set()
    for match in NUMBERS.finditer(text):
        try:
            result.add(Fraction(match.group().replace(",", "")))
        except (ValueError, ZeroDivisionError):
            pass
    return result


def bounded(value):
    if abs(value) > 10**12 or abs(value.numerator) > 10**24 or value.denominator > 10**24:
        raise ValueError("算术中间结果超限")
    return value


def expression_value(source):
    source = source.strip()
    if not source or len(source) > 128 or not re.fullmatch(r"[0-9.+*/() \t\-]+", source):
        raise ValueError("不支持的数字表达式")
    tree = ast.parse(source, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 63:
        raise ValueError("语法树节点超限")

    def visit(node, depth=1):
        if depth > 12:
            raise ValueError("语法树深度超限")
        if isinstance(node, ast.Expression):
            value = visit(node.body, depth + 1)
        elif isinstance(node, ast.Constant):
            literal = ast.get_source_segment(source, node)
            if (not re.fullmatch(r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)", literal)
                    or sum(c.isdigit() for c in literal) > 24):
                raise ValueError("数字字面量不支持或超限")
            value = Fraction(literal)  # 从原文转有理数，避免经过 float。
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand, depth + 1) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, right = visit(node.left, depth + 1), visit(node.right, depth + 1)
            if isinstance(node.op, ast.Add):
                value = left + right
            elif isinstance(node.op, ast.Sub):
                value = left - right
            elif isinstance(node.op, ast.Mult):
                value = left * right
            else:
                value = left / right
        else:
            raise ValueError("不支持的语法节点")
        return bounded(value)

    return visit(tree)


def arithmetic_counts(text):
    """返回通过/矛盾/未知数量；重叠区间去重，每候选最多检查 64 个。"""
    regions = []
    for pattern in REGIONS:
        for match in pattern.finditer(text):
            start, end = match.span()
            if not any(start < b and end > a for a, b, _ in regions):
                regions.append((start, end, match.group(match.lastindex)))
    offset = 0
    for line in text.splitlines(keepends=True):
        end = offset + len(line)
        if ("=" in line and NUMERIC_LINE.fullmatch(line.strip())
                and not any(offset < b and end > a for a, b, _ in regions)):
            regions.append((offset, end, line.strip()))
        offset = end
    passed = failed = unknown = 0
    for _, _, equation in sorted(regions)[:64]:
        try:
            equation = equation.replace("×", "*").replace("÷", "/")
            if equation.count("=") != 1:
                raise ValueError("需要恰好一个等号")
            left, right = map(expression_value, equation.split("="))
            good = abs(left - right) <= Fraction(1, 10**6) * max(1, abs(left), abs(right))
            passed += int(good)
            failed += int(not good)
        except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, RecursionError):
            unknown += 1
    return passed, failed, unknown


def build_evidence(question, candidates, groups, members, arithmetic=False, diagnostics=None):
    tick = time.perf_counter()
    question_words = set(LETTERS.findall(question.lower()))
    question_numbers = numbers_in(question)
    magnitude = max([Fraction(1), *(abs(n) for n in question_numbers)])
    lower = question.lower()
    percent = "%" in lower or bool(re.search(r"\b(?:percent|percentage)\b", lower))
    counting = any(phrase in lower for phrase in ("how many", "number of", "total number"))
    values, counts = {}, {}
    arithmetic_seconds = 0.0
    for j in (j for indices in members for j in indices):
        text = candidates[j]["text"]
        body = body_of(text)
        words = WORDS.findall(text.lower())
        bigrams = list(zip(words, words[1:]))
        body_words = set(LETTERS.findall(body.lower()))
        union = question_words | body_words
        values[j] = [
            math.log1p(len(text)), math.log1p(len(words)),
            sum("0" <= c <= "9" for c in text) / max(1, len(text)),
            1 - len(set(bigrams)) / len(bigrams) if bigrams else 0,
            len(question_words & body_words) / len(union) if union else 0,
            len(question_numbers & numbers_in(body)) / len(question_numbers) if question_numbers else 0,
        ]
        if arithmetic:
            started = time.perf_counter()
            passed, failed, unknown = arithmetic_counts(text)
            arithmetic_seconds += time.perf_counter() - started
            counts[j] = (passed, failed, unknown)
    result = np.zeros((len(groups), 24), dtype=np.float32)
    for g, (answer, indices) in enumerate(zip(groups, members)):
        local = np.array([values[j] for j in indices])
        result[g, :12:2], result[g, 1:12:2] = local.mean(axis=0), local.max(axis=0)
        answer = Fraction(answer)
        ratio = max(Fraction(-10**6), min(Fraction(10**6), answer / magnitude))
        result[g, 12:16] = [answer in question_numbers,
                            math.copysign(math.log1p(abs(float(ratio))), ratio),
                            percent and answer.denominator != 1,
                            counting and answer.denominator != 1]
        if arithmetic:
            local = []
            for j in indices:
                passed, failed, _ = counts[j]
                checked = passed + failed
                local.append([math.log1p(checked), passed / checked if checked else 0,
                              failed / checked if checked else 0, checked > 0])
            local = np.array(local)
            result[g, 16::2], result[g, 17::2] = local.mean(axis=0), local.max(axis=0)
    if diagnostics is not None:
        increments = {"questions": 1, "valid_candidates": len(values),
                      "evidence_seconds": time.perf_counter() - tick}
        if arithmetic:
            increments.update(arithmetic_seconds=arithmetic_seconds,
                              covered_questions=int(any(p + f for p, f, _ in counts.values())),
                              covered_candidates=sum(p + f > 0 for p, f, _ in counts.values()),
                              equations_passed=sum(p for p, _, _ in counts.values()),
                              equations_failed=sum(f for _, f, _ in counts.values()),
                              equations_unknown=sum(u for _, _, u in counts.values()))
        for key, value in increments.items():
            diagnostics[key] = diagnostics.get(key, 0) + value
    return result
