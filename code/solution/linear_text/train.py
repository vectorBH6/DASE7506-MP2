"""线性 + 文本特征训练入口；优化器和 OOF 流程复用基础线性版本。"""
from pathlib import Path

from solution.group_ranker.train_linear import main as _train


def main():
    _train(feature_set="text", method=Path(__file__).with_name("submission.py"))


if __name__ == "__main__":
    main()
