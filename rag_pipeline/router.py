import re
from dataclasses import dataclass

from .table_utils import is_numeric_finance_question


@dataclass
class RouteDecision:
    intent: str
    table_weight: float
    text_weight: float
    reason: str


COMPARISON_PATTERN = re.compile(
    r"同比|环比|对比|比较|增减|增长|下降|变化|差额|增速|变动"
)
NARRATIVE_PATTERN = re.compile(
    r"回购|计划|预案|风险|如何|为什么|说明|情况|政策|治理|募集资金|"
    r"发展战略|经营情况|重要事项|承诺|用途|是否|有哪些|怎么样"
)
STRONG_NARRATIVE_PATTERN = re.compile(
    r"回购|计划|预案|经营情况|重要事项|治理|发展战略|募集资金|承诺"
)


class QueryRouter:
    """Route finance questions to table-heavy or text-heavy retrieval."""

    def route(self, query) -> RouteDecision:
        query = str(query or "").strip()
        is_numeric = is_numeric_finance_question(query)
        is_comparison = bool(COMPARISON_PATTERN.search(query))
        is_narrative = bool(NARRATIVE_PATTERN.search(query))

        if is_comparison:
            return RouteDecision(
                intent="comparison",
                table_weight=0.55,
                text_weight=0.45,
                reason="检测到对比/变化类问题，启用表格+文本均衡检索",
            )

        if STRONG_NARRATIVE_PATTERN.search(query):
            return RouteDecision(
                intent="narrative_qa",
                table_weight=0.20,
                text_weight=0.80,
                reason="检测到回购/计划/经营叙述类问题，优先文本索引",
            )

        if is_narrative and not is_numeric:
            return RouteDecision(
                intent="narrative_qa",
                table_weight=0.20,
                text_weight=0.80,
                reason="检测到叙述/政策/计划类问题，优先文本索引",
            )

        if is_numeric:
            return RouteDecision(
                intent="numeric_lookup",
                table_weight=0.80,
                text_weight=0.20,
                reason="检测到数值类问题，优先表格索引",
            )

        return RouteDecision(
            intent="general",
            table_weight=0.50,
            text_weight=0.50,
            reason="通用问题，表格与文本均衡检索",
        )
