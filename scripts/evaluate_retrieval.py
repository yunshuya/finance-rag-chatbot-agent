import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_pipeline.eval import (
    evaluate_retrieval_case,
    format_comparison_table,
    load_eval_dataset,
    summarize_results,
)
from rag_pipeline.dual_index import load_dual_chroma_index
from rag_pipeline.reranker import BgeReranker
from rag_pipeline.retriever import FinanceHybridRetriever, RoutedDualIndexRetriever
from rag_pipeline.vectorstore import BgeM3Embeddings, load_chroma_vectorstore


DEFAULT_EVAL_PATH = PROJECT_ROOT / "data" / "eval" / "moutai_2024_retrieval_eval.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "docs" / "retrieval_eval_report.md"
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "data" / "eval" / "results"


def build_embeddings(name, model_cache_dir):
    if name == "bge-m3":
        return BgeM3Embeddings(cache_dir=model_cache_dir)
    raise ValueError(f"Retrieval eval requires real embeddings, got: {name}")


def build_retriever(
    strategy,
    vectorstore=None,
    dual_index=None,
    reranker=None,
    final_k=10,
):
    if strategy == "vector_baseline":
        return _VectorBaselineRetriever(vectorstore, k=final_k)
    if strategy == "finance_hybrid_no_rerank":
        return FinanceHybridRetriever(
            vectorstore=vectorstore,
            llm=None,
            reranker=None,
            vector_k=24,
            final_k=final_k,
            min_confidence_score=0.0,
        )
    if strategy == "finance_hybrid_full":
        return FinanceHybridRetriever(
            vectorstore=vectorstore,
            llm=None,
            reranker=reranker,
            vector_k=24,
            final_k=final_k,
            min_confidence_score=0.0,
        )
    if strategy == "routed_dual_index":
        return RoutedDualIndexRetriever(
            dual_index=dual_index,
            llm=None,
            reranker=reranker,
            vector_k=24,
            final_k=final_k,
            min_confidence_score=0.0,
        )
    raise ValueError(f"Unknown strategy: {strategy}")


class _VectorBaselineRetriever:
    """Plain vector retrieval without finance-specific optimizations."""

    def __init__(self, vectorstore, k=10):
        self.vectorstore = vectorstore
        self.k = k

    def invoke(self, query):
        return self.vectorstore.similarity_search(query, k=self.k)


def run_evaluation(args):
    payload, items = load_eval_dataset(args.eval_path)
    vectorstore_dir = PROJECT_ROOT / payload["vectorstore_dir"]
    embeddings = build_embeddings(args.embedding, args.model_cache_dir)
    vectorstore = load_chroma_vectorstore(
        vectorstore_dir,
        embeddings=embeddings,
        collection_name=payload["collection_name"],
    )

    dual_index = load_dual_chroma_index(
        vectorstore_dir,
        embeddings=embeddings,
        collection_name=payload["collection_name"],
    )

    reranker = None
    if (
        any(
            strategy in args.strategies
            for strategy in ("finance_hybrid_full", "routed_dual_index")
        )
        and not args.skip_reranker
    ):
        reranker = BgeReranker(cache_dir=args.model_cache_dir)

    all_results = {}
    per_case = {}
    for strategy in args.strategies:
        retriever = build_retriever(
            strategy=strategy,
            vectorstore=vectorstore,
            dual_index=dual_index,
            reranker=reranker,
            final_k=args.top_k,
        )
        case_results = []
        for item in items:
            documents = retriever.invoke(item.question)
            case_results.append(
                evaluate_retrieval_case(item, documents, k=args.top_k)
            )
        all_results[strategy] = summarize_results(case_results)
        per_case[strategy] = case_results

    report_lines = [
        "# 茅台2024年报检索评测报告",
        "",
        f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 评测集: `{args.eval_path.relative_to(PROJECT_ROOT)}`",
        f"- 向量库: `{payload['vectorstore_dir']}`",
        f"- Top-K: {args.top_k}",
        f"- Fact Store: {dual_index.fact_store.count() if dual_index.fact_store else 0} 条",
        "",
        "## 策略对比",
        "",
        format_comparison_table(all_results, args.top_k),
        "",
        "## 逐题结果",
        "",
    ]

    for item in items:
        report_lines.append(f"### {item.id} {item.question}")
        report_lines.append(
            f"- 期望页码: {item.expected_pages} | 类别: {item.category}"
        )
        if item.expected_answer_hint:
            report_lines.append(f"- 参考答案提示: {item.expected_answer_hint}")
        report_lines.append("")
        report_lines.append("| 策略 | Hit | 命中页码 | Top section |")
        report_lines.append("| --- | --- | --- | --- |")
        for strategy in args.strategies:
            result = next(
                case for case in per_case[strategy] if case.item_id == item.id
            )
            section_preview = " / ".join(result.top_sections[:2]) or "-"
            report_lines.append(
                f"| {strategy} | {'✅' if result.hit else '❌'} | "
                f"{result.retrieved_pages[:6]} | {section_preview} |"
            )
        report_lines.append("")

    report_text = "\n".join(report_lines)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(report_text, encoding="utf-8")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = args.results_dir / f"retrieval_eval_{timestamp}.json"
    results_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "eval_path": str(args.eval_path),
                "top_k": args.top_k,
                "strategies": args.strategies,
                "summary": all_results,
                "cases": {
                    strategy: [
                        {
                            "item_id": result.item_id,
                            "question": result.question,
                            "hit": result.hit,
                            "reciprocal_rank": result.reciprocal_rank,
                            "retrieved_pages": result.retrieved_pages,
                            "top_sections": result.top_sections,
                        }
                        for result in per_case[strategy]
                    ]
                    for strategy in args.strategies
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(report_text)
    print()
    print(f"Report saved to: {args.report_path}")
    print(f"Results saved to: {results_path}")
    return all_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval Hit@k on the finance QA benchmark.",
    )
    parser.add_argument(
        "--eval-path",
        type=Path,
        default=DEFAULT_EVAL_PATH,
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--embedding",
        choices=["bge-m3"],
        default="bge-m3",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "model_cache",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=[
            "vector_baseline",
            "finance_hybrid_full",
            "routed_dual_index",
        ],
        choices=[
            "vector_baseline",
            "finance_hybrid_no_rerank",
            "finance_hybrid_full",
            "routed_dual_index",
        ],
    )
    parser.add_argument(
        "--skip-reranker",
        action="store_true",
        help="Skip loading BGE reranker for finance_hybrid_full.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.skip_reranker and "finance_hybrid_full" in args.strategies:
        print("Note: finance_hybrid_full will run without reranker because --skip-reranker is set.")
    run_evaluation(args)


if __name__ == "__main__":
    main()
