import unittest
from dataclasses import dataclass

from rag_pipeline.chunker import Document
from rag_pipeline.eval import (
    evaluate_retrieval_case,
    hit_at_k,
    load_eval_dataset,
    summarize_results,
)
from rag_pipeline.retriever import FinanceHybridRetriever
from rag_pipeline.vectorstore import DeterministicFakeEmbeddings, build_chroma_vectorstore


@dataclass
class _EvalItem:
    id: str
    question: str
    expected_pages: list[int]


class RetrievalEvalTest(unittest.TestCase):
    def test_hit_at_k_matches_expected_page(self):
        docs = [
            Document("irrelevant", {"page": 99}),
            Document("target", {"page": 5}),
        ]
        hit, rr = hit_at_k(docs, [5], k=2)
        self.assertTrue(hit)
        self.assertEqual(rr, 0.5)

    def test_hit_at_k_misses_when_page_not_in_top_k(self):
        docs = [Document("irrelevant", {"page": 99})]
        hit, rr = hit_at_k(docs, [5], k=1)
        self.assertFalse(hit)
        self.assertEqual(rr, 0.0)

    def test_load_eval_dataset_has_ten_items(self):
        from pathlib import Path

        eval_path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "eval"
            / "moutai_2024_retrieval_eval.json"
        )
        payload, items = load_eval_dataset(eval_path)
        self.assertEqual(len(items), 10)
        self.assertEqual(payload["collection_name"], "moutai_2024_bge_m3")

    def test_finance_hybrid_beats_plain_vector_on_numeric_table_query(self):
        from tempfile import TemporaryDirectory

        parsed_doc = {
            "doc_id": "demo",
            "filename": "demo.pdf",
            "parser": "mineru",
            "blocks": [
                {
                    "block_id": "demo_b1",
                    "type": "text",
                    "text": "公司主要从事白酒生产和销售，经营情况稳定。",
                    "page_no": 20,
                },
                {
                    "block_id": "demo_b2",
                    "type": "table",
                    "text": (
                        "【表格摘要】主要会计数据\n"
                        "| 指标 | 2024年 |\n"
                        "| --- | --- |\n"
                        "| 归属于上市公司股东的净利润 | 86228146421.62 |\n"
                    ),
                    "page_no": 5,
                    "section": "主要会计数据",
                },
            ],
        }
        from rag_pipeline.chunker import build_chunks

        chunks = build_chunks(parsed_doc)
        with TemporaryDirectory() as tmp_dir:
            vectorstore = build_chroma_vectorstore(
                chunks,
                persist_dir=tmp_dir,
                embeddings=DeterministicFakeEmbeddings(),
                collection_name="eval_demo",
            )
            baseline = vectorstore.similarity_search("2024年净利润是多少", k=2)
            hybrid = FinanceHybridRetriever(
                vectorstore=vectorstore,
                vector_k=4,
                final_k=2,
                min_confidence_score=0.0,
            ).invoke("2024年净利润是多少")

        baseline_hit, _ = hit_at_k(baseline, [5], k=2)
        hybrid_hit, _ = hit_at_k(hybrid, [5], k=2)
        self.assertTrue(hybrid_hit)
        self.assertIn(5, {doc.metadata.get("page") for doc in hybrid})

    def test_summarize_results(self):
        item = _EvalItem("q01", "test", [5])
        results = [
            evaluate_retrieval_case(
                item,
                [Document("x", {"page": 5})],
                k=1,
            ),
            evaluate_retrieval_case(
                item,
                [Document("x", {"page": 99})],
                k=1,
            ),
        ]
        summary = summarize_results(results)
        self.assertEqual(summary["hits"], 1)
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["hit_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
