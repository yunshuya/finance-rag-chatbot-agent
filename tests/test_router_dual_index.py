import unittest
from tempfile import TemporaryDirectory

from rag_pipeline.chunker import Document, build_chunks
from rag_pipeline.dual_index import (
    build_dual_chroma_index,
    has_dual_chroma_index,
    load_dual_chroma_index,
    split_chunks_for_dual_index,
)
from rag_pipeline.retriever import RoutedDualIndexRetriever
from rag_pipeline.router import QueryRouter
from rag_pipeline.vectorstore import DeterministicFakeEmbeddings


class QueryRouterTest(unittest.TestCase):
    def test_route_numeric_question_to_table_index(self):
        route = QueryRouter().route("2024年净利润是多少")
        self.assertEqual(route.intent, "numeric_lookup")
        self.assertGreater(route.table_weight, route.text_weight)

    def test_route_narrative_question_to_text_index(self):
        route = QueryRouter().route("公司2024年股份回购计划的金额范围是多少")
        self.assertEqual(route.intent, "narrative_qa")
        self.assertGreater(route.text_weight, route.table_weight)


class DualIndexTest(unittest.TestCase):
    def test_split_chunks_for_dual_index(self):
        chunks = [
            Document("table", {"block_type": "table", "chunk_id": "t1"}),
            Document("text", {"block_type": "text", "chunk_id": "x1"}),
        ]
        table_chunks, text_chunks = split_chunks_for_dual_index(chunks)
        self.assertEqual(len(table_chunks), 1)
        self.assertEqual(len(text_chunks), 1)

    def test_build_and_load_dual_chroma_index(self):
        parsed_doc = {
            "doc_id": "demo",
            "filename": "demo.pdf",
            "parser": "mineru",
            "blocks": [
                {
                    "block_id": "demo_b1",
                    "type": "text",
                    "text": "公司启动股份回购计划，金额不低于30亿元且不超过60亿元。",
                    "page_no": 8,
                    "section": "报告期内主要经营情况",
                },
                {
                    "block_id": "demo_b2",
                    "type": "table",
                    "text": (
                        "【表格摘要】主要会计数据\n"
                        "| 指标 | 2024年 |\n| --- | --- |\n"
                        "| 归属于上市公司股东的净利润 | 86228146421.62 |\n"
                    ),
                    "page_no": 5,
                    "section": "主要会计数据",
                },
            ],
        }
        chunks = build_chunks(parsed_doc)
        with TemporaryDirectory() as tmp_dir:
            dual_index = build_dual_chroma_index(
                chunks,
                persist_dir=tmp_dir,
                embeddings=DeterministicFakeEmbeddings(),
                collection_name="demo_dual",
                parsed_docs=[parsed_doc],
            )
            self.assertTrue(has_dual_chroma_index(tmp_dir, "demo_dual"))
            self.assertEqual(dual_index.table_count(), 1)
            self.assertEqual(dual_index.text_count(), 1)
            self.assertGreater(dual_index.fact_store.count(), 0)

            loaded = load_dual_chroma_index(
                tmp_dir,
                embeddings=DeterministicFakeEmbeddings(),
                collection_name="demo_dual",
            )
            retriever = RoutedDualIndexRetriever(
                dual_index=loaded,
                vector_k=4,
                final_k=2,
                min_confidence_score=0.0,
            )
            numeric_results = retriever.invoke("2024年净利润是多少")
            narrative_results = retriever.invoke(
                "公司2024年股份回购计划的金额范围是多少"
            )

        self.assertTrue(numeric_results)
        self.assertEqual(numeric_results[0].metadata.get("block_type"), "table")
        self.assertTrue(narrative_results)
        top_index = narrative_results[0].metadata.get("index_name")
        self.assertIn(top_index, {"text", "fact_store"})
        self.assertEqual(narrative_results[0].metadata.get("page"), 8)


if __name__ == "__main__":
    unittest.main()
