import unittest

from rag_pipeline.chunker import build_chunks, prepare_parsed_doc
from rag_pipeline.retriever import FinanceHybridRetriever
from rag_pipeline.table_utils import (
    enrich_table_block,
    extract_relevant_table_facts,
    html_table_to_markdown,
    rewrite_finance_query,
)
from rag_pipeline.vectorstore import DeterministicFakeEmbeddings, build_chroma_vectorstore


class TableUtilsTest(unittest.TestCase):
    def test_html_table_to_markdown(self):
        html = "<table><tr><td>指标</td><td>2024年</td></tr><tr><td>净利润</td><td>100</td></tr></table>"
        markdown = html_table_to_markdown(html)
        self.assertIn("| 指标 | 2024年 |", markdown)
        self.assertIn("| 净利润 | 100 |", markdown)

    def test_enrich_table_block_adds_summary_and_markdown(self):
        block = enrich_table_block(
            {
                "type": "table",
                "text": "<table><tr><td>营业收入</td><td>1000</td></tr></table>",
                "table_caption": "主要会计数据",
                "section": "第二节 公司简介",
            }
        )
        self.assertIn("【表格摘要】", block["text"])
        self.assertIn("| 营业收入 | 1000 |", block["text"])
        self.assertTrue(block["table_summary"])

    def test_rewrite_finance_query_expands_aliases(self):
        variants = rewrite_finance_query("2024年净利润是多少")
        self.assertIn("2024年净利润是多少", variants)
        self.assertTrue(
            any("归属于上市公司股东的净利润" in variant for variant in variants)
        )

    def test_extract_relevant_table_facts(self):
        text = (
            "【表格摘要】主要会计数据\n"
            "| 指标 | 2024年 |\n"
            "| --- | --- |\n"
            "| 归属于上市公司股东的净利润 | 86228146421.62 |\n"
        )
        facts = extract_relevant_table_facts("2024年净利润是多少", text)
        self.assertIn("归属于上市公司股东的净利润", facts)

    def test_prepare_parsed_doc_enriches_legacy_table_blocks(self):
        parsed_doc = {
            "doc_id": "demo",
            "filename": "demo.pdf",
            "parser": "mineru",
            "blocks": [
                {
                    "block_id": "demo_b1",
                    "type": "table",
                    "text": "<table><tr><td>净利润</td><td>100</td></tr></table>",
                    "page_no": 5,
                }
            ],
        }
        prepared = prepare_parsed_doc(parsed_doc)
        self.assertIn("| 净利润 | 100 |", prepared["blocks"][0]["text"])


class FinanceHybridRetrieverTest(unittest.TestCase):
    def test_finance_hybrid_retriever_returns_table_with_extracted_facts(self):
        from tempfile import TemporaryDirectory

        parsed_doc = {
            "doc_id": "demo",
            "filename": "demo.pdf",
            "parser": "mineru",
            "blocks": [
                {
                    "block_id": "demo_b1",
                    "type": "text",
                    "text": "公司主要从事白酒生产和销售。",
                    "page_no": 2,
                },
                {
                    "block_id": "demo_b2",
                    "type": "table",
                    "text": enrich_table_block(
                        {
                            "type": "table",
                            "text": "<table><tr><td>指标</td><td>2024年</td></tr>"
                            "<tr><td>归属于上市公司股东的净利润</td><td>86228146421.62</td></tr></table>",
                            "section": "主要会计数据",
                            "page_no": 5,
                        }
                    )["text"],
                    "page_no": 5,
                    "section": "主要会计数据",
                },
            ],
        }
        chunks = build_chunks(parsed_doc)
        with TemporaryDirectory() as tmp_dir:
            vectorstore = build_chroma_vectorstore(
                chunks,
                persist_dir=tmp_dir,
                embeddings=DeterministicFakeEmbeddings(),
                collection_name="finance_hybrid",
            )
            retriever = FinanceHybridRetriever(
                vectorstore=vectorstore,
                vector_k=4,
                final_k=2,
            )
            results = retriever.invoke("2024年净利润是多少")

        self.assertTrue(results)
        table_results = [
            item for item in results if item.metadata.get("block_type") == "table"
        ]
        self.assertTrue(table_results)
        self.assertIn("结构化数值线索", table_results[0].page_content)

    def test_finance_hybrid_retriever_refuses_low_confidence(self):
        from tempfile import TemporaryDirectory

        class _EmptyVectorStore:
            def similarity_search_with_score(self, query, k=4):
                return []

        retriever = FinanceHybridRetriever(
            vectorstore=_EmptyVectorStore(),
            min_confidence_score=0.5,
        )
        results = retriever.invoke("2024年营业收入是多少")
        self.assertEqual(results, [])
        self.assertTrue(retriever.last_refused)


if __name__ == "__main__":
    unittest.main()
