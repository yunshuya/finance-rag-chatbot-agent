import unittest
from tempfile import TemporaryDirectory

from rag_pipeline.fact_store import (
    FactStore,
    build_fact_store,
    extract_facts_from_parsed_doc,
    load_fact_store,
    save_fact_store,
)


class FactStoreTest(unittest.TestCase):
    def setUp(self):
        self.parsed_doc = {
            "doc_id": "demo",
            "filename": "demo.pdf",
            "blocks": [
                {
                    "block_id": "demo_b1",
                    "type": "text",
                    "text": (
                        "五是市值管理再创新绩。首次制定《市值管理办法》，提升管理规范化水平；"
                        "发布2024-2026三年分红规划，建立长效回报机制；"
                        "首次启动股份回购计划，回购金额不低于30亿元（含）且不超过60亿元（含），"
                        "回购股份用于注销并减少公司注册资本。"
                    ),
                    "page_no": 8,
                    "section": "报告期内主要经营情况",
                },
                {
                    "block_id": "demo_b2",
                    "type": "table",
                    "text": "<table><tr><td>净利润</td><td>100</td></tr></table>",
                    "page_no": 5,
                },
            ],
        }

    def test_extract_facts_from_parsed_doc(self):
        facts = extract_facts_from_parsed_doc(self.parsed_doc)
        self.assertGreaterEqual(len(facts), 2)
        buyback_facts = [
            fact for fact in facts if "buyback" in fact.get("categories", [])
        ]
        self.assertTrue(buyback_facts)
        self.assertEqual(buyback_facts[0]["page_no"], 8)
        self.assertIn("30亿元", buyback_facts[0]["text"])

    def test_fact_store_search_buyback_query(self):
        facts = extract_facts_from_parsed_doc(self.parsed_doc)
        store = FactStore(facts)
        results = store.search("公司2024年股份回购计划的金额范围是多少", top_k=3)
        self.assertTrue(results)
        top = results[0]
        self.assertEqual(top.metadata.get("page"), 8)
        self.assertIn("回购", top.page_content)
        self.assertEqual(top.metadata.get("index_name"), "fact_store")

    def test_save_and_load_fact_store(self):
        store = build_fact_store(parsed_docs=[self.parsed_doc])
        with TemporaryDirectory() as tmp_dir:
            save_fact_store(store, tmp_dir)
            loaded = load_fact_store(tmp_dir)
            self.assertEqual(loaded.count(), store.count())
            self.assertGreater(loaded.count(), 0)
            results = loaded.search("股份回购计划金额", top_k=1)
            self.assertTrue(results)


if __name__ == "__main__":
    unittest.main()
