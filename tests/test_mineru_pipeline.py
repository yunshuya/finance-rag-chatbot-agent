import json
import unittest
from pathlib import Path

from rag_pipeline.chunker import build_chunks
from rag_pipeline.mineru_parser import find_content_list, run_mineru
from rag_pipeline.normalizer import normalize_mineru_content
from rag_pipeline.vectorstore import (
    BgeM3Embeddings,
    DeterministicFakeEmbeddings,
    build_chroma_vectorstore,
    load_chunks_from_parsed_json,
)
from scripts.inspect_mineru_output import inspect_content_list


class MinerUPipelineTest(unittest.TestCase):
    def test_run_mineru_uses_sync_parser_and_modelscope_defaults(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            pdf_path = tmp_path / "report.pdf"
            output_root = tmp_path / "mineru_outputs"
            pdf_path.write_bytes(b"%PDF-1.4 fake test pdf")
            calls = []

            def fake_do_parse(
                output_dir,
                pdf_file_names,
                pdf_bytes_list,
                p_lang_list,
                **kwargs,
            ):
                calls.append(
                    {
                        "output_dir": output_dir,
                        "pdf_file_names": pdf_file_names,
                        "pdf_bytes_list": pdf_bytes_list,
                        "p_lang_list": p_lang_list,
                        "kwargs": kwargs,
                    }
                )
                content_dir = Path(output_dir) / "report.pdf" / "txt"
                content_dir.mkdir(parents=True, exist_ok=True)
                (content_dir / "report.pdf_content_list.json").write_text(
                    "[]",
                    encoding="utf-8",
                )

            result = run_mineru(
                pdf_path,
                output_root,
                do_parse_func=fake_do_parse,
                cache_root=tmp_path / "cache",
            )

            self.assertEqual(
                result,
                output_root / "report.pdf" / "txt" / "report.pdf_content_list.json",
            )
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["pdf_file_names"], ["report.pdf"])
            self.assertEqual(calls[0]["pdf_bytes_list"], [b"%PDF-1.4 fake test pdf"])
            self.assertEqual(calls[0]["p_lang_list"], ["ch"])
            self.assertEqual(calls[0]["kwargs"]["backend"], "pipeline")
            self.assertEqual(calls[0]["kwargs"]["parse_method"], "txt")
            self.assertEqual(calls[0]["kwargs"]["formula_enable"], False)
            self.assertEqual(calls[0]["kwargs"]["table_enable"], True)
            self.assertEqual(calls[0]["kwargs"]["start_page_id"], 0)
            self.assertIsNone(calls[0]["kwargs"]["end_page_id"])

    def test_find_content_list_prefers_matching_pdf_name(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir) / "mineru_outputs"
            matching = output_root / "report" / "report_content_list.json"
            other = output_root / "other" / "other_content_list.json"
            matching.parent.mkdir(parents=True)
            other.parent.mkdir(parents=True)
            matching.write_text("[]", encoding="utf-8")
            other.write_text("[]", encoding="utf-8")

            result = find_content_list(output_root, "report.pdf")

            self.assertEqual(result, matching)

    def test_normalize_mineru_content_keeps_structured_blocks_and_skips_noise(self):
        from tempfile import TemporaryDirectory

        content_list = [
            {
                "type": "text",
                "text": "第三节 管理层讨论与分析",
                "page_idx": 0,
                "bbox": [8, 10, 300, 40],
                "text_level": 1,
            },
            {
                "type": "text",
                "text": "Annual revenue increased by 12%.",
                "page_idx": 0,
                "bbox": [10, 20, 300, 80],
                "text_level": 1,
            },
            {
                "type": "table",
                "table_body": "<table><tr><td>Revenue</td><td>100</td></tr></table>",
                "table_caption": ["主要财务数据"],
                "img_path": "tables/table-1.jpg",
                "page_idx": 1,
                "bbox": [12, 90, 500, 250],
            },
            {
                "type": "equation",
                "latex": "ROE = Net\\ Income / Equity",
                "page_idx": 2,
            },
            {
                "type": "image",
                "image_caption": ["控股股东结构图"],
                "image_footnote": [],
                "img_path": "images/control-structure.jpg",
                "page_idx": 3,
                "bbox": [20, 30, 400, 300],
            },
            {
                "type": "footer",
                "text": "Confidential",
                "page_idx": 2,
            },
        ]

        with TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "report_content_list.json"
            source.write_text(json.dumps(content_list), encoding="utf-8")

            parsed = normalize_mineru_content(
                source,
                "report.pdf",
                doc_id="doc-1",
                page_offset=10,
            )

            self.assertEqual(parsed["doc_id"], "doc-1")
            self.assertEqual(parsed["filename"], "report.pdf")
            self.assertEqual(parsed["parser"], "mineru")
            self.assertEqual(parsed["page_offset"], 10)
            self.assertEqual(
                parsed["block_counts"],
                {"equation": 1, "image": 1, "table": 1, "text": 2},
            )
            self.assertEqual(
                [block["type"] for block in parsed["blocks"]],
                ["text", "text", "table", "equation", "image"],
            )
            self.assertEqual(parsed["blocks"][0]["page_no"], 11)
            self.assertEqual(parsed["blocks"][0]["section"], "第三节 管理层讨论与分析")
            self.assertEqual(parsed["blocks"][1]["section"], "第三节 管理层讨论与分析")
            self.assertIn("| Revenue | 100 |", parsed["blocks"][2]["text"])
            self.assertTrue(parsed["blocks"][2]["text_html"].startswith("<table>"))
            self.assertEqual(parsed["blocks"][2]["table_id"], "doc-1_t1")
            self.assertEqual(parsed["blocks"][2]["table_caption"], "主要财务数据")
            self.assertEqual(parsed["blocks"][2]["asset_path"], "tables/table-1.jpg")
            self.assertEqual(parsed["blocks"][3]["text"], "ROE = Net\\ Income / Equity")
            self.assertEqual(parsed["blocks"][4]["page_no"], 14)
            self.assertEqual(parsed["blocks"][4]["asset_path"], "images/control-structure.jpg")
            self.assertEqual(parsed["blocks"][4]["text"], "控股股东结构图")

    def test_build_chunks_keeps_tables_separate_and_adds_retrieval_metadata(self):
        parsed_doc = {
            "doc_id": "doc-1",
            "filename": "report.pdf",
            "parser": "mineru",
            "blocks": [
                {
                    "block_id": "b1",
                    "type": "text",
                    "text": "Management discussion and analysis. " * 8,
                    "page_no": 1,
                    "section": "Management discussion",
                },
                {
                    "block_id": "b2",
                    "type": "table",
                    "text": "<table><tr><td>Cash</td><td>20</td></tr></table>",
                    "page_no": 2,
                    "section": "Financial statements",
                    "table_id": "doc-1_t1",
                    "asset_path": "tables/cash.jpg",
                },
                {
                    "block_id": "b3",
                    "type": "text",
                    "text": "Risk factors include market volatility.",
                    "page_no": 3,
                },
            ],
        }

        chunks = build_chunks(parsed_doc, chunk_size=120, chunk_overlap=20)

        self.assertGreaterEqual(len(chunks), 3)
        table_chunks = [doc for doc in chunks if doc.metadata["block_type"] == "table"]
        self.assertEqual(len(table_chunks), 1)
        self.assertIn("| Cash | 20 |", table_chunks[0].page_content)
        self.assertEqual(
            table_chunks[0].metadata,
            {
                "doc_id": "doc-1",
                "source": "report.pdf",
                "page": 2,
                "block_type": "table",
                "parser": "mineru",
                "chunk_id": "doc-1_chunk_2",
                "pre_chunked": True,
                "block_id": "b2",
                "table_id": "doc-1_t1",
                "table_summary": "Financial statements；表头字段: Cash、20",
                "section": "Financial statements",
                "asset_path": "tables/cash.jpg",
                "page_start": 2,
                "page_end": 2,
            },
        )

    def test_build_chunks_flushes_text_when_section_changes(self):
        parsed_doc = {
            "doc_id": "doc-section",
            "filename": "annual.pdf",
            "parser": "mineru",
            "blocks": [
                {
                    "block_id": "b1",
                    "type": "text",
                    "text": "Revenue growth remained stable.",
                    "page_no": 8,
                    "section": "一、经营情况讨论与分析",
                },
                {
                    "block_id": "b2",
                    "type": "text",
                    "text": "Gross margin stayed high.",
                    "page_no": 9,
                    "section": "一、经营情况讨论与分析",
                },
                {
                    "block_id": "b3",
                    "type": "text",
                    "text": "Risk factors include market competition.",
                    "page_no": 21,
                    "section": "二、风险因素",
                },
            ],
        }

        chunks = build_chunks(parsed_doc, chunk_size=500, chunk_overlap=0)

        self.assertEqual(len(chunks), 2)
        self.assertIn("Gross margin", chunks[0].page_content)
        self.assertNotIn("Risk factors", chunks[0].page_content)
        self.assertEqual(chunks[0].metadata["section"], "一、经营情况讨论与分析")
        self.assertEqual(chunks[0].metadata["page_start"], 8)
        self.assertEqual(chunks[0].metadata["page_end"], 9)
        self.assertEqual(chunks[1].metadata["section"], "二、风险因素")
        self.assertEqual(chunks[1].metadata["page_start"], 21)
        self.assertEqual(chunks[1].metadata["page_end"], 21)

    def test_inspect_mineru_output_reports_tables_and_images(self):
        from tempfile import TemporaryDirectory

        content_list = [
            {
                "type": "text",
                "text": "Revenue increased.",
                "page_idx": 0,
            },
            {
                "type": "table",
                "table_body": "<table><tr><td>Revenue</td></tr></table>",
                "img_path": "tables/table-1.jpg",
                "page_idx": 0,
            },
            {
                "type": "image",
                "img_path": "images/chart-1.jpg",
                "image_caption": ["股权结构图"],
                "page_idx": 1,
            },
            {
                "type": "text",
                "text": "",
            },
        ]

        with TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "content_list.json"
            source.write_text(json.dumps(content_list), encoding="utf-8")

            report = inspect_content_list(source)

            self.assertEqual(report["total_blocks"], 4)
            self.assertEqual(report["page_count"], 2)
            self.assertEqual(report["type_counts"]["table"], 1)
            self.assertEqual(report["table_count"], 1)
            self.assertEqual(report["tables_with_body"], 1)
            self.assertEqual(report["tables_with_asset"], 1)
            self.assertEqual(report["image_count"], 1)
            self.assertEqual(report["images_with_asset"], 1)
            self.assertEqual(report["images_with_caption"], 1)
            self.assertEqual(report["missing_page_blocks"], 1)
            self.assertEqual(report["empty_text_blocks"], 1)

    def test_build_chroma_vectorstore_retrieves_financial_table_with_metadata(self):
        from tempfile import TemporaryDirectory

        parsed_doc = {
            "doc_id": "moutai-test",
            "filename": "moutai.pdf",
            "parser": "mineru",
            "blocks": [
                {
                    "block_id": "b1",
                    "type": "text",
                    "text": "公司治理结构保持稳定。",
                    "page_no": 8,
                    "section": "一、公司治理",
                },
                {
                    "block_id": "b2",
                    "type": "table",
                    "text": "<table><tr><td>营业收入</td><td>1741.44亿元</td></tr></table>",
                    "page_no": 9,
                    "section": "二、主要财务数据",
                    "asset_path": "images/table-revenue.jpg",
                },
            ],
        }

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            parsed_json = tmp_path / "parsed.json"
            parsed_json.write_text(
                json.dumps(parsed_doc, ensure_ascii=False),
                encoding="utf-8",
            )

            chunks = load_chunks_from_parsed_json(parsed_json)
            vectorstore = build_chroma_vectorstore(
                chunks,
                persist_dir=tmp_path / "chroma",
                embeddings=DeterministicFakeEmbeddings(size=64),
                collection_name="finance_test",
            )
            results = vectorstore.similarity_search("营业收入是多少", k=2)

            self.assertGreaterEqual(len(results), 1)
            table_results = [
                result
                for result in results
                if result.metadata.get("block_type") == "table"
            ]
            self.assertGreaterEqual(len(table_results), 1)
            self.assertIn("1741.44亿元", table_results[0].page_content)
            self.assertEqual(table_results[0].metadata["source"], "moutai.pdf")
            self.assertEqual(table_results[0].metadata["page"], 9)
            self.assertEqual(table_results[0].metadata["section"], "二、主要财务数据")
            self.assertEqual(
                table_results[0].metadata["asset_path"],
                "images/table-revenue.jpg",
            )

    def test_bge_m3_embeddings_wraps_dense_vectors_and_uses_cache_dir(self):
        from tempfile import TemporaryDirectory

        calls = []

        class FakeBgeModel:
            def __init__(self, model_name, **kwargs):
                calls.append({"model_name": model_name, "kwargs": kwargs})

            def encode(self, texts, **kwargs):
                calls.append({"texts": texts, "encode_kwargs": kwargs})
                return {
                    "dense_vecs": [
                        [float(index), float(index + 1), float(index + 2)]
                        for index, _ in enumerate(texts, start=1)
                    ]
                }

        with TemporaryDirectory() as tmp_dir:
            embeddings = BgeM3Embeddings(
                cache_dir=Path(tmp_dir) / "model_cache",
                batch_size=2,
                use_fp16=False,
                model_factory=FakeBgeModel,
            )

            document_vectors = embeddings.embed_documents(["营业收入", "净利润"])
            query_vector = embeddings.embed_query("负债合计")

            self.assertEqual(
                calls[0],
                {
                    "model_name": "BAAI/bge-m3",
                    "kwargs": {
                        "cache_dir": str(Path(tmp_dir) / "model_cache"),
                        "use_fp16": False,
                    },
                },
            )
            self.assertEqual(calls[1]["texts"], ["营业收入", "净利润"])
            self.assertEqual(calls[1]["encode_kwargs"]["batch_size"], 2)
            self.assertTrue(calls[1]["encode_kwargs"]["return_dense"])
            self.assertFalse(calls[1]["encode_kwargs"]["return_sparse"])
            self.assertFalse(calls[1]["encode_kwargs"]["return_colbert_vecs"])
            self.assertEqual(document_vectors, [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
            self.assertEqual(query_vector, [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
