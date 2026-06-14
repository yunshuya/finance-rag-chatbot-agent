import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rag_pipeline.mineru_parser import MinerUParseError
from rag_pipeline.vectorstore import (
    DeterministicFakeEmbeddings,
    load_chroma_vectorstore,
)
from scripts.ingest_finance_pdfs import collect_pdf_paths, ingest_pdfs


class FinancePdfIngestTest(unittest.TestCase):
    def test_ingest_multiple_pdfs_writes_manifest_and_appends_vectors(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            pdf_a = tmp_path / "alpha-report.pdf"
            pdf_b = tmp_path / "beta-report.pdf"
            pdf_a.write_bytes(b"%PDF-1.4 alpha")
            pdf_b.write_bytes(b"%PDF-1.4 beta")

            def fake_run_mineru(pdf_path, output_root, **_kwargs):
                content_list = [
                    {
                        "type": "table",
                        "table_body": (
                            f"<table><tr><td>净利润</td><td>{pdf_path.stem}</td></tr></table>"
                        ),
                        "page_idx": 0,
                    }
                ]
                content_path = Path(output_root) / "content_list.json"
                content_path.parent.mkdir(parents=True, exist_ok=True)
                content_path.write_text(
                    json.dumps(content_list, ensure_ascii=False),
                    encoding="utf-8",
                )
                return content_path

            manifest = ingest_pdfs(
                inputs=[pdf_a, pdf_b],
                mineru_output_dir=tmp_path / "mineru",
                parsed_json_dir=tmp_path / "parsed",
                vectorstore_dir=tmp_path / "chroma",
                manifest_path=tmp_path / "manifest.json",
                collection_name="finance_multi",
                embeddings=DeterministicFakeEmbeddings(size=64),
                run_mineru_func=fake_run_mineru,
            )

            self.assertEqual(manifest["summary"]["total_files"], 2)
            self.assertEqual(manifest["summary"]["succeeded"], 2)
            self.assertEqual(manifest["summary"]["failed"], 0)
            self.assertEqual(manifest["summary"]["stored_vectors"], 2)
            self.assertTrue((tmp_path / "manifest.json").exists())
            self.assertEqual(len(list((tmp_path / "parsed").glob("*.json"))), 2)

            doc_ids = [document["doc_id"] for document in manifest["documents"]]
            self.assertEqual(len(set(doc_ids)), 2)
            self.assertTrue(doc_ids[0].startswith("alpha-report-"))
            self.assertTrue(doc_ids[1].startswith("beta-report-"))
            self.assertEqual(manifest["documents"][0]["status"], "success")
            self.assertEqual(manifest["documents"][0]["chunk_count"], 1)
            self.assertEqual(manifest["documents"][0]["block_counts"], {"table": 1})

            loaded = load_chroma_vectorstore(
                tmp_path / "chroma",
                embeddings=DeterministicFakeEmbeddings(size=64),
                collection_name="finance_multi",
            )
            self.assertEqual(loaded.count(), 2)

    def test_ingest_records_failed_pdf_and_continues(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            good_pdf = tmp_path / "good.pdf"
            bad_pdf = tmp_path / "bad.pdf"
            good_pdf.write_bytes(b"%PDF-1.4 good")
            bad_pdf.write_bytes(b"%PDF-1.4 bad")

            def fake_run_mineru(pdf_path, output_root, **_kwargs):
                if pdf_path.name == "bad.pdf":
                    raise MinerUParseError("cannot parse encrypted pdf")
                content_path = Path(output_root) / "content_list.json"
                content_path.parent.mkdir(parents=True, exist_ok=True)
                content_path.write_text(
                    json.dumps(
                        [
                            {
                                "type": "text",
                                "text": "营业收入稳定增长。",
                                "page_idx": 0,
                            }
                        ],
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                return content_path

            manifest = ingest_pdfs(
                inputs=[good_pdf, bad_pdf],
                mineru_output_dir=tmp_path / "mineru",
                parsed_json_dir=tmp_path / "parsed",
                vectorstore_dir=tmp_path / "chroma",
                manifest_path=tmp_path / "manifest.json",
                collection_name="finance_multi",
                embeddings=DeterministicFakeEmbeddings(size=64),
                run_mineru_func=fake_run_mineru,
            )

            self.assertEqual(manifest["summary"]["total_files"], 2)
            self.assertEqual(manifest["summary"]["succeeded"], 1)
            self.assertEqual(manifest["summary"]["failed"], 1)
            self.assertEqual(manifest["summary"]["stored_vectors"], 1)

            by_file = {Path(item["source_path"]).name: item for item in manifest["documents"]}
            self.assertEqual(by_file["good.pdf"]["status"], "success")
            self.assertEqual(by_file["bad.pdf"]["status"], "failed")
            self.assertIn("cannot parse encrypted pdf", by_file["bad.pdf"]["error_message"])

    def test_collect_pdf_paths_accepts_directories_and_rejects_invalid_inputs(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            docs_dir = tmp_path / "docs"
            docs_dir.mkdir()
            pdf_a = docs_dir / "a.pdf"
            pdf_b = docs_dir / "b.PDF"
            note = docs_dir / "note.txt"
            pdf_a.write_bytes(b"%PDF-1.4 a")
            pdf_b.write_bytes(b"%PDF-1.4 b")
            note.write_text("not a pdf", encoding="utf-8")

            self.assertEqual(collect_pdf_paths([docs_dir]), [pdf_a, pdf_b])

            with self.assertRaises(ValueError):
                collect_pdf_paths([note])
            with self.assertRaises(FileNotFoundError):
                collect_pdf_paths([tmp_path / "missing.pdf"])


if __name__ == "__main__":
    unittest.main()
