# 面向财报与公告的可信 RAG 智能问答系统

<div align="center">
  <img src="https://github.com/AlaGrine/RAG_chatabot_with_Langchain/blob/main/data/docs/RAG_architecture.png" >
  <figcaption>基于 LangChain 组件的 RAG 架构示意图。</figcaption>
</div>

## 项目概述 <a name="overview"></a>

本项目是在通用 LangChain RAG Chatbot 原型基础上，面向财报、年报、公告等金融文档场景进行改造的可信问答系统。金融文档通常篇幅较长、表格密集、数值信息多，通用大模型直接回答容易出现幻觉，也难以追溯到具体文件和页码。因此，本项目采用 RAG 流程：先解析和切分 PDF，再向量化存入 Chroma，用户提问时先检索相关原文证据，再由大模型生成回答。

项目目标包括：

- 支持上传 PDF 财报或公告，并完成 MinerU 解析、JSON 归一化、chunk 切分和 Chroma 向量化存储。
- 在回答中保留来源追踪信息，包括文件名、页码、章节、表格或图片块信息。
- 支持多个财报同时入库，为后续多公司、多年度对比问答打基础。
- 通过 Streamlit 前端提供可运行、可演示的交互界面。
- 通过 `manifest.json` 记录每个 PDF 的解析状态，使异常文件和失败原因可追踪。

## 安装环境 <a name="installation"></a>

建议使用 Python 3.10-3.13 环境运行 MinerU 相关流程。项目主要依赖包括：

```text
langchain
langchain-openai
langchain-google-genai
chromadb
streamlit
mineru[all]
FlagEmbedding
```

完整依赖版本见 `requirements.txt`。

安装依赖：

```bash
pip install -r requirements.txt
```

## 运行 Streamlit 应用 <a name="instructions"></a>

在项目根目录运行：

```bash
streamlit run RAG_app.py
```

基本使用流程：

1. 在侧边栏选择 LLM provider，例如 OpenAI、Google Generative AI 或 HuggingFace。
2. 填入对应 API key。
3. 选择 embedding 模型。推荐在中文财报场景中使用 `Local BGE-M3`。
4. 上传 PDF、TXT、CSV 或 DOCX 文件。
5. 创建或加载 Chroma 向量库。
6. 提问并查看回答及来源文档。

## MinerU PDF 解析与数据预处理

PDF 文件在进入 Chroma 之前，会先通过 [MinerU](https://github.com/opendatalab/MinerU) 解析。完整预处理流程如下：

```text
一个或多个 PDF -> MinerU content_list.json -> 归一化 JSON -> 检索 chunk -> Chroma 向量库 -> manifest.json
```

项目会将 MinerU 解析后的归一化 JSON 保存到 `data/parsed_json/`，方便复现和检查。每个文档会根据文件名和 SHA-256 哈希生成稳定的 `doc_id`。每个 block 会保留 `block_id`、`type`、`text`、`page_no`、`bbox`、`section` 等字段；表格块会额外生成稳定的 `table_id`，并可选保留 `asset_path`、`table_caption`、`image_caption` 等结构信息。

每个检索 chunk 会保留以下元数据：

```text
doc_id
source
page
page_start
page_end
block_type
section
block_id
table_id
asset_path
parser
chunk_id
```

这些字段用于支持可信问答中的来源追踪，例如展示文件名、页码、章节、表格证据和图片证据。

归一化 JSON 示例：

```json
{
  "doc_id": "kweichow-moutai-2024",
  "filename": "kweichow_moutai_2024_annual_report.pdf",
  "parser": "mineru",
  "parser_version": "3.2.1",
  "parse_status": "success",
  "file_hash": "748f...",
  "source_path": "data/docs/kweichow_moutai_2024_annual_report.pdf",
  "page_count": 143,
  "block_counts": {
    "text": 1200,
    "table": 80,
    "image": 2
  },
  "blocks": [
    {
      "block_id": "kweichow-moutai-2024_b42",
      "type": "table",
      "text": "<table>...</table>",
      "page_no": 15,
      "section": "2、 收入和成本分析",
      "table_id": "kweichow-moutai-2024_t1",
      "asset_path": "images/table-1.jpg"
    }
  ]
}
```

MinerU 使用 CPU 兼容的 pipeline backend：

```bash
mineru -p <input.pdf> -o data/mineru_outputs -b pipeline
```

如果 Streamlit 上传流程中 MinerU 解析失败，应用会回退到 LangChain 的 `PyPDFLoader`，并在页面中显示 warning，避免直接崩溃。

## 批量解析多个财报 PDF

推荐使用 `scripts/ingest_finance_pdfs.py` 执行完整批处理流程。它支持输入 PDF 文件或目录，对每个 PDF 依次执行解析、归一化、chunk 切分和向量化入库，并生成 `manifest.json` 记录每个文件的成功或失败状态。

使用 BGE-M3 构建真实向量库：

```bash
python scripts/ingest_finance_pdfs.py \
  data/docs \
  --mineru-output-dir data/mineru_outputs \
  --parsed-json-dir data/parsed_json \
  --vectorstore-dir data/vector_stores/finance_multi_bge_m3 \
  --manifest-path data/parsed_json/manifest.json \
  --collection-name finance_multi_bge_m3 \
  --embedding bge-m3
```

如果只想快速检查流程结构，不想加载 BGE-M3，可以使用 `fake` embedding：

```bash
python scripts/ingest_finance_pdfs.py \
  data/docs \
  --vectorstore-dir data/vector_stores/finance_multi_fake \
  --manifest-path data/parsed_json/manifest_fake.json \
  --collection-name finance_multi_fake \
  --embedding fake
```

`manifest.json` 示例：

```json
{
  "pipeline": "finance_pdf_ingest",
  "collection_name": "finance_multi_bge_m3",
  "summary": {
    "total_files": 2,
    "succeeded": 2,
    "failed": 0,
    "chunk_count": 1200,
    "stored_vectors": 1200
  },
  "documents": [
    {
      "doc_id": "report-a8f4d2e0c91b",
      "filename": "report.pdf",
      "status": "success",
      "page_count": 143,
      "block_counts": {
        "text": 1791,
        "table": 233,
        "image": 2
      },
      "chunk_count": 909
    }
  ]
}
```

## 从已归一化 JSON 构建 Chroma

如果已经有归一化 JSON，也可以直接用 `scripts/build_finance_vectorstore.py` 构建向量库。

使用 `fake` embedding 做结构测试：

```bash
python scripts/build_finance_vectorstore.py \
  --parsed-json data/parsed_json/kweichow-moutai-2024-pages-1-20.json \
  --persist-dir data/vector_stores/moutai_pages_1_20_fake \
  --collection-name moutai_pages_1_20 \
  --embedding fake \
  --query "营业收入是多少" \
  --top-k 3
```

使用本地 BGE-M3 构建真实向量库：

```bash
python scripts/build_finance_vectorstore.py \
  --parsed-json data/parsed_json/kweichow-moutai-2024-full.json \
  --persist-dir data/vector_stores/moutai_2024_bge_m3 \
  --collection-name moutai_2024_bge_m3 \
  --embedding bge-m3 \
  --query "2024年度公司净利润是多少" \
  --top-k 5
```

说明：

- `fake` embedding 是确定性的，仅用于本地结构测试。
- `bge-m3` 通过 `FlagEmbedding` 加载 `BAAI/bge-m3`，模型文件缓存于 `data/model_cache/`。
- 为了保证回答可追溯，不要删除 chunk metadata 中的文件名、页码、章节和 block 类型信息。

## 异常 PDF 鲁棒性测试

异常输入测试报告见：

```text
docs/abnormal_pdf_test_report.md
```

该测试覆盖：

- 加密 PDF
- 损坏 PDF
- 扫描版 PDF
- 有效但无表格 PDF

测试结果通过 `data/parsed_json/abnormal_pdf_test/manifest.json` 记录。当前结果显示：4 个样本中 2 个成功、2 个失败。失败文件不会进入 Chroma，但错误原因会被记录；扫描 PDF 可被 OCR 解析，但数值识别可能存在误差。

## 测试

运行 PDF 解析与数据预处理相关测试：

```bash
python -m unittest tests.test_mineru_pipeline tests.test_finance_ingest -v
```

已覆盖内容包括：

- MinerU 调用参数
- content_list 查找
- JSON 归一化
- 表格、图片、公式等结构块处理
- chunk 切分与 metadata
- Chroma 向量化入库
- BGE-M3 embedding 封装
- 多 PDF 批量 ingest
- manifest 成功和失败记录
- 非 PDF、缺失文件等无效输入

## 课程报告中的 AI 辅助说明

MinerU 集成、JSON 归一化、chunk 切分、多文档入库和异常 PDF 测试代码是在“为财报 RAG 系统实现 PDF 解析与数据预处理模块”的需求下，借助 AI 辅助完成的。人工确认和调整的内容包括：财报场景元数据设计、表格/图片结构保留、BGE-M3 本地 embedding 接入、异常 PDF 测试结论，以及 README 和测试报告中的项目说明。
