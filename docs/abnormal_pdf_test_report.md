# 异常 PDF 鲁棒性测试报告

测试日期：2026-06-05

## 测试目的

本测试用于验证财报 PDF 预处理流程在异常或边界输入下的处理能力，对应项目分工中的任务：

```text
PDF解析与数据预处理：MinerU解析、Chunk切分JSON结构设计、向量化存储
```

本次测试覆盖的流程为：

```text
PDF测试样本 -> MinerU解析 -> 归一化JSON -> Chunk切分 -> Chroma入库 -> manifest.json记录
```

本测试使用 `fake` embedding，目的是聚焦 PDF 解析鲁棒性和 `manifest.json` 记录能力，不测试 BGE-M3 向量效果。

## 测试命令

生成异常 PDF 样本：

```bash
.venv/bin/python scripts/generate_abnormal_pdf_samples.py
```

运行批量预处理：

```bash
.venv/bin/python scripts/ingest_finance_pdfs.py \
  data/test_pdfs/abnormal \
  --mineru-output-dir data/mineru_outputs/abnormal_pdf_test \
  --parsed-json-dir data/parsed_json/abnormal_pdf_test \
  --vectorstore-dir data/vector_stores/abnormal_pdf_test_fake \
  --manifest-path data/parsed_json/abnormal_pdf_test/manifest.json \
  --collection-name abnormal_pdf_test_fake \
  --embedding fake
```

## 测试样本

| 样本文件 | 类型 | 预期行为 |
|---|---|---|
| `corrupted_report.pdf` | 损坏 PDF | 解析失败，并在 `manifest.json` 中记录失败原因。 |
| `encrypted_report.pdf` | 加密 PDF | 解析失败，并在 `manifest.json` 中记录密码错误。 |
| `scanned_report.pdf` | 图片型扫描 PDF | MinerU 尝试 OCR；可能提取出文本，但可能存在 OCR 误差。 |
| `text_only_no_table_report.pdf` | 有效纯文本 PDF，无表格 | 解析成功，表格数量应为 0。 |

## Manifest 汇总

Manifest 路径：

```text
data/parsed_json/abnormal_pdf_test/manifest.json
```

整体结果：

| 指标 | 数值 |
|---|---:|
| PDF 总数 | 4 |
| 解析成功 | 2 |
| 解析失败 | 2 |
| Chunk 总数 | 2 |
| 入库向量数 | 2 |

逐文件结果：

| 样本文件 | 状态 | 页数 | Blocks | Tables | Images | Chunks | 错误信息 |
|---|---:|---:|---:|---:|---:|---:|---|
| `corrupted_report.pdf` | failed | - | 0 | 0 | 0 | 0 | `MinerU failed to parse corrupted_report.pdf: Failed to load document (PDFium: Data format error).` |
| `encrypted_report.pdf` | failed | - | 0 | 0 | 0 | 0 | `MinerU failed to parse encrypted_report.pdf: Failed to load document (PDFium: Incorrect password error).` |
| `scanned_report.pdf` | success | 1 | 5 | 0 | 0 | 1 | - |
| `text_only_no_table_report.pdf` | success | 1 | 6 | 0 | 0 | 1 | - |

## 结果观察

1. 损坏 PDF 和加密 PDF 没有导致整个批处理流程崩溃。它们被记录为失败文件，错误原因被写入 `manifest.json`，后续 PDF 仍然继续处理。

2. 扫描版 PDF 可以通过 MinerU 的 OCR 流程解析，并生成 1 个 chunk。人工抽查发现存在轻微 OCR 误差：原文 `Revenue: 100 million yuan` 被识别为 `Revenue: 10o million yuan`。因此，扫描版财报可以进入 RAG 流程，但数值类问答需要额外校验。

3. 有效的纯文本无表格 PDF 可以正常解析。`manifest.json` 正确记录其表格数量为 0，并生成 1 个 chunk。这说明“无表格”不是异常情况，而是一类有效输入。

4. Chroma 向量库只保存解析成功文件产生的 chunk。失败文件不会入库，但失败原因仍然可以通过 `manifest.json` 追踪。

## 结论

当前 PDF 预处理模块已经具备基础异常输入鲁棒性：

- 支持一次性批量处理多个 PDF。
- 支持逐文件记录成功和失败状态。
- 单个 PDF 失败时不会中断整个批处理流程。
- 能区分“有效但无表格 PDF”和“无法解析的异常 PDF”。
- 能对扫描 PDF 执行 OCR，但 OCR 结果的数值精度仍需在后续问答或人工验证中关注。

因此，在最终项目报告中，可以说明该 PDF 解析与数据预处理流程不仅能处理正常年报，也具备可复现、可追踪、失败可记录的工程化能力。
