# 面向财报与公告的可信 RAG 智能问答系统

> **课程项目**：《非结构化数据处理》期末 Project  
> **路径**：Option 1 — 工程应用路径（Business App）  
> **方向**：大模型知识库问答（RAG）  
> **仓库**：[yunshuya/finance-rag-chatbot-agent](https://github.com/yunshuya/finance-rag-chatbot-agent)  
> **本分支**：`rag-nk-local`（在课程 Streamlit RAG 模板 `main` 基础上完成的财报场景改造）

## 0. 仓库与分支说明

本 README 对应 **`rag-nk-local` 分支**，不是远程 `main` 上的英文课程模板版。

```bash
git clone https://github.com/yunshuya/finance-rag-chatbot-agent.git
cd finance-rag-chatbot-agent
git checkout rag-nk-local
```

| 分支 | 定位 | 适合谁 |
| --- | --- | --- |
| `main` | 课程下发的 LangChain Streamlit RAG **原始模板**（英文 README、通用文档问答） | 对照课程起点、查看模板结构 |
| `rag-nk-local` | 面向财报场景的**完整工程改造版**（本文件） | 直接演示、复现检索优化与 Docker 部署 |

`main` 分支仅保留 `RAG_app.py`、`RAG_notebook.ipynb`、`requirements.txt` 及演示向量库 `Vit_All_HF_Embeddings`；`rag-nk-local` 在其上新增 `rag_pipeline/`、`scripts/`、`tests/`、`docs/`、`Dockerfile` 等模块，并将预置向量库替换为茅台 2024 年报 BGE-M3 双索引。

## 1. 项目概述

本项目基于课程 Streamlit RAG 模板改造，面向 A 股年报、财报、公告等**长文档、表格密集、数值敏感**的金融场景，实现可运行、可复现、可容器化部署的问答应用。

核心目标：

- 将 PDF 解析为结构化 JSON，切分并向量化存入 Chroma，支持来源可追溯问答。
- 针对财报场景优化检索：表格增强、混合检索、查询路由、双索引、轻量 Fact Store、Reranker、置信度拒答。
- 通过 Streamlit 提供交互式演示界面，并提供 Dockerfile 支持一键容器化运行。

**演示数据**：仓库内已包含贵州茅台 2024 年报的归一化 JSON、双索引向量库（909 chunks = 233 表格 + 676 文本，另含 119 条 Fact Store 政策事实）及检索评测集，克隆后可直接演示问答。

### 1.1 与 `main` 分支的主要差异

| 维度 | `main`（课程模板） | `rag-nk-local`（本分支） |
| --- | --- | --- |
| 文档语言 | 英文通用 RAG 说明 | 中文课程交付文档 |
| 应用场景 | 任意 txt/pdf/csv/docx 上传问答 | 年报、财报、公告等**表格密集、数值敏感**长文档 |
| PDF 解析 | `PyPDFLoader` 等基础 Loader | **MinerU** 解析 + `normalizer` 归一化 + 章节感知 chunk |
| Embedding | HuggingFace API / Provider 默认 | **本地 BGE-M3**（`data/model_cache/` 缓存） |
| 预置向量库 | `Vit_All_HF_Embeddings` | `moutai_2024_bge_m3` 双索引 + `fact_store.json` |
| 检索策略 | 向量检索 / Cohere Rerank / 上下文压缩 | **Query Router** + **双索引** + **混合检索** + **Fact Store** + BGE Reranker |
| LLM 接入 | OpenAI / Google / HuggingFace | 上述保留，并新增 **DeepSeek**（推荐演示） |
| 可信输出 | 基础来源展示 | 页码、章节、`retrieval_score` / `rerank_score`、低置信度拒答（阈值 0.22） |
| 工程化 | 仅 `requirements.txt` | `Dockerfile`、`scripts/` 批处理与评测、`tests/`（30 项）、`docs/` 报告 |
| 可复现评测 | 无 | `data/eval/moutai_2024_retrieval_eval.json`，Hit@5 = **90%** |

保留自 `main` 的文件：`RAG_app.py`（大幅扩展）、`RAG_notebook.ipynb`、`requirements.txt`（增补 MinerU / BGE 等依赖）。

## 2. 课程算法与技术应用

本项目至少覆盖以下课程相关 NLP / 大模型技术（对应评分项「技术应用」）：

| 课程知识点 | 本项目实现 |
| --- | --- |
| 文本预处理与切分 | MinerU PDF 解析、HTML 表格转 Markdown、章节感知 chunk |
| 词向量 / 稠密向量检索 | BGE-M3 本地 Embedding + Chroma 向量库 |
| Transformer 表示与重排序 | BGE-Reranker-v2-m3 Cross-Encoder 精排 |
| 检索增强生成（RAG） | LangChain ConversationalRetrievalChain |
| 对话记忆与上下文压缩 | ConversationSummaryBufferMemory、财报追问改写 |
| 领域适配（非微调） | 查询改写、表格数值线索抽取、Query Router、Fact Store |

**使用的开源基础组件**（报告中需标注）：LangChain、Chroma、MinerU、FlagEmbedding / sentence-transformers、Streamlit、DeepSeek API（可选 LLM）。

**本项目自主改造部分**：`rag_pipeline/` 下的财报归一化、双索引、混合检索、路由、Fact Store、评测脚本及 Streamlit 集成逻辑。

## 3. 系统架构

```text
PDF / 已归一化 JSON
    -> MinerU 解析 & normalizer
    -> chunker（表格独立 chunk + 文本 chunk）
    -> 双索引 Chroma（tables / text）+ Fact Store（政策句预抽取）
    -> Query Router -> 混合检索 + Reranker
    -> LLM 生成（附页码 / 章节 / 检索分数来源）
```

主要模块：

```text
RAG_app.py                 # Streamlit 前端与对话链
RAG_notebook.ipynb         # 课程模板配套 Notebook（保留自 main）
rag_pipeline/
  mineru_parser.py         # MinerU CLI 封装与输出定位
  normalizer.py            # MinerU JSON 归一化
  chunker.py               # 财报感知切分
  vectorstore.py           # BGE-M3 Embedding + Chroma
  table_utils.py           # 表格 Markdown、查询改写、数值线索
  retriever.py             # FinanceHybrid / RoutedDualIndex 检索器
  router.py                # 数值 / 叙述 / 对比 查询路由
  dual_index.py            # 表格索引 + 文本索引
  fact_store.py            # 轻量政策事实库
  reranker.py              # BGE Reranker
  eval.py                  # Hit@k 评测
scripts/
  ingest_finance_pdfs.py   # 批量 PDF 入库
  build_finance_vectorstore.py
  evaluate_retrieval.py    # 检索评测
data/
  parsed_json/             # 归一化 JSON
  vector_stores/           # Chroma 持久化（含预置茅台 2024）
  eval/                    # 检索评测集
docs/
  retrieval_eval_report.md
  abnormal_pdf_test_report.md
```

## 4. 环境要求

- Python **3.10 – 3.13**（推荐 3.12）
- 操作系统：macOS / Linux / Windows
- 磁盘：建议 ≥ 8 GB（含 BGE 模型缓存）
- 可选：Docker 24+（容器化部署）
- 可选：NVIDIA GPU（加速 MinerU / Embedding，非必须）

## 5. 快速开始（推荐演示流程）

使用仓库内**预构建向量库**，无需重新解析 PDF，即可在 5 分钟内完成演示。

### 5.1 克隆与安装

```bash
git clone https://github.com/yunshuya/finance-rag-chatbot-agent.git
cd finance-rag-chatbot-agent
git checkout rag-nk-local

python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-docker.txt
# 若需本地 PDF 批处理，再安装完整依赖：
# pip install -r requirements.txt
```

### 5.2 配置 LLM（DeepSeek 示例）

```bash
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY
```

### 5.3 启动 Streamlit

```bash
streamlit run RAG_app.py
```

浏览器打开 `http://localhost:8501`，按以下配置：

| 配置项 | 推荐值 |
| --- | --- |
| LLM Provider | **DeepSeek**（或 OpenAI / Google） |
| Embedding | **Local BGE-M3** |
| Retriever | **Routed dual-index retriever** |
| Vector Store | 加载已有 `moutai_2024_bge_m3`（**勿选** `main` 分支遗留的 `Vit_All_HF_Embeddings`） |
| Language | chinese |

**示例问题**：

- `2024年归属于上市公司股东的净利润是多少？`
- `2024年年度利润分配预案中每10股派发现金红利多少元？`
- `公司2024年股份回购计划的金额范围是多少？`

回答应附带来源页码、章节、`retrieval_score` / `rerank_score`。

### 5.4 示例截图

> 答辩前请补充实际运行截图至 `docs/screenshots/` 并在下方引用。

| 截图 | 说明 |
| --- | --- |
| `docs/screenshots/streamlit_main.png` | Streamlit 主界面与侧边栏配置 |
| `docs/screenshots/qa_with_sources.png` | 问答结果带来源页码与检索分数 |
| `docs/screenshots/retrieval_eval.png` | 检索评测 Hit@5 结果 |

## 6. Docker 一键运行

Docker 镜像用于**平台无关的 Streamlit 演示**（使用预置向量库）。MinerU PDF 批处理建议在宿主机执行（依赖较重，见第 7 节）。

```bash
# 构建镜像
docker build -t finance-rag-chatbot .

# 运行（挂载 .env 提供 API Key）
docker run --rm -p 8501:8501 \
  --env-file .env \
  -v "$(pwd)/data/model_cache:/app/data/model_cache" \
  finance-rag-chatbot
```

访问 `http://localhost:8501`。首次使用 BGE-M3 时会下载模型到 `data/model_cache/`。

## 7. PDF 解析与向量化（完整流程）

### 7.1 数据流

```text
PDF -> MinerU content_list.json -> 归一化 JSON -> chunk -> 双索引 Chroma + fact_store.json -> manifest.json
```

### 7.2 单文件 MinerU 解析

```bash
mineru -p data/docs/your_report.pdf -o data/mineru_outputs -b pipeline
```

Streamlit 上传流程中若 MinerU 失败，会自动回退 `PyPDFLoader` 并显示 warning。

### 7.3 批量入库

```bash
python scripts/ingest_finance_pdfs.py \
  data/docs \
  --mineru-output-dir data/mineru_outputs \
  --parsed-json-dir data/parsed_json \
  --vectorstore-dir data/vector_stores/finance_multi_bge_m3 \
  --manifest-path data/parsed_json/manifest.json \
  --collection-name finance_multi_bge_m3 \
  --embedding bge-m3 \
  --dual-index
```

### 7.4 从已有 JSON 构建向量库

```bash
python scripts/build_finance_vectorstore.py \
  --parsed-json data/parsed_json/kweichow-moutai-2024-full.json \
  --persist-dir data/vector_stores/moutai_2024_bge_m3 \
  --collection-name moutai_2024_bge_m3 \
  --embedding bge-m3
```

结构测试（无需 GPU / 模型下载）：

```bash
python scripts/build_finance_vectorstore.py \
  --parsed-json data/parsed_json/kweichow-moutai-2024-pages-1-20.json \
  --persist-dir data/vector_stores/moutai_pages_1_20_fake \
  --collection-name moutai_pages_1_20 \
  --embedding fake
```

### 7.5 Chunk 元数据（来源追踪）

每个 chunk 保留：`doc_id`, `source`, `page`, `section`, `block_type`, `block_id`, `table_id`, `chunk_id` 等字段，用于可信问答中的证据引用。

## 8. 检索优化与算法复现

### 8.1 优化模块说明

| 模块 | 作用 |
| --- | --- |
| `table_utils` | HTML→Markdown、表头摘要、财务指标别名改写 |
| `FinanceHybridRetriever` | 向量 + 关键词混合分、表格加权 |
| `QueryRouter` | 数值 / 叙述 / 对比类问题分流 |
| `DualChromaIndex` | 表格索引与文本索引分离检索 |
| `FactStore` | 预抽取回购/分红/风险等政策句，关键词检索补强 |
| `BgeReranker` | Cross-Encoder 精排 Top 候选 |
| 置信度拒答 | `min_confidence_score=0.22`，低置信度不强行作答 |

### 8.2 检索评测复现

评测集：`data/eval/moutai_2024_retrieval_eval.json`（10 条茅台 2024 年报问题）

```bash
python scripts/evaluate_retrieval.py \
  --strategies routed_dual_index finance_hybrid_full vector_baseline \
  --top-k 5
```

结果写入 `docs/retrieval_eval_report.md`。

**当前结果（2026-06-11，routed_dual_index + Fact Store，无 reranker）**：

- Hit@5：**90%**（9/10）
- Fact Store：**119** 条政策事实
- 叙述类难题 q10（股份回购金额）已命中第 8 页

完整对比见 [`docs/retrieval_eval_report.md`](docs/retrieval_eval_report.md)。

## 9. 测试

### 9.1 单元测试（推荐答辩前执行）

```bash
python -m unittest discover -s tests -v
```

当前覆盖 **30** 项，包括：MinerU 归一化、chunk 切分、双索引、Fact Store、混合检索、评测逻辑、异常 ingest。

### 9.2 异常 PDF 鲁棒性测试

详见 [`docs/abnormal_pdf_test_report.md`](docs/abnormal_pdf_test_report.md)。

覆盖：加密 PDF、损坏 PDF、扫描版 PDF、无表格 PDF。失败文件不进入 Chroma，错误写入 `manifest.json`。

## 10. 项目交付物对照（Option 1）

| 交付物 | 本仓库位置 | 状态 |
| --- | --- | --- |
| 可运行代码仓库 | 本仓库 `rag-nk-local` 分支 | ✅ |
| README.md | 本文件（含与 `main` 对比说明） | ✅ |
| Dockerfile | `Dockerfile` | ✅ |
| Streamlit 前端 | `RAG_app.py` | ✅ |
| 技术报告（3–5 页） | 课程系统单独提交 | 待提交 |
| 示例截图 | `docs/screenshots/` | 待补充 |

## 11. 已知局限

- BGE 模型首次运行需联网下载，体积约数 GB。
- Docker 镜像不包含 MinerU 全量依赖，PDF 批处理请在宿主机完成。
- 扫描版 PDF OCR 数值可能有误差；复杂跨表计算问题仍依赖 LLM 推理。
- Reranker 在部分网络环境下可能回退到 FlagEmbedding，需确保模型缓存可用。
- 多公司多年度对比问答尚未做专门的跨文档聚合层。

## 12. AI 辅助开发说明（课程要求）

本课程允许使用 AI 辅助工具。以下为核心业务逻辑中借助 AI 完成或辅助完成的部分及人工校验说明。

### 12.1 MinerU 集成与数据预处理管线

- **需求描述（Prompt 摘要）**：为财报 RAG 实现 MinerU PDF 解析、JSON 归一化、chunk 切分、Chroma 入库及 manifest 追踪；保留页码、章节、表格结构元数据。
- **AI 生成部分**：`normalizer.py`、`chunker.py`、`ingest_finance_pdfs.py` 初版骨架。
- **人工修改**：财报元数据字段设计、表格 HTML 增强、`doc_id` 哈希策略、异常 PDF 测试与 manifest 错误记录。

### 12.2 财报检索优化（混合检索 / 双索引 / Fact Store）

- **需求描述**：针对年报数值题与政策叙述题优化检索；实现表格 Markdown、查询改写、Query Router、双索引、BGE Reranker、Hit@k 评测；为回购/分红类问题增加轻量 Fact Store。
- **AI 生成部分**：`table_utils.py`、`retriever.py`、`router.py`、`dual_index.py`、`fact_store.py`、`eval.py` 及对应测试。
- **人工修改**：评测集标注、茅台向量库重建、q10 回购案例验证、阈值调参（拒答 0.22）、Streamlit 集成与 DeepSeek API 接入。

### 12.3 文档与容器化

- **需求描述**：按课程 Project 指南重写 README，补充 Dockerfile 与可复现说明。
- **AI 生成部分**：README 结构、Dockerfile 模板。
- **人工修改**：答辩截图占位、团队信息、技术报告与 README 一致性校对。

## 13. 参考文献与开源声明

**课程模板来源**（`main` 分支）：基于 [AlaGrine/RAG_chatabot_with_Langchain](https://github.com/AlaGrine/RAG_chatabot_with_Langchain) Streamlit RAG 模板改造。

**本项目使用的主要开源组件**：

- [LangChain](https://github.com/langchain-ai/langchain)
- [Chroma](https://www.trychroma.com/)
- [MinerU](https://github.com/opendatalab/MinerU)
- [BGE-M3 / BGE-Reranker](https://github.com/FlagOpen/FlagEmbedding)
- [Streamlit](https://streamlit.io/)
- [DeepSeek API](https://platform.deepseek.com/)

## 14. 分工与联系（请按小组实际情况填写）

| 成员 | 分工 |
| --- | --- |
| 方妍 | PDF 解析与数据预处理、向量库构建 |
| 牛珂 | 检索优化、评测与 Streamlit 集成 |
| 朱婧怡 | Docker 部署、鲁棒性测试、技术报告 |

---

如有问题，请在课程仓库 Issue 中反馈，或联系助教。
