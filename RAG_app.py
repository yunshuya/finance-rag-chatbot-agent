####################################################################
#                         import
####################################################################

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import os, glob, shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")
DEEPSEEK_API_KEY_ENV = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# Import openai and google_genai as main LLM services
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# langchain prompts, memory, chains...
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, format_document
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory, ConversationSummaryBufferMemory


# document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    CSVLoader,
    Docx2txtLoader,
)

# text_splitter
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)

# OutputParser
from langchain_core.output_parsers import StrOutputParser

# Import chroma as the vector store
from langchain_community.vectorstores import Chroma

# Contextual_compression
from langchain_classic.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import (
    EmbeddingsRedundantFilter,
    LongContextReorder,
)
from langchain_classic.retrievers.document_compressors import EmbeddingsFilter
from langchain_classic.retrievers import ContextualCompressionRetriever

# Cohere
from langchain_classic.retrievers.document_compressors import CohereRerank
from langchain_community.llms import Cohere

# HuggingFace
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms import HuggingFaceHub

# Import streamlit
import streamlit as st

from rag_pipeline import (
    BgeM3Embeddings,
    BgeReranker,
    FinanceHybridRetriever,
    RoutedDualIndexRetriever,
    build_chunks,
    build_dual_chroma_index,
    compute_file_sha256,
    has_dual_chroma_index,
    load_dual_chroma_index,
    make_doc_id,
    normalize_mineru_content,
    run_mineru,
)
from rag_pipeline.normalizer import write_normalized_json

####################################################################
#              Config: LLM services, assistant language,...
####################################################################
list_LLM_providers = [
    "**DeepSeek**",
    ":rainbow[**OpenAI**]",
    "**Google Generative AI**",
    ":hugging_face: **HuggingFace**",
]

dict_welcome_message = {
    "english": "How can I assist you today?",
    "french": "Comment puis-je vous aider aujourd’hui ?",
    "spanish": "¿Cómo puedo ayudarle hoy?",
    "german": "Wie kann ich Ihnen heute helfen?",
    "russian": "Чем я могу помочь вам сегодня?",
    "chinese": "我今天能帮你什么？",
    "arabic": "كيف يمكنني مساعدتك اليوم؟",
    "portuguese": "Como posso ajudá-lo hoje?",
    "italian": "Come posso assistervi oggi?",
    "Japanese": "今日はどのようなご用件でしょうか?",
}

list_retriever_types = [
    "Routed dual-index retriever",
    "Finance hybrid retriever",
    "Cohere reranker",
    "Contextual compression",
    "Vectorstore backed retriever",
]

list_embedding_models = [
    "Local BGE-M3",
    "Provider default",
]

TMP_DIR = Path(__file__).resolve().parent.joinpath("data", "tmp")
LOCAL_VECTOR_STORE_DIR = (
    Path(__file__).resolve().parent.joinpath("data", "vector_stores")
)
MINERU_OUTPUT_DIR = Path(__file__).resolve().parent.joinpath("data", "mineru_outputs")
PARSED_JSON_DIR = Path(__file__).resolve().parent.joinpath("data", "parsed_json")
MODEL_CACHE_DIR = Path(__file__).resolve().parent.joinpath("data", "model_cache")

for data_dir in (
    TMP_DIR,
    LOCAL_VECTOR_STORE_DIR,
    MINERU_OUTPUT_DIR,
    PARSED_JSON_DIR,
    MODEL_CACHE_DIR,
):
    data_dir.mkdir(parents=True, exist_ok=True)

####################################################################
#            Create app interface with streamlit
####################################################################
st.set_page_config(page_title="Chat With Your Data")

st.title("🤖 RAG chatbot")


def init_session_state():
    """Initialize Streamlit session keys once per browser session."""
    defaults = {
        "openai_api_key": "",
        "google_api_key": "",
        "cohere_api_key": "",
        "hf_api_key": "",
        "deepseek_api_key": DEEPSEEK_API_KEY_ENV,
        "chain": None,
        "memory": None,
        "retriever": None,
        "vector_store": None,
        "dual_index": None,
        "selected_vectorstore_name": "",
        "vector_store_name": "",
        "error_message": "",
        "assistant_language": "chinese",
        "embedding_model": list_embedding_models[0],
        "retriever_type": list_retriever_types[0],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def is_vectorstore_ready():
    return st.session_state.get("chain") is not None


def has_llm_api_key():
    return bool(
        st.session_state.openai_api_key
        or st.session_state.google_api_key
        or st.session_state.hf_api_key
        or st.session_state.deepseek_api_key
    )


def list_local_vectorstores():
    """List persisted Chroma directories under data/vector_stores."""
    if not LOCAL_VECTOR_STORE_DIR.exists():
        return []
    stores = []
    for path in sorted(LOCAL_VECTOR_STORE_DIR.iterdir()):
        if path.is_dir() and not path.name.startswith("."):
            stores.append(path)
    return stores


def delete_vectorstore(persist_path):
    """Delete a persisted vectorstore directory from disk."""
    try:
        shutil.rmtree(persist_path)
        return True, None
    except Exception as error:
        return False, str(error)


def load_vectorstore_from_path(selected_vectorstore_path):
    """Load Chroma / dual-index retriever from a persisted directory."""
    selected_vectorstore_path = Path(selected_vectorstore_path)
    st.session_state.selected_vectorstore_name = selected_vectorstore_path.name
    embeddings = select_embeddings_model()
    collection_name = selected_vectorstore_path.name
    dual_index = None
    if st.session_state.retriever_type == list_retriever_types[0]:
        if not has_dual_chroma_index(selected_vectorstore_path, collection_name):
            st.error(
                "该向量库尚未构建双索引。请使用 "
                "`scripts/build_finance_vectorstore.py --dual-index` "
                "或重新 ingest 文档。"
            )
            return
        dual_index = load_dual_chroma_index(
            selected_vectorstore_path,
            embeddings=embeddings,
            collection_name=collection_name,
        )
        st.session_state.dual_index = dual_index
        st.session_state.vector_store = None
    else:
        st.session_state.dual_index = None
        st.session_state.vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory=str(selected_vectorstore_path),
        )

    st.session_state.retriever = create_retriever(
        vector_store=st.session_state.vector_store,
        dual_index=dual_index,
        embeddings=embeddings,
        retriever_type=st.session_state.retriever_type,
        base_retriever_search_type="similarity",
        base_retriever_k=16,
        compression_retriever_k=20,
        cohere_api_key=st.session_state.cohere_api_key,
        cohere_model="rerank-multilingual-v2.0",
        cohere_top_n=10,
    )
    st.session_state.chain, st.session_state.memory = create_ConversationalRetrievalChain(
        retriever=st.session_state.retriever,
        chain_type="stuff",
        language=st.session_state.assistant_language,
    )
    clear_chat_history()
    st.info(f"**{st.session_state.selected_vectorstore_name}** is loaded successfully.")


def create_deepseek_llm(model, temperature, top_p=None):
    kwargs = {
        "model": model,
        "api_key": st.session_state.deepseek_api_key,
        "base_url": DEEPSEEK_BASE_URL,
        "temperature": temperature,
    }
    if top_p is not None:
        kwargs["model_kwargs"] = {"top_p": top_p}
    return ChatOpenAI(**kwargs)


def expander_model_parameters(
    LLM_provider="OpenAI",
    text_input_API_key="OpenAI API Key - [Get an API key](https://platform.openai.com/account/api-keys)",
    list_models=["gpt-3.5-turbo-0125", "gpt-3.5-turbo", "gpt-4-turbo-preview"],
):
    """Add a text_input (for API key) and a streamlit expander containing models and parameters."""
    st.session_state.LLM_provider = LLM_provider

    if LLM_provider == "DeepSeek":
        st.session_state.deepseek_api_key = st.text_input(
            text_input_API_key,
            value=st.session_state.deepseek_api_key,
            type="password",
            placeholder="insert your API key",
        )
        st.session_state.openai_api_key = ""
        st.session_state.google_api_key = ""
        st.session_state.hf_api_key = ""

    if LLM_provider == "OpenAI":
        st.session_state.openai_api_key = st.text_input(
            text_input_API_key,
            type="password",
            placeholder="insert your API key",
        )
        st.session_state.deepseek_api_key = ""
        st.session_state.google_api_key = ""
        st.session_state.hf_api_key = ""

    if LLM_provider == "Google":
        st.session_state.google_api_key = st.text_input(
            text_input_API_key,
            type="password",
            placeholder="insert your API key",
        )
        st.session_state.openai_api_key = ""
        st.session_state.deepseek_api_key = ""
        st.session_state.hf_api_key = ""

    if LLM_provider == "HuggingFace":
        st.session_state.hf_api_key = st.text_input(
            text_input_API_key,
            type="password",
            placeholder="insert your API key",
        )
        st.session_state.openai_api_key = ""
        st.session_state.google_api_key = ""
        st.session_state.deepseek_api_key = ""

    with st.expander("**Models and parameters**"):
        st.session_state.selected_model = st.selectbox(
            f"Choose {LLM_provider} model", list_models
        )

        # model parameters
        st.session_state.temperature = st.slider(
            "temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
        )
        st.session_state.top_p = st.slider(
            "top_p",
            min_value=0.0,
            max_value=1.0,
            value=0.95,
            step=0.05,
        )


def sidebar_and_documentChooser():
    """Create the sidebar and the a tabbed pane: the first tab contains a document chooser (create a new vectorstore);
    the second contains a vectorstore chooser (open an old vectorstore)."""

    with st.sidebar:
        st.caption(
            "🚀 A retrieval augmented generation chatbot powered by 🔗 Langchain, Cohere, OpenAI, Google Generative AI and 🤗"
        )
        st.write("")

        llm_chooser = st.radio(
            "Select provider",
            list_LLM_providers,
            captions=[
                "[DeepSeek platform](https://platform.deepseek.com/)",
                "[OpenAI pricing page](https://openai.com/pricing)",
                "Rate limit: 60 requests per minute.",
                "**Free access.**",
            ],
        )

        st.divider()
        if llm_chooser == list_LLM_providers[0]:
            expander_model_parameters(
                LLM_provider="DeepSeek",
                text_input_API_key="DeepSeek API Key - [Get an API key](https://platform.deepseek.com/api_keys)",
                list_models=["deepseek-chat", "deepseek-reasoner"],
            )

        if llm_chooser == list_LLM_providers[1]:
            expander_model_parameters(
                LLM_provider="OpenAI",
                text_input_API_key="OpenAI API Key - [Get an API key](https://platform.openai.com/account/api-keys)",
                list_models=[
                    "gpt-3.5-turbo-0125",
                    "gpt-3.5-turbo",
                    "gpt-4-turbo-preview",
                ],
            )

        if llm_chooser == list_LLM_providers[2]:
            expander_model_parameters(
                LLM_provider="Google",
                text_input_API_key="Google API Key - [Get an API key](https://makersuite.google.com/app/apikey)",
                list_models=["gemini-pro"],
            )
        if llm_chooser == list_LLM_providers[3]:
            expander_model_parameters(
                LLM_provider="HuggingFace",
                text_input_API_key="HuggingFace API key - [Get an API key](https://huggingface.co/settings/tokens)",
                list_models=["mistralai/Mistral-7B-Instruct-v0.2"],
            )
        # Assistant language
        st.write("")
        st.session_state.assistant_language = st.selectbox(
            f"Assistant language", list(dict_welcome_message.keys())
        )

        st.write("")
        st.session_state.embedding_model = st.selectbox(
            "Embedding model",
            list_embedding_models,
            help="Use Local BGE-M3 for reproducible Chinese finance retrieval.",
        )

        st.divider()
        st.subheader("Retrievers")
        retrievers = list_retriever_types
        if st.session_state.selected_model == "gpt-3.5-turbo":
            # for "gpt-3.5-turbo", we will not use the vectorstore backed retriever
            # there is a high risk of exceeding the max tokens limit (4096).
            retrievers = list_retriever_types[:-1]

        st.session_state.retriever_type = st.selectbox(
            f"Select retriever type", retrievers
        )
        st.write("")
        if st.session_state.retriever_type == list_retriever_types[2]:  # Cohere
            st.session_state.cohere_api_key = st.text_input(
                "Coher API Key - [Get an API key](https://dashboard.cohere.com/api-keys)",
                type="password",
                placeholder="insert your API key",
            )

        if st.session_state.retriever_type == list_retriever_types[0]:
            st.caption(
                "Routed dual-index retriever 会根据问题类型在表格索引与文本索引之间动态路由。"
            )

        st.write("\n\n")
        st.write(
            f"ℹ _Your {st.session_state.LLM_provider} API key, '{st.session_state.selected_model}' parameters, \
            and {st.session_state.retriever_type} are only considered when loading or creating a vectorstore._"
        )

    # Tabbed Pane: Create a new Vectorstore | Open a saved Vectorstore

    tab_new_vectorstore, tab_open_vectorstore = st.tabs(
        ["Create a new Vectorstore", "Open a saved Vectorstore"]
    )
    with tab_new_vectorstore:
        # 1. Select documnets
        st.session_state.uploaded_file_list = st.file_uploader(
            label="**Select documents**",
            accept_multiple_files=True,
            type=(["pdf", "txt", "docx", "csv"]),
        )
        # 2. Process documents
        st.session_state.vector_store_name = st.text_input(
            label="**Documents will be loaded, embedded and ingested into a vectorstore (Chroma dB). Please provide a valid dB name.**",
            placeholder="Vectorstore name",
        )
        # 3. Add a button to process documnets and create a Chroma vectorstore

        st.button("Create Vectorstore", on_click=chain_RAG_blocks)
        try:
            if st.session_state.error_message != "":
                st.warning(st.session_state.error_message)
        except:
            pass

    with tab_open_vectorstore:
        st.write("请选择 `data/vector_stores/` 下已保存的向量库：")
        available_stores = list_local_vectorstores()
        if not available_stores:
            st.info(
                f"暂无可用向量库。请先创建向量库，或将已有 Chroma 目录放入 "
                f"`{LOCAL_VECTOR_STORE_DIR}`。"
            )
        else:
            store_options = {path.name: path for path in available_stores}
            default_index = 0
            if "moutai_2024_bge_m3" in store_options:
                default_index = list(store_options.keys()).index("moutai_2024_bge_m3")
            selected_name = st.selectbox(
                "Vectorstore",
                options=list(store_options.keys()),
                index=default_index,
            )
            if st.button("Load Vectorstore"):
                error_messages = []
                if not has_llm_api_key():
                    error_messages.append(
                        f"insert your {st.session_state.LLM_provider} API key"
                    )
                if (
                    st.session_state.retriever_type == list_retriever_types[2]
                    and not st.session_state.cohere_api_key
                ):
                    error_messages.append("insert your Cohere API key")

                if len(error_messages) == 1:
                    st.session_state.error_message = "Please " + error_messages[0] + "."
                    st.warning(st.session_state.error_message)
                elif len(error_messages) > 1:
                    st.session_state.error_message = (
                        "Please "
                        + ", ".join(error_messages[:-1])
                        + ", and "
                        + error_messages[-1]
                        + "."
                    )
                    st.warning(st.session_state.error_message)
                else:
                    with st.spinner("Loading vectorstore..."):
                        try:
                            load_vectorstore_from_path(store_options[selected_name])
                        except Exception as e:
                            st.error(e)

            st.write("")
            st.markdown("**Delete existing vectorstore**")
            st.write("如果向量库不再需要，可以删除它来释放空间。")
            if st.checkbox("确认删除选中向量库", key="confirm_delete_vectorstore"):
                if st.button("Delete Vectorstore", key="delete_vectorstore"):
                    success, error = delete_vectorstore(store_options[selected_name])
                    if success:
                        st.success(
                            f"向量库 '{selected_name}' 已删除。请刷新页面以更新列表。"
                        )
                        st.session_state.selected_vectorstore_name = ""
                        st.session_state.retriever = None
                        st.session_state.chain = None
                        st.session_state.memory = None
                    else:
                        st.error(
                            f"删除向量库 '{selected_name}' 失败：{error}"
                        )


####################################################################
#        Process documents and create vectorstor (Chroma dB)
####################################################################
def delte_temp_files():
    """delete files from the './data/tmp' folder"""
    files = glob.glob(TMP_DIR.as_posix() + "/*")
    for f in files:
        try:
            os.remove(f)
        except:
            pass


def langchain_document_loader():
    """
    Crete documnet loaders for PDF, TXT and CSV files.
    https://python.langchain.com/docs/modules/data_connection/document_loaders/file_directory
    """

    documents = []

    txt_loader = DirectoryLoader(
        TMP_DIR.as_posix(), glob="**/*.txt", loader_cls=TextLoader, show_progress=True
    )
    documents.extend(txt_loader.load())

    documents.extend(load_pdf_documents_with_mineru())

    csv_loader = DirectoryLoader(
        TMP_DIR.as_posix(), glob="**/*.csv", loader_cls=CSVLoader, show_progress=True,
        loader_kwargs={"encoding":"utf8"}
    )
    documents.extend(csv_loader.load())

    doc_loader = DirectoryLoader(
        TMP_DIR.as_posix(),
        glob="**/*.docx",
        loader_cls=Docx2txtLoader,
        show_progress=True,
    )
    documents.extend(doc_loader.load())
    return documents


def load_pdf_documents_with_mineru():
    """Load PDFs through MinerU and fall back to PyPDFLoader on parser failure."""
    documents = []
    for pdf_path in sorted(TMP_DIR.glob("**/*.pdf")):
        try:
            content_list_path = run_mineru(
                pdf_path=pdf_path,
                output_root=MINERU_OUTPUT_DIR.joinpath(pdf_path.stem),
                backend="pipeline",
            )
            file_hash = compute_file_sha256(pdf_path)
            parsed_doc = normalize_mineru_content(
                content_list_path=content_list_path,
                filename=pdf_path.name,
                doc_id=make_doc_id(pdf_path.name, file_hash),
                file_hash=file_hash,
                source_path=str(pdf_path),
            )
            write_normalized_json(
                parsed_doc,
                PARSED_JSON_DIR.joinpath(f"{parsed_doc['doc_id']}.json"),
            )
            documents.extend(build_chunks(parsed_doc))
        except Exception as error:
            st.warning(
                f"MinerU failed to parse {pdf_path.name}; falling back to PyPDFLoader. Error: {error}"
            )
            documents.extend(PyPDFLoader(pdf_path.as_posix()).load())
    return documents


def split_documents_to_chunks(documents):
    """Split documents to chunks using RecursiveCharacterTextSplitter."""

    pre_chunked_documents = [
        document
        for document in documents
        if document.metadata.get("pre_chunked") is True
    ]
    raw_documents = [
        document
        for document in documents
        if document.metadata.get("pre_chunked") is not True
    ]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=200)
    chunks = text_splitter.split_documents(raw_documents)
    return pre_chunked_documents + chunks


def select_embeddings_model():
    """Select embeddings models: OpenAIEmbeddings or GoogleGenerativeAIEmbeddings."""
    if st.session_state.get("embedding_model") == "Local BGE-M3":
        return BgeM3Embeddings(cache_dir=MODEL_CACHE_DIR)

    if st.session_state.LLM_provider == "DeepSeek":
        return BgeM3Embeddings(cache_dir=MODEL_CACHE_DIR)

    if st.session_state.LLM_provider == "OpenAI":
        embeddings = OpenAIEmbeddings(api_key=st.session_state.openai_api_key)

    if st.session_state.LLM_provider == "Google":
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", google_api_key=st.session_state.google_api_key
        )

    if st.session_state.LLM_provider == "HuggingFace":
        embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=st.session_state.hf_api_key, model_name="thenlper/gte-large"
        )

    return embeddings


def create_rewrite_llm():
    """Create a lightweight LLM used for finance query rewriting."""
    if st.session_state.LLM_provider == "DeepSeek":
        return create_deepseek_llm(
            model=st.session_state.selected_model,
            temperature=0.0,
        )
    if st.session_state.LLM_provider == "OpenAI":
        return ChatOpenAI(
            api_key=st.session_state.openai_api_key,
            model=st.session_state.selected_model,
            temperature=0.0,
        )
    if st.session_state.LLM_provider == "Google":
        return ChatGoogleGenerativeAI(
            google_api_key=st.session_state.google_api_key,
            model=st.session_state.selected_model,
            temperature=0.0,
            convert_system_message_to_human=True,
        )
    return None


def create_finance_hybrid_retriever(vector_store, enable_llm_rewrite=True):
    reranker = None
    if st.session_state.get("embedding_model") == "Local BGE-M3":
        reranker = BgeReranker(cache_dir=MODEL_CACHE_DIR)
    return FinanceHybridRetriever(
        vectorstore=vector_store,
        llm=create_rewrite_llm() if enable_llm_rewrite else None,
        reranker=reranker,
        vector_k=24,
        final_k=10,
        min_confidence_score=0.22,
    )


def create_routed_dual_index_retriever(dual_index, enable_llm_rewrite=True):
    reranker = None
    if st.session_state.get("embedding_model") == "Local BGE-M3":
        reranker = BgeReranker(cache_dir=MODEL_CACHE_DIR)
    return RoutedDualIndexRetriever(
        dual_index=dual_index,
        llm=create_rewrite_llm() if enable_llm_rewrite else None,
        reranker=reranker,
        vector_k=24,
        final_k=10,
        min_confidence_score=0.22,
    )


def create_retriever(
    vector_store=None,
    dual_index=None,
    embeddings=None,
    retriever_type="Contextual compression",
    base_retriever_search_type="semilarity",
    base_retriever_k=16,
    compression_retriever_k=20,
    cohere_api_key="",
    cohere_model="rerank-multilingual-v2.0",
    cohere_top_n=10,
):
    """
    create a retriever which can be a:
        - Vectorstore backed retriever: this is the base retriever.
        - Contextual compression retriever: We wrap the the base retriever in a ContextualCompressionRetriever.
            The compressor here is a Document Compressor Pipeline, which splits documents
            to smaller chunks, removes redundant documents, filters the top relevant documents,
            and reorder the documents so that the most relevant are at beginning / end of the list.
        - Cohere_reranker: CohereRerank endpoint is used to reorder the results based on relevance.

    Parameters:
        vector_store: Chroma vector database.
        embeddings: OpenAIEmbeddings or GoogleGenerativeAIEmbeddings.

        retriever_type (str): in [Vectorstore backed retriever,Contextual compression,Cohere reranker]. default = Cohere reranker

        base_retreiver_search_type: search_type in ["similarity", "mmr", "similarity_score_threshold"], default = similarity.
        base_retreiver_k: The most similar vectors are returned (default k = 16).

        compression_retriever_k: top k documents returned by the compression retriever, default = 20

        cohere_api_key: Cohere API key
        cohere_model (str): model used by Cohere, in ["rerank-multilingual-v2.0","rerank-english-v2.0"]
        cohere_top_n: top n documents returned bu Cohere, default = 10

    """

    if retriever_type == "Routed dual-index retriever":
        if dual_index is None:
            raise ValueError("Routed dual-index retriever requires a dual Chroma index.")
        return create_routed_dual_index_retriever(dual_index)

    if retriever_type == "Finance hybrid retriever":
        if vector_store is None:
            raise ValueError("Finance hybrid retriever requires a vector store.")
        return create_finance_hybrid_retriever(vector_store)

    base_retriever = Vectorstore_backed_retriever(
        vectorstore=vector_store,
        search_type=base_retriever_search_type,
        k=base_retriever_k,
        score_threshold=None,
    )

    if retriever_type == "Vectorstore backed retriever":
        return base_retriever

    elif retriever_type == "Contextual compression":
        compression_retriever = create_compression_retriever(
            embeddings=embeddings,
            base_retriever=base_retriever,
            k=compression_retriever_k,
        )
        return compression_retriever

    elif retriever_type == "Cohere reranker":
        cohere_retriever = CohereRerank_retriever(
            base_retriever=base_retriever,
            cohere_api_key=cohere_api_key,
            cohere_model=cohere_model,
            top_n=cohere_top_n,
        )
        return cohere_retriever
    else:
        pass


def Vectorstore_backed_retriever(
    vectorstore, search_type="similarity", k=4, score_threshold=None
):
    """create a vectorsore-backed retriever
    Parameters:
        search_type: Defines the type of search that the Retriever should perform.
            Can be "similarity" (default), "mmr", or "similarity_score_threshold"
        k: number of documents to return (Default: 4)
        score_threshold: Minimum relevance threshold for similarity_score_threshold (default=None)
    """
    search_kwargs = {}
    if k is not None:
        search_kwargs["k"] = k
    if score_threshold is not None:
        search_kwargs["score_threshold"] = score_threshold

    retriever = vectorstore.as_retriever(
        search_type=search_type, search_kwargs=search_kwargs
    )
    return retriever


def create_compression_retriever(
    embeddings, base_retriever, chunk_size=500, k=16, similarity_threshold=None
):
    """Build a ContextualCompressionRetriever.
    We wrap the the base_retriever (a Vectorstore-backed retriever) in a ContextualCompressionRetriever.
    The compressor here is a Document Compressor Pipeline, which splits documents
    to smaller chunks, removes redundant documents, filters the top relevant documents,
    and reorder the documents so that the most relevant are at beginning / end of the list.

    Parameters:
        embeddings: OpenAIEmbeddings or GoogleGenerativeAIEmbeddings.
        base_retriever: a Vectorstore-backed retriever.
        chunk_size (int): Docs will be splitted into smaller chunks using a CharacterTextSplitter with a default chunk_size of 500.
        k (int): top k relevant documents to the query are filtered using the EmbeddingsFilter. default =16.
        similarity_threshold : similarity_threshold of the  EmbeddingsFilter. default =None
    """

    # 1. splitting docs into smaller chunks
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=0, separator=". "
    )

    # 2. removing redundant documents
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)

    # 3. filtering based on relevance to the query
    relevant_filter = EmbeddingsFilter(
        embeddings=embeddings, k=k, similarity_threshold=similarity_threshold
    )

    # 4. Reorder the documents

    # Less relevant document will be at the middle of the list and more relevant elements at beginning / end.
    # Reference: https://python.langchain.com/docs/modules/data_connection/retrievers/long_context_reorder
    reordering = LongContextReorder()

    # 5. create compressor pipeline and retriever
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter, reordering]
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=base_retriever
    )

    return compression_retriever


def CohereRerank_retriever(
    base_retriever, cohere_api_key, cohere_model="rerank-multilingual-v2.0", top_n=10
):
    """Build a ContextualCompressionRetriever using CohereRerank endpoint to reorder the results
    based on relevance to the query.

    Parameters:
       base_retriever: a Vectorstore-backed retriever
       cohere_api_key: the Cohere API key
       cohere_model: the Cohere model, in ["rerank-multilingual-v2.0","rerank-english-v2.0"], default = "rerank-multilingual-v2.0"
       top_n: top n results returned by Cohere rerank. default = 10.
    """

    compressor = CohereRerank(
        cohere_api_key=cohere_api_key, model=cohere_model, top_n=top_n
    )

    retriever_Cohere = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )
    return retriever_Cohere


def chain_RAG_blocks():
    """The RAG system is composed of:
    - 1. Retrieval: includes document loaders, text splitter, vectorstore and retriever.
    - 2. Memory.
    - 3. Converstaional Retreival chain.
    """
    with st.spinner("Creating vectorstore..."):
        # Check inputs
        error_messages = []
        if not has_llm_api_key():
            error_messages.append(
                f"insert your {st.session_state.LLM_provider} API key"
            )

        if (
            st.session_state.retriever_type == list_retriever_types[2]
            and not st.session_state.cohere_api_key
        ):
            error_messages.append(f"insert your Cohere API key")
        if not st.session_state.uploaded_file_list:
            error_messages.append("select documents to upload")
        if st.session_state.vector_store_name == "":
            error_messages.append("provide a Vectorstore name")

        if len(error_messages) == 1:
            st.session_state.error_message = "Please " + error_messages[0] + "."
        elif len(error_messages) > 1:
            st.session_state.error_message = (
                "Please "
                + ", ".join(error_messages[:-1])
                + ", and "
                + error_messages[-1]
                + "."
            )
        else:
            st.session_state.error_message = ""
            try:
                # 1. Delete old temp files
                delte_temp_files()

                # 2. Upload selected documents to temp directory
                if st.session_state.uploaded_file_list is not None:
                    for uploaded_file in st.session_state.uploaded_file_list:
                        error_message = ""
                        try:
                            temp_file_path = os.path.join(
                                TMP_DIR.as_posix(), uploaded_file.name
                            )
                            with open(temp_file_path, "wb") as temp_file:
                                temp_file.write(uploaded_file.read())
                        except Exception as e:
                            error_message += e
                    if error_message != "":
                        st.warning(f"Errors: {error_message}")

                    # 3. Load documents with Langchain loaders
                    documents = langchain_document_loader()

                    # 4. Split documents to chunks
                    chunks = split_documents_to_chunks(documents)
                    # 5. Embeddings
                    embeddings = select_embeddings_model()

                    # 6. Create a vectorstore
                    persist_directory = (
                        LOCAL_VECTOR_STORE_DIR.as_posix()
                        + "/"
                        + st.session_state.vector_store_name
                    )

                    try:
                        st.session_state.vector_store = Chroma.from_documents(
                            documents=chunks,
                            embedding=embeddings,
                            persist_directory=persist_directory,
                        )
                        st.session_state.dual_index = build_dual_chroma_index(
                            chunks,
                            persist_dir=persist_directory,
                            embeddings=embeddings,
                            collection_name=st.session_state.vector_store_name,
                        )
                        st.info(
                            f"Vectorstore **{st.session_state.vector_store_name}** is created succussfully."
                        )

                        # 7. Create retriever
                        dual_index = (
                            st.session_state.dual_index
                            if st.session_state.retriever_type
                            == list_retriever_types[0]
                            else None
                        )
                        st.session_state.retriever = create_retriever(
                            vector_store=st.session_state.vector_store,
                            dual_index=dual_index,
                            embeddings=embeddings,
                            retriever_type=st.session_state.retriever_type,
                            base_retriever_search_type="similarity",
                            base_retriever_k=16,
                            compression_retriever_k=20,
                            cohere_api_key=st.session_state.cohere_api_key,
                            cohere_model="rerank-multilingual-v2.0",
                            cohere_top_n=10,
                        )

                        # 8. Create memory and ConversationalRetrievalChain
                        (
                            st.session_state.chain,
                            st.session_state.memory,
                        ) = create_ConversationalRetrievalChain(
                            retriever=st.session_state.retriever,
                            chain_type="stuff",
                            language=st.session_state.assistant_language,
                        )

                        # 9. Cclear chat_history
                        clear_chat_history()

                    except Exception as e:
                        st.error(e)

            except Exception as error:
                st.error(f"An error occurred: {error}")


####################################################################
#                       Create memory
####################################################################


def create_memory(model_name="gpt-3.5-turbo", memory_max_token=None):
    """Creates a ConversationSummaryBufferMemory for gpt-3.5-turbo
    Creates a ConversationBufferMemory for the other models"""

    if model_name == "gpt-3.5-turbo":
        if memory_max_token is None:
            memory_max_token = 1024  # max_tokens for 'gpt-3.5-turbo' = 4096
        memory = ConversationSummaryBufferMemory(
            max_token_limit=memory_max_token,
            llm=ChatOpenAI(
                model_name="gpt-3.5-turbo",
                openai_api_key=st.session_state.openai_api_key,
                temperature=0.1,
            ),
            return_messages=True,
            memory_key="chat_history",
            output_key="answer",
            input_key="question",
        )
    else:
        memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",
            output_key="answer",
            input_key="question",
        )
    return memory


####################################################################
#          Create ConversationalRetrievalChain with memory
####################################################################


def answer_template(language="english"):
    """Pass the standalone question along with the chat history and context
    to the `LLM` wihch will answer."""

    template = f"""Answer the question at the end, using only the following context (delimited by <context></context>).
If the context is empty or insufficient, clearly say that the annual report does not contain enough evidence and do not guess numbers.
When citing evidence, mention file name, page number, and section when available.
Your answer must be in the language at the end. 

<context>
{{chat_history}}

{{context}} 
</context>

Question: {{question}}

Language: {language}.
"""
    return template


def finance_condense_question_template():
    return """你是金融财报对话助手。请根据聊天历史，把追问改写成可独立检索年报的完整问题。
必须尽量保留公司名、年份、财务指标（如净利润、营业收入、现金流）。
使用中国A股年报中的常用表述，保持中文。

Chat History:
{chat_history}

Follow Up Input: {question}

Standalone question:"""


def create_ConversationalRetrievalChain(
    retriever,
    chain_type="stuff",
    language="english",
):
    """Create a ConversationalRetrievalChain.
    First, it passes the follow-up question along with the chat history to an LLM which rephrases
    the question and generates a standalone query.
    This query is then sent to the retriever, which fetches relevant documents (context)
    and passes them along with the standalone question and chat history to an LLM to answer.
    """

    # 1. Define the standalone_question prompt.
    # Pass the follow-up question along with the chat history to the `condense_question_llm`
    # which rephrases the question and generates a standalone question.

    use_finance_condense = isinstance(
        retriever, (FinanceHybridRetriever, RoutedDualIndexRetriever)
    )
    condense_question_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template=(
            finance_condense_question_template()
            if use_finance_condense
            else """Given the following conversation and a follow up question, 
rephrase the follow up question to be a standalone question, in its original language.\n\n
Chat History:\n{chat_history}\n
Follow Up Input: {question}\n
Standalone question:"""
        ),
    )

    # 2. Define the answer_prompt
    # Pass the standalone question + the chat history + the context (retrieved documents)
    # to the `LLM` wihch will answer

    answer_prompt = ChatPromptTemplate.from_template(answer_template(language=language))

    # 3. Add ConversationSummaryBufferMemory for gpt-3.5, and ConversationBufferMemory for the other models
    memory = create_memory(st.session_state.selected_model)

    # 4. Instantiate LLMs: standalone_query_generation_llm & response_generation_llm
    if st.session_state.LLM_provider == "DeepSeek":
        standalone_query_generation_llm = create_deepseek_llm(
            model=st.session_state.selected_model,
            temperature=0.1,
        )
        response_generation_llm = create_deepseek_llm(
            model=st.session_state.selected_model,
            temperature=st.session_state.temperature,
            top_p=st.session_state.top_p,
        )
    if st.session_state.LLM_provider == "OpenAI":
        standalone_query_generation_llm = ChatOpenAI(
            api_key=st.session_state.openai_api_key,
            model=st.session_state.selected_model,
            temperature=0.1,
        )
        response_generation_llm = ChatOpenAI(
            api_key=st.session_state.openai_api_key,
            model=st.session_state.selected_model,
            temperature=st.session_state.temperature,
            model_kwargs={"top_p": st.session_state.top_p},
        )
    if st.session_state.LLM_provider == "Google":
        standalone_query_generation_llm = ChatGoogleGenerativeAI(
            google_api_key=st.session_state.google_api_key,
            model=st.session_state.selected_model,
            temperature=0.1,
            convert_system_message_to_human=True,
        )
        response_generation_llm = ChatGoogleGenerativeAI(
            google_api_key=st.session_state.google_api_key,
            model=st.session_state.selected_model,
            temperature=st.session_state.temperature,
            top_p=st.session_state.top_p,
            convert_system_message_to_human=True,
        )

    if st.session_state.LLM_provider == "HuggingFace":
        standalone_query_generation_llm = HuggingFaceHub(
            repo_id=st.session_state.selected_model,
            huggingfacehub_api_token=st.session_state.hf_api_key,
            model_kwargs={
                "temperature": 0.1,
                "top_p": 0.95,
                "do_sample": True,
                "max_new_tokens": 1024,
            },
        )
        response_generation_llm = HuggingFaceHub(
            repo_id=st.session_state.selected_model,
            huggingfacehub_api_token=st.session_state.hf_api_key,
            model_kwargs={
                "temperature": st.session_state.temperature,
                "top_p": st.session_state.top_p,
                "do_sample": True,
                "max_new_tokens": 1024,
            },
        )

    # 5. Create the ConversationalRetrievalChain

    chain = ConversationalRetrievalChain.from_llm(
        condense_question_prompt=condense_question_prompt,
        combine_docs_chain_kwargs={"prompt": answer_prompt},
        condense_question_llm=standalone_query_generation_llm,
        llm=response_generation_llm,
        memory=memory,
        retriever=retriever,
        chain_type=chain_type,
        verbose=False,
        return_source_documents=True,
    )

    return chain, memory


def clear_chat_history():
    """clear chat history and memory."""
    # 1. re-initialize messages
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": dict_welcome_message[st.session_state.assistant_language],
        }
    ]
    # 2. Clear memory (history)
    try:
        st.session_state.memory.clear()
    except:
        pass


def get_response_from_LLM(prompt):
    """invoke the LLM, get response, and display results (answer and source documents)."""
    try:
        chain = st.session_state.get("chain")
        if chain is None:
            st.info(
                "请先在侧边栏 **Open a saved Vectorstore** 中选择向量库并点击 "
                "**Load Vectorstore**，或先 **Create Vectorstore** 完成入库。"
            )
            return

        retriever = st.session_state.get("retriever")

        # 1. Invoke LLM
        response = chain.invoke({"question": prompt})
        answer = response["answer"]

        if isinstance(
            retriever, (FinanceHybridRetriever, RoutedDualIndexRetriever)
        ) and (retriever.last_refused or not response.get("source_documents")):
            answer = (
                "未在已加载年报中找到足够相关的证据，系统已拒绝作答。"
                f"（检索置信度 {retriever.last_top_score:.2f}，"
                f"阈值 {retriever.min_confidence_score:.2f}）\n\n"
                "请补充年份、公司名或财务指标名称后重试，"
                "例如：「2024年归属于上市公司股东的净利润是多少？」"
            )
            response = {"answer": answer, "source_documents": []}

        if st.session_state.LLM_provider == "HuggingFace":
            answer = answer[answer.find("\nAnswer: ") + len("\nAnswer: ") :]

        # 2. Display results
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            # 2.1. Display anwser:
            st.markdown(answer)

            # 2.2. Display source documents:
            with st.expander("**Source documents**"):
                st.markdown(format_source_documents(response["source_documents"]))

    except Exception as e:
        st.warning(e)


def format_source_documents(source_documents):
    """Format retrieved sources with finance-report metadata for traceability."""
    formatted_sources = []
    for index, document in enumerate(source_documents, start=1):
        metadata = document.metadata or {}
        page = metadata.get("page")
        page_start = metadata.get("page_start")
        page_end = metadata.get("page_end")
        if page_start and page_end and page_start != page_end:
            page_text = f"{page_start}-{page_end}"
        elif page:
            page_text = str(page)
        else:
            page_text = ""

        preview = " ".join(document.page_content.split())
        if len(preview) > 900:
            preview = preview[:900] + "..."

        lines = [
            f"**Source {index}**",
            f"- File: `{metadata.get('source', '')}`",
        ]
        if page_text:
            lines.append(f"- Page: {page_text}")
        if metadata.get("section"):
            lines.append(f"- Section: {metadata.get('section')}")
        if metadata.get("block_type"):
            lines.append(f"- Block type: `{metadata.get('block_type')}`")
        if metadata.get("table_id"):
            lines.append(f"- Table ID: `{metadata.get('table_id')}`")
        if metadata.get("table_summary"):
            lines.append(f"- Table summary: {metadata.get('table_summary')}")
        if metadata.get("extracted_facts"):
            lines.append(f"- Extracted facts: {metadata.get('extracted_facts')}")
        if metadata.get("retrieval_score") is not None:
            lines.append(f"- Retrieval score: {metadata.get('retrieval_score')}")
        if metadata.get("rerank_score") is not None:
            lines.append(f"- Rerank score: {metadata.get('rerank_score')}")
        if metadata.get("route_intent"):
            lines.append(f"- Route intent: `{metadata.get('route_intent')}`")
        if metadata.get("index_name"):
            lines.append(f"- Index: `{metadata.get('index_name')}`")
        if metadata.get("categories"):
            lines.append(f"- Categories: `{metadata.get('categories')}`")
        if metadata.get("fact_id"):
            lines.append(f"- Fact ID: `{metadata.get('fact_id')}`")
        if metadata.get("asset_path"):
            lines.append(f"- Asset: `{metadata.get('asset_path')}`")
        lines.append(f"> {preview}")
        formatted_sources.append("\n".join(lines))

    return "\n\n".join(formatted_sources)


####################################################################
#                         Chatbot
####################################################################
def chatbot():
    init_session_state()
    sidebar_and_documentChooser()
    st.divider()
    col1, col2 = st.columns([7, 3])
    with col1:
        st.subheader("Chat with your data")
    with col2:
        st.button("Clear Chat History", on_click=clear_chat_history)

    if is_vectorstore_ready():
        loaded_name = st.session_state.get("selected_vectorstore_name") or st.session_state.get(
            "vector_store_name", ""
        )
        if loaded_name:
            st.caption(f"已加载向量库：`{loaded_name}`")
    else:
        st.warning(
            "尚未加载向量库。请先在侧边栏 **Open a saved Vectorstore** 选择向量库并点击 **Load Vectorstore**。"
        )

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": dict_welcome_message[st.session_state.assistant_language],
            }
        ]
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        if not has_llm_api_key():
            st.info(
                f"Please insert your {st.session_state.LLM_provider} API key to continue."
            )
            st.stop()
        if not is_vectorstore_ready():
            st.info(
                "请先在侧边栏加载向量库后再提问。"
            )
            st.stop()
        with st.spinner("Running..."):
            get_response_from_LLM(prompt=prompt)


if __name__ == "__main__":
    chatbot()
