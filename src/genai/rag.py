"""Retrieval-Augmented Generation (RAG) over the Telco Churn dataset.

This module builds a question-answering system over the Telco Churn data by
combining:

- A **vector store** (ChromaDB) that indexes document chunks describing the
  dataset (data dictionary, feature summaries and key statistics derived from
  the raw data).
- A **HuggingFace embedding model** (``sentence-transformers``) used to embed
  the chunks and the users' questions into the same vector space.
- A **retriever** (LangChain) that fetches the most relevant chunks for a
  question.
- A **generator** that produces an answer grounded on the retrieved context.
  The generator uses a locally-run instruction-tuned HuggingFace model
  (``TinyLlama-Chat`` by default) so that the system works without any
  external API key.

The persistence directory is ``models/rag_chroma/`` which is gitignored as
part of ``models/`` artifacts.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline

from src.data.make_dataset import load_raw_data

PROJECT_DIR = Path(__file__).resolve().parents[2]
CHROMA_DIR = PROJECT_DIR / "models" / "rag_chroma"
DATA_DICT_PATH = PROJECT_DIR / "references" / "data-dictionary.md"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 80
COLLECTION_NAME = "telco_churn_docs"

SYSTEM_PROMPT = """Eres un asistente que responde preguntas sobre el dataset \
"Telco Customer Churn" de telecomunicaciones. Usa SOLO el contexto proporcionado \
para responder. Si el contexto no contiene la respuesta, indícalo con \
"El contexto disponible no contiene esta información". Responde en español, de \
forma concisa y clara."""

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "Contexto:\n{context}\n\nPregunta: {question}\n\nRespuesta:",
        ),
    ]
)


def build_dataset_context() -> List[Document]:
    """Build a list of documents describing the Telco Churn dataset.

    The context combines the data dictionary with aggregate statistics and
    related summary sentences derived from the raw data, so that open-ended
    questions about features and the target variable can be answered.

    Returns:
        List[Document]: Ready-to-embed document chunks.
    """
    dictionary_text = DATA_DICT_PATH.read_text(encoding="utf-8")
    df = load_raw_data()

    stats_lines = [
        f"El dataset tiene {len(df)} clientes y {df.shape[1]} columnas.",
        f"Distribución de la variable objetivo Churn: "
        f"{df['Churn'].value_counts().to_dict()}.",
        "La tasa de abandono (churn) es de aproximadamente "
        f"{(df['Churn'] == 'Yes').mean() * 100:.1f}%.",
        f"La antigüedad (tenure) media es de {df['tenure'].mean():.1f} meses.",
        f"El cargo mensual medio es de {df['MonthlyCharges'].mean():.2f} dólares.",
        "Los clientes con contrato mes a mes (month-to-month) tienden a "
        "abandonar más que los de contrato anual o bienal.",
        "El género está balanceado entre 'Male' y 'Female'.",
        "La mayoría de los clientes contratan servicio telefónico y de "
        f"internet (fibras ópticas o DSL).",
    ]

    feature_stats = []
    for col in ["Contract", "InternetService", "PaymentMethod"]:
        top = df[col].value_counts().head(3)
        feature_stats.append(
            f"Distribución de '{col}': {top.to_dict()}."
        )

    raw_text = "\n".join(
        [dictionary_text, "\n## Estadísticas clave\n"] + stats_lines + feature_stats
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents([Document(page_content=raw_text)])


def build_vector_store(
    docs: Optional[List[Document]] = None,
    persist_directory: Optional[Path] = None,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """Create or load the ChromaDB vector store for the Telco Churn docs.

    If the vector store already exists at the given directory, it is loaded
    directly (skipping re-embedding). Otherwise the documents are embedded and
    persisted.

    Args:
        docs: Documents to index. Only used when the store does not yet exist.
        persist_directory: Directory of the Chroma DB. Defaults to
            ``models/rag_chroma/``.
        collection_name: Name of the Chroma collection.

    Returns:
        Chroma: The configured vector store.
    """
    persist_directory = persist_directory or CHROMA_DIR
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )
    if docs is not None and store._collection.count() == 0:
        store.add_documents(docs)
    return store


def build_generator(temperature: float = 0.1) -> "HuggingFacePipeline":
    """Build the local text-generation pipeline used as the generator.

    A lightweight local instruction-tuned model (``TinyLlama`` by default) is
    used so that the whole RAG system runs offline without an API key. The
    model applies its chat template so that the system/user prompt is followed
    properly.

    Returns:
        HuggingFacePipeline: LangChain wrapper around the local LLM.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    model_name = os.getenv("RAG_GENERATION_MODEL", GENERATION_MODEL)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32
    )
    generation = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1,
        max_new_tokens=200,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        repetition_penalty=1.1,
    )
    return HuggingFacePipeline(pipeline=generation)


def _strip_prompt_echo(text: str) -> str:
    """Extract the generated answer, removing any echoed prompt prefix.

    Local causal language models return ``generated_text`` that includes the
    full input prompt. This helper keeps only the text following the final
    assistant marker so the user sees just the model's answer.

    Args:
        text: Raw ``generated_text`` string returned by the model.

    Returns:
        str: Cleaned answer text.
    """
    marker = "<|assistant|>"
    idx = text.rfind(marker)
    if idx != -1:
        text = text[idx + len(marker):]
    return text.strip()


def _format_chat(messages) -> str:
    """Format a list of chat messages for a TinyLlama-style local model.

    Builds the Llama-2 chat template string so the generation model treats the
    retrieved context as system instructions and produces a grounded answer.

    Args:
        messages: List of LangChain ``BaseMessage`` objects.

    Returns:
        str: Chat-formatted prompt ready for the local LLM.
    """
    mapping = {"system": "<|system|>", "human": "<|user|>", "ai": "<|assistant|>"}
    blocks = []
    for msg in messages:
        role = mapping.get(msg.type, "<|user|>")
        blocks.append(f"{role}\n{msg.content}</s>")
    blocks.append("<|assistant|>")
    return "\n".join(blocks)


def build_chain(
    store: Chroma,
    generator: Optional[object] = None,
    k: int = 3,
    max_context_chars: int = 1500,
):
    """Assemble the retrieval + generation chain.

    Args:
        store: Ready Chroma vector store.
        generator: LangChain-compatible LLM to generate answers. Defaults to a
            local HuggingFace pipeline.
        k: Number of retrieved chunks passed as context.
        max_context_chars: Maximum number of characters of context fed to the
            generator (keeps prompts within the local model's token budget).

    Returns:
        A callable ``answer(question: str) -> Dict``.
    """
    if generator is None:
        generator = build_generator()

    retriever = store.as_retriever(search_kwargs={"k": k})

    def answer(question: str) -> Dict[str, object]:
        """Retrieve context and generate a grounded answer.

        Args:
            question: User question in natural language.

        Returns:
            Dict with keys ``question``, ``answer``, ``sources`` (list of
            retrieved documents) and ``context`` (joined retrieved chunks).
        """
        docs = retriever.invoke(question)

        if not docs:
            return {
                "question": question,
                "answer": "No se encontró contexto suficiente.",
                "sources": [],
                "context": "",
            }

        context = "\n\n".join(d.page_content for d in docs)
        context = context[:max_context_chars]
        messages = QA_PROMPT.format_messages(context=context, question=question)
        prompt = _format_chat(messages)

        if hasattr(generator, "invoke"):
            answer_text = generator.invoke(prompt)
        else:
            answer_text = generator(prompt)

        answer_text = _strip_prompt_echo(answer_text)

        return {
            "question": question,
            "answer": answer_text,
            "sources": [d.page_content for d in docs],
            "context": context,
        }

    return answer


def setup_rag(rebuild: bool = False):
    """Build (and optionally rebuild) the end-to-end RAG pipeline.

    Args:
        rebuild: If True, clears the existing Chroma store and re-indexes the
            documents.

    Returns:
        Callable ``answer(question)`` ready to use.
    """
    if rebuild and CHROMA_DIR.exists():
        import shutil

        shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    docs = build_dataset_context() if rebuild or not CHROMA_DIR.exists() else None
    store = build_vector_store(docs=docs)
    return build_chain(store)


def main() -> None:
    """Demo entry point: ask a few sample questions through the CLI."""
    answer = setup_rag(rebuild=False)

    questions = [
        "¿Qué significa la variable Churn?",
        "¿Qué variables describen los servicios contratados?",
        "¿Cuál es la tasa de abandono del dataset?",
    ]

    for q in questions:
        result = answer(q)
        print(f"\nPregunta: {q}")
        print(f"Respuesta: {result['answer']}")


if __name__ == "__main__":
    main()
