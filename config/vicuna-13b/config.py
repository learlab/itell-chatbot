from functools import lru_cache
from pathlib import Path
from typing import Optional
import os
import pickle

import torch
import faiss
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from langchain import HuggingFacePipeline
from langchain.llms import BaseLLM
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.actions.actions import ActionResult
from nemoguardrails.llm.helpers import get_llm_instance_wrapper
from nemoguardrails.llm.providers import register_llm_provider

def get_model_config(config: RailsConfig, type: str):
    """Quick helper to return the config for a specific model type."""
    for model_config in config.models:
        if model_config.type == type:
            return model_config


@lru_cache
def get_vicuna():
    """Loads the Vicuna 7B LLM."""
    repo_id = "lmsys/vicuna-7b-v1.5"
    # repo_id = "TheBloke/vicuna-13B-v1.5-GGUF"

    # model_params_gguf = {
    #     "model_file": "vicuna-13b-v1.5.Q4_K_M.gguf",
    #     "model_type": "llama",
    #     "gpu_layers": 50,
    #     "hf": True,
    # }

    model_params = {
        "device_map": "auto",
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        ),
        "low_cpu_mem_usage": True,
        "do_sample": True,
    }

    pipeline_params = {"max_new_tokens": 128, "do_sample": True, "temperature": 0.2}

    model = AutoModelForCausalLM.from_pretrained(repo_id, **model_params)
    tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=False)

    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, **pipeline_params
    )

    llm = HuggingFacePipeline(pipeline=pipe, model_kwargs=pipeline_params)

    return llm


def make_faiss_gpu(data_path: Path, out_path: Path, embeddings):
    # Here we process the txt files under the data_path folder.
    ps = list(data_path.glob("**/*.txt"))
    data = []
    sources = []
    for p in ps:
        with open(p) as f:
            data.append(f.read())
        sources.append(p)

    # We do this due to the context limits of the LLMs.
    # Here we split the documents, as needed, into smaller chunks.
    # We do this due to the context limits of the LLMs.
    text_splitter = CharacterTextSplitter(chunk_size=200, separator="\n")
    docs = []
    metadatas = []
    for i, d in enumerate(data):
        splits = text_splitter.split_text(d)
        docs.extend(splits)
        metadatas.extend([{"source": sources[i]}] * len(splits))

    # Here we create a vector store from the documents and save it to disk.
    store = FAISS.from_texts(docs, embeddings, metadatas=metadatas)
    out_path.mkdir(exist_ok=True)
    faiss.write_index(store.index, str(out_path / "docs.index"))
    store.index = None
    with open(out_path / "faiss_store.pkl", "wb") as f:
        pickle.dump(store, f)
    return store


def get_vector_db(model_name: str, data_path: str, persist_path: str):
    """Creates a vector DB for a given data path.

    If it's the first time, the index will be persisted at the given path.
    Otherwise, it will be loaded directly (if the `persist_path` exists).
    """
    data_path = Path(data_path)
    persist_path = Path(persist_path)
    model_kwargs = {"device": "cuda"}

    hf_embedding = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs
    )

    if os.path.exists(persist_path):
        with open(persist_path / "faiss_store.pkl", "rb") as f:
            vectordb = pickle.load(f)
    else:
        vectordb = make_faiss_gpu(data_path, persist_path, hf_embedding)

    vectordb.index = faiss.read_index(str(persist_path / "docs.index"))

    return vectordb


def init_vectordb_model(config: RailsConfig):
    global vectordb
    model_config = get_model_config(config, "vectordb")
    vectordb = get_vector_db(
        model_name=model_config.model,
        data_path=config.custom_data["kb_data_path"],
        persist_path=model_config.parameters.get("persist_path"),
    )

    register_llm_provider("faiss", vectordb)


@action(is_system_action=True)
async def retrieve_relevant_chunks(
    context: Optional[dict] = None,
    llm: Optional[BaseLLM] = None,
):
    """Retrieve relevant chunks from the knowledge base and add them to the context."""
    user_message = context.get("last_user_message")

    retriever = vectordb.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    out = qa_chain(user_message)
    result = out["result"]
    citing_text = out["source_documents"][0].page_content
    source_ref = str(out["source_documents"][0].metadata["source"])

    context_updates = {
        "relevant_chunks": f"""
            Question: {user_message}
            Answer: {result},
            Citing : {citing_text},
            Source : {source_ref}
    """
    }

    return ActionResult(
        return_value=context_updates["relevant_chunks"],
        context_updates=context_updates,
    )


def init_main_llm():
    HFPipelineVicuna = get_llm_instance_wrapper(
        llm_instance=get_vicuna(), llm_type="hf_pipeline_vicuna"
    )
    
    register_llm_provider("hf_pipeline_vicuna", HFPipelineVicuna)


def init(llm_rails: LLMRails):
    config = llm_rails.config

    # Initialize the models
    init_main_llm()
    init_vectordb_model(config)

    # Register the custom `retrieve_relevant_chunks` for custom retrieval
    llm_rails.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")