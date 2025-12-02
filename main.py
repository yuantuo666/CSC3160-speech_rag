import os
import time
import base64
from typing import List, Optional

# LangChain imports
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

# ==========================================
# Configuration
# ==========================================

# 1. Remote vLLM Server Configuration
# Please ensure this address matches the server address started by start_remote_vllm.sh
# If the server is on a remote machine, please replace localhost with the remote IP
REMOTE_VLLM_HOST = "https://gender-distribution-pencil-communicate.trycloudflare.com/v1"
REMOTE_VLLM_API_KEY = "EMPTY"  # vLLM does not require an API Key by default
REMOTE_MODEL_NAME = (
    "microsoft/Phi-4-multimodal-instruct"  # Must match MODEL_NAME in the startup script
)

# 2. Embedding Model Configuration
# Running locally, used for RAG retrieval
EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
CHROMA_PERSIST_DIR = "./chroma_db"
COURSEWARE_DIR = "./courseware"

# ==========================================
# Module Definitions
# ==========================================


class RAGSystem:
    """
    RAG Management System: Responsible for document loading, vectorization, storage, and retrieval
    """

    def __init__(self):
        print("[Init] Initializing Embedding Model...")
        # Load Qwen3-Embedding using HuggingFaceEmbeddings
        # device='cpu' (default) or 'cuda' (if local GPU is available)
        # For Qwen3-Embedding-0.6B, CPU speed is acceptable
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_ID,
            model_kwargs={"device": "cpu", "trust_remote_code": True},
        )
        self.vector_store = None

    def build_knowledge_base(self):
        """
        Load PDF documents from COURSEWARE_DIR and build the vector database
        """
        if not os.path.exists(COURSEWARE_DIR):
            os.makedirs(COURSEWARE_DIR)
            print(
                f"[RAG] Created directory {COURSEWARE_DIR}. Please put PDF files there."
            )
            return

        print(f"[RAG] Loading documents from {COURSEWARE_DIR}...")

        docs = []
        # Attempt to load Text (converted by pdftotext)
        try:
            loader_txt = DirectoryLoader(
                COURSEWARE_DIR, glob="**/*.txt", loader_cls=TextLoader
            )
            docs.extend(loader_txt.load())
        except Exception as e:
            print(f"[RAG] Warning loading Text files: {e}")

        if not docs:
            print("[RAG] No documents found (PDF or TXT) in courseware directory.")
            return

        print(f"[RAG] Splitting {len(docs)} documents...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs)

        print(f"[RAG] Creating ChromaDB at {CHROMA_PERSIST_DIR}...")
        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        print("[RAG] Knowledge base built successfully.")

    def load_knowledge_base(self):
        """
        Load existing vector database
        """
        if os.path.exists(CHROMA_PERSIST_DIR):
            print(f"[RAG] Loading existing ChromaDB from {CHROMA_PERSIST_DIR}...")
            self.vector_store = Chroma(
                persist_directory=CHROMA_PERSIST_DIR, embedding_function=self.embeddings
            )
        else:
            print("[RAG] ChromaDB not found. Please run build_knowledge_base() first.")

    def retrieve(self, query: str, k: int = 3) -> str:
        """
        Retrieve relevant document fragments based on Query
        """
        if not self.vector_store:
            return ""

        print(f"[RAG] Searching for: {query}")
        docs = self.vector_store.similarity_search(query, k=k)

        # Concatenate retrieved context
        context = "\n\n".join([f"[Document Content]: {d.page_content}" for d in docs])
        return context


class RemoteLLMClient:
    """
    Client for interacting with the remote vLLM server (OpenAI Compatible)
    """

    def __init__(self, base_url, api_key, model_name):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model_name = model_name

    def process_audio_to_query(self, audio_path: str) -> str:
        """
        Sends audio to Phi-4-MM to perform ASR (Speech-to-Text).
        """
        print(f"[LLM] Processing audio: {audio_path}")

        # Check file existence
        if not os.path.exists(audio_path):
            return "Error: Audio file not found."

        # Encode audio to base64
        with open(audio_path, "rb") as audio_file:
            encoded_audio = base64.b64encode(audio_file.read()).decode("utf-8")

        # Construct Multimodal Prompt for ASR
        system_prompt = "You are an accurate ASR system. Transcribe the user's speech verbatim. Do not add any commentary."

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe this audio."},
                    {
                        "type": "audio_url",
                        "audio_url": {"url": f"data:audio/wav;base64,{encoded_audio}"},
                    },
                ],
            },
        ]

        print("[LLM] Sending audio to remote vLLM server for ASR...")
        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0,  # Use 0 for deterministic transcription
                frequency_penalty=1.5,  # Discourage repetition in ASR
                max_tokens=200,
            )
            content = response.choices[0].message.content
            print(
                f"[LLM] Transcription: '{content}' (Latency: {time.time() - start_time:.2f}s)"
            )
            return content if content else ""
        except Exception as e:
            print(f"[LLM] Error processing audio: {e}")
            return f"Error: {e}"

    def generate_response(self, context: str, question: str) -> str:
        """
        Generate final answer based on context and question.
        """
        system_prompt = "You are a helpful teaching assistant. Answer the question based on the provided context."
        user_prompt = f"""Context:
{context}

Question:
{question}

Answer:"""

        print("[LLM] Sending text context to remote vLLM server for generation...")
        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=500,
            )
            content = response.choices[0].message.content
            print(f"[LLM] Response received (Latency: {time.time() - start_time:.2f}s)")
            return content if content else ""
        except Exception as e:
            print(f"[LLM] Error communicating with remote server: {e}")
            return "Sorry, I encountered an error connecting to the model server."


# ==========================================
# Main Pipeline
# ==========================================


def main():
    # 1. Initialize components
    rag = RAGSystem()
    llm_client = RemoteLLMClient(
        REMOTE_VLLM_HOST, REMOTE_VLLM_API_KEY, REMOTE_MODEL_NAME
    )

    # 2. Check if knowledge base needs to be built
    if not os.path.exists(CHROMA_PERSIST_DIR):
        print(
            "\n[Setup] ChromaDB not found. Attempting to build from './courseware'..."
        )
        rag.build_knowledge_base()
    else:
        rag.load_knowledge_base()

    print("\n" + "=" * 50)
    print("Speech RAG System Ready! (Audio Input Mode - ASR Only)")
    print(f"Remote Server: {REMOTE_VLLM_HOST}")
    print("Type 'exit' or 'quit' to stop.")
    print("Please provide the path to your audio file (e.g., query.wav).")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("Student (Audio Path): ").strip()
        except KeyboardInterrupt:
            break

        if user_input.lower() in ["exit", "quit"]:
            break

        if not user_input:
            continue

        if not os.path.exists(user_input):
            print(f"File not found: {user_input}")
            continue

        # 3. Audio Processing (ASR)
        transcription = llm_client.process_audio_to_query(user_input)
        if transcription.startswith("Error"):
            print(transcription)
            continue

        # 4. RAG Retrieval
        context = rag.retrieve(transcription)

        # 5. Generate Response (Remote vLLM)
        response_text = llm_client.generate_response(context, transcription)
        print(f"\nAssistant: {response_text}\n")


if __name__ == "__main__":
    main()
