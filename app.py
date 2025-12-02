import os
import gradio as gr
from main import (
    RAGSystem,
    RemoteLLMClient,
    REMOTE_VLLM_HOST,
    REMOTE_VLLM_API_KEY,
    REMOTE_MODEL_NAME,
    CHROMA_PERSIST_DIR,
)

# Initialize Global Components
print("Initializing RAG System and LLM Client...")
rag_system = RAGSystem()
llm_client = RemoteLLMClient(REMOTE_VLLM_HOST, REMOTE_VLLM_API_KEY, REMOTE_MODEL_NAME)

# Check if Knowledge Base exists
if not os.path.exists(CHROMA_PERSIST_DIR):
    print("ChromaDB not found. Building knowledge base...")
    rag_system.build_knowledge_base()
else:
    print("Loading existing ChromaDB...")
    rag_system.load_knowledge_base()


def process_pipeline(audio_path):
    """
    Gradio callback: Audio -> Phi-4-MM (ASR) -> RAG -> Phi-4-MM (QA)
    """
    if not audio_path:
        return "Please record or upload an audio file.", "", ""

    # 1. Audio Processing (ASR)
    print(f"[Gradio] Processing audio: {audio_path}")
    transcription = llm_client.process_audio_to_query(audio_path)

    if transcription.startswith("Error"):
        return transcription, "", ""

    # 2. RAG Retrieval
    context = rag_system.retrieve(transcription)

    # 3. Final Generation
    answer = llm_client.generate_response(context, transcription)

    return transcription, context, answer


# Define Gradio Interface
with gr.Blocks(title="Speech RAG Assistant") as demo:
    gr.Markdown("# Speech RAG Assistant")
    gr.Markdown("Speak into the microphone to query the course knowledge base.")

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources=["microphone", "upload"], type="filepath", label="Voice Input"
            )
            submit_btn = gr.Button("Submit", variant="primary")

        with gr.Column():
            query_output = gr.Textbox(label="Transcribed Text (ASR)", interactive=False)
            answer_output = gr.Textbox(label="Assistant Answer", interactive=False)
            with gr.Accordion("Retrieved Context", open=False):
                context_output = gr.Textbox(
                    label="Context", interactive=False, lines=10
                )

    submit_btn.click(
        fn=process_pipeline,
        inputs=[audio_input],
        outputs=[query_output, context_output, answer_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
