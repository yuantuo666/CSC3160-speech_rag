import json
import re
import os
import sys
import time
from xfinder.eval import Evaluator
from main import (
    RAGSystem,
    RemoteLLMClient,
    REMOTE_VLLM_HOST,
    REMOTE_VLLM_API_KEY,
    REMOTE_MODEL_NAME,
    CHROMA_PERSIST_DIR,
)

# Configuration
IS_USING_RAG = False
INPUT_MODE = "audio"  # "audio" or "text"


def load_dataset(path):
    with open(path, "r") as f:
        return json.load(f)


def evaluate():
    # Initialize systems
    print("Initializing RAG System...")
    rag = RAGSystem()

    # Ensure Knowledge Base is loaded
    if os.path.exists(CHROMA_PERSIST_DIR):
        rag.load_knowledge_base()
    else:
        print("Knowledge base not found. Please run main.py to build it first.")
        # Alternatively, we could build it here, but usually evaluation assumes setup is done.
        # Let's try to build if not exists, similar to main.py
        rag.build_knowledge_base()

    print("Initializing LLM Client...")
    llm_client = RemoteLLMClient(
        REMOTE_VLLM_HOST, REMOTE_VLLM_API_KEY, REMOTE_MODEL_NAME
    )

    print("Initializing xFinder Evaluator...")
    evaluator = Evaluator(
        model_name="xFinder-qwen1505",  # Model name
        inference_mode="local",  # Inference mode, 'local' or 'api'
        model_path_or_url="IAAR-Shanghai/xFinder-qwen1505",  # Anonymized model path or URL
    )

    dataset_path = "./test_dataset/qa_pairs.json"
    qa_pairs = load_dataset(dataset_path)

    correct_count = 0
    total_count = len(qa_pairs)

    results = []

    print(f"Starting evaluation on {total_count} items...")
    print(f"Mode: {'RAG + ASR' if IS_USING_RAG else 'Text-only (No RAG)'}")

    total_start_time = time.time()

    for i, item in enumerate(qa_pairs):
        print(f"\nProcessing Question {i+1}/{total_count}")

        question_start_time = time.time()

        audio_path = item["audio_path"]
        correct_answer = item["correct_answer"]
        options = item["options"]

        question = item["question"]
        query_text = ""
        context = ""

        if INPUT_MODE == "audio":
            # 1. Transcribe
            transcription = llm_client.process_audio_to_query(audio_path)

            if not transcription or transcription.startswith("Error"):
                print("Skipping due to transcription error")
                results.append(
                    {
                        "question_id": i + 1,
                        "status": "failed",
                        "reason": "transcription_error",
                    }
                )
                continue

            query_text = transcription
        else:
            # Direct text input from dataset
            query_text = item["text_to_speak"]
            print(f"Question (Text): {query_text}")

        if IS_USING_RAG:
            # 2. Retrieve
            context = rag.retrieve(query_text)
        else:
            # No retrieval, context is empty
            context = ""

        # 3. Generate Answer
        # Note: main.py generate_response doesn't take options.
        # The model generates a free-text answer.
        response = llm_client.generate_response(context, query_text)
        print(f"Model Response: {response}")

        # 4. Extract and Evaluate using xFinder
        # Convert options to alphabet format: [['A', 'opt1'], ['B', 'opt2'], ...]
        formatted_options = [[chr(65 + idx), opt] for idx, opt in enumerate(options)]
        standard_answer_range = json.dumps(formatted_options)
        key_answer_type = "alphabet_option"

        # Find the correct label (A, B, C...)
        correct_label = None
        for label, opt_text in formatted_options:
            if opt_text == correct_answer:
                correct_label = label
                break

        if not correct_label:
            print(
                f"Warning: Correct answer '{correct_answer}' not found in options. Skipping xFinder evaluation."
            )
            eval_result = "Error: Correct answer not found in options"
            is_correct = False
        else:
            try:
                eval_result = evaluator.evaluate_single_example(
                    question,
                    response,
                    standard_answer_range,
                    key_answer_type,
                    correct_label,
                )
                print(f"Evaluation Result: {eval_result}")

                # Check the result string directly
                is_correct = eval_result == "Correct"

                if is_correct:
                    correct_count += 1
                    print("Result: CORRECT")
                else:
                    print(f"Result: INCORRECT (Expected: {correct_label})")

            except Exception as e:
                print(f"Error during xFinder evaluation: {e}")
                is_correct = False
                eval_result = str(e)

        question_duration = time.time() - question_start_time
        print(f"Time taken: {question_duration:.2f}s")

        results.append(
            {
                "question_id": i + 1,
                "query_text": query_text,
                "response": response,
                "eval_result": eval_result,
                "correct_answer": correct_answer,
                "is_correct": is_correct,
                "duration_seconds": question_duration,
                "mode": "RAG" if IS_USING_RAG else "No-RAG",
            }
        )

    total_duration = time.time() - total_start_time
    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    print("\n" + "=" * 50)
    print(f"Evaluation Complete")
    print(f"Total Time: {total_duration:.2f}s")
    print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{total_count})")
    print("=" * 50)

    rag_name = "rag" if IS_USING_RAG else "no_rag"
    save_name = f"eval/evaluation_results_{rag_name}_{INPUT_MODE}.json"

    # Save detailed results
    with open(save_name, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Detailed results saved to {save_name}")


if __name__ == "__main__":
    evaluate()
