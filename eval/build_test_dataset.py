import os
import json
import time
import wave
import random
from pathlib import Path
from gtts import gTTS
from google import genai
from google.genai import types

# Configuration
COURSEWARE_DIR = "./courseware"
TEST_DATASET_DIR = "./test_dataset"
QA_FILE = os.path.join(TEST_DATASET_DIR, "qa_pairs.json")
AUDIO_DIR = os.path.join(TEST_DATASET_DIR, "audio")
NUM_GENERATION = 100  # Total questions to generate
RANDOM_SEED = 42

VOICE_NAMES = [
    "Zephyr",
    "Puck",
    "Charon",
    "Kore",
    "Fenrir",
    "Leda",
    "Orus",
    "Aoede",
    "Callirrhoe",
    "Autonoe",
    "Enceladus",
    "Iapetus",
    "Umbriel",
    "Algieba",
    "Despina",
    "Erinome",
    "Algenib",
    "Rasalgethi",
    "Laomedeia",
    "Achernar",
    "Alnilam",
    "Schedar",
    "Gacrux",
    "Pulcherrima",
    "Achird",
    "Zubenelgenubi",
    "Vindemiatrix",
    "Sadachbia",
    "Sadaltager",
    "Sulafat",
]

# Proxy Configuration
os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"

# Ensure directories exist
os.makedirs(TEST_DATASET_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)


def load_courseware_content():
    """Reads all text files from the courseware directory and returns them as a list."""
    chunks = []
    print(f"Loading courseware from {COURSEWARE_DIR}...")
    files = list(Path(COURSEWARE_DIR).glob("**/*.txt"))
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                if len(text.strip()) > 500:  # Ignore very short files
                    chunks.append(f"--- File: {file_path.name} ---\n{text}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    return chunks


def generate_qa_pairs(client, chunks, num_total=50):
    """Generates Multiple Choice Q&A pairs iteratively until num_total is reached."""
    print(f"Generating {num_total} Q&A pairs...")
    all_qa_pairs = []

    # We generate in batches
    batch_size = 5

    while len(all_qa_pairs) < num_total:
        # Pick random chunks (e.g., 2 files) to provide context
        selected_chunks = random.sample(chunks, min(len(chunks), 2))
        context = "\n\n".join(selected_chunks)

        # Calculate how many more we need, max batch_size
        needed = num_total - len(all_qa_pairs)
        current_batch = min(needed, batch_size)

        print(
            f"Requesting {current_batch} questions (Progress: {len(all_qa_pairs)}/{num_total})..."
        )

        prompt = f"""
        You are a teaching assistant. Based on the provided course materials, generate {current_batch} multiple-choice questions (MCQs).
        
        Each question must have:
        1. A clear question stem.
        2. 4 options (A, B, C, D). 
           IMPORTANT: The options must be concise, ideally 2-5 words each.
        3. A single correct answer (the text of the correct option).

        Output strictly in valid JSON format as a list of objects with keys: 
        "question", "options" (list of strings), "correct_answer" (string).
        
        Do not include explanations or keywords.
        
        Course Material:
        {context[:80000]} 
        """

        try:
            response = client.models.generate_content(
                model="gemini-flash-latest",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                ),
            )

            text_response = response.text.strip()
            if text_response.startswith("```json"):
                text_response = text_response[7:]
            if text_response.startswith("```"):
                text_response = text_response[3:]
            if text_response.endswith("```"):
                text_response = text_response[:-3]

            batch_pairs = json.loads(text_response)

            # Validate and add
            if isinstance(batch_pairs, list):
                all_qa_pairs.extend(batch_pairs)
            else:
                print("Unexpected JSON format, skipping batch.")

            # Sleep briefly to avoid rate limits
            time.sleep(2)

        except Exception as e:
            print(f"Error generating batch: {e}")
            time.sleep(5)  # Wait longer on error

    return all_qa_pairs[:num_total]


def generate_audio(client, text, output_path, voice_name="Kore"):
    """Generates audio using Gemini TTS model (google-genai SDK), falling back to gTTS."""
    print(f"Generating audio for: {text[:30]}... using voice {voice_name}")

    # Try Gemini TTS first
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice_name,
                        )
                    )
                ),
            ),
        )

        # Extract audio data
        if response.candidates and response.candidates[0].content.parts:
            part = response.candidates[0].content.parts[0]
            if part.inline_data and part.inline_data.data:
                data = part.inline_data.data
                # Save as WAV using user provided logic
                with wave.open(output_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(24000)
                    wf.writeframes(data)

                print(f"Gemini Audio saved to {output_path}")
                return True

        print("Gemini TTS response did not contain expected audio data.")

    except Exception as e:
        print(f"Gemini TTS failed: {e}. Falling back to gTTS...")

    try:
        # Fallback to gTTS
        tts = gTTS(text=text, lang="en")
        tts.save(output_path)
        print(f"gTTS Audio saved to {output_path}")
        return True

    except Exception as e:
        print(f"Error generating audio: {e}")
        return False


def main():
    random.seed(RANDOM_SEED)
    api_key = input("Please enter your Google AI Studio API Key: ").strip()
    if not api_key:
        print("API Key is required.")
        return

    # Set env vars for new SDK
    os.environ["GOOGLE_API_KEY"] = api_key
    os.environ["GEMINI_API_KEY"] = api_key  # User snippet mentioned this

    # Initialize Client
    try:
        client = genai.Client()
    except Exception as e:
        print(f"Failed to initialize Google GenAI Client: {e}")
        return

    # 1. Load Content
    course_chunks = load_courseware_content()
    if not course_chunks:
        print("No course content found. Please check 'courseware' directory.")
        return

    # 2. Generate QA
    existing_qa_pairs = []
    if os.path.exists(QA_FILE):
        try:
            with open(QA_FILE, "r", encoding="utf-8") as f:
                existing_qa_pairs = json.load(f)
            print(f"Loaded {len(existing_qa_pairs)} existing Q&A pairs.")
        except Exception as e:
            print(f"Error loading existing Q&A pairs: {e}")

    num_needed = NUM_GENERATION - len(existing_qa_pairs)

    if num_needed > 0:
        print(f"Need to generate {num_needed} more Q&A pairs.")
        new_qa_pairs = generate_qa_pairs(client, course_chunks, num_total=num_needed)

        if new_qa_pairs:
            qa_pairs = existing_qa_pairs + new_qa_pairs
            with open(QA_FILE, "w", encoding="utf-8") as f:
                json.dump(qa_pairs, f, indent=4)
            print(f"Saved {len(qa_pairs)} Q&A pairs to {QA_FILE}")
        else:
            print("Failed to generate new Q&A pairs.")
            qa_pairs = existing_qa_pairs
    else:
        print("Enough Q&A pairs already exist. Skipping generation.")
        qa_pairs = existing_qa_pairs[:NUM_GENERATION]

    if not qa_pairs:
        print("No Q&A pairs available.")
        return

    # 3. Generate Audio for Questions
    print("Generating audio files for questions...")

    for i, item in enumerate(qa_pairs):
        # Format text to include question and options
        text_to_speak = f"{item['question']}\n"
        labels = ["A", "B", "C", "D"]
        for idx, option in enumerate(item.get("options", [])):
            if idx < 4:
                if option[-1] not in [".", "!", "?"]:
                    option += "."
                text_to_speak += f"Option {labels[idx]}: {option} "

        file_name = f"question_{i+1}.wav"
        output_path = os.path.join(AUDIO_DIR, file_name)

        # Pick a random voice if not already assigned (for reproducibility if re-running)
        if "voice_name" not in item:
            item["voice_name"] = random.choice(VOICE_NAMES)

        # Generate audio
        if os.path.exists(output_path):
            print(f"Audio {file_name} already exists. Skipping.")
        else:
            generate_audio(
                client, text_to_speak, output_path, voice_name=item["voice_name"]
            )
            # Respect rate limits
            time.sleep(1)

        # Add audio path to the item for reference
        item["audio_path"] = output_path
        item["text_to_speak"] = text_to_speak

    # Save updated QA pairs with audio paths
    with open(QA_FILE, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=4)
    print("Dataset generation complete.")


if __name__ == "__main__":
    main()
