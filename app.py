import uvicorn
import torch
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from contextlib import asynccontextmanager
from typing import Dict, Any, List

# --- Model Loading & Lifespan ---
# (This part is the same as your code... no changes needed)
models: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronous context manager to load models on startup and clear them on shutdown.
    This is the recommended way to manage models in production.
    """
    print("--- Loading models... ---")
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Using device: {device} with dtype: {torch_dtype}")

    # 1. Load Whisper ASR Model (using pipeline)
    try:
        models["whisper_pipeline"] = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3",
            torch_dtype=torch_dtype,
            device=device,
            generate_kwargs={"task": "translate"} 
        )
        print("Whisper model loaded successfully.")
    except Exception as e:
        print(f"Error loading Whisper model: {e}")

    # 2. Load mBART Translation Model (Model + Tokenizer)
    try:
        model_name = "facebook/mbart-large-50-many-to-many-mmt"
        models["mbart_tokenizer"] = AutoTokenizer.from_pretrained(model_name)
        models["mbart_model"] = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        models["mbart_model"].eval() # Set model to evaluation mode
        print("mBART model loaded successfully.")
    except Exception as e:
        print(f"Error loading mBART model: {e}")
    
    models["device"] = device
    
    yield
    
    # --- Cleanup on shutdown ---
    print("--- Clearing models from memory... ---")
    models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --- FastAPI App Initialization ---
# (This part is the same as your code... no changes needed)
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper Functions ---
# (This part is the same as your code... no changes needed)
def process_audio(audio_bytes: bytes) -> np.ndarray:
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    return audio_float32

def translate_text(text: str, target_lang_code: str) -> str:
    if "mbart_model" not in models or "mbart_tokenizer" not in models:
        return "Error: Translation model not loaded."
    tokenizer = models["mbart_tokenizer"]
    model = models["mbart_model"]
    device = models["device"]
    try:
        tokenizer.src_lang = "en_XX"
        forced_bos_token_id = tokenizer.lang_code_to_id[target_lang_code]
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id
            )
        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translated_text
    except KeyError:
        return f"Error: Invalid target language code '{target_lang_code}'."
    except Exception as e:
        print(f"Translation error: {e}")
        return "Error during translation."

# --- WebSocket Endpoint (MODIFIED) ---

# We define constants for our audio buffer
SAMPLE_RATE = 16000  # 16kHz
BYTES_PER_SAMPLE = 2 # int16 = 2 bytes
CHANNELS = 1
BUFFER_SECONDS = 5   # Process audio every 5 seconds

# Calculate buffer size in bytes
# 16000 samples/sec * 2 bytes/sample * 5 seconds
BUFFER_SIZE = SAMPLE_RATE * BYTES_PER_SAMPLE * BUFFER_SECONDS 

@app.websocket("/ws/{target_language_code}")
async def websocket_endpoint(websocket: WebSocket, target_language_code: str):
    """
    Main WebSocket endpoint for real-time ASR and Translation.
    Now buffers audio for 5 seconds before processing.
    """
    await websocket.accept()
    print(f"WebSocket connected. Target language: {target_language_code}")
    
    if "whisper_pipeline" not in models:
        await websocket.send_json({"error": "ASR model is not loaded."})
        await websocket.close()
        return

    # Create an empty byte array to buffer audio
    audio_buffer = bytearray()

    try:
        while True:
            # 1. Receive audio data from Flutter client
            audio_bytes = await websocket.receive_bytes()
            
            # 2. Add new audio data to our buffer
            audio_buffer.extend(audio_bytes)

            # 3. Check if buffer is full enough to process
            if len(audio_buffer) >= BUFFER_SIZE:
                print(f"Buffer full. Processing {len(audio_buffer)} bytes...")
                
                # 4. Process the *entire* buffer
                # We pass a copy of the buffer so we can clear it
                audio_input = process_audio(bytes(audio_buffer))
                
                # 5. Clear the buffer for the next chunk
                audio_buffer.clear()
                
                if audio_input.size == 0:
                    continue

                # 6. Run ASR (on the full 5-second clip)
                asr_input = {"raw": audio_input, "sampling_rate": SAMPLE_RATE}
                asr_result = models["whisper_pipeline"](asr_input, batch_size=1)
                english_text = asr_result.get("text", "").strip()
                print(f"ASR Result: {english_text}")

                translated_text = ""
                if english_text:
                    # 7. Run Translation
                    translated_text = translate_text(english_text, target_language_code)
                    print(f"Translation Result: {translated_text}")
                
                # 8. Send the full sentence results back to client
                await websocket.send_json({
                    "asr_text": english_text,
                    "translated_text": translated_text
                })
            
            # If buffer is not full, loop and wait for more audio

    except WebSocketDisconnect:
        print("WebSocket disconnected.")
    except Exception as e:
        print(f"An error occurred: {e}")
        await websocket.close(code=1011, reason=f"Internal error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)