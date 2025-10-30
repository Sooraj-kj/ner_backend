import spacy
import uvicorn
import os
import json
import google.generativeai as genai
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from pydantic import BaseModel
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# =====================================================
# 1️⃣ LOAD MED7 MODEL (for prescriptions, dosage, route, etc.)
# =====================================================
try:
    med7_nlp = spacy.load("en_core_med7_lg")
    print("✅ Med7 model loaded successfully.")
except IOError:
    print("❌ Error: 'en_core_med7_lg' model not found.")
    print("Please run:")
    print("pip install \"en-core-med7-lg @ https://huggingface.co/kormilitzin/en_core_med7_lg/resolve/main/en_core_med7_lg-any-py3-none-any.whl\"")
    exit()

# =====================================================
# 2️⃣ ✨ LOAD OPENMED DISEASE MODEL (for diseases)
# =====================================================
try:
    hf_auth_token = os.environ.get("HF_TOKEN")
    # ✨ NEW MODEL: This is the specialized disease detection model
    model_name = "OpenMed/OpenMed-NER-DiseaseDetect-BioMed-335M" 
    
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name,token = hf_auth_token)
    hf_model = AutoModelForTokenClassification.from_pretrained(model_name, token = hf_auth_token)
    
    # We use "token-classification" (or "ner") and an aggregation strategy
    hf_pipeline = pipeline(
        "token-classification", 
        model=hf_model, 
        tokenizer=hf_tokenizer, 
        aggregation_strategy="max" # "max" is often better than "simple"
    )
    print(f"✅ {model_name} model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load Hugging Face model: {e}")
    exit()

# =====================================================
# 3️⃣ FASTAPI SETUP
# =====================================================
app = FastAPI(
    title="Combined Medical NER WebSocket API",
    description="Extracts drugs, dosage, route (Med7) + diseases (OpenMed-NER)",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# ✨ 4️⃣ NEW: ENTITY FILTERING CONSTANTS
# =====================================================

# Filter 1: ✨ UPDATED LABELS
# This new model is specialized and uses the "DISEASE" label.
ALLOWED_HF_LABELS = {
    "DISEASE"
}

# Filter 2: Set a minimum confidence score (0.0 to 1.0)
CONFIDENCE_THRESHOLD = 0.5

# Filter 3: Ignore common stop words that are not medical entities
MEDICAL_STOP_WORDS = {
    "hello", "good", "see", "it", "that", "now", "when", "so",
    "is", "a", "an", "the", "be", "to", "of", "and", "in",
    "have", "let's", "let", "wasn't", "other", "some", "?",
    "issue", "issues", "people", "two", "for"
}

# =====================================================
# 5️⃣ WEBSOCKET ENDPOINT
# =====================================================
@app.websocket("/ws/ner")
async def ner_websocket(websocket: WebSocket):
    """
    Accepts a WebSocket connection.
    Receives text, processes it with Med7 + OpenMed-NER,
    applies filters, and sends back JSON of all extracted entities.
    """
    await websocket.accept()
    print("🩺 NER WebSocket client connected.")
    try:
        while True:
            text = await websocket.receive_text()
            print("\n" + "="*50)
            
            if text.strip():
                # --- ✨ 1. PRINT WHAT TEXT WAS RECEIVED ---
                print(f"--- 1. RECEIVED TEXT: {text}")

                # --- Run Med7 ---
                med7_doc = med7_nlp(text)
                med7_entities = [
                    {
                        "text": ent.text,
                        "label": ent.label_,
                        "source": "Med7",
                        "start_char": ent.start_char,
                        "end_char": ent.end_char
                    }
                    for ent in med7_doc.ents
                ]
                
                # --- ✨ 2. PRINT WHAT MED7 FOUND ---
                print(f"--- 2. MED7 FOUND (pre-filter): {med7_entities}")

                # --- ✨ Run OpenMed-NER ---
                openmed_entities = []
                try:
                    openmed_output = hf_pipeline(text)
                    print(f"--- 2.5. OPENMED RAW OUTPUT (pre-filter): {openmed_output}")
                    
                    openmed_entities = [
                        {
                            "text": ent["word"],
                            "label": ent["entity_group"], # This will be "DISEASE"
                            "score": float(ent["score"]),
                            "source": "OpenMed-NER", # ✨ Updated source name
                            "start_char": int(ent["start"]),
                            "end_char": int(ent["end"])
                        }
                        for ent in openmed_output
                        if ent["entity_group"] in ALLOWED_HF_LABELS
                        and float(ent["score"]) >= CONFIDENCE_THRESHOLD
                        and ent["word"].lower() not in MEDICAL_STOP_WORDS
                    ]
                    
                    # --- ✨ 3. PRINT WHAT OPENMED FOUND ---
                    print(f"--- 3. OPENMED FOUND (post-filter): {openmed_entities}")
                    
                except Exception as e:
                    print(f"⚠️ Error processing OpenMed-NER: {e}")

                # --- Combine results ---
                all_entities = med7_entities + openmed_entities

                # --- ✨ 4. PRINT WHAT IS BEING SENT ---
                print(f"--- 4. SENDING ALL ENTITIES: {all_entities}")
                print("="*50 + "\n")

                # --- Send back JSON ---
                await websocket.send_json(all_entities)

    except WebSocketDisconnect:
        print("🔌 WebSocket client disconnected.")
    except Exception as e:
        print(f"NER WebSocket Error: {e}")
        await websocket.close(code=1011)  # Internal error

# =====================================================
# 6️⃣ GEMINI CHAT ENDPOINT (Unchanged)
# =====================================================

# --- Configure Gemini ---
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel('models/gemini-2.0-flash-lite') # Use a fast model
    print("✅ Gemini AI model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load Gemini: {e}")
    print("Make sure you have set the GEMINI_API_KEY environment variable.")
    gemini_model = None

# --- Define the prompt for Gemini ---
GEMINI_SYSTEM_PROMPT = """
You are an AI assistant helping a doctor edit a list of medical entities (NER).
The user will give you a command and the current list of entities in JSON format.
Your task is to understand the command and return the NEW, updated list of entities as a valid JSON list.

RULES:
- Only return the JSON list. Do NOT add any conversational text like 'Here is the list...'.
- If the command is to "add 'fever'", add a new object: {"text": "fever", "label": "CUSTOM", "source": "User"}.
- If the command is to "remove 'paracetamol'", find the object with "text": "paracetamol" and remove it from the list.
- If the user asks a question (e.g., "what is paracetamol?"), return a JSON list containing a SINGLE entity with the answer: 
  [{"text": "Paracetamol is an analgesic used for fever and pain.", "label": "ASSISTANT", "source": "Gemini"}]
- If you cannot understand the command, return the original list.
"""

# --- Define the data models for the request ---
class NerEntity(BaseModel):
    text: str
    label: str
    source: str
    # Add optional fields if they exist, otherwise Pydantic will error
    score: float | None = None
    start_char: int | None = None
    end_char: int | None = None

class ChatRequest(BaseModel):
    command: str
    context: List[NerEntity]

# --- Create the HTTP POST endpoint ---\;.
@app.post("/chat")
async def chat_with_gemini(request: ChatRequest):
    if not gemini_model:
        return {"error": "Gemini model is not initialized."}

    try:
        # 1. Convert the context list to a JSON string
        # Use .model_dump() for Pydantic v2+ (which FastAPI uses)
        # or .dict() for Pydantic v1
        context_list = []
        for entity in request.context:
            if hasattr(entity, 'model_dump'):
                 context_list.append(entity.model_dump(exclude_none=True))
            else:
                 context_list.append(entity.dict(exclude_none=True)) # Fallback for v1
        
        context_json = json.dumps(context_list)

        # 2. Build the full prompt
        prompt = f"""
        Current NER List:
        {context_json}

        User Command:
        "{request.command}"

        Updated JSON List:
        """

        # 3. Call the Gemini API
        response = await gemini_model.generate_content_async(GEMINI_SYSTEM_PROMPT + prompt)
        
        # 4. Clean and parse the response
        clean_json = response.text.strip().replace("```json", "").replace("```", "").strip()
        
        # 5. Return the new list (as valid JSON)
        return json.loads(clean_json) 

    except Exception as e:
        print(f"❌ Error in Gemini chat: {e}")
        # On error, just return the original list
        return request.context

# =====================================================
# 7️⃣ RUN SERVER
# =====================================================
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)