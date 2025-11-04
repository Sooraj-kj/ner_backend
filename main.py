import spacy
import uvicorn
import os
import json
import string 
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# Initialize FastAPI app and add CORS middleware so routes and websockets work
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# 1️⃣ LOAD MED7 MODEL (Unchanged)
# =====================================================
try:
    med7_nlp = spacy.load("en_core_med7_lg")
    med7_nlp.add_pipe('sentencizer')
    print("✅ Med7 model loaded successfully (with sentencizer).")
except IOError:
    print("❌ 'en_core_med7_lg' model not found. Please run the install command.")
    exit()

# =====================================================
# 2️⃣ ✨ UPDATED: LOAD GENERAL BIOMEDICAL MODEL
# =====================================================
try:
    hf_auth_token = os.environ.get("HF_TOKEN")
    
    # ✨ NEW MODEL: This model is broader and finds procedures
    model_name = "d4data/biomedical-ner-all" 
    
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name,token = hf_auth_token)
    hf_model = AutoModelForTokenClassification.from_pretrained(model_name, token = hf_auth_token)
    
    hf_pipeline = pipeline(
        "token-classification", 
        model=hf_model, 
        tokenizer=hf_tokenizer, 
        aggregation_strategy="max"
    )
    print(f"✅ {model_name} model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load Hugging Face model: {e}")
    exit()

# =====================================================
# 3️⃣ ✨ UPDATED: ENTITY FILTERING
# =====================================================

# ✨ NEW LABELS: We now look for procedures, diseases, and symptoms
ALLOWED_HF_LABELS = {
    "Disease_disorder",     # Replaces 'DISEASE'
    "Diagnostic_procedure", # <-- THIS IS THE FIX (for MRI, CT scan)
    "Sign_symptom"          # Catches things like 'fever', 'headache'
}

CONFIDENCE_THRESHOLD = 0.5 

# We keep the same stop words as they are still useful
MEDICAL_STOP_WORDS = {
    "hello", "hi", "good", "doctor", "ok", "fine", "thank", "you",
    "see", "it", "that", "now", "when", "so", "is", "a", "an", "the",
    "be", "to", "of", "and", "in", "have", "let's", "let", "wasn't",
    "other", "some", "?", "what", "all", "this", "are", "these",
    "issue", "issues", "people", "two", "for", "symptoms", "symptom",
    "dolo", "strongness", "thiruvananthapuram", "thenga", "coconut", "hra",
    "malayalam"
}

# =====================================================
# 4️⃣ PYDANTIC MODELS (Unchanged)
# =====================================================
class PrescriptionItem(BaseModel):
    medication: str
    strength: Optional[str] = None
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    duration: Optional[str] = None
    form: Optional[str] = None
    route: Optional[str] = None

class SymptomItem(BaseModel):
    symptom: str
    duration: Optional[str] = None

class StructuredSummary(BaseModel):
    prescriptions: List[PrescriptionItem] = []
    symptoms: List[SymptomItem] = []
    other: List[Dict[str, Any]] = []

# =====================================================
# 5️⃣ ✨ UPDATED: RULE-BASED STRUCTURING FUNCTION
# =====================================================
def build_structured_summary_rules(text: str, med7_entities: list, openmed_entities: list) -> dict:
    
    doc = med7_nlp(text)
    summary = StructuredSummary()
    used_entity_spans = set()
    punctuation_to_strip = string.punctuation + " "

    # --- 1. Iterate through each sentence (Unchanged) ---
    for sent in doc.sents:
        sent_start = sent.start_char
        sent_end = sent.end_char

        sent_med7_ents = [
            ent for ent in med7_entities
            if ent['start_char'] >= sent_start and ent['end_char'] <= sent_end
        ]
        sent_openmed_ents = [
            ent for ent in openmed_entities
            if ent['start_char'] >= sent_start and ent['end_char'] <= sent_end
        ]
        
        # --- 3. Apply Prescription Rule (Unchanged) ---
        # (This logic for Med7 drugs and modifiers remains the same)
        drugs_in_sentence = [e for e in sent_med7_ents if e['label'] == 'DRUG']
        modifiers = {
            "strength": [e for e in sent_med7_ents if e['label'] == 'STRENGTH'],
            "dosage": [e for e in sent_med7_ents if e['label'] == 'DOSAGE'],
            "duration": [e for e in sent_med7_ents if e['label'] == 'DURATION'],
            "frequency": [e for e in sent_med7_ents if e['label'] == 'FREQUENCY'],
            "form": [e for e in sent_med7_ents if e['label'] == 'FORM'],
            "route": [e for e in sent_med7_ents if e['label'] == 'ROUTE'],
        }

        if len(drugs_in_sentence) == 1:
            drug_ent = drugs_in_sentence[0]
            prescription = PrescriptionItem(medication=drug_ent['text'])
            used_entity_spans.add((drug_ent['start_char'], drug_ent['end_char']))
            
            if modifiers['strength']:
                mod = modifiers['strength'][0]
                prescription.strength = mod['text']
                used_entity_spans.add((mod['start_char'], mod['end_char']))
            if modifiers['dosage']:
                mod = modifiers['dosage'][0]
                prescription.dosage = mod['text']
                used_entity_spans.add((mod['start_char'], mod['end_char']))
            if modifiers['duration']:
                mod = modifiers['duration'][0]
                prescription.duration = mod['text']
                used_entity_spans.add((mod['start_char'], mod['end_char']))
            if modifiers['frequency']:
                mod = modifiers['frequency'][0]
                prescription.frequency = mod['text']
                used_entity_spans.add((mod['start_char'], mod['end_char']))
            if modifiers['form']:
                mod = modifiers['form'][0]
                prescription.form = mod['text']
                used_entity_spans.add((mod['start_char'], mod['end_char']))
            if modifiers['route']:
                mod = modifiers['route'][0]
                prescription.route = mod['text']
                used_entity_spans.add((mod['start_char'], mod['end_char']))
            summary.prescriptions.append(prescription)

        elif len(drugs_in_sentence) > 1:
            for drug_ent in drugs_in_sentence:
                summary.prescriptions.append(PrescriptionItem(medication=drug_ent['text']))
                used_entity_spans.add((drug_ent['start_char'], drug_ent['end_char']))

        
        # --- 4. Apply Symptom Rule (✨ UPDATED LOGIC) ---
        
        # ✨ Use the new labels from the d4data model
        symptom_labels = {"Disease_disorder", "Sign_symptom"}
        symptoms_in_sentence = [e for e in sent_openmed_ents if e['label'] in symptom_labels]
        
        symptom_durations = [
            e['text'] for e in sent_med7_ents
            if e['label'] == 'DURATION'
            and (e['start_char'], e['end_char']) not in used_entity_spans
        ]
        duration_text = ", ".join(symptom_durations) if symptom_durations else None
        
        if duration_text:
            for e in sent_med7_ents:
                if e['label'] == 'DURATION' and e['text'] in symptom_durations:
                     used_entity_spans.add((e['start_char'], e['end_char']))

        for sym_ent in symptoms_in_sentence:
            clean_text = sym_ent['text'].lower().strip(punctuation_to_strip)
            if clean_text in MEDICAL_STOP_WORDS:
                continue 
                
            summary.symptoms.append(SymptomItem(
                symptom=sym_ent['text'],
                duration=duration_text
            ))
            used_entity_spans.add((sym_ent['start_char'], sym_ent['end_char']))

    # --- 5. Collect all "other" entities (✨ UPDATED LOGIC) ---
    # This logic now correctly catches "Diagnostic_procedure"
    all_entities = med7_entities + openmed_entities
    for ent in all_entities:
        if (ent['start_char'], ent['end_char']) not in used_entity_spans:
            clean_text = ent['text'].lower().strip(punctuation_to_strip)
            if clean_text in MEDICAL_STOP_WORDS:
                continue 
            
            if ent.get("source") == "d4data-biomedical-ner-all": # Source name changed
                 if ent.get('score', 0) < CONFIDENCE_THRESHOLD:
                     continue
            
            summary.other.append(ent)
            
    # Use .dict() for Pydantic v1
    return summary.dict(exclude_none=True)


# =====================================================
# 6️⃣ WEBSOCKET ENDPOINT (Unchanged, but logic is updated)
# =====================================================
@app.websocket("/ws/ner")
async def ner_websocket(websocket: WebSocket):
    await websocket.accept()
    print("🩺 NER WebSocket client connected.")
        
    try:
        while True:
            text = await websocket.receive_text()
            print("\n" + "="*50)
            
            if not text.strip():
                continue

            print(f"--- 1. RECEIVED TEXT: {text}")

            # --- 2. Run Med7 (Unchanged) ---
            med7_doc = med7_nlp(text) 
            med7_entities = [
                {
                    "text": ent.text, "label": ent.label_, "source": "Med7",
                    "start_char": ent.start_char, "end_char": ent.end_char
                }
                for ent in med7_doc.ents
            ]
            print(f"--- 2. MED7 FOUND: {med7_entities}")

            # --- 3. Run General HF Model (Unchanged) ---
            # This pipeline now uses the new model and will find procedures
            openmed_entities = []
            try:
                openmed_output = hf_pipeline(text)
                openmed_entities = [
                    {
                        "text": ent["word"], "label": ent["entity_group"],
                        "score": float(ent["score"]), 
                        "source": "d4data-biomedical-ner-all", # ✨ Updated source name
                        "start_char": int(ent["start"]), "end_char": int(ent["end"])
                    }
                    for ent in openmed_output
                    # This line now checks against the new ALLOWED_HF_LABELS
                    if ent["entity_group"] in ALLOWED_HF_LABELS
                ]
                print(f"--- 3. HUGGINGFACE FOUND: {openmed_entities}")
            except Exception as e:
                print(f"⚠️ Error processing HuggingFace pipeline: {e}")

            # --- 4. Call Rule-Based Structuring Function (Unchanged) ---
            print("--- 4. CALLING RULE-BASED STRUCTURING...")
            structured_data = build_structured_summary_rules(
                text, med7_entities, openmed_entities
            )
            
            print(f"--- 5. SENDING STRUCTURED DATA: {structured_data}")
            await websocket.send_json(structured_data)
            print("="*50 + "\n")

    except WebSocketDisconnect:
        print("🔌 WebSocket client disconnected.")
    except Exception as e:
        print(f"NER WebSocket Error: {e}") 
        await websocket.close(code=1011)

# =====================================================
# 7️⃣ RUN SERVER (Unchanged)
# =====================================================
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)