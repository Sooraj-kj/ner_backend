import spacy
import uvicorn
import os
import json
import string 
# ✨ REMOVED: import re
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
# 2️⃣ LOAD GENERAL SPACY MODEL (Unchanged)
# =====================================================
try:
    # This model is good at finding general entities like DATE and TIME
    # that the medical models might miss in conversational text.
    general_nlp = spacy.load("en_core_web_lg")
    print("✅ en_core_web_lg model loaded successfully.")
except IOError:
    print("❌ 'en_core_web_lg' model not found. Run: python -m spacy download en_core_web_lg")
    exit()

# =====================================================
# 3️⃣ LOAD GENERAL BIOMEDICAL MODEL (Unchanged)
# =====================================================
try:
    hf_auth_token = os.environ.get("HF_TOKEN")
    
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
# 4️⃣ ENTITY FILTERING (Unchanged)
# =====================================================

ALLOWED_HF_LABELS = {
    "Disease_disorder",
    "Sign_symptom",
    "Diagnostic_procedure",
    "Clinical_event",
    "Time",
    "Duration"
}

CONFIDENCE_THRESHOLD = 0.5 

MEDICAL_STOP_WORDS = {
    "hello", "hi", "good", "doctor", "ok", "fine", "thank", "you",
    "see", "it", "that", "now", "when", "so", "is", "a", "an", "the",
    "be", "to", "of", "and", "in", "have", "let's", "let", "wasn't",
    "other", "some", "?", "what", "all", "this", "are", "these",
    "issue", "issues", "people", "two", "for", "symptoms", "symptom",
    
}

# =====================================================
# 5️⃣ PYDANTIC MODELS (Unchanged)
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

class ScanItem(BaseModel):
    procedure: str
    
class FollowUpItem(BaseModel):
    event: str
    timeframe: Optional[str] = None

class StructuredSummary(BaseModel):
    prescriptions: List[PrescriptionItem] = []
    symptoms: List[SymptomItem] = []
    scans: List[ScanItem] = []
    follow_ups: List[FollowUpItem] = []
    other: List[Dict[str, Any]] = []

# =====================================================
# 6️⃣ ✨ UPDATED: RULE-BASED STRUCTURING FUNCTION
# =====================================================

# ✨ UPDATED: Function signature now accepts 'general_entities'
def build_structured_summary_rules(
    text: str, med7_entities: list, openmed_entities: list, general_entities: list
) -> dict:
    
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
        # ✨ NEW: Get general entities for this sentence
        sent_general_ents = [
            ent for ent in general_entities
            if ent['start_char'] >= sent_start and ent['end_char'] <= sent_end
        ]
        
        # --- 2. Apply Prescription Rule (Unchanged) ---
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
        # (Prescription logic for 'if len(drugs...) == 1' etc. is unchanged)
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

        
        # --- 3. Apply Symptom Rule (Unchanged) ---
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

        # --- 4. Apply Scan/Procedure Rule (Unchanged) ---
        procedure_labels = {"Diagnostic_procedure"}
        procedures_in_sentence = [e for e in sent_openmed_ents if e['label'] in procedure_labels]

        for proc_ent in procedures_in_sentence:
            clean_text = proc_ent['text'].lower().strip(punctuation_to_strip)
            if clean_text in MEDICAL_STOP_WORDS:
                continue 
            
            if (proc_ent['start_char'], proc_ent['end_char']) not in used_entity_spans:
                summary.scans.append(ScanItem(procedure=proc_ent['text']))
                used_entity_spans.add((proc_ent['start_char'], proc_ent['end_char']))


        # --- 5. ✨ UPDATED: Apply Follow-up Rule (NER-Only) ---
        
        # 1. Find timeframes (from ALL models)
        follow_up_times_hf = [
            e for e in sent_openmed_ents 
            if e['label'] in {'Time', 'Duration'}
            and (e['start_char'], e['end_char']) not in used_entity_spans
        ]
        follow_up_durations_med7 = [
            e for e in sent_med7_ents
            if e['label'] == 'DURATION'
            and (e['start_char'], e['end_char']) not in used_entity_spans
        ]
        follow_up_times_general = [
            e for e in sent_general_ents
            if e['label'] in {'DATE', 'TIME'} # DATE (e.g., "two days")
            and (e['start_char'], e['end_char']) not in used_entity_spans
        ]
        all_timeframes = follow_up_times_hf + follow_up_durations_med7 + follow_up_times_general
        
        # 2. Find events (from HF model)
        follow_up_events_hf = [
            e for e in sent_openmed_ents 
            if e['label'] == 'Clinical_event'
            and (e['start_char'], e['end_char']) not in used_entity_spans
        ]

        # 3. ✨ REMOVED: Rule-based keyword matching
        # (The re.finditer and FOLLOW_UP_KEYWORDS logic has been removed)
            
        # ✨ CHANGED: 'all_events' now only uses NER-based events
        all_events = follow_up_events_hf
        
        # 4. Apply the rule
        if all_events and all_timeframes:
            # Only add if we have BOTH an event and a timeframe
            event_ent = all_events[0] # Take the first one found
            time_ent = all_timeframes[0]
            time_text = time_ent['text']
            
            # Mark both as used
            used_entity_spans.add((time_ent['start_char'], time_ent['end_char']))
            # We know the event is not 'Rule-based' so we can safely add it
            used_entity_spans.add((event_ent['start_char'], event_ent['end_char']))

            summary.follow_ups.append(FollowUpItem(
                event=event_ent['text'].capitalize(),
                timeframe=time_text
            ))


    # --- 6. ✨ UPDATED: Collect "other" entities ---
    
    # ✨ Combine entities from ALL sources
    all_entities = med7_entities + openmed_entities + general_entities
    
    # ✨ NEW: Labels we use for rules, but don't want in "other"
    LABELS_TO_SKIP_IN_OTHER = {"Time", "Duration", "DATE", "TIME"}
    
    for ent in all_entities:
        if (ent['start_char'], ent['end_char']) not in used_entity_spans:
            clean_text = ent['text'].lower().strip(punctuation_to_strip)
            if clean_text in MEDICAL_STOP_WORDS:
                continue 
            
            # ✨ NEW: Skip labels we don't want in "other"
            if ent['label'] in LABELS_TO_SKIP_IN_OTHER:
                continue

            # ✨ NEW: Skip entities that are too long (likely ASR noise)
            if len(ent['text'].split()) > 5:
                continue

            # --- Confidence Check for HF Model ---
            if ent.get("source") == "d4data-biomedical-ner-all":
                if ent.get('score', 0) < CONFIDENCE_THRESHOLD:
                    continue
            
            # --- Add the entity ---
            if ent.get("source") == "Med7":
                 summary.other.append(ent)
            elif ent.get("source") == "en_core_web_lg":
                 # This will only be PER, ORG, etc. since we skip DATE/TIME
                 summary.other.append(ent)
            elif ent.get("label") in ALLOWED_HF_LABELS:
                 summary.other.append(ent)
            
            
    # Use .dict() for Pydantic v1
    return summary.dict(exclude_none=True)


# =====================================================
# 7️⃣ ✨ UPDATED: WEBSOCKET ENDPOINT
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
            print("--- PROCESSING NER...")
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

            # --- 3. ✨ NEW: Run General Spacy Model ---
            general_doc = general_nlp(text)
            general_entities = [
                {
                    "text": ent.text, "label": ent.label_, "source": "en_core_web_lg",
                    "start_char": ent.start_char, "end_char": ent.end_char
                }
                # ✨ Get DATE/TIME but also general stuff in case 'other' is useful
                for ent in general_doc.ents
                if ent.label_ in {"DATE", "TIME", "PERSON", "ORG", "GPE"}
            ]
            print(f"--- 3. GENERAL SPACY FOUND: {general_entities}")


            # --- 4. Run General HF Model (Re-numbered) --- 
            openmed_entities = []
            try:
                openmed_output = hf_pipeline(text)
                openmed_entities = [
                    {
                        "text": ent["word"], "label": ent["entity_group"],
                        "score": float(ent["score"]), 
                        "source": "d4data-biomedical-ner-all",
                        "start_char": int(ent["start"]), "end_char": int(ent["end"])
                    }
                    for ent in openmed_output
                    if ent["entity_group"] in ALLOWED_HF_LABELS
                ]
                print(f"--- 4. HUGGINGFACE FOUND: {openmed_entities}")
            except Exception as e:
                print(f"⚠️ Error processing HuggingFace pipeline: {e}")

            # --- 5. ✨ UPDATED: Call Rule-Based Structuring ---
            print("--- 5. CALLING RULE-BASED STRUCTURING...")
            structured_data = build_structured_summary_rules(
                text, med7_entities, openmed_entities, general_entities # ✨ Pass new list
            )
            
            print(f"--- 6. SENDING STRUCTURED DATA: {structured_data}")
            await websocket.send_json(structured_data)
            print("="*50 + "\n")

    except WebSocketDisconnect:
        print("🔌 WebSocket client disconnected.")
    except Exception as e:
        print(f"NER WebSocket Error: {e}") 
        await websocket.close(code=1011)

# =====================================================
# 8️⃣ RUN SERVER (Unchanged)
# =====================================================
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)