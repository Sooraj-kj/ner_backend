import uvicorn
import string 
import re
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Initialize FastAPI app and add CORS middleware
app = FastAPI(title="Medical NER API - Blaze999")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# 1️⃣ LOAD BLAZE999/MEDICAL-NER MODEL
# =====================================================
try:
    model_name = "blaze999/Medical-NER"
    
    print(f"📥 Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    
    ner_pipeline = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        ignore_labels=["O"]  # Ignore "Outside" labels
    )
    
    # Extract all entity types from model config
    entity_labels = list(model.config.id2label.values())
    entity_types = set()
    for label in entity_labels:
        if label.startswith('B-') or label.startswith('I-'):
            entity_types.add(label[2:])
        elif label != 'O':
            entity_types.add(label)
    
    print(f"✅ Model loaded successfully!")
    print(f"📋 Detects {len(entity_types)} entity types:")                  
    for i, ent_type in enumerate(sorted(entity_types), 1):
        print(f"    {i}. {ent_type}")
    print()  # Empty line for readability
    
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit()

# =====================================================
# 2️⃣ CONFIGURATION
# =====================================================

# Confidence threshold for entity extraction
CONFIDENCE_THRESHOLD = 0.40  # Lowered to capture more entities

# Words to filter out (common non-medical words)
MEDICAL_STOP_WORDS = {
    "hello", "hi", "good", "doctor", "ok", "fine", "thank", "you",
    "see", "it", "that", "now", "when", "so", "is", "a", "an", "the",
    "be", "to", "of", "and", "in", "have", "let's", "let", "wasn't",
    "other", "some", "what", "all", "this", "are", "these", "issue",
    "issues", "people", "two", "for", "symptoms", "symptom"
}

# Entity type mappings - Based on blaze999/Medical-NER actual output
# The model uses UPPERCASE labels
MEDICATION_LABELS = {"MEDICATION", "Medication", "Drug", "DRUG", "Medicine"}
SYMPTOM_LABELS = {"SIGN_SYMPTOM", "Sign_symptom", "DISEASE_DISORDER", "Disease_disorder", "Symptom", "Disease", "Clinical_finding"}
PROCEDURE_LABELS = {"DIAGNOSTIC_PROCEDURE", "Procedure", "Test", "Diagnostic_procedure", "Therapeutic_procedure"}
DURATION_LABELS = {"DURATION", "Duration", "Time"}
FREQUENCY_LABELS = {"FREQUENCY", "Frequency"}
DOSAGE_LABELS = {"DOSAGE", "Dosage", "STRENGTH", "Strength", "Dose"}
TIME_LABELS = {"DATE", "Time", "TIME", "Date"}
EVENT_LABELS = {"CLINICAL_EVENT", "Clinical_event", "Event", "Activity"}

# =====================================================
# 3️⃣ PYDANTIC MODELS
# =====================================================

class PrescriptionItem(BaseModel):
    medication: str
    strength: Optional[str] = None
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    duration: Optional[str] = None

class SymptomItem(BaseModel):
    symptom: str
    duration: Optional[str] = None

class ProcedureItem(BaseModel):
    procedure: str
    
class FollowUpItem(BaseModel):
    event: str
    timeframe: Optional[str] = None

class StructuredSummary(BaseModel):
    prescriptions: List[PrescriptionItem] = []
    symptoms: List[SymptomItem] = []
    procedures: List[ProcedureItem] = []
    follow_ups: List[FollowUpItem] = []
    other_entities: List[Dict[str, Any]] = []

# =====================================================
# 4️⃣ HELPER FUNCTIONS
# =====================================================

def clean_entity_text(text: str) -> str:
    """Remove extra whitespace and punctuation"""
    return text.strip(string.punctuation + " ")

def is_valid_entity(entity: dict) -> bool:
    """Check if entity is valid and not a stop word"""
    clean_text = clean_entity_text(entity['word']).lower()
    
    # Skip stop words
    if clean_text in MEDICAL_STOP_WORDS:
        return False
    
    # Skip very short entities (likely noise)
    if len(clean_text) < 2:
        return False
    
    # Skip low confidence entities
    if entity.get('score', 0) < CONFIDENCE_THRESHOLD:
        return False
    
    # Skip very long entities (likely ASR errors)
    if len(entity['word'].split()) > 6:
        return False
    
    return True

def group_entities_by_sentence(text: str, entities: list) -> list:
    """Group entities by sentence for better context"""
    # Simple sentence splitting (you can use spaCy for better results)
    sentences = re.split(r'[.!?]+', text)
    
    sentence_groups = []
    char_count = 0
    
    for sentence in sentences:
        if not sentence.strip():
            continue
            
        sent_start = char_count
        sent_end = char_count + len(sentence)
        
        sent_entities = [
            ent for ent in entities
            if ent['start'] >= sent_start and ent['end'] <= sent_end
        ]
        
        sentence_groups.append({
            'text': sentence.strip(),
            'start': sent_start,
            'end': sent_end,
            'entities': sent_entities
        })
        
        char_count = sent_end + 1  # +1 for the delimiter
    
    return sentence_groups

# =====================================================
# 5️⃣ STRUCTURING LOGIC
# =====================================================

def build_structured_summary(text: str, raw_entities: list) -> dict:
    """Convert raw NER output into structured medical information"""
    
    summary = StructuredSummary()
    used_entities = set()
    
    # Filter valid entities
    valid_entities = [ent for ent in raw_entities if is_valid_entity(ent)]
    
    # Debug: Print what entities were found
    print(f"   📌 Valid entities after filtering:")
    for ent in valid_entities:
        print(f"      - '{ent['word']}' [{ent['entity_group']}] (score: {ent['score']:.2f})")
    
    # Group by sentence for context
    sentence_groups = group_entities_by_sentence(text, valid_entities)
    
    # Process each sentence
    for sent_group in sentence_groups:
        entities = sent_group['entities']
        
        # Separate entities by type
        medications = [e for e in entities if e['entity_group'] in MEDICATION_LABELS]
        symptoms = [e for e in entities if e['entity_group'] in SYMPTOM_LABELS]
        procedures = [e for e in entities if e['entity_group'] in PROCEDURE_LABELS]
        durations = [e for e in entities if e['entity_group'] in DURATION_LABELS]
        frequencies = [e for e in entities if e['entity_group'] in FREQUENCY_LABELS]
        dosages = [e for e in entities if e['entity_group'] in DOSAGE_LABELS]
        times = [e for e in entities if e['entity_group'] in TIME_LABELS]
        events = [e for e in entities if e['entity_group'] in EVENT_LABELS]
        
        # --- Build Prescriptions ---
        if medications:
            for med in medications:
                if id(med) in used_entities:
                    continue
                    
                prescription = PrescriptionItem(
                    medication=clean_entity_text(med['word'])
                )
                
                # Try to find associated modifiers
                for dosage in dosages:
                    if id(dosage) not in used_entities:
                        prescription.dosage = clean_entity_text(dosage['word'])
                        used_entities.add(id(dosage))
                        break
                
                for freq in frequencies:
                    if id(freq) not in used_entities:
                        prescription.frequency = clean_entity_text(freq['word']) 
                        used_entities.add(id(freq))
                        break
                
                for dur in durations:
                    if id(dur) not in used_entities:
                        prescription.duration = clean_entity_text(dur['word'])
                        used_entities.add(id(dur))
                        break
                
                summary.prescriptions.append(prescription)
                used_entities.add(id(med))
        
        # --- Build Symptoms ---
        if symptoms:
            # Find duration for symptoms
            duration_text = None
            for dur in durations:
                if id(dur) not in used_entities:
                    duration_text = clean_entity_text(dur['word'])
                    used_entities.add(id(dur))
                    break
            
            for sym in symptoms:
                if id(sym) not in used_entities:
                    summary.symptoms.append(SymptomItem(
                        symptom=clean_entity_text(sym['word']),
                        duration=duration_text
                    ))
                    used_entities.add(id(sym))
        
        # --- Build Procedures ---
        for proc in procedures:
            if id(proc) not in used_entities:
                summary.procedures.append(ProcedureItem(
                    procedure=clean_entity_text(proc['word'])
                ))
                used_entities.add(id(proc))
        
        # --- Build Follow-ups ---
        # Check for follow-up keywords
        follow_up_pattern = r'\b(follow up|come back|return|see me|visit)\b'
        has_follow_up_keyword = bool(re.search(follow_up_pattern, sent_group['text'].lower()))
        
        if (events or has_follow_up_keyword) and times:
            event_text = None
            
            if events and id(events[0]) not in used_entities:
                event_text = clean_entity_text(events[0]['word'])
                used_entities.add(id(events[0]))
            elif has_follow_up_keyword:
                match = re.search(follow_up_pattern, sent_group['text'].lower())
                event_text = match.group(0).capitalize() if match else "Follow up"
            
            if event_text:
                for time_ent in times:
                    if id(time_ent) not in used_entities:
                        summary.follow_ups.append(FollowUpItem(
                            event=event_text,
                            timeframe=clean_entity_text(time_ent['word'])
                        ))
                        used_entities.add(id(time_ent))
                        break
    
    # --- Collect Other Entities ---
    for entity in valid_entities:
        if id(entity) not in used_entities:
            # Skip time/duration entities in "other" category
            if entity['entity_group'] not in (TIME_LABELS | DURATION_LABELS):
                summary.other_entities.append({
                    'text': clean_entity_text(entity['word']),
                    'label': entity['entity_group'],
                    'confidence': float(entity['score'])  # Convert to native Python float
                })
    
    return summary.dict(exclude_none=True)

# =====================================================
# 6️⃣ WEBSOCKET ENDPOINT
# =====================================================

@app.websocket("/ws/ner")
async def ner_websocket(websocket: WebSocket):
    """Real-time medical NER via WebSocket"""
    await websocket.accept()
    print("🩺 WebSocket client connected")
    
    try:
        while True:
            # Receive text from client
            text = await websocket.receive_text()
            
            if not text.strip():
                continue
            
            print("\n" + "="*60)
            print(f"📝 INPUT: {text}")
            
            # Run NER
            try:
                raw_entities = ner_pipeline(text)
                
                # Debug: Print raw entities
                print(f"🔍 FOUND {len(raw_entities)} raw entities:")
                for ent in raw_entities:
                    print(f"   - '{ent['word']}' [{ent['entity_group']}] (score: {ent['score']:.3f})")
                
                # Structure the entities
                structured_data = build_structured_summary(text, raw_entities)
                
                print(f"✅ STRUCTURED OUTPUT:")
                print(f"   - Prescriptions: {len(structured_data['prescriptions'])}")
                print(f"   - Symptoms: {len(structured_data['symptoms'])}")
                print(f"   - Procedures: {len(structured_data['procedures'])}")
                print(f"   - Follow-ups: {len(structured_data['follow_ups'])}")
                print(f"   - Other: {len(structured_data['other_entities'])}")
                
                # Send to client
                await websocket.send_json(structured_data)
                
            except Exception as e:
                error_msg = {"error": str(e)}
                print(f"❌ ERROR: {e}")
                await websocket.send_json(error_msg)
            
            print("="*60 + "\n")
    
    except WebSocketDisconnect:
        print("🔌 WebSocket client disconnected")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        await websocket.close(code=1011)

# =====================================================
# 7️⃣ REST API ENDPOINTS
# =====================================================

@app.post("/api/extract")
async def extract_entities(data: dict):
    """Extract medical entities from text (REST API)"""
    text = data.get("text", "")
    
    if not text.strip():
        return {"error": "No text provided"}
    
    try:
        raw_entities = ner_pipeline(text)
        structured_data = build_structured_summary(text, raw_entities)
        return structured_data
    except Exception as e:
        return {"error": str(e)}
  
@app.get("/api/entities")
async def get_entity_types():
    """Get all supported entity types"""
    return {
        "model": "blaze999/Medical-NER",
        "total_entities": len(entity_types),
        "entity_types": sorted(entity_types)
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "blaze999/Medical-NER",
        "model_loaded": ner_pipeline is not None,
        "entity_types": len(entity_types)
    }

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "Medical NER API",
        "model": "blaze999/Medical-NER",
        "version": "1.0.0",
        "endpoints": {
            "websocket": "/ws/ner",
            "rest_api": "/api/extract",
            "entity_types": "/api/entities",
            "health": "/health"
        }
    }

# =====================================================
# 8️⃣ RUN SERVER
# =====================================================

if __name__ == "__main__":
    print("\n🚀 Starting Medical NER Server...")
    print(f"📍 WebSocket: ws://127.0.0.1:8000/ws/ner")
    print(f"📍 REST API: http://127.0.0.1:8000/api/extract")
    print(f"📍 Health Check: http://127.0.0.1:8000/health\n")
    
    uvicorn.run(app, host="127.0.0.1", port=8000)