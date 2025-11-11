import uvicorn
import string 
import re
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from collections import defaultdict

# Initialize FastAPI app and add CORS middleware
app = FastAPI(title="Medical NER API - Complete Entity Extraction")
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
    print()
    
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit()

# =====================================================
# 2️⃣ COMPREHENSIVE ENTITY CATEGORIZATION
# =====================================================

# Confidence threshold for entity extraction
CONFIDENCE_THRESHOLD = 0.35

# Medical stop words to filter out
MEDICAL_STOP_WORDS = {
    "hello", "hi", "good", "doctor", "ok", "fine", "thank", "you",
    "see", "it", "that", "now", "when", "so", "is", "a", "an", "the",
    "be", "to", "of", "and", "in", "have", "let's", "let", "wasn't",
    "other", "some", "what", "all", "this", "are", "these", "issue",
    "issues", "people", "two", "for", "symptoms", "symptom", "he", "she",
    "his", "her", "was", "were", "been", "being", "has", "had", "do",
    "does", "did", "will", "would", "should", "could", "may", "might"
}

# Complete entity type categorization based on actual model labels
ENTITY_CATEGORIES = {
    # Primary Medical Information
    "medications": {"MEDICATION"},
    "symptoms": {"SIGN_SYMPTOM"},
    "diseases": {"DISEASE_DISORDER"},
    "diagnostic_procedures": {"DIAGNOSTIC_PROCEDURE"},
    "therapeutic_procedures": {"THERAPEUTIC_PROCEDURE"},
    
    # Temporal Information
    "dates": {"DATE"},
    "times": {"TIME"},
    "durations": {"DURATION"},
    "frequencies": {"FREQUENCY"},
    "ages": {"AGE"},
    
    # Dosage & Administration
    "dosages": {"DOSAGE"},
    "administrations": {"ADMINISTRATION"},
    
    # Measurements
    "lab_values": {"LAB_VALUE"},
    "masses": {"MASS"},
    "heights": {"HEIGHT"},
    "weights": {"WEIGHT"},
    "volumes": {"VOLUME"},
    "distances": {"DISTANCE"},
    
    # Anatomical & Structural
    "biological_structures": {"BIOLOGICAL_STRUCTURE"},
    "areas": {"AREA"},
    "biological_attributes": {"BIOLOGICAL_ATTRIBUTE"},
    
    # Clinical Context
    "clinical_events": {"CLINICAL_EVENT"},
    "outcomes": {"OUTCOME"},
    "severities": {"SEVERITY"},
    "activities": {"ACTIVITY"},
    
    # Patient Information
    "sex": {"SEX"},
    "occupations": {"OCCUPATION"},
    "family_history": {"FAMILY_HISTORY"},
    "personal_background": {"PERSONAL_BACKGROUND"},
    "history": {"HISTORY"},
    
    # Descriptive Information
    "colors": {"COLOR"},
    "shapes": {"SHAPE"},
    "textures": {"TEXTURE"},
    "detailed_descriptions": {"DETAILED_DESCRIPTION"},
    "qualitative_concepts": {"QUALITATIVE_CONCEPT"},
    "quantitative_concepts": {"QUANTITATIVE_CONCEPT"},
    
    # Location & Context
    "nonbiological_locations": {"NONBIOLOGICAL_LOCATION"},
    "subjects": {"SUBJECT"},
    "coreferences": {"COREFERENCE"},
    
    # Events & Other
    "other_events": {"OTHER_EVENT"},
    "other_entities": {"OTHER_ENTITY"}
}

# Create reverse mapping for quick lookup
LABEL_TO_CATEGORY = {}
for category, labels in ENTITY_CATEGORIES.items():
    for label in labels:
        LABEL_TO_CATEGORY[label] = category

# =====================================================
# 3️⃣ PYDANTIC MODELS
# =====================================================

class EntityItem(BaseModel):
    text: str
    label: str
    confidence: float
    start: int
    end: int

class PrescriptionItem(BaseModel):
    medication: str
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    duration: Optional[str] = None
    administration: Optional[str] = None

class SymptomItem(BaseModel):
    symptom: str
    severity: Optional[str] = None
    duration: Optional[str] = None
    area: Optional[str] = None

class ProcedureItem(BaseModel):
    procedure: str
    procedure_type: str  # diagnostic or therapeutic
    date: Optional[str] = None

class FollowUpItem(BaseModel):
    event: str
    timeframe: Optional[str] = None

class ComprehensiveNERResult(BaseModel):
    # Structured medical information
    prescriptions: List[PrescriptionItem] = []
    symptoms: List[SymptomItem] = []
    procedures: List[ProcedureItem] = []
    follow_ups: List[FollowUpItem] = []
    
    # All categorized entities
    medications: List[EntityItem] = []
    diseases: List[EntityItem] = []
    diagnostic_procedures: List[EntityItem] = []
    therapeutic_procedures: List[EntityItem] = []
    symptoms_raw: List[EntityItem] = []
    
    # Temporal information
    dates: List[EntityItem] = []
    times: List[EntityItem] = []
    durations: List[EntityItem] = []
    frequencies: List[EntityItem] = []
    ages: List[EntityItem] = []
    
    # Dosage & Administration
    dosages: List[EntityItem] = []
    administrations: List[EntityItem] = []
    
    # Measurements
    lab_values: List[EntityItem] = []
    masses: List[EntityItem] = []
    heights: List[EntityItem] = []
    weights: List[EntityItem] = []
    volumes: List[EntityItem] = []
    distances: List[EntityItem] = []
    
    # Anatomical
    biological_structures: List[EntityItem] = []
    areas: List[EntityItem] = []
    biological_attributes: List[EntityItem] = []
    
    # Clinical Context
    clinical_events: List[EntityItem] = []
    outcomes: List[EntityItem] = []
    severities: List[EntityItem] = []
    activities: List[EntityItem] = []
    
    # Patient Information
    sex: List[EntityItem] = []
    occupations: List[EntityItem] = []
    family_history: List[EntityItem] = []
    personal_background: List[EntityItem] = []
    history: List[EntityItem] = []
    
    # Descriptive
    colors: List[EntityItem] = []
    shapes: List[EntityItem] = []
    textures: List[EntityItem] = []
    detailed_descriptions: List[EntityItem] = []
    qualitative_concepts: List[EntityItem] = []
    quantitative_concepts: List[EntityItem] = []
    
    # Location & Context
    nonbiological_locations: List[EntityItem] = []
    subjects: List[EntityItem] = []
    coreferences: List[EntityItem] = []
    
    # Events & Other
    other_events: List[EntityItem] = []
    other_entities: List[EntityItem] = []
    
    # Summary statistics
    total_entities: int = 0
    entity_count_by_category: Dict[str, int] = {}

# =====================================================
# 4️⃣ HELPER FUNCTIONS
# =====================================================

def clean_entity_text(text: str) -> str:
    """Remove extra whitespace and punctuation"""
    cleaned = text.strip(string.punctuation + " ")
    # Remove multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned

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
    if len(entity['word'].split()) > 8:
        return False
    
    return True

def create_entity_item(entity: dict) -> EntityItem:
    """Create an EntityItem from raw entity"""
    return EntityItem(
        text=clean_entity_text(entity['word']),
        label=entity['entity_group'],
        confidence=float(entity['score']),
        start=entity['start'],
        end=entity['end']
    )

# =====================================================
# 5️⃣ COMPREHENSIVE ENTITY EXTRACTION
# =====================================================

def extract_all_entities(text: str, raw_entities: list) -> dict:
    """Extract and categorize ALL entities from the text"""
    
    result = ComprehensiveNERResult()
    
    # Filter valid entities
    valid_entities = [ent for ent in raw_entities if is_valid_entity(ent)]
    
    # Categorize all entities
    categorized = defaultdict(list)
    
    for entity in valid_entities:
        entity_item = create_entity_item(entity)
        label = entity['entity_group']
        
        # Find the category for this label
        category = LABEL_TO_CATEGORY.get(label, "other_entities")
        categorized[category].append(entity_item)
    
    # Populate the result object with all categorized entities
    result.medications = categorized.get("medications", [])
    result.diseases = categorized.get("diseases", [])
    result.diagnostic_procedures = categorized.get("diagnostic_procedures", [])
    result.therapeutic_procedures = categorized.get("therapeutic_procedures", [])
    result.symptoms_raw = categorized.get("symptoms", [])
    
    # Temporal
    result.dates = categorized.get("dates", [])
    result.times = categorized.get("times", [])
    result.durations = categorized.get("durations", [])
    result.frequencies = categorized.get("frequencies", [])
    result.ages = categorized.get("ages", [])
    
    # Dosage & Administration
    result.dosages = categorized.get("dosages", [])
    result.administrations = categorized.get("administrations", [])
    
    # Measurements
    result.lab_values = categorized.get("lab_values", [])
    result.masses = categorized.get("masses", [])
    result.heights = categorized.get("heights", [])
    result.weights = categorized.get("weights", [])
    result.volumes = categorized.get("volumes", [])
    result.distances = categorized.get("distances", [])
    
    # Anatomical
    result.biological_structures = categorized.get("biological_structures", [])
    result.areas = categorized.get("areas", [])
    result.biological_attributes = categorized.get("biological_attributes", [])
    
    # Clinical Context
    result.clinical_events = categorized.get("clinical_events", [])
    result.outcomes = categorized.get("outcomes", [])
    result.severities = categorized.get("severities", [])
    result.activities = categorized.get("activities", [])
    
    # Patient Information
    result.sex = categorized.get("sex", [])
    result.occupations = categorized.get("occupations", [])
    result.family_history = categorized.get("family_history", [])
    result.personal_background = categorized.get("personal_background", [])
    result.history = categorized.get("history", [])
    
    # Descriptive
    result.colors = categorized.get("colors", [])
    result.shapes = categorized.get("shapes", [])
    result.textures = categorized.get("textures", [])
    result.detailed_descriptions = categorized.get("detailed_descriptions", [])
    result.qualitative_concepts = categorized.get("qualitative_concepts", [])
    result.quantitative_concepts = categorized.get("quantitative_concepts", [])
    
    # Location & Context
    result.nonbiological_locations = categorized.get("nonbiological_locations", [])
    result.subjects = categorized.get("subjects", [])
    result.coreferences = categorized.get("coreferences", [])
    
    # Events & Other
    result.other_events = categorized.get("other_events", [])
    result.other_entities = categorized.get("other_entities", [])
    
    # Build structured prescriptions
    result.prescriptions = build_prescriptions(
        result.medications,
        result.dosages,
        result.frequencies,
        result.durations,
        result.administrations
    )
    
    # Build structured symptoms
    result.symptoms = build_symptoms(
        result.symptoms_raw,
        result.severities,
        result.durations,
        result.areas
    )
    
    # Build structured procedures
    result.procedures = build_procedures(
        result.diagnostic_procedures,
        result.therapeutic_procedures,
        result.dates
    )
    
    # Build follow-ups
    result.follow_ups = build_followups(
        result.clinical_events,
        result.other_events,
        result.dates,
        result.times,
        text
    )
    
    # Calculate statistics
    result.total_entities = len(valid_entities)
    result.entity_count_by_category = {
        category: len(entities) 
        for category, entities in categorized.items()
        if len(entities) > 0
    }
    
    return result.dict(exclude_none=True)

# =====================================================
#6️⃣ STRUCTURED DATA BUILDERS
# =====================================================

def build_prescriptions(medications, dosages, frequencies, durations, administrations) -> List[PrescriptionItem]:
    """Build structured prescription items"""
    prescriptions = []
    used_modifiers = set()
    
    for med in medications:
        prescription = PrescriptionItem(medication=med.text)
        
        # Find closest dosage
        for dosage in dosages:
            if id(dosage) not in used_modifiers and abs(dosage.start - med.end) < 50:
                prescription.dosage = dosage.text
                used_modifiers.add(id(dosage))
                break
        
        # Find closest frequency
        for freq in frequencies:
            if id(freq) not in used_modifiers and abs(freq.start - med.end) < 50:
                prescription.frequency = freq.text
                used_modifiers.add(id(freq))
                break
        
        # Find closest duration
        for dur in durations:
            if id(dur) not in used_modifiers and abs(dur.start - med.end) < 50:
                prescription.duration = dur.text
                used_modifiers.add(id(dur))
                break
        
        # Find closest administration
        for admin in administrations:
            if id(admin) not in used_modifiers and abs(admin.start - med.end) < 50:
                prescription.administration = admin.text
                used_modifiers.add(id(admin))
                break
        
        prescriptions.append(prescription)
    
    return prescriptions

def build_symptoms(symptoms, severities, durations, areas) -> List[SymptomItem]:
    """Build structured symptom items"""
    symptom_items = []
    used_modifiers = set()
    
    for symptom in symptoms:
        symptom_item = SymptomItem(symptom=symptom.text)
        
        # Find closest severity
        for sev in severities:
            if id(sev) not in used_modifiers and abs(sev.start - symptom.start) < 30:
                symptom_item.severity = sev.text
                used_modifiers.add(id(sev))
                break
        
        # Find closest duration
        for dur in durations:
            if id(dur) not in used_modifiers and abs(dur.start - symptom.end) < 50:
                symptom_item.duration = dur.text
                used_modifiers.add(id(dur))
                break
        
        # Find closest area
        for area in areas:
            if id(area) not in used_modifiers and abs(area.start - symptom.start) < 30:
                symptom_item.area = area.text
                used_modifiers.add(id(area))
                break
        
        symptom_items.append(symptom_item)
    
    return symptom_items

def build_procedures(diagnostic_procs, therapeutic_procs, dates) -> List[ProcedureItem]:
    """Build structured procedure items"""
    procedures = []
    used_dates = set()
    
    # Process diagnostic procedures
    for proc in diagnostic_procs:
        procedure = ProcedureItem(procedure=proc.text, procedure_type="diagnostic")
        
        for date in dates:
            if id(date) not in used_dates and abs(date.start - proc.end) < 50:
                procedure.date = date.text
                used_dates.add(id(date))
                break
        
        procedures.append(procedure)
    
    # Process therapeutic procedures
    for proc in therapeutic_procs:
        procedure = ProcedureItem(procedure=proc.text, procedure_type="therapeutic")
        
        for date in dates:
            if id(date) not in used_dates and abs(date.start - proc.end) < 50:
                procedure.date = date.text
                used_dates.add(id(date))
                break
        
        procedures.append(procedure)
    
    return procedures

def build_followups(clinical_events, other_events, dates, times, text) -> List[FollowUpItem]:
    """Build structured follow-up items"""
    followups = []
    used_timeframes = set()
    
    # Check for follow-up keywords
    follow_up_pattern = r'\b(follow up|come back|return|see me|visit|appointment)\b'
    has_follow_up = bool(re.search(follow_up_pattern, text.lower()))
    
    all_events = clinical_events + other_events
    all_timeframes = dates + times
    
    if has_follow_up or all_events:
        for event in all_events:
            followup = FollowUpItem(event=event.text)
            
            for timeframe in all_timeframes:
                if id(timeframe) not in used_timeframes and abs(timeframe.start - event.end) < 50:
                    followup.timeframe = timeframe.text
                    used_timeframes.add(id(timeframe))
                    break
            
            followups.append(followup)
        
        # If no events but has follow-up keyword, create generic follow-up
        if not followups and has_follow_up:
            for timeframe in all_timeframes:
                if id(timeframe) not in used_timeframes:
                    followups.append(FollowUpItem(
                        event="Follow up",
                        timeframe=timeframe.text
                    ))
                    used_timeframes.add(id(timeframe))
                    break
    
    return followups

# =====================================================
# 7️⃣ WEBSOCKET ENDPOINT
# =====================================================

@app.websocket("/ws/ner")
async def ner_websocket(websocket: WebSocket):
    """Real-time medical NER via WebSocket"""
    await websocket.accept()
    print("🩺 WebSocket client connected")
    
    try:
        while True:
            text = await websocket.receive_text()
            
            if not text.strip():
                continue
            
            print("\n" + "="*80)
            print(f"📝 INPUT: {text}")
            
            try:
                raw_entities = ner_pipeline(text)
                
                print(f"🔍 FOUND {len(raw_entities)} raw entities:")
                for ent in raw_entities:
                    print(f"   - '{ent['word']}' [{ent['entity_group']}] (score: {ent['score']:.3f})")
                
                result = extract_all_entities(text, raw_entities)
                
                print(f"\n✅ EXTRACTED ENTITIES:")
                print(f"   Total: {result['total_entities']}")
                print(f"   Categories: {len(result['entity_count_by_category'])}")
                for cat, count in result['entity_count_by_category'].items():
                    print(f"      - {cat}: {count}")
                
                await websocket.send_json(result)
                
            except Exception as e:
                error_msg = {"error": str(e)}
                print(f"❌ ERROR: {e}")
                import traceback
                traceback.print_exc()
                await websocket.send_json(error_msg)
            
            print("="*80 + "\n")
    
    except WebSocketDisconnect:
        print("🔌 WebSocket client disconnected")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        await websocket.close(code=1011)

# =====================================================
# 8️⃣ REST API ENDPOINTS
# =====================================================

@app.post("/api/extract")
async def extract_entities(data: dict):
    """Extract all medical entities from text (REST API)"""
    text = data.get("text", "")
    
    if not text.strip():
        return {"error": "No text provided"}
    
    try:
        raw_entities = ner_pipeline(text)
        result = extract_all_entities(text, raw_entities)
        return result
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@app.get("/api/categories")
async def get_entity_categories():
    """Get all entity categories and their labels"""
    return {
        "model": "blaze999/Medical-NER",
        "total_categories": len(ENTITY_CATEGORIES),
        "categories": {
            category: list(labels) 
            for category, labels in ENTITY_CATEGORIES.items()
        }
    }

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
        "total_entity_types": len(entity_types),
        "total_categories": len(ENTITY_CATEGORIES)
    }

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "Comprehensive Medical NER API",
        "model": "blaze999/Medical-NER",
        "version": "2.0.0",
        "description": "Extracts ALL 41 medical entity types with comprehensive categorization",
        "endpoints": {
            "websocket": "/ws/ner",
            "rest_api": "/api/extract",
            "categories": "/api/categories",
            "entity_types": "/api/entities",
            "health": "/health"
        },
        "features": [
            "41 entity types",
            f"{len(ENTITY_CATEGORIES)} categories",
            "Structured prescriptions, symptoms, procedures",
            "Real-time WebSocket support",
            "Comprehensive entity extraction"
        ]
    }

# =====================================================
# 9️⃣ RUN SERVER
# =====================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 COMPREHENSIVE MEDICAL NER SERVER")
    print("="*80)
    print(f"📊 Model: {model_name}")
    print(f"📋 Entity Types: {len(entity_types)}")
    print(f"🗂️  Categories: {len(ENTITY_CATEGORIES)}")
    print(f"\n📍 Endpoints:")
    print(f"   WebSocket: ws://127.0.0.1:8000/ws/ner")
    print(f"   REST API: http://127.0.0.1:8000/api/extract")
    print(f"   Categories: http://127.0.0.1:8000/api/categories")
    print(f"   Health: http://127.0.0.1:8000/health")
    print("="*80 + "\n")
    
    uvicorn.run(app, host="127.0.0.1", port=8000)