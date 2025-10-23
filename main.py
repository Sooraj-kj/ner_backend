import spacy
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

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
# 2️⃣ LOAD HUNFLAIR BC5CDR MODEL (for diseases & symptoms)
# =====================================================
try:
    model_name = "d4data/biomedical-ner-all"
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForTokenClassification.from_pretrained(model_name)
    hf_pipeline = pipeline("ner", model=hf_model, tokenizer=hf_tokenizer, aggregation_strategy="max")
    print("✅ BioSyn-SapBERT (BC5CDR) model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load BioSyn-SapBERT model: {e}")
    exit()

# =====================================================
# 3️⃣ FASTAPI SETUP
# =====================================================
app = FastAPI(
    title="Combined Medical NER WebSocket API",
    description="Extracts drugs, dosage, route, and symptoms/diseases using Med7 + BioSyn-SapBERT",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# 4️⃣ WEBSOCKET ENDPOINT
# =====================================================
@app.websocket("/ws/ner")
async def ner_websocket(websocket: WebSocket):
    """
    Accepts a WebSocket connection.
    Receives text, processes it with Med7 + BioSyn-SapBERT,
    and sends back JSON of all extracted entities.
    """
    await websocket.accept()
    print("🩺 NER WebSocket client connected.")
    try:
        while True:
            text = await websocket.receive_text()

            if text.strip():
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

                # --- Run BioSyn-SapBERT (BC5CDR) ---
                biosyn_entities = []
                try:
                    biosyn_output = hf_pipeline(text)
                    biosyn_entities = [
                        {
                            "text": ent["word"],
                            "label": ent["entity_group"],
                            "score": float(ent["score"]),
                            "source": "BioSyn-SapBERT",
                            "start_char": int(ent["start"]),
                            "end_char": int(ent["end"])
                        }
                        for ent in biosyn_output
                    ]
                except Exception as e:
                    print(f"⚠️ Error processing BioSyn-SapBERT: {e}")

                # --- Combine results ---
                all_entities = med7_entities + biosyn_entities

                # --- Send back JSON ---
                await websocket.send_json(all_entities)

    except WebSocketDisconnect:
        print("🔌 WebSocket client disconnected.")
    except Exception as e:
        print(f"NER WebSocket Error: {e}")
        await websocket.close(code=1011)  # Internal error


# =====================================================
# 5️⃣ RUN SERVER
# =====================================================
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
