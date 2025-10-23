import spacy
from typing import List
from models.schemas import MedicalEntity
import logging

logger = logging.getLogger(__name__)

class NERService:
    def __init__(self):
        try:
            # Load Med7 model - specifically trained for medical entities
            self.nlp = spacy.load("en_core_med7_lg")
            logger.info("Med7 Medical NER model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Med7 model: {e}")
            logger.error("Make sure to install: pip install 'en-core-med7-lg @ https://huggingface.co/kormilitzin/en_core_med7_lg/resolve/main/en_core_med7_lg-any-py3-none-any.whl'")
            raise

    def extract_entities(self, text: str) -> List[MedicalEntity]:
        """Extract medical entities from text using Med7"""
        try:
            doc = self.nlp(text)
            entities = []
            
            for ent in doc.ents:
                entities.append(MedicalEntity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.92  # Med7 has high accuracy
                ))
            
            logger.debug(f"Extracted {len(entities)} entities from: {text[:50]}...")
            return entities
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []

    def get_entity_labels(self) -> dict:
        """Return available entity labels and their descriptions"""
        return {
            "DRUG": "Medications and pharmaceutical compounds",
            "DISEASE": "Medical conditions and diseases",
            "SYMPTOM": "Clinical symptoms and signs",
            "PROCEDURE": "Medical procedures and interventions",
            "ANATOMY": "Body parts and anatomical structures",
            "DOSAGE": "Medication dosages and frequencies",
            "DURATION": "Treatment duration and time periods"
        }