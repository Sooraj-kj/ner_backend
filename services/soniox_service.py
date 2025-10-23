import asyncio
import websockets
import json
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class SonioxService:
    """
    Real Soniox WebSocket service implementation
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.ws_url = "wss://stt-rt.soniox.com/transcribe-websocket"
        self.connection: Optional[websockets.WebSocketClientProtocol] = None
        self.connected = False
        self.message_count = 0

    async def connect(self, language: str = "en"):
        """Connect to Soniox WebSocket API"""
        try:
            # Connect to Soniox WebSocket
            logger.info(f"Connecting to Soniox at {self.ws_url}")
            self.connection = await websockets.connect(self.ws_url)
            logger.info("WebSocket connection established")
            
            # Send initial configuration with API key
            config = {
                "api_key": self.api_key,
                "model": "en_v2",
                "include_nonfinal": True,
                "enable_endpoint_detection": True,
            }
            
            # Add language if not auto-detect
            if language and language != "auto":
                config["language"] = language
            
            logger.info(f"Sending config to Soniox: {config}")
            await self.connection.send(json.dumps(config))
            
            # Wait for initial response from Soniox
            try:
                initial_response = await asyncio.wait_for(self.connection.recv(), timeout=5.0)
                logger.info(f"Soniox initial response: {initial_response}")
            except asyncio.TimeoutError:
                logger.warning("No initial response from Soniox (this might be normal)")
            
            self.connected = True
            logger.info(f"✓ Connected to Soniox API with language: {language}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Soniox: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.connected = False
            return False

    async def send_audio(self, audio_data: bytes):
        """Send audio data to Soniox"""
        if not self.connection or not self.connected:
            raise ConnectionError("Not connected to Soniox")
        
        try:
            # Send raw audio bytes directly
            await self.connection.send(audio_data)
            self.message_count += 1
            
            # Log every 50 messages
            if self.message_count % 50 == 0:
                logger.debug(f"Sent {self.message_count} audio chunks to Soniox")
                
        except Exception as e:
            logger.error(f"Error sending audio: {e}")
            self.connected = False
            raise

    async def receive_transcription(self) -> Optional[dict]:
        """Receive transcription from Soniox"""
        if not self.connection or not self.connected:
            return None
        
        try:
            # Receive message from Soniox with timeout
            message = await asyncio.wait_for(self.connection.recv(), timeout=0.1)
            
            # Parse JSON response
            data = json.loads(message)
            
            logger.info(f"📨 Received from Soniox: {json.dumps(data, indent=2)}")
            
            # Soniox returns results with 'words' array
            if "words" in data and len(data["words"]) > 0:
                # Combine words into full text
                words = data["words"]
                text = " ".join([word["text"] for word in words])
                
                # Check for 'final' field (Soniox format)
                is_final = data.get("final", False)
                
                logger.info(f"✓ Transcription: '{text}' (final={is_final})")
                
                return {
                    "text": text,
                    "is_final": is_final,
                    "language": data.get("language", "en"),
                    "confidence": data.get("confidence", 1.0)
                }
            
            # Check for status messages
            if "status" in data:
                logger.info(f"Soniox status: {data['status']}")
            
            # Check for error messages
            if "error" in data:
                logger.error(f"Soniox error: {data['error']}")
            
            return None
            
        except asyncio.TimeoutError:
            # No data yet, that's okay - this is normal
            return None
        except websockets.exceptions.ConnectionClosed:
            logger.warning("⚠️ Soniox connection closed")
            self.connected = False
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Soniox response: {e}")
            logger.error(f"Raw message: {message}")
            return None
        except Exception as e:
            logger.error(f"Error receiving transcription: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    async def close(self):
        """Close the connection"""
        self.connected = False
        if self.connection:
            try:
                await self.connection.close()
                logger.info("Soniox connection closed")
            except:
                pass
            self.connection = None