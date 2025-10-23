import os
import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from websockets import connect as ws_connect
from dotenv import load_dotenv

load_dotenv() 

app = FastAPI()

SONIOX_WS_URL = 'wss://stt-rt.soniox.com/transcribe-websocket'
SONIOX_API_KEY = os.getenv("SONIOX_API_KEY") 

if SONIOX_API_KEY:
    print(f"✅ SONIOX_API_KEY loaded successfully: {SONIOX_API_KEY[:4]}...{SONIOX_API_KEY[-4:]}")
else:
    print("❌ FATAL ERROR: SONIOX_API_KEY environment variable not found.")

@app.websocket("/ws/soniox")
async def soniox_proxy(websocket: WebSocket):
    await websocket.accept()
    
    if not SONIOX_API_KEY:
        print("Closing connection: Server API key not configured.")
        await websocket.close(code=1008, reason="Server API key not configured.")
        return

    soniox_ws = None # Define soniox_ws in the outer scope
    try:
        # 1. Receive config from client
        start_config_json = await websocket.receive_text()
        start_config = json.loads(start_config_json)
        
        # 2. Build Soniox URL with API key as query param
        soniox_url_with_key = f"{SONIOX_WS_URL}?key={SONIOX_API_KEY}"
        
        # 3. Remove 'api_key' from the config, as it's now in the URL
        start_config.pop('api_key', None) 
        
        config_to_send = json.dumps(start_config)
        
        # --- NEW DEBUG LOGS ---
        print("--- Connecting to Soniox URL: ---")
        print(f"{SONIOX_WS_URL}?key={SONIOX_API_KEY[:4]}...{SONIOX_API_KEY[-4:]}")
        print("--- Sending this config to Soniox: ---")
        print(config_to_send)
        print("----------------------------------------")
        
        async with ws_connect(soniox_url_with_key) as soniox_ws:
            # 4. Send start config to Soniox (without API key in the JSON)
            await soniox_ws.send(config_to_send)
            
            print("✅ Connection to Soniox successful, config sent.")
            
            async def client_to_soniox():
                while True:
                    msg = await websocket.receive()
                    if msg.get("type") == "websocket.disconnect":
                        print("Client disconnected, closing Soniox connection.")
                        await soniox_ws.close()
                        break
                    
                    if "text" in msg:
                        await soniox_ws.send(msg["text"])
                    elif "bytes" in msg:
                        await soniox_ws.send(msg["bytes"])
            
            async def soniox_to_client():
                async for message in soniox_ws:
                    await websocket.send_text(message)
            
            await asyncio.gather(client_to_soniox(), soniox_to_client())
    
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error in WebSocket proxy: {e}")
        if websocket.client_state != "disconnected":
            await websocket.close()
    finally:
        # Ensure Soniox connection is closed if an error happens
        if soniox_ws and not soniox_ws.closed:
            await soniox_ws.close()
        print("Proxy connection fully closed.")