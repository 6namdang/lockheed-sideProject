import json
import io
import base64
import logging
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from fer import FER

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware with more permissive settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
    expose_headers=["*"]  # Expose all headers
)

# Add a health check endpoint
@app.get("/")
async def health_check():
    return JSONResponse({"status": "healthy"})

detector = FER()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_host = websocket.client.host
    logger.info(f"New WebSocket connection attempt from {client_host}")
    
    try:
        await websocket.accept()
        logger.info(f"WebSocket connection established with {client_host}")
        
        while True:  # Keep connection open
            try:
                payload = await websocket.receive_text()
                logger.info("Received payload")
                
                payload = json.loads(payload)
                imageByt64 = payload['data']['image'].split(',')[1]
                
                # decode and convert into image
                image = np.fromstring(base64.b64decode(imageByt64), np.uint8)
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                
                # Detect Emotion via Tensorflow model
                prediction = detector.detect_emotions(image)
                if prediction and len(prediction) > 0:
                    response = {
                        "predictions": prediction[0]['emotions'],
                        "emotion": max(prediction[0]['emotions'], key=prediction[0]['emotions'].get)
                    }
                    await websocket.send_json(response)
                    logger.info("Sent emotion prediction response")
                else:
                    logger.warning("No face detected in image")
                    await websocket.send_json({"error": "No face detected"})
                    
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                await websocket.send_json({"error": "Invalid JSON format"})
            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}")
                await websocket.send_json({"error": f"Error processing frame: {str(e)}"})
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket connection closed with {client_host}")
    except Exception as e:
        logger.error(f"WebSocket error with {client_host}: {str(e)}")
    finally:
        try:
            await websocket.close()
        except:
            pass