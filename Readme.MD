# Interview Grader

## Overview
Interview Grader is a full-stack application for real-time facial emotion recognition. The frontend is built with React.js, and the backend uses FastAPI. The emotion prediction model is developed with TensorFlow Keras, while real-time face detection on the frontend is handled using TensorFlow.js.

## Technologies Used
- **ReactJS**
  - Builds the front-end UI.
  - Displays real-time facial emotions.
  - Connects to WebSocket for video frame transmission and emotion predictions.
- **FastAPI**
  - Manages WebSocket connections for real-time data exchange.
  - Hosts the API for model predictions.
- **TensorFlow Keras**
  - Deploys a pre-trained emotion classification model.
  - Predicts seven emotions: happy, sad, neutral, disgusted, surprised, fear, and angry.
  - Outputs emotion probabilities for each frame.
- **TensorFlow.js**
  - Detects faces in real-time using a Single Shot Detector architecture (Blazeface).

## Implementation
This project enables real-time emotion recognition through a full-stack pipeline.

### Conda
Make sure that you have conda install in your machine. You can install it with conda or miniconda (both works fine). After that,
add the conda into your environment. If you don't know how to create conda environment, call me.

```sh
conda create --name env python==3.9
cd interview-grader
cd server
pip install -r requirements.txt
```

### Start the API Server
```sh
cd server
uvicorn main:app --reload
```

### Start the React Server
Create another terminal
```sh
cd emotion-recognition
npm install
npm start
```

## Features
- Real-time emotion detection from webcam feed.
- Progress bars visualizing emotion intensity.
- Bounding box around detected faces, color-coded based on dominant emotion.
- WebSocket-based real-time data exchange between frontend and backend.

## Architecture
- Video frames are sent from React to FastAPI via WebSockets.
- The FastAPI server processes frames and returns emotion predictions.
- TensorFlow.js detects faces, and TensorFlow Keras predicts emotions.
- The frontend dynamically updates UI elements based on received predictions.

## Conclusion
- Successfully streams and analyzes emotions in real-time.
- WebSockets facilitate seamless backend-frontend communication.
- Scope for improvement with lighter models for better performance.

## Future Improvements
- Body movement detection.
- Dockerization.
- Integration with a database.

