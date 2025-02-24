import React, { useRef, useEffect, useState } from "react";
import "./App.css";
import logo from "./logo.svg";
import * as tf from "@tensorflow/tfjs";
import Webcam from "react-webcam";
import { drawMesh } from "./utilities";
import { Pose } from "@mediapipe/pose/pose.js";
import { Camera } from "@mediapipe/camera_utils/camera_utils.js";

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const blazeface = require("@tensorflow-models/blazeface");
  const [alert, setAlert] = useState("");
  const pose = useRef(null);
  const movementHistory = useRef([]);
  const alertTimeout = useRef(null);

  // MediaPipe Pose initialization (unchanged)
  useEffect(() => {
    pose.current = new Pose({
      locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
    });

    pose.current.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    pose.current.onResults((results) => {
      if (results.poseLandmarks) {
        const leftShoulder = results.poseLandmarks[11];
        const rightShoulder = results.poseLandmarks[12];
        checkMovement(leftShoulder, rightShoulder);
      }
    });

    if (webcamRef.current) {
      const camera = new Camera(webcamRef.current.video, {
        onFrame: async () => {
          await pose.current.send({ image: webcamRef.current.video });
        },
        width: 640,
        height: 480,
      });
      camera.start();
    }
  }, []);

  // Modified movement detection (ONLY THIS SECTION CHANGED)
  const checkMovement = (left, right) => {
    const sensitivity = 0.03; // More sensitive threshold
    const maxHistory = 5; // Check movement over 5 frames

    movementHistory.current = [
      ...movementHistory.current.slice(-(maxHistory - 1)),
      { left, right, timestamp: Date.now() },
    ];

    if (movementHistory.current.length < 2) return;

    const current = movementHistory.current[movementHistory.current.length - 1];
    const previous = movementHistory.current[0];

    const leftMovement = Math.hypot(
      current.left.x - previous.left.x,
      current.left.y - previous.left.y
    );

    const rightMovement = Math.hypot(
      current.right.x - previous.right.x,
      current.right.y - previous.right.y
    );

    if (leftMovement > sensitivity || rightMovement > sensitivity) {
      if (!alertTimeout.current) {
        setAlert("ALERT! Fidgeting Detected");
        alertTimeout.current = setTimeout(() => {
          setAlert("");
          alertTimeout.current = null;
        }, 1000);
      }
    }
  };

  // Everything below remains EXACTLY AS ORIGINAL
  const runFaceDetectorModel = async () => {
    const model = await blazeface.load();
    console.log("FaceDetection Model is Loaded..");
    setInterval(() => {
      detect(model);
    }, 100);
  };

  const detect = async (net) => {
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4
    ) {
      const video = webcamRef.current.video;
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;

      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;
      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;

      const face = await net.estimateFaces(video);
      var socket = new WebSocket("ws://localhost:8000");
      var imageSrc = webcamRef.current.getScreenshot();
      var apiCall = {
        event: "localhost:subscribe",
        data: { image: imageSrc },
      };

      socket.onopen = () => socket.send(JSON.stringify(apiCall));
      socket.onmessage = function (event) {
        var pred_log = JSON.parse(event.data);
        document.getElementById("Angry").value = Math.round(
          pred_log["predictions"]["angry"] * 100
        );
        document.getElementById("Neutral").value = Math.round(
          pred_log["predictions"]["neutral"] * 100
        );
        document.getElementById("Happy").value = Math.round(
          pred_log["predictions"]["happy"] * 100
        );
        document.getElementById("Fear").value = Math.round(
          pred_log["predictions"]["fear"] * 100
        );
        document.getElementById("Surprise").value = Math.round(
          pred_log["predictions"]["surprise"] * 100
        );
        document.getElementById("Sad").value = Math.round(
          pred_log["predictions"]["sad"] * 100
        );
        document.getElementById("Disgust").value = Math.round(
          pred_log["predictions"]["disgust"] * 100
        );
        document.getElementById("emotion_text").value = pred_log["emotion"];

        const ctx = canvasRef.current.getContext("2d");
        requestAnimationFrame(() => {
          drawMesh(face, pred_log, ctx);
        });
      };
    }
  };

  useEffect(() => {
    runFaceDetectorModel();
  }, []);

  return (
    <div className="App">
      {alert && (
        <div
          style={{
            position: "absolute",
            left: "20px",
            top: "20px",
            color: "red",
            fontSize: "24px",
            fontWeight: "bold",
            zIndex: 1000,
          }}
        >
          {alert}
        </div>
      )}

      <Webcam
        ref={webcamRef}
        style={{
          position: "absolute",
          marginLeft: "auto",
          marginRight: "auto",
          left: 0,
          right: 600,
          top: 20,
          textAlign: "center",
          zindex: 9,
          width: 640,
          height: 480,
        }}
      />

      <canvas
        ref={canvasRef}
        style={{
          position: "absolute",
          marginLeft: "auto",
          marginRight: "auto",
          left: 0,
          right: 600,
          top: 20,
          textAlign: "center",
          zindex: 9,
          width: 640,
          height: 480,
        }}
      />
      <header className="App-header">
        <img
          src={logo}
          className="App-logo"
          alt="logo"
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            bottom: 10,
            left: 0,
            right: 0,
            width: 150,
            height: 150,
          }}
        />
        <div
          className="Prediction"
          style={{
            position: "absolute",
            right: 100,
            width: 500,
            top: 60,
          }}
        >
          <label htmlFor="Angry" style={{ color: "red" }}>
            Angry{" "}
          </label>
          <progress id="Angry" value="0" max="100">
            10%
          </progress>
          <br />
          <br />
          <label htmlFor="Neutral" style={{ color: "lightgreen" }}>
            Neutral{" "}
          </label>
          <progress id="Neutral" value="0" max="100"></progress>
          <br />
          <br />
          <label htmlFor="Happy" style={{ color: "orange" }}>
            Happy{" "}
          </label>
          <progress id="Happy" value="0" max="100"></progress>
          <br />
          <br />
          <label htmlFor="Fear" style={{ color: "lightblue" }}>
            Fear{" "}
          </label>
          <progress id="Fear" value="0" max="100"></progress>
          <br />
          <br />
          <label htmlFor="Surprise" style={{ color: "yellow" }}>
            Surprised{" "}
          </label>
          <progress id="Surprise" value="0" max="100"></progress>
          <br />
          <br />
          <label htmlFor="Sad" style={{ color: "gray" }}>
            Sad{" "}
          </label>
          <progress id="Sad" value="0" max="100"></progress>
          <br />
          <br />
          <label htmlFor="Disgust" style={{ color: "pink" }}>
            Disgusted{" "}
          </label>
          <progress id="Disgust" value="0" max="100"></progress>
        </div>
        <input
          id="emotion_text"
          name="emotion_text"
          value="Neutral"
          style={{
            position: "absolute",
            width: 200,
            height: 50,
            bottom: 60,
            left: 300,
            fontSize: "30px",
          }}
        />
      </header>
    </div>
  );
}

export default App;