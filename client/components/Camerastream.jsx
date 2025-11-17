import { useEffect, useRef, useState } from "react";

export default function CameraStream() {
  const videoRef = useRef(null);
  const [status, setStatus] = useState("waiting...");

  useEffect(() => {
    async function initCamera() {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;
    }
    initCamera();

    const interval = setInterval(async () => {
      if (!videoRef.current) return;

      const canvas = document.createElement("canvas");
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      canvas.getContext("2d").drawImage(videoRef.current, 0, 0);

      const dataUrl = canvas.toDataURL("image/jpeg");

      const API_URL = process.env.NEXT_PUBLIC_API_URL;
      const res = await fetch(`${API_URL}/api/messages/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ frame: dataUrl }),
      });

      const data = await res.json();
      setStatus(data.result);
    }, 500);

    return () => clearInterval(interval);
  }, []);

  return (
    <div>
      <video ref={videoRef} autoPlay style={{ width: "400px" }} />
      <h2 style={{ color: status === "pass" ? "green" : "red" }}>{status}</h2>
    </div>
  );
}
