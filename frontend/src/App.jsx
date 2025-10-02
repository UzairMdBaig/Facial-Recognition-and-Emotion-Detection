import { useRef, useState, useEffect } from "react";
import Webcam from "react-webcam";

function App() {
  const webcamRef = useRef(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // capture and send every 5 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      capture();
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  const capture = async () => {
    if (!webcamRef.current) return;

    const imageSrc = webcamRef.current.getScreenshot();
    if (!imageSrc) return;

    setLoading(true);

    try {
      // Convert base64 → blob
      const res = await fetch(imageSrc);
      const blob = await res.blob();

      // Send to backend
      const formData = new FormData();
      formData.append("file", blob, "capture.jpg");

      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errText = await response.text();
        console.error("Backend error:", errText);
        setResult({ ID: "Error", Emotion: "Backend failed" });
      } else {
        const data = await response.json();
        console.log("Backend response:", data);
        setResult(data);
      }
    } catch (err) {
      console.error("Frontend error:", err);
      setResult({ ID: "Error", Emotion: "Network issue" });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center h-screen w-screen bg-[rgb(15,15,16)] gap-4">
      <h1 className="text-[rgb(209,197,173)] text-3xl font-bold mt-4">
        Facial Recognition & Emotion Detection
      </h1>

      <Webcam
        ref={webcamRef}
        className="rounded-xl shadow-lg"
        mirrored={true}
        screenshotFormat="image/jpeg"
      />

      <div className="text-[rgb(209,197,173)] text-xl font-bold mt-2 flex gap-5">
        {result && !loading && (
          <>
            <p>Person: {result.ID}</p>
            <p>Emotion: {result.Emotion}</p>
          </>
        )}
      </div>
    </div>
  );
}

export default App;
