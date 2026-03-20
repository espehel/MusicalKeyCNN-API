import { useState, useRef } from "react";

interface KeyResult {
  file: string;
  camelot: string;
  key: string;
  id: number;
}

interface BpmResult {
  file: string;
  bpm: number;
  [key: string]: unknown;
}

type Status = "idle" | "loading" | "done" | "error";

export default function App() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [keyResult, setKeyResult] = useState<KeyResult | null>(null);
  const [bpmResult, setBpmResult] = useState<BpmResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function analyze() {
    if (!file) return;
    setStatus("loading");
    setError(null);
    setKeyResult(null);
    setBpmResult(null);

    const form1 = new FormData();
    form1.append("file", file);
    const form2 = new FormData();
    form2.append("file", file);

    try {
      const [keyRes, bpmRes] = await Promise.all([
        fetch("/predict", { method: "POST", body: form1 }),
        fetch("/analyze/bpm", { method: "POST", body: form2 }),
      ]);

      const [key, bpm] = await Promise.all([
        keyRes.ok ? keyRes.json() : keyRes.json().then((e: { detail: string }) => Promise.reject(e.detail)),
        bpmRes.ok ? bpmRes.json() : bpmRes.json().then((e: { detail: string }) => Promise.reject(e.detail)),
      ]);

      setKeyResult(key);
      setBpmResult(bpm);
      setStatus("done");
    } catch (e) {
      setError(String(e));
      setStatus("error");
    }
  }

  return (
    <div style={{ fontFamily: "sans-serif", maxWidth: 520, margin: "80px auto", padding: "0 16px" }}>
      <h1 style={{ fontSize: 24, marginBottom: 8 }}>MusicalKeyCNN</h1>
      <p style={{ color: "#666", marginBottom: 32 }}>Predict the key and BPM of an audio file.</p>

      <input
        ref={inputRef}
        type="file"
        accept=".mp3,.wav,.flac,.aiff,.ogg"
        style={{ display: "none" }}
        onChange={(e) => setFile(e.target.files?.[0] ?? null)}
      />

      <div
        onClick={() => inputRef.current?.click()}
        style={{
          border: "2px dashed #ccc",
          borderRadius: 8,
          padding: "40px 24px",
          textAlign: "center",
          cursor: "pointer",
          marginBottom: 16,
          background: file ? "#f0fff4" : "#fafafa",
        }}
      >
        {file ? file.name : "Click to select an audio file"}
      </div>

      <button
        onClick={analyze}
        disabled={!file || status === "loading"}
        style={{
          width: "100%",
          padding: "12px 0",
          fontSize: 16,
          borderRadius: 6,
          border: "none",
          background: "#1a1a1a",
          color: "#fff",
          cursor: file && status !== "loading" ? "pointer" : "not-allowed",
          opacity: file && status !== "loading" ? 1 : 0.5,
        }}
      >
        {status === "loading" ? "Analyzing..." : "Analyze"}
      </button>

      {status === "done" && keyResult && bpmResult && (
        <div
          style={{
            marginTop: 32,
            padding: 24,
            borderRadius: 8,
            background: "#f8f8f8",
            display: "grid",
            gridTemplateColumns: "1fr 1fr 1fr",
            gap: 16,
            textAlign: "center",
          }}
        >
          <div>
            <div style={{ fontSize: 12, color: "#888", textTransform: "uppercase", letterSpacing: 1 }}>Key</div>
            <div style={{ fontSize: 28, fontWeight: 700, marginTop: 4 }}>{keyResult.key}</div>
          </div>
          <div>
            <div style={{ fontSize: 12, color: "#888", textTransform: "uppercase", letterSpacing: 1 }}>Camelot</div>
            <div style={{ fontSize: 28, fontWeight: 700, marginTop: 4 }}>{keyResult.camelot}</div>
          </div>
          <div>
            <div style={{ fontSize: 12, color: "#888", textTransform: "uppercase", letterSpacing: 1 }}>BPM</div>
            <div style={{ fontSize: 28, fontWeight: 700, marginTop: 4 }}>{bpmResult.bpm}</div>
          </div>
        </div>
      )}

      {status === "error" && (
        <div style={{ marginTop: 24, color: "#c00", padding: 16, background: "#fff0f0", borderRadius: 6 }}>
          {error}
        </div>
      )}
    </div>
  );
}
