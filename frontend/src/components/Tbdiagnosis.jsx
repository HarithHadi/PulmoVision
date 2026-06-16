import { useState, useRef, useCallback } from "react";

const MOCK = false;

const MOCK_RESULT = {
  prediction: "TB",
  tb_probability: 83.2,
  normal_probability: 16.8,
  overlay_image: "",
  report: `FINDINGS: The chest X-ray demonstrates increased opacity in the right upper lobe with poorly defined margins. There is evidence of consolidation with possible cavitation. The left lung appears clear. No pleural effusion identified. Cardiac silhouette is within normal limits.

IMPRESSION: Findings are consistent with active pulmonary tuberculosis involving the right upper lobe. Clinical correlation and sputum AFB culture are recommended.`,
};

const API_URL = "https://humorous-headache-reenter.ngrok-free.dev/report";

export default function TBDiagnosis() {
  const [file, setFile]             = useState(null);
  const [preview, setPreview]       = useState(null);
  const [result, setResult]         = useState(null);
  const [loading, setLoading]       = useState(false);
  const [loadingStep, setLoadingStep] = useState("");
  const [error, setError]           = useState(null);
  const [dragging, setDragging]     = useState(false);
  const [reportOpen, setReportOpen] = useState(true);  
  const fileInputRef                = useRef(null);
  const stepTimerRef                = useRef(null);

  const steps = [
    "Initializing RAD-DINO...",
    "Extracting patch tokens...",
    "Computing GradCAM heatmap...",
    "Generating visual tokens...",
    "LLaMA-3 generating report...",
    "Almost done...",
  ];

  const startSteps = () => {
    let i = 0;
    setLoadingStep(steps[0]);
    stepTimerRef.current = setInterval(() => {
      i = Math.min(i + 1, steps.length - 1);
      setLoadingStep(steps[i]);
    }, 4000);
  };

  const stopSteps = () => {
    clearInterval(stepTimerRef.current);
    setLoadingStep("");
  };

  const loadFile = (f) => {
    if (!f || !f.type.startsWith("image/")) return;
    setFile(f);
    setResult(null);
    setError(null);
    setReportOpen(true);  // reset on new file
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target.result);
    reader.readAsDataURL(f);
  };

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    loadFile(e.dataTransfer.files[0]);
  }, []);

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    startSteps();

    if (MOCK) {
      await new Promise(r => setTimeout(r, 6000));
      stopSteps();
      setResult(MOCK_RESULT);
      setLoading(false);
      return;
    }

    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "ngrok-skip-browser-warning": "true" },
        body: form,
      });
      if (!res.ok) throw new Error(`Server error ${res.status}`);
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      stopSteps();
      setLoading(false);
    }
  };

  const isTB   = result?.prediction === "TB";
  const tbProb = result?.tb_probability;
  const nrProb = result?.normal_probability;

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 pt-20 pb-8 sm:pt-24 sm:pb-12">

        {/* Hero */}
        <div className="mb-10 sm:mb-12">
          <span className="inline-block text-xs font-semibold tracking-widest text-primary bg-background border border-border/60 px-3 py-1 rounded-full mb-4">
            RAD-DINO · GradCAM · LLaMA-3
          </span>
          <h1 className="text-2xl sm:text-4xl font-bold text-primary leading-tight mb-3">
            Tuberculosis Detection
          </h1>
          <p className="text-sm sm:text-base text-card max-w-lg">
            Upload a chest X-ray to detect TB, visualize suspicious regions, and generate an AI radiology report.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 lg:gap-10 items-start">

          {/* Left */}
          <div className="space-y-4">
            <h2 className="text-xs font-semibold text-card uppercase tracking-widest">Input</h2>

            <div
              onClick={() => fileInputRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              className={`
                border-2 border-dashed rounded-2xl p-8 sm:p-10 text-center cursor-pointer transition-all
                ${dragging
                  ? "border-primary bg-background"
                  : "border-border bg-card hover:border-border-hover hover:bg-card-hover"}
              `}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => loadFile(e.target.files[0])}
              />
              {file ? (
                <>
                  <div className="text-green-400 text-4xl mb-3">✓</div>
                  <p className="text-sm font-medium text-card break-all">{file.name}</p>
                  <p className="text-xs text-slate-500 mt-1">{(file.size / 1024).toFixed(0)} KB — ready to analyze</p>
                  <p className="text-xs text-blue-400 mt-3 underline underline-offset-2">Click to change</p>
                </>
              ) : (
                <>
                  <div className="w-14 h-14 rounded-2xl bg-card border border-card flex items-center justify-center mx-auto mb-4">
                    <svg className="w-6 h-6 text-card" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                      <polyline points="17 8 12 3 7 8"/>
                      <line x1="12" y1="3" x2="12" y2="15"/>
                    </svg>
                  </div>
                  <p className="text-sm font-medium text-foreground">Drop a chest X-ray here</p>
                  <p className="text-xs text-foreground mt-1.5">or click to browse — PNG, JPG</p>
                </>
              )}
            </div>

            {preview && !result && (
              <div className="rounded-xl overflow-hidden border border-border bg-card aspect-video">
                <img src={preview} alt="preview" className="w-full h-full object-contain" />
              </div>
            )}

            <button
              onClick={analyze}
              disabled={!file || loading}
              className="w-full py-3.5 rounded-xl font-semibold text-sm bg-green-400 hover:bg-green-300 active:bg-green-500 text-white transition-all disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {loading && <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />}
              {loading ? "Analyzing..." : "Run Analysis"}
            </button>

            {error && (
              <div className="bg-danger border border-danger-800 text-red-100 text-sm px-4 py-3 rounded-xl">
                ⚠ {error} — is the backend running?
              </div>
            )}

            <div className="grid grid-cols-2 gap-3 pt-2">
              {[
                { label: "Visual Encoder", value: "RAD-DINO" },
                { label: "Localization",   value: "GradCAM" },
                { label: "Report Model",   value: "LLaMA-3.2-3B" },
                { label: "Dataset",        value: "Montgomery + Shenzhen" },
              ].map(({ label, value }) => (
                <div key={label} className="bg-card border border-border rounded-xl px-3 py-2.5">
                  <p className="text-xs text-foreground mb-0.5">{label}</p>
                  <p className="text-xs font-semibold text-foreground">{value}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Right */}
          <div className="space-y-4">
            <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-widest">Results</h2>

            {!result && !loading && (
              <div className="border-2 border-dashed border-border rounded-2xl p-12 text-center">
                <p className="text-foreground text-sm">Results will appear here after analysis</p>
              </div>
            )}

            {loading && (
              <div className="border border-border bg-card rounded-2xl p-12 text-center space-y-3">
                <span className="w-8 h-8 border-2 border-slate-700 border-t-blue-500 rounded-full animate-spin inline-block" />
                <p className="text-sm text-slate-400">Running analysis...</p>
                {loadingStep && <p className="text-xs text-slate-600">{loadingStep}</p>}
              </div>
            )}

            {result && (
              <div className="space-y-4">

                {/* Prediction Banner */}
                <div className={`rounded-2xl px-5 py-4 border-2 border-dashed ${
                  isTB ? "bg-danger/50 border-danger" : "bg-success/50 border-success"
                }`}>
                  <div className="flex items-center justify-between flex-wrap gap-2 mb-3">
                    <div>
                      <p className="text-xs text-background mb-0.5">Prediction</p>
                      <p className={`text-2xl font-bold ${isTB ? "text-red-900" : "text-green-900"}`}>
                        {result.prediction}
                      </p>
                    </div>
                    <div className={`text-right ${isTB ? "text-red-900" : "text-green-900"}`}>
                      <p className="text-3xl font-bold">{isTB ? tbProb : nrProb}%</p>
                      <p className="text-xs text-background">confidence</p>
                    </div>
                  </div>
                  <div className="space-y-2 font-bold">
                    {[
                      { label: "TB",     value: tbProb, color: "from-red-700 to-red-500" },
                      { label: "Normal", value: nrProb, color: "from-green-700 to-green-500" },
                    ].map(({ label, value, color }) => (
                      <div key={label} className="flex items-center gap-3">
                        <span className="text-xs text-background w-12 shrink-0">{label}</span>
                        <div className="flex-1 h-1.5 bg-background rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full bg-gradient-to-r ${color} transition-all duration-700`}
                            style={{ width: `${value}%` }}
                          />
                        </div>
                        <span className="text-xs text-background w-10 text-right shrink-0">{value}%</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Image Panels */}
                <div className="grid grid-cols-2 gap-3">
                  {[
                    { label: "Original",        src: preview },
                    { label: "GradCAM Heatmap", src: `data:image/png;base64,${result.overlay_image}` },
                  ].map(({ label, src }) => (
                    <div key={label} className="rounded-xl overflow-hidden border border-border bg-card">
                      <div className="px-3 py-2 border-b border-border text-xs font-medium text-foreground uppercase tracking-widest">
                        {label}
                      </div>
                      <div className="aspect-square bg-card">
                        <img src={src} alt={label} className="w-full h-full object-contain" />
                      </div>
                    </div>
                  ))}
                </div>

                {/* Legend */}
                <div className="flex items-center gap-3 px-4 py-2.5 bg-card border border-border rounded-xl">
                  <span className="text-xs text-card shrink-0">Low</span>
                  <div className="flex-1 h-2 rounded-full" style={{
                    background: "linear-gradient(to right, #30123b, #4454c4, #1f9f88, #a2da37, #fdc527, #f05b12, #7a0403)"
                  }} />
                  <span className="text-xs text-card shrink-0">High</span>
                </div>

                {/* ── Collapsible Report ───────────────────────────────── */}
                {result.report && (
                  <div className="bg-card border border-border rounded-2xl overflow-hidden">
                    
                    {/* Header — clickable toggle */}
                    <button
                      onClick={() => setReportOpen(o => !o)}
                      className="w-full flex items-center justify-between px-5 py-3.5 hover:bg-card-hover transition-colors"
                    >
                      <div className="flex items-center gap-3">
                        <span className="text-xs font-semibold text-slate-500 uppercase tracking-widest">
                          AI Radiology Report
                        </span>
                        <span className="text-xs text-primary bg-card border border-border rounded px-2 py-0.5">
                          LLaMA-3 · LoRA
                        </span>
                      </div>

                      {/* Chevron icon — rotates when open */}
                      <svg
                        className={`w-4 h-4 text-foreground transition-transform duration-300 ${
                          reportOpen ? "rotate-180" : ""
                        }`}
                        fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"
                      >
                        <path d="M6 9l6 6 6-6"/>
                      </svg>
                    </button>

                    {/* Collapsible body with Smooth Animation */}
                    <div
                      className={`grid transition-all duration-300 ease-in-out ${
                        reportOpen ? "grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0"
                      }`}
                    >
                      {/* 
                        This inner div requires overflow-hidden so the text 
                        doesn't spill out while the grid row collapses to 0fr 
                      */}
                      <div className="overflow-hidden">
                        <div className="px-5 pb-5 pt-1 border-t border-border">
                          <p className="text-sm text-foreground leading-relaxed whitespace-pre-wrap">
                            {result.report}
                          </p>
                        </div>
                      </div>
                    </div>

                  </div>
                )}

                <p className="text-center text-xs text-slate-600 pb-2 mt-4">
                  ⚕ For research purposes only. Not a substitute for clinical diagnosis.
                </p>

              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}