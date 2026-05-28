import { useState, useRef, useCallback } from "react";

const API_URL = "http://localhost:8000/diagnose";

export default function TBDiagnosis() {
  const [file, setFile]         = useState(null);
  const [preview, setPreview]   = useState(null);
  const [result, setResult]     = useState(null);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState(null);
  const [dragging, setDragging] = useState(false);
  const fileInputRef            = useRef(null);

  const loadFile = (f) => {
    if (!f || !f.type.startsWith("image/")) return;
    setFile(f);
    setResult(null);
    setError(null);
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
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(API_URL, { method: "POST", body: form });
      if (!res.ok) throw new Error(`Server error ${res.status}`);
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const isTB   = result?.prediction === "TB";
  const tbProb = result?.tb_probability;
  const nrProb = result?.normal_probability;

  return (
    <div className="min-h-screen bg-background text-foreground">

      {/* Page */}
      <div className="max-w-5xl mx-auto px-4 sm:px-6 pt-20 pb-8 sm:pt-24 sm:pb-12">

        {/* Hero */}
        <div className="mb-10 sm:mb-12">
          <span className="inline-block text-xs font-semibold tracking-widest text-prmiary bg-background border border-border/60 px-3 py-1 rounded-full mb-4">
            RAD-DINO + GradCAM
          </span>
          <h1 className="text-2xl sm:text-4xl font-bold text-primary leading-tight mb-3">
            Tuberculosis Detection
          </h1>
          <p className="text-sm sm:text-base text-card max-w-lg">
            Upload a chest X-ray to detect TB and visualize the regions the model focuses on using gradient-weighted class activation mapping.
          </p>
        </div>

        {/* Two column layout on large screens */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 lg:gap-10 items-start">

          {/* Left — Upload + Controls */}
          <div className="space-y-4">
            <h2 className="text-xs font-semibold text-card uppercase tracking-widest">Input</h2>

            {/* Drop Zone */}
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

            {/* Preview thumbnail */}
            {preview && !result && (
              <div className="rounded-xl overflow-hidden border border-border bg-card/100 aspect-video">
                <img src={preview} alt="preview" className="w-full h-full object-contain" />
              </div>
            )}

            {/* Analyze Button */}
            <button
              onClick={analyze}
              disabled={!file || loading}
              className="w-full py-3.5 rounded-xl font-semibold text-sm bg-green-400 hover:bg-green-300 active:bg-green-500 text-white transition-all disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {loading && (
                <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              )}
              {loading ? "Analyzing..." : "Run Analysis"}
            </button>

            {/* Error */}
            {error && (
              <div className="bg-danger border border-danger-800 text-red-100 text-sm px-4 py-3 rounded-xl">
                ⚠ {error} — is the backend running?
              </div>
            )}

            {/* Info cards */}
            <div className="grid grid-cols-2 gap-3 pt-2">
              {[
                { label: "Model",    value: "RAD-DINO" },
                { label: "Method",   value: "GradCAM" },
                { label: "Dataset",  value: "Montgomery + Shenzhen" },
                { label: "Classes",  value: "TB / Normal" },
              ].map(({ label, value }) => (
                <div key={label} className="bg-card border border-border rounded-xl px-3 py-2.5">
                  <p className="text-xs text-foreground mb-0.5">{label}</p>
                  <p className="text-xs font-semibold text-foreground">{value}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Right — Results */}
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
                <p className="text-sm text-slate-400">Running RAD-DINO inference...</p>
              </div>
            )}

            {result && (
              <div className="space-y-4">

                {/* Prediction Banner */}
                <div className={`rounded-2xl px-5 py-4 border-2 border-dashed ${
                  isTB
                    ? "bg-danger/50 border-danger"
                    : "bg-success/50 border-success"
                }`}>
                  <div className="flex items-center justify-between flex-wrap gap-2 mb-3">
                    <div>
                      <p className="text-xs text-background mb-0.5">Prediction</p>
                      <p className={`text-2xl font-bold ${isTB ? "text-red-900" : "text-green-900"}`}>
                        {result.prediction}
                      </p>
                    </div>
                    <div className={`text-right text-xs ${isTB ? "text-red-900" : "text-green-900"}`}>
                      <p className="text-3xl font-bold">{isTB ? tbProb : nrProb}%</p>
                      <p className="text-background">confidence</p>
                    </div>
                  </div>

                  {/* Bars */}
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

                {/* Disclaimer */}
                <p className="text-center text-xs text-slate-600 pb-2">
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