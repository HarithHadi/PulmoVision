import { useState, useRef, useCallback, useEffect } from "react";

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
const BASE_URL = "https://humorous-headache-reenter.ngrok-free.dev";

export default function TBDiagnosis() {
  const [file, setFile]               = useState(null);
  const [preview, setPreview]         = useState(null);
  const [result, setResult]           = useState(null);
  const [loading, setLoading]         = useState(false);
  const [loadingStep, setLoadingStep] = useState("");
  const [error, setError]             = useState(null);
  const [dragging, setDragging]       = useState(false);
  const [reportOpen, setReportOpen]   = useState(true);
  const fileInputRef                  = useRef(null);
  const stepTimerRef                  = useRef(null);

  // Auth state
  const [token, setToken]             = useState(null);
  const [radiologistName, setRadiologistName] = useState(null);
  const [showWelcome, setShowWelcome] = useState(true);  // show on first load
  const [showLogin, setShowLogin]     = useState(false);
  const [showSave, setShowSave]       = useState(false);
  const [loginError, setLoginError]   = useState("");
  const [loginLoading, setLoginLoading] = useState(false);
  const [saveLoading, setSaveLoading] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [loginContext, setLoginContext] = useState("welcome"); // "welcome" | "save"

  // Login form
  const [radId, setRadId]             = useState("");
  const [password, setPassword]       = useState("");

  // Save form
  const [patientName, setPatientName] = useState("");
  const [patientAge, setPatientAge]   = useState("");
  const [patientSex, setPatientSex]   = useState("M");
  const [patientContact, setPatientContact] = useState("");
  const [radDiagnosis, setRadDiagnosis] = useState("");
  const [radNotes, setRadNotes]       = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [selectedPatientId, setSelectedPatientId] = useState(null);

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

  const resetDiagnosis = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setReportOpen(true);
    setSaveSuccess(false);
  };

  const loadFile = (f) => {
    if (!f || !f.type.startsWith("image/")) return;
    setFile(f);
    setResult(null);
    setError(null);
    setReportOpen(true);
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
    setSaveSuccess(false);
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

  // Login submit — works for both welcome and save contexts
  const handleLogin = async () => {
    setLoginLoading(true);
    setLoginError("");
    try {
      const res = await fetch(`${BASE_URL}/auth/login`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "ngrok-skip-browser-warning": "true",
        },
        body: JSON.stringify({ radiologist_id: radId, password }),
      });
      if (!res.ok) {
        setLoginError("Invalid credentials. Please try again.");
        return;
      }
      const data = await res.json();
      setToken(data.access_token);
      setRadiologistName(`Dr. ${data.radiologist_id}`);
      setShowLogin(false);
      setShowWelcome(false);
      setRadId("");
      setPassword("");

      // If they logged in from the save button, open save modal immediately
      if (loginContext === "save") {
        setShowSave(true);
      }
    } catch {
      setLoginError("Could not connect to server.");
    } finally {
      setLoginLoading(false);
    }
  };

  const handleSaveClick = () => {
    setSaveSuccess(false);
    if (!token) {
      setLoginContext("save");
      setShowLogin(true);
    } else {
      setShowSave(true);
    }
  };

  const handleNameSearch = async (name) => {
    setPatientName(name);
    setSelectedPatientId(null);
    if (name.length < 2) { setSearchResults([]); return; }
    try {
      const res = await fetch(`${BASE_URL}/patients/search?name=${encodeURIComponent(name)}`, {
        headers: {
          "Authorization": `Bearer ${token}`,
          "ngrok-skip-browser-warning": "true",
        },
      });
      const data = await res.json();
      setSearchResults(data);
    } catch {
      setSearchResults([]);
    }
  };

  const handleSelectPatient = (patient) => {
    setSelectedPatientId(patient.id);
    setPatientName(patient.name);
    setPatientAge(String(patient.age));
    setPatientSex(patient.sex);
    setPatientContact(patient.contact_number || "");
    setSearchResults([]);
  };

  const handleSave = async () => {
    setSaveLoading(true);
    try {
      let patientId = selectedPatientId;

      if (!patientId) {
        const patRes = await fetch(`${BASE_URL}/patients/`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${token}`,
            "ngrok-skip-browser-warning": "true",
          },
          body: JSON.stringify({
            name: patientName,
            age: parseInt(patientAge),
            sex: patientSex,
            contact_number: patientContact,
          }),
        });
        const patData = await patRes.json();
        patientId = patData.id;
      }

      const form = new FormData();
      form.append("patient_id", patientId);
      form.append("tb_probability", result.tb_probability / 100);
      form.append("llama_diagnosis", result.report || "");
      form.append("radiologist_diagnosis", radDiagnosis);
      form.append("radiologist_notes", radNotes);

      if (file) form.append("xray_file", file);

      if (result.overlay_image) {
        const byteString = atob(result.overlay_image);
        const ab = new ArrayBuffer(byteString.length);
        const ia = new Uint8Array(ab);
        for (let i = 0; i < byteString.length; i++) ia[i] = byteString.charCodeAt(i);
        const heatmapBlob = new Blob([ab], { type: "image/png" });
        form.append("heatmap_file", heatmapBlob, "heatmap.png");
      }

      await fetch(`${BASE_URL}/diagnoses/save`, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${token}`,
          "ngrok-skip-browser-warning": "true",
        },
        body: form,
      });

      setSaveSuccess(true);
      setShowSave(false);
      setPatientName(""); setPatientAge(""); setPatientSex("M");
      setPatientContact(""); setRadDiagnosis(""); setRadNotes("");
      setSelectedPatientId(null);

    } catch (err) {
      console.error(err);
    } finally {
      setSaveLoading(false);
    }
  };

  const isTB   = result?.prediction === "TB";
  const tbProb = result?.tb_probability;
  const nrProb = result?.normal_probability;

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 pt-20 pb-8 sm:pt-24 sm:pb-12">

        {/* Hero — show radiologist name if logged in */}
        <div className="mb-10 sm:mb-12">
          <div className="flex items-center justify-between flex-wrap gap-3 mb-4">
            <span className="inline-block text-xs font-semibold tracking-widest text-primary bg-background border border-border/60 px-3 py-1 rounded-full">
              RAD-DINO · GradCAM · LLaMA-3
            </span>
            {token && (
              <div className="flex items-center gap-3">
                <span className="text-xs text-slate-500">
                  Logged in as <span className="text-foreground font-medium">{radiologistName}</span>
                </span>
                <button
                  onClick={() => { setToken(null); setRadiologistName(null); setShowWelcome(true); }}
                  className="text-xs text-slate-500 hover:text-red-400 transition-colors underline underline-offset-2"
                >
                  Logout
                </button>
              </div>
            )}
          </div>
          <h1 className="text-2xl sm:text-4xl font-bold text-primary leading-tight mb-3">
            Tuberculosis Detection
          </h1>
          <p className="text-sm sm:text-base text-slate-400 max-w-lg">
            Upload a chest X-ray to detect TB, visualize suspicious regions, and generate an AI radiology report.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 lg:gap-10 items-start">

          {/* Left */}
          <div className="space-y-4">
            <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-widest">Input</h2>
            <div
              onClick={() => fileInputRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              className={`border-2 border-dashed rounded-2xl p-8 sm:p-10 text-center cursor-pointer transition-all ${
                dragging ? "border-primary bg-background" : "border-border bg-background hover:border-border-hover"
              }`}
            >
              <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={(e) => loadFile(e.target.files[0])} />
              {file ? (
                <>
                  <div className="text-green-400 text-4xl mb-3">✓</div>
                  <p className="text-sm font-medium text-foreground break-all">{file.name}</p>
                  <p className="text-xs text-slate-500 mt-1">{(file.size / 1024).toFixed(0)} KB — ready to analyze</p>
                  <p className="text-xs text-blue-400 mt-3 underline underline-offset-2">Click to change</p>
                </>
              ) : (
                <>
                  <div className="w-14 h-14 rounded-2xl bg-background border border-border flex items-center justify-center mx-auto mb-4">
                    <svg className="w-6 h-6 text-slate-400" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                      <polyline points="17 8 12 3 7 8"/>
                      <line x1="12" y1="3" x2="12" y2="15"/>
                    </svg>
                  </div>
                  <p className="text-sm font-medium text-foreground">Drop a chest X-ray here</p>
                  <p className="text-xs text-slate-500 mt-1.5">or click to browse — PNG, JPG</p>
                </>
              )}
            </div>

            {preview && !result && (
              <div className="rounded-xl overflow-hidden border border-border bg-background aspect-video">
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

            {/* Diagnose another button — only shown after a result when logged in */}
            {result && token && (
              <button
                onClick={resetDiagnosis}
                className="w-full py-3 rounded-xl font-semibold text-sm border border-border text-slate-400 hover:bg-background hover:text-foreground transition-all"
              >
                Diagnose another chest X-ray
              </button>
            )}

            {error && (
              <div className="bg-red-500/10 border border-red-500/30 text-red-400 text-sm px-4 py-3 rounded-xl">
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
                <div key={label} className="bg-background border border-border rounded-xl px-3 py-2.5">
                  <p className="text-xs text-slate-500 mb-0.5">{label}</p>
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
                <p className="text-slate-500 text-sm">Results will appear here after analysis</p>
              </div>
            )}

            {loading && (
              <div className="border border-border bg-background rounded-2xl p-12 text-center space-y-3">
                <span className="w-8 h-8 border-2 border-slate-700 border-t-blue-500 rounded-full animate-spin inline-block" />
                <p className="text-sm text-slate-400">Running analysis...</p>
                {loadingStep && <p className="text-xs text-slate-600">{loadingStep}</p>}
              </div>
            )}

            {result && (
              <div className="space-y-4">

                <div className={`rounded-2xl px-5 py-4 border-2 border-dashed ${
                  isTB ? "bg-red-500/10 border-red-500/40" : "bg-green-500/10 border-green-500/40"
                }`}>
                  <div className="flex items-center justify-between flex-wrap gap-2 mb-3">
                    <div>
                      <p className="text-xs text-slate-500 mb-0.5">Prediction</p>
                      <p className={`text-2xl font-bold ${isTB ? "text-red-400" : "text-green-400"}`}>{result.prediction}</p>
                    </div>
                    <div className={`text-right ${isTB ? "text-red-400" : "text-green-400"}`}>
                      <p className="text-3xl font-bold">{isTB ? tbProb : nrProb}%</p>
                      <p className="text-xs text-slate-500">confidence</p>
                    </div>
                  </div>
                  <div className="space-y-2">
                    {[
                      { label: "TB",     value: tbProb, color: "from-red-700 to-red-500" },
                      { label: "Normal", value: nrProb, color: "from-green-700 to-green-500" },
                    ].map(({ label, value, color }) => (
                      <div key={label} className="flex items-center gap-3">
                        <span className="text-xs text-slate-500 w-12 shrink-0">{label}</span>
                        <div className="flex-1 h-1.5 bg-background rounded-full overflow-hidden">
                          <div className={`h-full rounded-full bg-gradient-to-r ${color} transition-all duration-700`} style={{ width: `${value}%` }} />
                        </div>
                        <span className="text-xs text-slate-400 w-10 text-right shrink-0">{value}%</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  {[
                    { label: "Original",        src: preview },
                    { label: "GradCAM Heatmap", src: `data:image/png;base64,${result.overlay_image}` },
                  ].map(({ label, src }) => (
                    <div key={label} className="rounded-xl overflow-hidden border border-border bg-background">
                      <div className="px-3 py-2 border-b border-border text-xs font-medium text-slate-500 uppercase tracking-widest">{label}</div>
                      <div className="aspect-square bg-background">
                        <img src={src} alt={label} className="w-full h-full object-contain" />
                      </div>
                    </div>
                  ))}
                </div>

                <div className="flex items-center gap-3 px-4 py-2.5 bg-background border border-border rounded-xl">
                  <span className="text-xs text-slate-500 shrink-0">Low</span>
                  <div className="flex-1 h-2 rounded-full" style={{ background: "linear-gradient(to right, #30123b, #4454c4, #1f9f88, #a2da37, #fdc527, #f05b12, #7a0403)" }} />
                  <span className="text-xs text-slate-500 shrink-0">High</span>
                </div>

                {result.report && (
                  <div className="bg-background border border-border rounded-2xl overflow-hidden">
                    <button
                      onClick={() => setReportOpen(o => !o)}
                      className="w-full flex items-center justify-between px-5 py-3.5 hover:bg-background transition-colors"
                    >
                      <div className="flex items-center gap-3">
                        <span className="text-xs font-semibold text-slate-500 uppercase tracking-widest">AI Radiology Report</span>
                        <span className="text-xs text-primary bg-background border border-border rounded px-2 py-0.5">LLaMA-3 · LoRA</span>
                      </div>
                      <svg className={`w-4 h-4 text-foreground transition-transform duration-300 ${reportOpen ? "rotate-180" : ""}`} fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                        <path d="M6 9l6 6 6-6"/>
                      </svg>
                    </button>
                    <div className={`grid transition-all duration-300 ease-in-out ${reportOpen ? "grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0"}`}>
                      <div className="overflow-hidden">
                        <div className="px-5 pb-5 pt-1 border-t border-border">
                          <p className="text-sm text-foreground leading-relaxed whitespace-pre-wrap">{result.report}</p>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Save button */}
                {saveSuccess ? (
                  <div className="w-full py-3.5 rounded-xl text-sm font-semibold text-center bg-green-500/10 border border-green-500/30 text-green-400">
                    ✓ Diagnosis saved successfully
                  </div>
                ) : (
                  <button
                    onClick={handleSaveClick}
                    className="w-full py-3.5 rounded-xl font-semibold text-sm bg-blue-600 hover:bg-blue-500 text-white transition-all flex items-center justify-center gap-2"
                  >
                    {token ? "Save Diagnosis" : "Save Diagnosis — Login Required"}
                  </button>
                )}

                <p className="text-center text-xs text-slate-600 pb-2 mt-4">
                  ⚕ For research purposes only. Not a substitute for clinical diagnosis.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ── Welcome Modal ── */}
      {showWelcome && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 px-4">
          <div className="bg-background border border-border rounded-2xl p-6 w-full max-w-sm space-y-5">
            <div>
              <h2 className="text-base font-semibold text-foreground">Welcome to PulmoVision</h2>
              <p className="text-xs text-slate-500 mt-1">How would you like to continue?</p>
            </div>

            <div className="space-y-3">
              <button
                onClick={() => { setShowWelcome(false); setShowLogin(true); setLoginContext("welcome"); }}
                className="w-full py-3 rounded-xl text-sm font-semibold bg-blue-600 hover:bg-blue-500 text-white transition-all"
              >
                Login as Radiologist
              </button>
              <button
                onClick={() => setShowWelcome(false)}
                className="w-full py-3 rounded-xl text-sm font-medium border border-border text-slate-400 hover:text-foreground hover:border-border-hover transition-all"
              >
                Continue as Guest
              </button>
            </div>

            <p className="text-xs text-slate-600 text-center">
              Guests can run analyses but cannot save records
            </p>
          </div>
        </div>
      )}

      {/* ── Login Modal ── */}
      {showLogin && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 px-4">
          <div className="bg-background border border-border rounded-2xl p-6 w-full max-w-sm space-y-4">
            <div>
              <h2 className="text-base font-semibold text-foreground">Radiologist Login</h2>
              <p className="text-xs text-slate-500 mt-1">
                {loginContext === "save" ? "Login to save this diagnosis" : "Login to access full features"}
              </p>
            </div>
            <div className="space-y-3">
              <div>
                <label className="text-xs text-slate-500 mb-1 block">Radiologist ID</label>
                <input
                  type="text"
                  value={radId}
                  onChange={e => setRadId(e.target.value)}
                  placeholder="e.g. 1"
                  className="w-full bg-background border border-border rounded-xl px-4 py-2.5 text-sm text-foreground outline-none focus:border-blue-500"
                />
              </div>
              <div>
                <label className="text-xs text-slate-500 mb-1 block">Password</label>
                <input
                  type="password"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className="w-full bg-background border border-border rounded-xl px-4 py-2.5 text-sm text-foreground outline-none focus:border-blue-500"
                  onKeyDown={e => e.key === "Enter" && handleLogin()}
                />
              </div>
              {loginError && <p className="text-xs text-red-400">{loginError}</p>}
            </div>
            <div className="flex gap-3 pt-1">
              <button
                onClick={() => { setShowLogin(false); if (loginContext === "welcome") setShowWelcome(true); }}
                className="flex-1 py-2.5 rounded-xl text-sm border border-border text-slate-400 hover:text-foreground transition-colors"
              >
                {loginContext === "welcome" ? "Back" : "Cancel"}
              </button>
              <button
                onClick={handleLogin}
                disabled={loginLoading}
                className="flex-1 py-2.5 rounded-xl text-sm bg-blue-600 hover:bg-blue-500 text-white font-semibold transition-all disabled:opacity-50 flex items-center justify-center gap-2"
              >
                {loginLoading && <span className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />}
                {loginLoading ? "Logging in..." : "Login"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Save Modal ── */}
      {showSave && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 px-4">
          <div className="bg-background border border-border rounded-2xl p-6 w-full max-w-md space-y-4 max-h-[90vh] overflow-y-auto">
            <div>
              <h2 className="text-base font-semibold text-foreground">Save Diagnosis</h2>
              <p className="text-xs text-slate-500 mt-1">Search for an existing patient or enter new details</p>
            </div>

            <div className="relative">
              <label className="text-xs text-slate-500 mb-1 block">Patient Name</label>
              <input
                type="text"
                value={patientName}
                onChange={e => handleNameSearch(e.target.value)}
                placeholder="Start typing to search..."
                className="w-full bg-background border border-border rounded-xl px-4 py-2.5 text-sm text-foreground outline-none focus:border-blue-500"
              />
              {searchResults.length > 0 && (
                <div className="absolute top-full left-0 right-0 mt-1 bg-background border border-border rounded-xl overflow-hidden z-10 shadow-lg">
                  {searchResults.map(p => (
                    <button
                      key={p.id}
                      onClick={() => handleSelectPatient(p)}
                      className="w-full text-left px-4 py-2.5 text-sm text-foreground hover:bg-background transition-colors border-b border-border last:border-0"
                    >
                      <span className="font-medium">{p.name}</span>
                      <span className="text-xs text-slate-500 ml-2">{p.age}y · {p.sex}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs text-slate-500 mb-1 block">Age</label>
                <input
                  type="number"
                  value={patientAge}
                  onChange={e => setPatientAge(e.target.value)}
                  placeholder="e.g. 35"
                  className="w-full bg-background border border-border rounded-xl px-4 py-2.5 text-sm text-foreground outline-none focus:border-blue-500"
                />
              </div>
              <div>
                <label className="text-xs text-slate-500 mb-1 block">Sex</label>
                <select
                  value={patientSex}
                  onChange={e => setPatientSex(e.target.value)}
                  className="w-full bg-background border border-border rounded-xl px-4 py-2.5 text-sm text-foreground outline-none focus:border-blue-500"
                >
                  <option value="M">Male</option>
                  <option value="F">Female</option>
                  <option value="Other">Other</option>
                </select>
              </div>
            </div>

            <div>
              <label className="text-xs text-slate-500 mb-1 block">Contact Number</label>
              <input
                type="text"
                value={patientContact}
                onChange={e => setPatientContact(e.target.value)}
                placeholder="e.g. 0123456789"
                className="w-full bg-background border border-border rounded-xl px-4 py-2.5 text-sm text-foreground outline-none focus:border-blue-500"
              />
            </div>

            <div>
              <label className="text-xs text-slate-500 mb-1 block">Your Diagnosis</label>
              <textarea
                value={radDiagnosis}
                onChange={e => setRadDiagnosis(e.target.value)}
                placeholder="Enter your clinical diagnosis..."
                rows={2}
                className="w-full bg-background border border-border rounded-xl px-4 py-2.5 text-sm text-foreground outline-none focus:border-blue-500 resize-none"
              />
            </div>

            <div>
              <label className="text-xs text-slate-500 mb-1 block">Notes</label>
              <textarea
                value={radNotes}
                onChange={e => setRadNotes(e.target.value)}
                placeholder="Follow-up instructions, additional notes..."
                rows={2}
                className="w-full bg-background border border-border rounded-xl px-4 py-2.5 text-sm text-foreground outline-none focus:border-blue-500 resize-none"
              />
            </div>

            <div className="bg-background border border-border rounded-xl px-4 py-3 text-xs text-slate-400 space-y-1">
              <p>AI Prediction: <span className="text-foreground font-medium">{result?.prediction}</span></p>
              <p>TB Probability: <span className="text-foreground font-medium">{result?.tb_probability}%</span></p>
              <p>X-ray + heatmap will be saved automatically</p>
            </div>

            <div className="flex gap-3 pt-1">
              <button onClick={() => setShowSave(false)} className="flex-1 py-2.5 rounded-xl text-sm border border-border text-slate-400 hover:text-foreground transition-colors">
                Cancel
              </button>
              <button
                onClick={handleSave}
                disabled={saveLoading || !patientName || !patientAge}
                className="flex-1 py-2.5 rounded-xl text-sm bg-blue-600 hover:bg-blue-500 text-white font-semibold transition-all disabled:opacity-50 flex items-center justify-center gap-2"
              >
                {saveLoading && <span className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />}
                {saveLoading ? "Saving..." : "Save"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}