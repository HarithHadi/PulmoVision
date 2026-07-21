// src/Pages/DiagnosisDetail.jsx
import { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../AuthContext";

const BASE_URL = "http://localhost:8000";

export default function DiagnosisDetail() {
  const { state: d } = useLocation();
  const navigate = useNavigate();
  const { radiologistName, token } = useAuth();
  const [xrayUrl, setXrayUrl] = useState(null);
  const [heatmapUrl, setHeatmapUrl] = useState(null);

  useEffect(() => {
    if (!d) return;

    const fetchImage = async (path, setter) => {
      if (!path) return;
      try {
        const res = await fetch(
          `${BASE_URL}/diagnoses/image?path=${encodeURIComponent(path)}`,
          {
            headers: {
              "Authorization": `Bearer ${token}`,
            },
          }
        );
        if (!res.ok) return;
        const blob = await res.blob();
        setter(URL.createObjectURL(blob));
      } catch {
        console.error("Failed to load image");
      }
    };

    fetchImage(d.xray_path, setXrayUrl);
    fetchImage(d.heatmap_path, setHeatmapUrl);
  }, [d]);

  if (!d) {
    navigate("/patients");
    return null;
  }

  const isTB = d.tb_probability >= 0.5;

  const statusColor = (status) => {
    if (status === "finalized") return "text-green-400 bg-green-500/10 border-green-500/20";
    if (status === "reviewed")  return "text-blue-400 bg-blue-500/10 border-blue-500/20";
    return "text-yellow-400 bg-yellow-500/10 border-yellow-500/20";
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="max-w-3xl mx-auto px-4 sm:px-6 pt-20 pb-12 space-y-6">

        {/* Header */}
        <div className="flex items-center justify-between">
          <button
            onClick={() => navigate("/patients")}
            className="flex items-center gap-2 text-sm text-slate-400 hover:text-foreground border border-border hover:border-border-hover px-4 py-2 rounded-xl transition-all"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 19.5L3 12m0 0l7.5-7.5M3 12h18"/>
            </svg>
            Patient Records
          </button>
          <span className="text-xs text-slate-500 border border-border px-2.5 py-1 rounded-lg">
            {new Date(d.created_at).toLocaleDateString("en-MY", {
              day: "numeric", month: "long", year: "numeric"
            })}
          </span>
        </div>

        {/* Patient Info */}
        <div className="bg-background border border-border rounded-2xl p-5 space-y-3">
          <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-widest">Patient Information</h2>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            {[
              { label: "Name",    value: d.patients?.name ?? "—" },
              { label: "Age",     value: d.patients?.age ?? "—" },
              { label: "Sex",     value: d.patients?.sex ?? "—" },
              { label: "Contact", value: d.patients?.contact_number || "—" },
            ].map(({ label, value }) => (
              <div key={label}>
                <p className="text-xs text-slate-500 mb-0.5">{label}</p>
                <p className="text-sm font-medium text-foreground">{value}</p>
              </div>
            ))}
          </div>
        </div>

        {/* AI Result */}
        <div className={`rounded-2xl px-5 py-4 border-2 border-dashed ${
          isTB ? "bg-red-500/10 border-red-500/40" : "bg-green-500/10 border-green-500/40"
        }`}>
          <div className="flex items-center justify-between flex-wrap gap-2 mb-4">
            <div>
              <p className="text-xs text-slate-500 mb-0.5">AI Prediction</p>
              <p className={`text-3xl font-bold ${isTB ? "text-red-400" : "text-green-400"}`}>
                {isTB ? "TB Detected" : "Normal"}
              </p>
            </div>
            <div className={`text-right ${isTB ? "text-red-400" : "text-green-400"}`}>
              <p className="text-4xl font-bold">{(d.tb_probability * 100).toFixed(1)}%</p>
              <p className="text-xs text-slate-500">TB probability</p>
            </div>
          </div>
          <div className="space-y-2">
            {[
              { label: "TB",     value: d.tb_probability * 100,       color: "from-red-700 to-red-500" },
              { label: "Normal", value: (1 - d.tb_probability) * 100, color: "from-green-700 to-green-500" },
            ].map(({ label, value, color }) => (
              <div key={label} className="flex items-center gap-3">
                <span className="text-xs text-slate-500 w-12 shrink-0">{label}</span>
                <div className="flex-1 h-1.5 bg-background rounded-full overflow-hidden">
                  <div className={`h-full rounded-full bg-gradient-to-r ${color} transition-all duration-700`} style={{ width: `${value}%` }} />
                </div>
                <span className="text-xs text-slate-400 w-12 text-right shrink-0">{value.toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>
 

        {/* Clinical Data */}
        {d.clinical_data && (
          <div className="bg-background border border-border rounded-2xl p-5 space-y-3">
            <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-widest">Clinical Assessment</h2>
            <div className="space-y-2">
              {[
                { key: "cough",       label: "Cough ≥3 weeks" },
                { key: "weightLoss",  label: "Weight loss / loss of appetite" },
                { key: "nightSweats", label: "Night sweats" },
                { key: "fever",       label: "Low-grade fever" },
                { key: "fatigue",     label: "Persistent fatigue" },
                { key: "bloodSputum", label: "Haemoptysis" },
                { key: "contactTB",   label: "Known TB contact" },
              ].map(({ key, label }) => {
                const val = d.clinical_data[key];
                if (!val) return null;
                return (
                  <div key={key} className="flex items-center justify-between">
                    <span className="text-xs text-slate-400">{label}</span>
                    <span className={`text-xs font-medium px-2.5 py-1 rounded-lg border ${
                      val === "Yes"     ? "text-red-400 bg-red-500/10 border-red-500/20" :
                      val === "No"      ? "text-green-400 bg-green-500/10 border-green-500/20" :
                                          "text-slate-400 bg-background border-border"
                    }`}>
                      {val}
                    </span>
                  </div>
                );
              })}
              {d.clinical_data.duration && (
                <div className="pt-2 border-t border-border">
                  <p className="text-xs text-slate-500 mb-1">Additional notes</p>
                  <p className="text-xs text-slate-400">{d.clinical_data.duration}</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* X-ray Images */}
        {(d.xray_path || d.heatmap_path) && (
          <div className="bg-background border border-border rounded-2xl p-5 space-y-3">
            <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-widest">Imaging</h2>
            <div className="grid grid-cols-2 gap-3">
              {d.xray_path && (
                <div className="rounded-xl overflow-hidden border border-border">
                  <div className="px-3 py-2 border-b border-border text-xs font-medium text-slate-500 uppercase tracking-widest">Original X-ray</div>
                  <div className="aspect-square bg-background flex items-center justify-center">
                    {xrayUrl ? (
                      <img src={xrayUrl} alt="X-ray" className="w-full h-full object-contain" />
                    ) : (
                      <span className="w-5 h-5 border-2 border-slate-700 border-t-blue-500 rounded-full animate-spin" />
                    )}
                  </div>
                </div>
              )}
              {d.heatmap_path && (
                <div className="rounded-xl overflow-hidden border border-border">
                  <div className="px-3 py-2 border-b border-border text-xs font-medium text-slate-500 uppercase tracking-widest">GradCAM Heatmap</div>
                  <div className="aspect-square bg-background flex items-center justify-center">
                    {heatmapUrl ? (
                      <img src={heatmapUrl} alt="Heatmap" className="w-full h-full object-contain" />
                    ) : (
                      <span className="w-5 h-5 border-2 border-slate-700 border-t-blue-500 rounded-full animate-spin" />
                    )}
                  </div>
                </div>
              )}
            </div>
            {/* GradCAM legend */}
            <div className="flex items-center gap-3 px-4 py-2.5 bg-background border border-border rounded-xl">
              <span className="text-xs text-slate-500 shrink-0">Low</span>
              <div className="flex-1 h-2 rounded-full" style={{ background: "linear-gradient(to right, #30123b, #4454c4, #1f9f88, #a2da37, #fdc527, #f05b12, #7a0403)" }} />
              <span className="text-xs text-slate-500 shrink-0">High</span>
            </div>
          </div>
        )}
        {/* AI Report */}
        {d.llama_diagnosis && (
          <div className="bg-background border border-border rounded-2xl p-5 space-y-4">
            <div className="flex items-center gap-3">
              <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-widest">AI Radiology Report</h2>
              <span className="text-xs text-primary bg-background border border-border rounded px-2 py-0.5">LLaMA-3 · Groq</span>
            </div>

            {(() => {
              const text = d.llama_diagnosis;
              const impressionIndex = text.indexOf("IMPRESSION");
              const findings = impressionIndex !== -1 ? text.slice(0, impressionIndex).replace("FINDINGS:", "").trim() : text.replace("FINDINGS:", "").trim();
              const impression = impressionIndex !== -1 ? text.slice(impressionIndex).replace("IMPRESSION:", "").trim() : null;

              return (
                <div className="space-y-3">
                  {/* Findings */}
                  <div className="rounded-xl border border-blue-500/20 bg-blue-500/5 p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <div className="w-1.5 h-1.5 rounded-full bg-blue-400" />
                      <p className="text-xs font-semibold text-blue-400 uppercase tracking-widest">Findings</p>
                    </div>
                    <p className="text-sm text-foreground leading-relaxed">{findings}</p>
                  </div>

                  {/* Impression */}
                  {impression && (
                    <div className={`rounded-xl border p-4 ${
                      d.tb_probability >= 0.5
                        ? "border-red-500/20 bg-red-500/5"
                        : "border-green-500/20 bg-green-500/5"
                    }`}>
                      <div className="flex items-center gap-2 mb-2">
                        <div className={`w-1.5 h-1.5 rounded-full ${d.tb_probability >= 0.5 ? "bg-red-400" : "bg-green-400"}`} />
                        <p className={`text-xs font-semibold uppercase tracking-widest ${d.tb_probability >= 0.5 ? "text-red-400" : "text-green-400"}`}>
                          Impression
                        </p>
                      </div>
                      <p className="text-sm text-foreground leading-relaxed">{impression}</p>
                    </div>
                  )}
                </div>
              );
            })()}
          </div>
        )}

        {/* Radiologist Diagnosis */}
        {d.radiologist_diagnosis && (
          <div className="bg-background border border-border rounded-2xl p-5 space-y-3">
            <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-widest">Radiologist Diagnosis</h2>
            <p className="text-sm text-foreground leading-relaxed">{d.radiologist_diagnosis}</p>
          </div>
        )}

        {/* Notes */}
        {d.radiologist_notes && (
          <div className="bg-background border border-border rounded-2xl p-5 space-y-3">
            <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-widest">Notes</h2>
            <p className="text-sm text-foreground leading-relaxed">{d.radiologist_notes}</p>
          </div>
        )}

      </div>
    </div>
  );
}