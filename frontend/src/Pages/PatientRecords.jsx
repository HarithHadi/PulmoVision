// src/pages/PatientRecords.jsx
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../AuthContext";

const BASE_URL = "http://localhost:8000";

export default function PatientRecords() {
  const { token, radiologistName } = useAuth();
  const navigate = useNavigate();

  const [diagnoses, setDiagnoses] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState(null); // selected diagnosis for detail view

  // Redirect to diagnose if not logged in
  useEffect(() => {
    if (!token) navigate("/diagnose");
  }, [token]);

  // Fetch all diagnoses on mount
  useEffect(() => {
    if (!token) return;
    fetchDiagnoses();
  }, [token]);

  const fetchDiagnoses = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${BASE_URL}/diagnoses/all`, {
        headers: {
          "Authorization": `Bearer ${token}`,
          "ngrok-skip-browser-warning": "true",
        },
      });
      if (!res.ok) throw new Error("Failed to fetch records");
      const data = await res.json();
      setDiagnoses(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Filter by patient name
  const filtered = diagnoses.filter(d =>
    d.patients?.name?.toLowerCase().includes(search.toLowerCase())
  );

  const statusColor = (status) => {
    if (status === "finalized") return "text-green-400 bg-green-500/10 border-green-500/20";
    if (status === "reviewed")  return "text-blue-400 bg-blue-500/10 border-blue-500/20";
    return "text-yellow-400 bg-yellow-500/10 border-yellow-500/20";
  };

  const tbColor = (prob) => {
    const p = prob * 100;
    if (p >= 70) return "text-red-400";
    if (p >= 40) return "text-yellow-400";
    return "text-green-400";
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 pt-20 pb-12">

        {/* Header */}
        <div className="flex items-center justify-between mb-8">
        <div>
            <h1 className="text-2xl sm:text-3xl font-bold text-primary mb-1">Patient Records</h1>
            <p className="text-sm text-slate-500">
            Logged in as <span className="text-foreground font-medium">{radiologistName}</span>
            </p>
        </div>
        <button
            onClick={() => navigate("/diagnose")}
            className="flex items-center gap-2 text-sm text-slate-400 hover:text-foreground border border-border hover:border-border-hover px-4 py-2 rounded-xl transition-all"
        >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 19.5L3 12m0 0l7.5-7.5M3 12h18"/>
            </svg>
            TB Detection
        </button>
        </div>

        {/* Search + Refresh */}
        <div className="flex items-center gap-3 mb-6">
          <div className="flex-1 relative">
            <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-4.35-4.35M17 11A6 6 0 1 1 5 11a6 6 0 0 1 12 0z"/>
            </svg>
            <input
              type="text"
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="Search by patient name..."
              className="w-full bg-background border border-border rounded-xl pl-9 pr-4 py-2.5 text-sm text-foreground outline-none focus:border-blue-500"
            />
          </div>
          <button
            onClick={fetchDiagnoses}
            className="px-4 py-2.5 rounded-xl border border-border text-sm text-slate-400 hover:text-foreground hover:border-border-hover transition-all flex items-center gap-2"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 1 13.803-3.7l3.181 3.182m0-4.991v4.99"/>
            </svg>
            Refresh
          </button>
        </div>

        {/* Stats row */}
        <div className="grid grid-cols-3 gap-3 mb-6">
          {[
            { label: "Total Diagnoses", value: diagnoses.length },
            { label: "TB Detected",     value: diagnoses.filter(d => d.tb_probability >= 0.5).length },
            { label: "Pending Review",  value: diagnoses.filter(d => d.status === "pending").length },
          ].map(({ label, value }) => (
            <div key={label} className="bg-background border border-border rounded-xl px-4 py-3">
              <p className="text-xs text-slate-500 mb-1">{label}</p>
              <p className="text-2xl font-bold text-foreground">{value}</p>
            </div>
          ))}
        </div>

        {/* Error */}
        {error && (
          <div className="bg-red-500/10 border border-red-500/30 text-red-400 text-sm px-4 py-3 rounded-xl mb-4">
            ⚠ {error}
          </div>
        )}

        {/* Loading */}
        {loading && (
          <div className="flex items-center justify-center py-20">
            <span className="w-8 h-8 border-2 border-slate-700 border-t-blue-500 rounded-full animate-spin" />
          </div>
        )}

        {/* Table */}
        {!loading && (
          <div className="bg-background border border-border rounded-2xl overflow-hidden">
            {filtered.length === 0 ? (
              <div className="text-center py-16 text-slate-500 text-sm">
                {search ? "No patients match your search" : "No diagnosis records yet"}
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border text-xs text-slate-500 uppercase tracking-widest">
                      <th className="text-left px-5 py-3 font-medium">Patient</th>
                      <th className="text-left px-5 py-3 font-medium">Age / Sex</th>
                      <th className="text-left px-5 py-3 font-medium">TB Probability</th>
                      <th className="text-left px-5 py-3 font-medium">Prediction</th>
                      <th className="text-left px-5 py-3 font-medium">Status</th>
                      <th className="text-left px-5 py-3 font-medium">Date</th>
                      <th className="px-5 py-3" />
                    </tr>
                  </thead>
                  <tbody>
                    {filtered.map((d, i) => (
                      <tr
                        key={d.id}
                        className={`border-b border-border last:border-0 hover:bg-white/[0.02] transition-colors ${
                          selected?.id === d.id ? "bg-white/[0.03]" : ""
                        }`}
                      >
                        <td className="px-5 py-4 font-medium text-foreground">
                          {d.patients?.name ?? "—"}
                        </td>
                        <td className="px-5 py-4 text-slate-400">
                          {d.patients?.age ?? "—"} / {d.patients?.sex ?? "—"}
                        </td>
                        <td className="px-5 py-4">
                          <span className={`font-bold ${tbColor(d.tb_probability)}`}>
                            {(d.tb_probability * 100).toFixed(1)}%
                          </span>
                        </td>
                        <td className="px-5 py-4 text-slate-400">
                          {d.tb_probability >= 0.5 ? (
                            <span className="text-red-400 font-medium">TB</span>
                          ) : (
                            <span className="text-green-400 font-medium">Normal</span>
                          )}
                        </td>
                        <td className="px-5 py-4">
                          <span className={`text-xs font-medium px-2.5 py-1 rounded-lg border capitalize ${statusColor(d.status)}`}>
                            {d.status}
                          </span>
                        </td>
                        <td className="px-5 py-4 text-slate-500 text-xs">
                          {new Date(d.created_at).toLocaleDateString("en-MY", {
                            day: "numeric", month: "short", year: "numeric"
                          })}
                        </td>
                        <td className="px-5 py-4">
                          <button
                            onClick={() => setSelected(selected?.id === d.id ? null : d)}
                            className="text-xs text-blue-400 hover:text-blue-300 transition-colors"
                          >
                            {selected?.id === d.id ? "Close" : "View"}
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>

      {/* ── Detail Panel ── */}
      {selected && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 px-4">
          <div className="bg-background border border-border rounded-2xl p-6 w-full max-w-lg space-y-4 max-h-[90vh] overflow-y-auto">

            {/* Header */}
            <div className="flex items-start justify-between">
              <div>
                <h2 className="text-base font-semibold text-foreground">{selected.patients?.name}</h2>
                <p className="text-xs text-slate-500 mt-0.5">
                  {selected.patients?.age}y · {selected.patients?.sex} · {selected.patients?.contact_number || "No contact"}
                </p>
              </div>
              <button onClick={() => setSelected(null)} className="text-slate-500 hover:text-foreground transition-colors">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12"/>
                </svg>
              </button>
            </div>

            {/* Result summary */}
            <div className={`rounded-xl px-4 py-3 border ${
              selected.tb_probability >= 0.5
                ? "bg-red-500/10 border-red-500/20"
                : "bg-green-500/10 border-green-500/20"
            }`}>
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs text-slate-500 mb-0.5">AI Prediction</p>
                  <p className={`text-xl font-bold ${selected.tb_probability >= 0.5 ? "text-red-400" : "text-green-400"}`}>
                    {selected.tb_probability >= 0.5 ? "TB Detected" : "Normal"}
                  </p>
                </div>
                <p className={`text-3xl font-bold ${selected.tb_probability >= 0.5 ? "text-red-400" : "text-green-400"}`}>
                  {(selected.tb_probability * 100).toFixed(1)}%
                </p>
              </div>
            </div>

            {/* Status */}
            <div className="flex items-center justify-between">
              <span className="text-xs text-slate-500">Status</span>
              <span className={`text-xs font-medium px-2.5 py-1 rounded-lg border capitalize ${statusColor(selected.status)}`}>
                {selected.status}
              </span>
            </div>

            {/* LLaMA Report */}
            {selected.llama_diagnosis && (
              <div className="space-y-2">
                <p className="text-xs text-slate-500 uppercase tracking-widest font-medium">AI Report</p>
                {(() => {
                  const text = selected.llama_diagnosis;
                  const impressionIndex = text.indexOf("IMPRESSION");
                  const findings = impressionIndex !== -1
                    ? text.slice(0, impressionIndex).replace("FINDINGS:", "").trim()
                    : text.replace("FINDINGS:", "").trim();
                  const impression = impressionIndex !== -1
                    ? text.slice(impressionIndex).replace("IMPRESSION:", "").trim()
                    : null;
                  const isTB = selected.tb_probability >= 0.5;

                  return (
                    <>
                      <div className="rounded-xl border border-blue-500/20 bg-blue-500/5 p-3">
                        <div className="flex items-center gap-2 mb-1.5">
                          <div className="w-1.5 h-1.5 rounded-full bg-blue-400" />
                          <p className="text-xs font-semibold text-blue-400 uppercase tracking-widest">Findings</p>
                        </div>
                        <p className="text-xs text-slate-400 leading-relaxed">{findings}</p>
                      </div>

                      {impression && (
                        <div className={`rounded-xl border p-3 ${
                          isTB ? "border-red-500/20 bg-red-500/5" : "border-green-500/20 bg-green-500/5"
                        }`}>
                          <div className="flex items-center gap-2 mb-1.5">
                            <div className={`w-1.5 h-1.5 rounded-full ${isTB ? "bg-red-400" : "bg-green-400"}`} />
                            <p className={`text-xs font-semibold uppercase tracking-widest ${isTB ? "text-red-400" : "text-green-400"}`}>
                              Impression
                            </p>
                          </div>
                          <p className="text-xs text-slate-400 leading-relaxed">{impression}</p>
                        </div>
                      )}
                    </>
                  );
                })()}
              </div>
            )}

            {/* Radiologist Diagnosis */}
            {selected.radiologist_diagnosis && (
              <div>
                <p className="text-xs text-slate-500 mb-2 uppercase tracking-widest font-medium">Radiologist Diagnosis</p>
                <div className="bg-background border border-border rounded-xl px-4 py-3">
                  <p className="text-xs text-slate-400 leading-relaxed">{selected.radiologist_diagnosis}</p>
                </div>
              </div>
            )}

            {/* Notes */}
            {selected.radiologist_notes && (
              <div>
                <p className="text-xs text-slate-500 mb-2 uppercase tracking-widest font-medium">Notes</p>
                <div className="bg-background border border-border rounded-xl px-4 py-3">
                  <p className="text-xs text-slate-400 leading-relaxed">{selected.radiologist_notes}</p>
                </div>
              </div>
            )}

            {/* Date */}
            <p className="text-xs text-slate-600">
              Recorded on {new Date(selected.created_at).toLocaleDateString("en-MY", {
                day: "numeric", month: "long", year: "numeric"
              })}
            </p>

            <button
              onClick={() => navigate(`/diagnosis/${selected.id}`, { state: selected })}
              className="w-full py-2.5 rounded-xl text-sm bg-blue-600 hover:bg-blue-500 text-white font-semibold transition-all"
            >
              View Full Report
            </button>

            <button
              onClick={() => setSelected(null)}
              className="w-full py-2.5 rounded-xl text-sm border border-border text-slate-400 hover:text-foreground transition-colors"
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}