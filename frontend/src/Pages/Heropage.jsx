import { useNavigate } from "react-router-dom";

const features = [
  {
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
        <path d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909m-18 3.75h16.5a1.5 1.5 0 001.5-1.5V6a1.5 1.5 0 00-1.5-1.5H3.75A1.5 1.5 0 002.25 6v12a1.5 1.5 0 001.5 1.5zm10.5-11.25h.008v.008h-.008V8.25zm.375 0a.375.375 0 11-.75 0 .375.375 0 01.75 0z"/>
      </svg>
    ),
    title: "Visual Encoder",
    desc: "RAD-DINO Vision Transformer pre-trained on 900,000+ chest X-rays for expert-level feature extraction.",
    tag: "microsoft/rad-dino",
  },
  {
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
        <path d="M8.25 3v1.5M4.5 8.25H3m18 0h-1.5M4.5 12H3m18 0h-1.5m-15 3.75H3m18 0h-1.5M8.25 19.5V21M12 3v1.5m0 15V21m3.75-18v1.5m0 15V21m-9-1.5h10.5a2.25 2.25 0 002.25-2.25V6.75a2.25 2.25 0 00-2.25-2.25H6.75A2.25 2.25 0 004.5 6.75v10.5a2.25 2.25 0 002.25 2.25zm.75-12h9v9h-9v-9z"/>
      </svg>
    ),
    title: "Reasoning Engine",
    desc: "LLaMA-3-8B language model generates structured clinical reports from visual embeddings.",
    tag: "meta-llama/llama-3-8b",
  },
  {
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
        <path d="M7.5 14.25v2.25m3-4.5v4.5m3-6.75v6.75m3-9v9M6 20.25h12A2.25 2.25 0 0020.25 18V6A2.25 2.25 0 0018 3.75H6A2.25 2.25 0 003.75 6v12A2.25 2.25 0 006 20.25z"/>
      </svg>
    ),
    title: "GradCAM Localization",
    desc: "Gradient-weighted class activation maps highlight suspicious lung regions directly on the X-ray.",
    tag: "Explainable AI",
  },
  {
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
        <path d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z"/>
      </svg>
    ),
    title: "Clinical Reports",
    desc: "Automated radiology reports with findings, impressions and diagnostic confidence scores.",
    tag: "BLEU / ROUGE scored",
  },
];

const stats = [
  { value: "900K+", label: "X-rays trained on" },
  { value: "8B", label: "LLM parameters" },
  { value: "97%+", label: "TB detection accuracy" },
  { value: "<3s", label: "Inference time" },
];

const pipeline = [
  { step: "01", label: "Upload X-ray", desc: "Radiologist uploads chest X-ray and optional clinical history" },
  { step: "02", label: "RAD-DINO Encoding", desc: "Vision transformer extracts 1,369 spatial patch embeddings" },
  { step: "03", label: "LLaMA Reasoning", desc: "Language model generates diagnostic report with lesion coordinates" },
  { step: "05", label: "GradCAM Overlay", desc: "Heatmap rendered over X-ray to show model attention regions" },
];

export default function HeroPage() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-background text-foreground overflow-x-hidden">      

      {/* Hero */}
      <section className="pt-32 pb-20 px-4 sm:px-6 relative">
        {/* Background glow */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute top-20 left-1/2 -translate-x-1/2 w-[600px] h-[400px] bg-orange-100/100 rounded-full blur-3xl" />
        </div>

        <div className="max-w-4xl mx-auto text-center relative">
          <div className="inline-flex items-center gap-2 text-xs font-medium text-primary bg-background/60 border border-border px-3 py-1.5 rounded-full mb-6">
            <span className="w-1.5 h-1.5 bg-primary rounded-full animate-pulse" />
            UiTM Final Year Project — Muhamad Harith Bin Hadi
          </div>

          <h1 className="text-4xl sm:text-6xl font-bold text-foreground leading-tight mb-6 tracking-tight">
            AI-Powered{" "}
            <span className="text-transparent bg-clip-text bg-primary/100">
              TB Detection
            </span>
            <br />from Chest X-Rays
          </h1>

          <p className="text-base sm:text-lg text-muted max-w-2xl mx-auto mb-10 leading-relaxed">
            PulmoVision is a Large Vision-Language Model system that automatically detects and localizes
            tuberculosis from chest X-rays, generating clinical radiology reports with explainable AI visualization.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
            <button
              onClick={() => navigate("/diagnose")}
              className="w-full sm:w-auto px-6 py-3 btn-primary text-sm flex items-center justify-center gap-2"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <path d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"/>
              </svg>
              Upload X-Ray
            </button>
            <a
              href="#pipeline"
              className="w-full sm:w-auto px-6 py-3 border border-border hover:border-slate-500 text-mute hover:text-foreground font-medium rounded-xl transition-all text-sm text-center"
            >
              How it works
            </a>
          </div>
        </div>

        {/* Mock UI preview */}
        <div className="max-w-3xl mx-auto mt-16 relative">
          <div className="rounded-2xl border border-border/60 bg-card/60 backdrop-blur overflow-hidden shadow-2xl">
            <div className="flex items-center gap-1.5 px-4 py-3 border-b border-border">
              <div className="w-2.5 h-2.5 rounded-full bg-red-500/70" />
              <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/70" />
              <div className="w-2.5 h-2.5 rounded-full bg-green-500/70" />
              <span className="ml-3 text-xs text-slate-500">pulmovision.app / diagnose</span>
            </div>
            <div className="grid grid-cols-2 gap-4 p-4">
              <div className="rounded-xl bg-surface-2 border border-border overflow-hidden">
                <div className="px-3 py-2 border-b border-border text-xs text-primary uppercase tracking-widest">Original</div>
                <div className="aspect-square bg-card flex items-center justify-center">
                  <img 
                    src="/Tuberculosis-118.png" 
                    alt="Original Chest X-Ray" 
                    className="w-full h-full object-cover"
                  />
                </div>
              </div>
              <div className="rounded-xl bg-surface-2 border border-border overflow-hidden">
                <div className="px-3 py-2 border-b border-border text-xs text-primary uppercase tracking-widest">GradCAM</div>
                <div className="aspect-square bg-card flex items-center justify-center relative">
                  <img 
                    src="/resultTB.png" 
                    alt="Original Chest X-Ray" 
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute top-1/4 left-1/3 w-8 h-8 rounded-full bg-red-500/30 blur-sm" />
                  <div className="absolute top-1/3 right-1/4 w-6 h-6 rounded-full bg-orange-500/30 blur-sm" />
                </div>
              </div>
            </div>
            <div className="px-4 pb-4">
              <div className="rounded-xl bg-danger/40 border border-danger/50 px-4 py-3 flex items-center justify-between">
                <span className="text-danger font-bold text-lg">TB Detected</span>
                <span className="text-danger text-sm font-semibold">97.5% confidence</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Stats */}
      <section id="stats" className="py-16 px-4 sm:px-6 border-y border-border/60">
        <div className="max-w-4xl mx-auto grid grid-cols-2 sm:grid-cols-4 gap-6">
          {stats.map(({ value, label }) => (
            <div key={label} className="text-center">
              <p className="text-3xl sm:text-4xl font-bold text-foreground mb-1">{value}</p>
              <p className="text-xs text-slate-500 uppercase tracking-widest">{label}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Features */}
      <section id="features" className="py-20 px-4 sm:px-6">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-14">
            <span className="text-xs font-semibold text-primary uppercase tracking-widest mb-3 block">Architecture</span>
            <h2 className="text-2xl sm:text-3xl font-bold text-foreground mb-3">Built on state-of-the-art models</h2>
            <p className="text-sm text-muted max-w-lg mx-auto">
              PulmoVision combines specialized medical AI components into a unified diagnostic pipeline.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {features.map(({ icon, title, desc, tag }) => (
              <div key={title} className="bg-surface border border-border rounded-2xl p-5 hover:border-border transition-all">
                <div className="w-10 h-10 rounded-xl bg-primary/10 border border-primary/20 flex items-center justify-center text-primary mb-4">
                  {icon}
                </div>
                <h3 className="text-sm font-semibold text-foreground mb-1.5">{title}</h3>
                <p className="text-xs text-muted leading-relaxed mb-3">{desc}</p>
                <span className="inline-block text-xs font-mono text-foreground bg-card/50 px-2 py-0.5 rounded">{tag}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Pipeline */}
      <section id="pipeline" className="py-20 px-4 sm:px-6 bg-surface/30">
        <div className="max-w-3xl mx-auto">
          <div className="text-center mb-14">
            <span className="text-xs font-semibold text-primary uppercase tracking-widest mb-3 block">How it works</span>
            <h2 className="text-2xl sm:text-3xl font-bold text-foreground mb-3">The diagnostic pipeline</h2>
            <p className="text-sm text-muted max-w-lg mx-auto">
              From raw chest X-ray to clinical report in under 3 seconds.
            </p>
          </div>

          <div className="space-y-3">
            {pipeline.map(({ step, label, desc }, i) => (
              <div key={step} className="flex gap-4 items-start">
                <div className="flex flex-col items-center shrink-0">
                  <div className="w-9 h-9 rounded-xl bg-card border border-border flex items-center justify-center text-xs font-bold text-primary">
                    {step}
                  </div>
                  {i < pipeline.length - 1 && (
                    <div className="w-px h-6 bg-card mt-1" />
                  )}
                </div>
                <div className="bg-card border border-border rounded-xl px-4 py-3 flex-1 mb-1">
                  <p className="text-sm font-semibold text-foreground mb-0.5">{label}</p>
                  <p className="text-xs text-muted">{desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-20 px-4 sm:px-6">
        <div className="max-w-2xl mx-auto text-center">
          <div className="bg-gradient-to-br from-white to-orange-50/50 border border-blue-900/50 rounded-3xl p-10">
            <h2 className="text-2xl sm:text-3xl font-bold text-foreground mb-3">Ready to analyze a chest X-ray?</h2>
            <p className="text-sm text-muted mb-8 max-w-md mx-auto">
              Upload a chest X-ray image and get an AI-generated TB assessment with GradCAM localization in seconds.
            </p>
            <button
              onClick={() => navigate("/diagnose")}
              className="px-8 py-3.5 btn-primary text-sm inline-flex items-center gap-2"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <path d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"/>
              </svg>
              Start Diagnosis
            </button>
            <p className="text-xs text-muted-2 mt-4">For research purposes only — not a substitute for clinical diagnosis</p>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border py-8 px-4 sm:px-6">
        <div className="max-w-5xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-slate-600">
          <div className="flex items-center gap-2">
            <div className="w-5 h-5 rounded bg-primary  flex items-center justify-center">
              <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <path d="M12 21.35 l-1.45-1.32 C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3 c1.74 0 3.41.81 4.5 2.09 C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5 c0 3.78-3.4 6.86-8.55 11.54 Z"/>
              </svg>
            </div>
            <span className="text-card font-medium">PulmoVision</span>
          </div>
          <span>Muhamad Harith Bin Hadi · 2024699436 · UiTM Faculty of Computer and Mathematical Sciences</span>
        </div>
      </footer>
    </div>
  );
}