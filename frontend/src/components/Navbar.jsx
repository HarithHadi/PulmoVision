import { Link, useLocation } from "react-router-dom";
import { useAuth } from "../AuthContext";

const navLinks = [
  { label: "Features",    href: "/#features" },
  { label: "Pipeline",    href: "/#pipeline" },
  { label: "Performance", href: "/#stats" },
];

export default function Navbar() {
  const location = useLocation();
  const isHome = location.pathname === "/";
  const { token, radiologistName, logout } = useAuth();

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 border-b border-border/60 bg-card/80 backdrop-blur-md">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 h-14 grid grid-cols-3 items-center">

        {/* Left — Logo */}
        <Link to="/" className="flex items-center gap-2.5 shrink-0">
          <div className="w-7 h-7 rounded-lg bg-primary/90 flex items-center justify-center">
            <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
              <path d="M12 21.35 l-1.45-1.32 C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3 c1.74 0 3.41.81 4.5 2.09 C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5 c0 3.78-3.4 6.86-8.55 11.54 Z"/>
            </svg>
          </div>
          <span className="font-semibold text-sm text-card tracking-tight">PulmoVision</span>
        </Link>

        {/* Center — Nav links or page buttons */}
        <div className="hidden md:flex items-center justify-center gap-3 text-sm">
          {isHome ? (
            navLinks.map(({ label, href }) => (
              <a key={label} href={href} className="text-slate-400 hover:text-primary transition-colors">
                {label}
              </a>
            ))
          ) : (
            <div className="flex items-center gap-2">
              <Link
                to="/diagnose"
                className={`text-sm font-medium px-4 py-1.5 rounded-lg transition-all border ${
                  location.pathname === "/diagnose"
                    ? "border-primary text-primary"
                    : "border-border text-slate-400 hover:text-foreground hover:border-border-hover"
                }`}
              >
                TB Detection
              </Link>
              {token && (
                <Link
                  to="/patients"
                  className={`text-sm font-medium px-4 py-1.5 rounded-lg transition-all border ${
                    location.pathname === "/patients"
                      ? "border-primary text-primary"
                      : "border-border text-slate-400 hover:text-foreground hover:border-border-hover"
                  }`}
                >
                  Patient Records
                </Link>
              )}
            </div>
          )}
        </div>

        {/* Right — Auth */}
        <div className="flex items-center justify-end gap-3">
          {token ? (
            <>
              <span className="hidden sm:block text-xs text-slate-500">
                {radiologistName}
              </span>
              <button
                onClick={logout}
                className="text-sm font-medium border border-border text-slate-400 hover:text-red-400 hover:border-red-400/40 px-4 py-1.5 rounded-lg transition-all"
              >
                Logout
              </button>
            </>
          ) : (
            <Link
              to="/diagnose"
              className="text-sm font-medium bg-primary/90 hover:bg-primary-hover text-white px-4 py-1.5 rounded-lg transition-all"
            >
              Launch App
            </Link>
          )}
        </div>

      </div>
    </nav>
  );
}