import { Settings2, LogOut } from "lucide-react";
import { Link, useNavigate } from "@tanstack/react-router";
import type { BackendStatus } from "@/hooks/use-backend";
import { useAuth } from "@/hooks/use-auth";
import { StatusDot } from "./StatusDot";

function AccountControl() {
  const { user, ready, signOut } = useAuth();
  const navigate = useNavigate();
  if (!ready) return null;
  if (!user)
    return <Link to="/login" className="btn btn-ghost text-xs uppercase tracking-[0.14em]">Sign in</Link>;
  return (
    <button onClick={async () => { await signOut(); navigate({ to: "/login", replace: true }); }} className="btn btn-ghost text-xs" aria-label="Log out" title={user.email ?? "Log out"}>
      <LogOut className="size-4" /> <span className="hidden sm:inline">Log out</span>
    </button>
  );
}

export function Logo() {
  return (
    <svg width="28" height="28" viewBox="0 0 32 32" aria-hidden="true">
      <path d="M3 24 V14 a6 6 0 0 1 12 0 v10" fill="none" stroke="var(--color-foreground)" strokeWidth="2.2" />
      <path d="M17 24 V14 a6 6 0 0 1 12 0 v10" fill="none" stroke="var(--color-primary)" strokeWidth="2.2" />
      <path d="M9 18 H23" stroke="var(--color-foreground)" strokeWidth="2.2" />
    </svg>
  );
}

const statusLabel: Record<BackendStatus, string> = {
  checking: "Checking",
  connected: "Backend online",
  disconnected: "Backend offline",
  unconfigured: "Backend not configured",
};

export function Nav({ status, onRecheck }: { status: BackendStatus; onRecheck: () => void }) {
  return (
    <header className="border-b">
      <nav className="mx-auto flex h-16 max-w-7xl items-center gap-6 px-5 md:px-10" aria-label="Main">
        <a href="#top" className="flex items-center gap-3">
          <Logo />
          <span className="text-sm font-bold tracking-[0.32em]">SIGNBRIDGE</span>
        </a>
        <ul className="mx-auto hidden gap-8 text-xs font-medium uppercase tracking-[0.14em] text-muted-foreground md:flex">
          <li><a className="transition-colors hover:text-primary" href="#translate">Translate</a></li>
          <li><a className="transition-colors hover:text-primary" href="#how">How it works</a></li>
          <li><a className="transition-colors hover:text-primary" href="#signs">Vocabulary</a></li>
        </ul>
        <div className="ml-auto flex items-center gap-1 md:ml-0">
          <button onClick={onRecheck} className="btn btn-ghost text-xs" title="Recheck connection">
            <StatusDot tone={status === "connected" ? "ok" : status === "checking" ? "idle" : "off"} />
            <span className="hidden sm:inline">{statusLabel[status]}</span>
          </button>
          <button className="btn btn-ghost" aria-label="Settings" title="Configured via VITE_API_BASE_URL">
            <Settings2 className="size-4" />
          </button>
          <AccountControl />
        </div>
      </nav>
    </header>
  );
}
