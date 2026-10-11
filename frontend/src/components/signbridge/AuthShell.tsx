import { useState, type InputHTMLAttributes, type ReactNode } from "react";
import { Link } from "@tanstack/react-router";
import { Eye, EyeOff } from "lucide-react";
import { Logo } from "./Nav";

export function AuthShell({ eyebrow, title, children }: { eyebrow: string; title: ReactNode; children: ReactNode }) {
  return (
    <div className="min-h-screen">
      <header className="border-b animate-fade">
        <div className="mx-auto flex h-16 max-w-7xl items-center px-5 md:px-10">
          <Link to="/" className="group flex items-center gap-3">
            <span className="transition-transform duration-500 group-hover:-translate-y-0.5"><Logo /></span>
            <span className="text-sm font-bold tracking-[0.32em]">SIGNBRIDGE</span>
          </Link>
        </div>
      </header>
      <main className="mx-auto grid max-w-7xl gap-12 px-5 py-14 md:px-10 lg:grid-cols-[1.1fr_1fr] lg:py-20">
        <div>
          <p className="eyebrow animate-rise [--d:60ms]">{eyebrow}</p>
          <h1 className="mt-6 font-serif text-5xl leading-[0.95] tracking-tight animate-rise [--d:140ms] sm:text-6xl lg:text-7xl">{title}</h1>
        </div>
        <div className="stagger bg-paper p-6 text-paper-foreground animate-rise [--d:120ms] md:p-10">{children}</div>
      </main>
    </div>
  );
}

const inputCls =
  "h-12 w-full border border-paper-border bg-transparent px-3 text-base text-paper-foreground outline-none transition-[border-color,box-shadow] duration-300 placeholder:text-paper-muted hover:border-paper-muted focus:border-paper-foreground focus:shadow-[0_0_0_3px_color-mix(in_oklab,var(--color-paper-foreground)_10%,transparent)] aria-[invalid=true]:border-destructive";

interface FieldProps extends InputHTMLAttributes<HTMLInputElement> {
  label: string;
  error?: string | undefined;
  aside?: ReactNode;
}

export function Field({ label, error, aside, id, type, ...rest }: FieldProps) {
  const [show, setShow] = useState(false);
  const isPw = type === "password";
  return (
    <div className="mb-5">
      <div className="mb-2 flex items-baseline justify-between">
        <label htmlFor={id} className="text-[0.6875rem] font-bold uppercase tracking-[0.16em]">{label}</label>
        {aside}
      </div>
      <div className="relative">
        <input id={id} type={isPw && show ? "text" : type} aria-invalid={!!error} aria-describedby={error ? `${id}-err` : undefined}
          className={inputCls + (isPw ? " pr-12" : "")} {...rest} />
        {isPw && (
          <button type="button" onClick={() => setShow((s) => !s)} aria-label={show ? "Hide password" : "Show password"}
            className="absolute inset-y-0 right-0 flex w-12 items-center justify-center text-paper-muted transition-colors hover:text-paper-foreground">
            {show ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
          </button>
        )}
      </div>
      {error && <p id={`${id}-err`} className="mt-1.5 text-xs text-destructive animate-drop">{error}</p>}
    </div>
  );
}

export function FormMessage({ tone, children }: { tone: "error" | "success"; children: ReactNode }) {
  return (
    <p role={tone === "error" ? "alert" : "status"}
      className={"mb-5 border-l-2 px-3 py-2 text-sm animate-drop " + (tone === "error" ? "border-destructive text-destructive" : "border-paper-foreground")}>
      {children}
    </p>
  );
}
