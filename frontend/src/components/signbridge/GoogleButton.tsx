import { useState } from "react";
import { Loader2 } from "lucide-react";
import { lovable } from "@/integrations/lovable/index";
import { FormMessage } from "@/components/signbridge/AuthShell";
import { storePostLoginRedirect } from "@/lib/post-login-redirect";

function GoogleLogo() {
  return (
    <svg className="size-4 shrink-0" viewBox="0 0 24 24" aria-hidden="true">
      <path fill="#4285F4" d="M23.49 12.27c0-.79-.07-1.54-.19-2.27H12v4.51h6.47a5.57 5.57 0 0 1-2.4 3.58v3h3.86c2.26-2.09 3.56-5.17 3.56-8.82Z" />
      <path fill="#34A853" d="M12 24c3.24 0 5.95-1.08 7.93-2.91l-3.86-3c-1.08.72-2.45 1.16-4.07 1.16-3.13 0-5.78-2.11-6.73-4.96H1.29v3.09A11.99 11.99 0 0 0 12 24Z" />
      <path fill="#FBBC05" d="M5.27 14.29A7.2 7.2 0 0 1 4.89 12c0-.8.14-1.57.38-2.29V6.62H1.29a12 12 0 0 0 0 10.76l3.98-3.09Z" />
      <path fill="#EA4335" d="M12 4.75c1.77 0 3.35.61 4.6 1.8l3.42-3.42A11.97 11.97 0 0 0 12 0 11.99 11.99 0 0 0 1.29 6.62l3.98 3.09C6.22 6.86 8.87 4.75 12 4.75Z" />
    </svg>
  );
}

export function GoogleButton({ redirect }: { redirect?: string | undefined }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const signIn = async () => {
    if (loading) return;
    setError(null);
    setLoading(true);
    storePostLoginRedirect(redirect);
    const result = await lovable.auth.signInWithOAuth("google", {
      redirect_uri: window.location.origin,
    });
    if (result.error) {
      setLoading(false);
      setError(
        /popup|closed|cancel/i.test(result.error.message)
          ? "Sign-in was cancelled."
          : "Google sign-in is unavailable right now. Please try again or use email.",
      );
      return;
    }
    if (result.redirected) return; // browser is leaving for Google
    setLoading(false);
  };

  return (
    <div className="mt-6">
      <div className="mb-6 flex items-center gap-3 text-[0.6875rem] font-bold uppercase tracking-[0.16em] text-paper-muted">
        <span className="h-px flex-1 bg-paper-muted/30" />
        or
        <span className="h-px flex-1 bg-paper-muted/30" />
      </div>
      {error && <FormMessage tone="error">{error}</FormMessage>}
      <button
        type="button"
        onClick={signIn}
        disabled={loading}
        className="press group flex w-full items-center justify-center gap-3 border border-paper-foreground/25 px-6 py-3 text-sm font-semibold tracking-wide hover:border-paper-foreground hover:bg-paper-foreground/5 disabled:opacity-60"
      >
        {loading ? <Loader2 className="size-4 animate-spin" /> : <GoogleLogo />}
        Continue with Google
      </button>
    </div>
  );
}
