import { useState, type FormEvent } from "react";
import { createFileRoute, Link } from "@tanstack/react-router";
import { Loader2 } from "lucide-react";
import { z } from "zod";
import { supabase } from "@/integrations/supabase/client";
import { AuthShell, Field, FormMessage } from "@/components/signbridge/AuthShell";
import { submitCls } from "@/lib/auth-styles";

export const Route = createFileRoute("/forgot-password")({
  head: () => ({
    meta: [
      { title: "Reset password — SignBridge" },
      { name: "description", content: "Request a SignBridge password reset link." },
      { property: "og:title", content: "Reset password — SignBridge" },
      { property: "og:description", content: "Request a SignBridge password reset link." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
  }),
  component: ForgotPage,
});

function ForgotPage() {
  const [email, setEmail] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [sent, setSent] = useState(false);
  const [loading, setLoading] = useState(false);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (loading) return;
    const parsed = z.string().trim().email().max(255).safeParse(email);
    if (!parsed.success) return setError("Enter a valid email address");
    setError(null);
    setLoading(true);
    const { error } = await supabase.auth.resetPasswordForEmail(parsed.data, {
      redirectTo: `${window.location.origin}/reset-password`,
    });
    setLoading(false);
    if (error) return setError("We couldn't send the reset link. Please try again.");
    setSent(true);
  };

  return (
    <AuthShell eyebrow="Account / Recovery" title={<>Lost your<br /><span className="italic text-primary">way in?</span></>}>
      <h2 className="mb-8 text-[0.6875rem] font-bold uppercase tracking-[0.16em]">Reset password / 01</h2>
      {sent ? (
        <FormMessage tone="success">If an account exists for that email, a reset link is on its way.</FormMessage>
      ) : (
        <form onSubmit={submit} noValidate>
          <p className="mb-6 text-sm text-paper-muted">Enter your email and we'll send you a link to choose a new password.</p>
          <Field id="email" label="Email" type="email" autoComplete="email" value={email} onChange={(e) => setEmail(e.target.value)} error={error ?? undefined} />
          <button type="submit" className={submitCls} disabled={loading}>
            {loading && <Loader2 className="size-4 animate-spin" />} Send reset link
          </button>
        </form>
      )}
      <p className="mt-6 text-sm text-paper-muted">
        <Link to="/login" className="font-semibold text-paper-foreground underline-offset-4 hover:underline">Back to sign in</Link>
      </p>
    </AuthShell>
  );
}
