import { useEffect, useState, type FormEvent } from "react";
import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { Loader2 } from "lucide-react";
import { supabase } from "@/integrations/supabase/client";
import { AuthShell, Field, FormMessage } from "@/components/signbridge/AuthShell";
import { submitCls } from "@/lib/auth-styles";

export const Route = createFileRoute("/reset-password")({
  head: () => ({
    meta: [
      { title: "Choose a new password — SignBridge" },
      { name: "description", content: "Set a new password for your SignBridge account." },
      { property: "og:title", content: "Choose a new password — SignBridge" },
      { property: "og:description", content: "Set a new password for your SignBridge account." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
  }),
  component: ResetPage,
});

function ResetPage() {
  const navigate = useNavigate();
  const [ready, setReady] = useState(false);
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [formError, setFormError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const isRecovery = window.location.hash.includes("type=recovery");
    const { data } = supabase.auth.onAuthStateChange((event) => {
      if (event === "PASSWORD_RECOVERY") setReady(true);
    });
    supabase.auth.getSession().then(({ data: s }) => { if (s.session && isRecovery) setReady(true); else if (s.session) setReady(true); });
    return () => data.subscription.unsubscribe();
  }, []);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (loading) return;
    const errs: Record<string, string> = {};
    if (password.length < 8) errs["password"] = "Use at least 8 characters";
    if (password !== confirm) errs["confirm"] = "Passwords don't match";
    setErrors(errs);
    if (Object.keys(errs).length) return;
    setLoading(true);
    const { error } = await supabase.auth.updateUser({ password });
    setLoading(false);
    if (error) return setFormError(error.message);
    navigate({ to: "/dashboard" });
  };

  return (
    <AuthShell eyebrow="Account / Recovery" title={<>A fresh<br /><span className="italic text-primary">start.</span></>}>
      <h2 className="mb-8 text-[0.6875rem] font-bold uppercase tracking-[0.16em]">New password / 02</h2>
      {!ready ? (
        <div>
          <FormMessage tone="error">This reset link is invalid or has expired.</FormMessage>
          <Link to="/forgot-password" className="text-sm font-semibold underline-offset-4 hover:underline">Request a new link</Link>
        </div>
      ) : (
        <form onSubmit={submit} noValidate>
          {formError && <FormMessage tone="error">{formError}</FormMessage>}
          <Field id="password" label="New password" type="password" autoComplete="new-password" value={password} onChange={(e) => setPassword(e.target.value)} error={errors["password"]} />
          <Field id="confirm" label="Confirm password" type="password" autoComplete="new-password" value={confirm} onChange={(e) => setConfirm(e.target.value)} error={errors["confirm"]} />
          <button type="submit" className={submitCls} disabled={loading}>
            {loading && <Loader2 className="size-4 animate-spin" />} Update password
          </button>
        </form>
      )}
    </AuthShell>
  );
}
