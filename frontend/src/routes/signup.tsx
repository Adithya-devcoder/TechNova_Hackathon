import { useState, type FormEvent } from "react";
import { createFileRoute, Link } from "@tanstack/react-router";
import { Loader2 } from "lucide-react";
import { z } from "zod";
import { supabase } from "@/integrations/supabase/client";
import { AuthShell, Field, FormMessage } from "@/components/signbridge/AuthShell";
import { submitCls } from "@/lib/auth-styles";
import { GoogleButton } from "@/components/signbridge/GoogleButton";

export const Route = createFileRoute("/signup")({
  head: () => ({
    meta: [
      { title: "Create account — SignBridge" },
      { name: "description", content: "Create a SignBridge account." },
      { property: "og:title", content: "Create account — SignBridge" },
      { property: "og:description", content: "Create a SignBridge account." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
  }),
  component: SignupPage,
});

const schema = z
  .object({
    name: z.string().trim().min(1, "Enter your full name").max(100, "Name is too long"),
    email: z.string().trim().email("Enter a valid email address").max(255),
    password: z.string().min(8, "Use at least 8 characters").max(72, "Password is too long"),
    confirm: z.string(),
  })
  .refine((v) => v.password === v.confirm, { path: ["confirm"], message: "Passwords don't match" });

function SignupPage() {
  const [form, setForm] = useState({ name: "", email: "", password: "", confirm: "" });
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [formError, setFormError] = useState<string | null>(null);
  const [sentTo, setSentTo] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const set = (k: keyof typeof form) => (e: React.ChangeEvent<HTMLInputElement>) => setForm((f) => ({ ...f, [k]: e.target.value }));

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (loading) return;
    setFormError(null);
    const parsed = schema.safeParse(form);
    if (!parsed.success) {
      setErrors(Object.fromEntries(parsed.error.issues.map((i) => [i.path[0], i.message])));
      return;
    }
    setErrors({});
    setLoading(true);
    const { error } = await supabase.auth.signUp({
      email: parsed.data.email,
      password: parsed.data.password,
      options: { emailRedirectTo: window.location.origin, data: { full_name: parsed.data.name } },
    });
    setLoading(false);
    if (error) {
      setFormError(error.message);
      return;
    }
    setSentTo(parsed.data.email);
  };

  return (
    <AuthShell eyebrow="Account / Create" title={<>Start the<br /><span className="italic text-primary">conversation.</span></>}>
      <h2 className="mb-8 text-[0.6875rem] font-bold uppercase tracking-[0.16em]">Create account / 01</h2>
      {sentTo ? (
        <div>
          <FormMessage tone="success">Check your inbox — we sent a confirmation link to <strong>{sentTo}</strong>.</FormMessage>
          <p className="text-sm text-paper-muted">Once confirmed, you can sign in.</p>
          <Link to="/login" className={submitCls + " mt-6"}>Go to sign in</Link>
        </div>
      ) : (
        <form onSubmit={submit} noValidate>
          {formError && <FormMessage tone="error">{formError}</FormMessage>}
          <Field id="name" label="Full name" autoComplete="name" value={form.name} onChange={set("name")} error={errors["name"]} />
          <Field id="email" label="Email" type="email" autoComplete="email" value={form.email} onChange={set("email")} error={errors["email"]} />
          <Field id="password" label="Password" type="password" autoComplete="new-password" value={form.password} onChange={set("password")} error={errors["password"]} />
          <Field id="confirm" label="Confirm password" type="password" autoComplete="new-password" value={form.confirm} onChange={set("confirm")} error={errors["confirm"]} />
          <button type="submit" className={submitCls + " mt-2"} disabled={loading}>
            {loading && <Loader2 className="size-4 animate-spin" />} Create account
          </button>
        </form>
      )}
      {!sentTo && <GoogleButton />}
      <p className="mt-6 text-sm text-paper-muted">
        Already have an account? <Link to="/login" className="font-semibold text-paper-foreground underline-offset-4 hover:underline">Sign in</Link>
      </p>
    </AuthShell>
  );
}
