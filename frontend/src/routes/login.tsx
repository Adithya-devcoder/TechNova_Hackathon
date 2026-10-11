import { useState, type FormEvent } from "react";
import { createFileRoute, Link, redirect, useNavigate } from "@tanstack/react-router";
import { Loader2 } from "lucide-react";
import { z } from "zod";
import { supabase } from "@/integrations/supabase/client";
import { setRememberMe } from "@/hooks/use-auth";
import { AuthShell, Field, FormMessage } from "@/components/signbridge/AuthShell";
import { submitCls } from "@/lib/auth-styles";
import { GoogleButton } from "@/components/signbridge/GoogleButton";

/** Only same-origin relative paths are accepted as post-login destinations. */
function safeRedirect(r: unknown): string | undefined {
  if (typeof r !== "string" || !r.startsWith("/") || r.startsWith("//")) return undefined;
  if (r === "/" || r.startsWith("/login")) return undefined;
  return r;
}

export const Route = createFileRoute("/login")({
  ssr: false,
  validateSearch: (s: Record<string, unknown>): { redirect?: string } => {
    const r = safeRedirect(s["redirect"]);
    return r ? { redirect: r } : {};
  },
  beforeLoad: async ({ search }) => {
    const { data } = await supabase.auth.getUser();
    if (data.user) throw redirect({ href: search.redirect ?? "/dashboard" });
  },
  head: () => ({
    meta: [
      { title: "Sign in — SignBridge" },
      { name: "description", content: "Sign in to your SignBridge account." },
      { property: "og:title", content: "Sign in — SignBridge" },
      { property: "og:description", content: "Sign in to your SignBridge account." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
  }),
  component: LoginPage,
});

const schema = z.object({
  email: z.string().trim().email("Enter a valid email address").max(255),
  password: z.string().min(1, "Enter your password"),
});

function LoginPage() {
  const navigate = useNavigate();
  const { redirect: dest } = Route.useSearch();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [remember, setRemember] = useState(true);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [formError, setFormError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (loading) return;
    setFormError(null);
    const parsed = schema.safeParse({ email, password });
    if (!parsed.success) {
      setErrors(Object.fromEntries(parsed.error.issues.map((i) => [i.path[0], i.message])));
      return;
    }
    setErrors({});
    setLoading(true);
    const { error } = await supabase.auth.signInWithPassword(parsed.data);
    setLoading(false);
    if (error) {
      setFormError(
        error.message.toLowerCase().includes("confirm")
          ? "Please confirm your email first — check your inbox for the link."
          : "Email or password is incorrect.",
      );
      return;
    }
    setRememberMe(remember);
    navigate({ to: "/" });
  };

  return (
    <AuthShell eyebrow="Account / Sign in" title={<>Welcome<br /><span className="italic text-primary">back.</span></>}>
      <h2 className="mb-8 text-[0.6875rem] font-bold uppercase tracking-[0.16em]">Sign in / 01</h2>
      <form onSubmit={submit} noValidate>
        {formError && <FormMessage tone="error">{formError}</FormMessage>}
        <Field id="email" label="Email" type="email" autoComplete="email" value={email} onChange={(e) => setEmail(e.target.value)} error={errors["email"]} />
        <Field id="password" label="Password" type="password" autoComplete="current-password" value={password}
          onChange={(e) => setPassword(e.target.value)} error={errors["password"]}
          aside={<Link to="/forgot-password" className="text-xs text-paper-muted underline-offset-4 hover:text-paper-foreground hover:underline">Forgot password?</Link>} />
        <label className="mb-6 flex cursor-pointer items-center gap-2 text-sm">
          <input type="checkbox" checked={remember} onChange={(e) => setRemember(e.target.checked)} className="size-4 accent-paper-foreground" />
          Remember me
        </label>
        <button type="submit" className={submitCls} disabled={loading}>
          {loading && <Loader2 className="size-4 animate-spin" />} Sign in
        </button>
      </form>
      <GoogleButton redirect={dest} />
      <p className="mt-6 text-sm text-paper-muted">
        New to SignBridge? <Link to="/signup" className="font-semibold text-paper-foreground underline-offset-4 hover:underline">Create an account</Link>
      </p>
    </AuthShell>
  );
}
