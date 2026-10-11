import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";
import { requireSupabaseAuth } from "@/integrations/supabase/auth-middleware";

const NAMES = { en: "English", ta: "Tamil", hi: "Hindi", te: "Telugu" } as const;

export type TranslateOutcome =
  | { ok: true; text: string }
  | { ok: false; reason: "unconfigured" | "rate_limited" | "credits" | "failed"; message: string };

export const translateText = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .validator((input) =>
    z.object({ text: z.string().trim().min(1).max(2000), target: z.enum(["en", "ta", "hi", "te"]) }).parse(input),
  )
  .handler(async ({ data }): Promise<TranslateOutcome> => {
    const key = process.env["LOVABLE_API_KEY"];
    if (!key) return { ok: false, reason: "unconfigured", message: "Translation service is not configured." };

    const res = await fetch("https://ai.gateway.lovable.dev/v1/responses", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Lovable-API-Key": key,
        "X-Lovable-AIG-SDK": "fetch",
      },
      body: JSON.stringify({
        model: "openai/gpt-6-astra",
        stream: true,
        store: false,
        reasoning: { effort: "low" },
        instructions: `You are a professional translator. Translate the user's text into ${NAMES[data.target]} using its native script. Output only the translation — no quotes, notes or explanations. If the text is already in ${NAMES[data.target]}, return it unchanged.`,
        input: data.text,
      }),
    });

    if (!res.ok || !res.body) {
      const body = await res.text().catch(() => "");
      console.error("translate failed", res.status, body.slice(0, 500));
      if (res.status === 429) return { ok: false, reason: "rate_limited", message: "Too many requests — please wait a moment." };
      if (res.status === 402 || res.status === 403)
        return { ok: false, reason: "credits", message: "Translation is paused: AI usage limit reached for this workspace." };
      return { ok: false, reason: "failed", message: "Translation failed — please try again." };
    }

    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    let out = "";
    let failed = false;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, i).trim();
        buf = buf.slice(i + 1);
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (!payload || payload === "[DONE]") continue;
        try {
          const ev = JSON.parse(payload) as { type?: string; delta?: string };
          if (ev.type === "response.output_text.delta" && ev.delta) out += ev.delta;
          if (ev.type === "response.failed" || ev.type === "error") failed = true;
        } catch {
          /* partial line */
        }
      }
    }
    const text = out.trim();
    if (failed || !text) return { ok: false, reason: "failed", message: "Translation failed — please try again." };
    return { ok: true, text };
  });
