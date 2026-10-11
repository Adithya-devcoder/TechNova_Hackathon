/**
 * Sign vocabulary data layer.
 * Source of truth: the ML model's class labels, served by the Python backend at GET /api/vocabulary.
 * Nothing here invents signs, meanings or media — absent fields stay absent.
 */
import { queryOptions } from "@tanstack/react-query";
import { API_BASE_URL, ApiError } from "@/lib/api";

export interface SignMedia {
  type: "image" | "video";
  url: string;
}

export interface SignEntry {
  label: string;
  meaning?: string | undefined;
  category?: string | undefined;
  example?: string | undefined;
  media?: SignMedia | undefined;
}

export interface Vocabulary {
  signs: SignEntry[];
  signLanguage?: string | undefined;
}

/** Expected backend shape (strings are accepted for label-only models). */
type RawSign =
  | string
  | { label: string; meaning?: string; category?: string; example_sentence?: string; example?: string; media?: { type?: string; url?: string } };
type RawVocabulary = RawSign[] | { signs: RawSign[]; sign_language?: string };

function normalize(raw: RawVocabulary): Vocabulary {
  const list = Array.isArray(raw) ? raw : raw.signs ?? [];
  const signs: SignEntry[] = list
    .map((s): SignEntry | null => {
      if (typeof s === "string") return s.trim() ? { label: s.trim() } : null;
      if (!s?.label) return null;
      const m = s.media;
      return {
        label: s.label,
        meaning: s.meaning || undefined,
        category: s.category || undefined,
        example: s.example_sentence || s.example || undefined,
        media: m?.url && (m.type === "image" || m.type === "video") ? { type: m.type, url: m.url } : undefined,
      };
    })
    .filter((s): s is SignEntry => s !== null);
  return { signs, signLanguage: Array.isArray(raw) ? undefined : raw.sign_language };
}

export async function fetchVocabulary(): Promise<Vocabulary> {
  if (!API_BASE_URL) throw new ApiError("Backend not configured", 0);
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}/api/vocabulary`);
  } catch {
    throw new ApiError("Backend not reachable");
  }
  if (!res.ok) throw new ApiError(`Request failed (${res.status})`, res.status);
  return normalize((await res.json()) as RawVocabulary);
}

export const vocabularyQuery = queryOptions({
  queryKey: ["vocabulary"],
  queryFn: fetchVocabulary,
  retry: false,
  staleTime: 5 * 60_000,
});

export const signSlug = (label: string) => encodeURIComponent(label);
