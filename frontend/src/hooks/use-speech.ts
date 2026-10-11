
import { useCallback, useEffect, useRef, useState } from "react";
import type { Language } from "@/lib/api";

const LOCALES: Record<Language, string> = {
  en: "en-US",
  ta: "ta-IN",
  hi: "hi-IN",
  te: "te-IN",
};

export function useSpeech(lang: Language) {
  const [supported, setSupported] = useState<boolean | null>(null);
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [speaking, setSpeaking] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const fallbackTried = useRef(false);

  useEffect(() => {
    const available =
      typeof window !== "undefined" &&
      "speechSynthesis" in window &&
      "SpeechSynthesisUtterance" in window;

    setSupported(available);
    if (!available) return;

    const loadVoices = () => {
      setVoices(window.speechSynthesis.getVoices());
    };

    loadVoices();
    window.speechSynthesis.addEventListener("voiceschanged", loadVoices);

    return () => {
      window.speechSynthesis.removeEventListener("voiceschanged", loadVoices);
      window.speechSynthesis.cancel();
      audioRef.current?.pause();
    };
  }, []);

  const voice =
    voices.find(
      (v) => v.lang.toLowerCase() === LOCALES[lang].toLowerCase(),
    ) ??
    voices.find((v) => v.lang.toLowerCase().startsWith(lang + "-")) ??
    null;

  const stop = useCallback(() => {
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }

    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.src = "";
      audioRef.current = null;
    }

    fallbackTried.current = false;
    setSpeaking(false);
  }, []);

  const speak = useCallback(
    (text: string) => {
      const value = text.trim();
      if (!value) return false;

      stop();

      const playOnlineFallback = () => {
        if (fallbackTried.current) return;
        fallbackTried.current = true;

        const url =
          "https://translate.google.com/translate_tts?ie=UTF-8&client=tw-ob&tl=" +
          lang +
          "&q=" +
          encodeURIComponent(value.slice(0, 180));

        const audio = new Audio(url);
        audioRef.current = audio;

        audio.onplay = () => setSpeaking(true);
        audio.onended = () => {
          audioRef.current = null;
          setSpeaking(false);
        };
        audio.onerror = () => {
          audioRef.current = null;
          setSpeaking(false);
          console.error("Online voice playback failed.");
        };

        void audio.play().catch((error) => {
          console.error("Voice playback failed:", error);
          audioRef.current = null;
          setSpeaking(false);
        });
      };

      if (
        typeof window === "undefined" ||
        !("speechSynthesis" in window)
      ) {
        playOnlineFallback();
        return true;
      }

      const utterance = new SpeechSynthesisUtterance(value);
      utterance.lang = LOCALES[lang];

      if (voice) utterance.voice = voice;

      utterance.onstart = () => setSpeaking(true);
      utterance.onend = () => setSpeaking(false);
      utterance.onerror = (event) => {
        setSpeaking(false);
        if (event.error !== "canceled" && event.error !== "interrupted") {
          playOnlineFallback();
        }
      };

      window.speechSynthesis.speak(utterance);
      return true;
    },
    [lang, stop, voice],
  );

  return {
    supported,
    voicesLoaded: voices.length > 0,
    voice,
    speaking,
    speak,
    stop,
  };
}
