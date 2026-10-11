import { useEffect, useState } from "react";
import type { User } from "@supabase/supabase-js";
import { supabase } from "@/integrations/supabase/client";

const EPHEMERAL_FLAG = "signbridge-ephemeral-session";

/** Marks the session as "don't remember me": it is dropped when the browser session ends. */
export function setRememberMe(remember: boolean) {
  if (remember) {
    localStorage.removeItem(EPHEMERAL_FLAG);
    sessionStorage.removeItem(EPHEMERAL_FLAG);
  } else {
    localStorage.setItem(EPHEMERAL_FLAG, "1");
    sessionStorage.setItem(EPHEMERAL_FLAG, "1");
  }
}

export function useAuth() {
  const [user, setUser] = useState<User | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const { data: sub } = supabase.auth.onAuthStateChange((_e, session) => {
      setUser(session?.user ?? null);
    });
    (async () => {
      // Browser was closed since a "don't remember me" sign-in: end that session.
      if (localStorage.getItem(EPHEMERAL_FLAG) && !sessionStorage.getItem(EPHEMERAL_FLAG)) {
        localStorage.removeItem(EPHEMERAL_FLAG);
        await supabase.auth.signOut();
      }
      const { data } = await supabase.auth.getUser();
      setUser(data.user ?? null);
      setReady(true);
    })();
    return () => sub.subscription.unsubscribe();
  }, []);

  const signOut = async () => {
    await supabase.auth.signOut();
    localStorage.removeItem(EPHEMERAL_FLAG);
    sessionStorage.removeItem(EPHEMERAL_FLAG);
  };

  return { user, ready, signOut };
}
