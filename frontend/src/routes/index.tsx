import { createFileRoute, redirect } from "@tanstack/react-router";
import { supabase } from "@/integrations/supabase/client";
import { consumePostLoginRedirect } from "@/lib/post-login-redirect";

export const Route = createFileRoute("/")({
  ssr: false,
  head: () => ({
    meta: [
      { title: "SignBridge — Live sign language to text" },
      { name: "description", content: "Turn continuous signs into clear, readable text with SignBridge." },
      { property: "og:title", content: "SignBridge — Live sign language to text" },
      { property: "og:description", content: "Turn continuous signs into clear, readable text." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  beforeLoad: async () => {
    const { data } = await supabase.auth.getUser();
    if (!data.user) throw redirect({ to: "/login" });
    const pending = consumePostLoginRedirect();
    throw redirect({ href: pending ?? "/dashboard" });
  },
});
