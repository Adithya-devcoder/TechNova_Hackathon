import { cn } from "@/lib/utils";

export function StatusDot({ tone }: { tone: "ok" | "off" | "idle" | "live" | "warn" }) {
  return (
    <span
      aria-hidden="true"
      className={cn(
        "inline-block size-1.5 rounded-full",
        tone === "ok" && "bg-primary",
        tone === "off" && "bg-destructive",
        tone === "idle" && "bg-muted-foreground",
        tone === "warn" && "bg-warning",
        tone === "live" && "bg-destructive animate-pulse",
      )}
    />
  );
}
