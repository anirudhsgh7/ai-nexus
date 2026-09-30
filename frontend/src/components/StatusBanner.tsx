import type { HealthPayload } from "../types";

interface StatusBannerProps {
  health: HealthPayload | null;
  error: string | null;
}

/** Hidden when healthy; surfaces backend problems before anything is clicked. */
export function StatusBanner({ health, error }: StatusBannerProps) {
  if (error !== null) {
    return (
      <div className="banner banner-error" role="status">
        {error}
      </div>
    );
  }
  if (health === null || health.status === "ok") return null;
  const suffix =
    health.hint !== undefined && health.hint !== ""
      ? ` — ${health.hint}`
      : "";
  return (
    <div
      className={
        health.status === "unavailable"
          ? "banner banner-error"
          : "banner banner-warn"
      }
      role="status"
    >
      {health.status === "unavailable" ? "Backend unavailable" : "Backend degraded"}
      {suffix}
    </div>
  );
}
