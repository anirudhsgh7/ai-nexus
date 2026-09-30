import { describe, expect, it } from "vitest";

import { agentLabel, formatDuration, formatElapsed, pluralize } from "../format";

describe("formatDuration", () => {
  it("renders sub-minute durations with one decimal", () => {
    expect(formatDuration(100)).toBe("0.1s");
    expect(formatDuration(59_900)).toBe("59.9s");
  });

  it("renders minute durations", () => {
    expect(formatDuration(60_000)).toBe("1m 0.0s");
    expect(formatDuration(118_900)).toBe("1m 58.9s");
    expect(formatDuration(178_900)).toBe("2m 58.9s");
  });

  it("renders missing durations as a dash", () => {
    expect(formatDuration(null)).toBe("—");
    expect(formatDuration(Number.NaN)).toBe("—");
  });
});

describe("formatElapsed", () => {
  it("uses the finished timestamp when present", () => {
    expect(
      formatElapsed("2026-09-30T12:00:00.000Z", "2026-09-30T12:02:00.000Z", 0),
    ).toBe("2m 0.0s");
  });

  it("uses the provided now while a run is live", () => {
    const start = "2026-09-30T12:00:00.000Z";
    const now = Date.parse(start) + 5_000;
    expect(formatElapsed(start, null, now)).toBe("5.0s");
  });

  it("is dash-safe for missing or invalid timestamps", () => {
    expect(formatElapsed(null, null, Date.now())).toBe("—");
    expect(formatElapsed("not-a-date", null, Date.now())).toBe("—");
  });
});

describe("agentLabel and pluralize", () => {
  it("labels roles", () => {
    expect(agentLabel("manager")).toBe("Manager");
    expect(agentLabel("skeptic")).toBe("Skeptic");
  });

  it("pluralizes correctly", () => {
    expect(pluralize(1, "claim")).toBe("1 claim");
    expect(pluralize(2, "claim")).toBe("2 claims");
    expect(pluralize(0, "verdict")).toBe("0 verdicts");
  });
});
