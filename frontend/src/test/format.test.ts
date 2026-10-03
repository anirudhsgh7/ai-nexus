import { describe, expect, it } from "vitest";

import {
  ACCOUNTABILITY_STATUS_LABELS,
  ACCOUNTABILITY_STATUS_TONES,
  AGENT_LABELS,
  CLAIM_STATUS_LABELS,
  CLAIM_STATUS_TONES,
  FLAG_SEVERITY_LABELS,
  FLAG_SEVERITY_TONES,
  VERDICT_LABELS,
  VERIFICATION_STATUS_LABELS,
  VERIFICATION_STATUS_TONES,
  agentLabel,
  formatConfidence,
  formatDuration,
  formatElapsed,
  pluralize,
  verdictLabel,
} from "../format";

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

// ------------------------------------------------- Phase 9 evidence helpers

describe("claim status labels", () => {
  it("labels and tones all six statuses", () => {
    expect(Object.keys(CLAIM_STATUS_LABELS)).toHaveLength(6);
    for (const status of Object.keys(CLAIM_STATUS_LABELS) as Array<
      keyof typeof CLAIM_STATUS_LABELS
    >) {
      expect(CLAIM_STATUS_LABELS[status]).not.toBe("");
      expect(CLAIM_STATUS_TONES[status]).toMatch(/^claim-/);
    }
    expect(CLAIM_STATUS_LABELS.fact).toBe("Fact");
    expect(CLAIM_STATUS_LABELS.unverified).toBe("Unverified");
  });

  it("labels all verdicts and the unevaluated case", () => {
    expect(VERDICT_LABELS).toEqual({
      supported: "Supported",
      refuted: "Refuted",
      unverifiable: "Unverifiable",
    });
    expect(verdictLabel("refuted")).toBe("Refuted");
    expect(verdictLabel(null)).toBe("Not evaluated");
  });
});

describe("formatConfidence", () => {
  it("renders percentages to the nearest integer", () => {
    expect(formatConfidence(0.555)).toBe("56%");
    expect(formatConfidence(0)).toBe("0%");
    expect(formatConfidence(1)).toBe("100%");
  });

  it("renders missing and out-of-range values as a dash", () => {
    expect(formatConfidence(null)).toBe("—");
    expect(formatConfidence(1.01)).toBe("—");
    expect(formatConfidence(-0.01)).toBe("—");
    expect(formatConfidence(Number.NaN)).toBe("—");
  });
});


// --------------------------------------------- Phase 11 audit display labels

describe("audit display labels", () => {
  it("labels all four verification statuses with paired tones", () => {
    expect(Object.keys(VERIFICATION_STATUS_LABELS)).toHaveLength(4);
    for (const status of Object.keys(VERIFICATION_STATUS_LABELS) as Array<
      keyof typeof VERIFICATION_STATUS_LABELS
    >) {
      expect(VERIFICATION_STATUS_LABELS[status]).not.toBe("");
      expect(VERIFICATION_STATUS_TONES[status]).toMatch(/^vf-/);
    }
    expect(VERIFICATION_STATUS_LABELS.partially_verified).toBe(
      "Partially verified",
    );
  });

  it("labels flag severities with paired tones", () => {
    expect(FLAG_SEVERITY_LABELS).toEqual({
      info: "info",
      warning: "warning",
      violation: "violation",
    });
    for (const severity of ["info", "warning", "violation"] as const) {
      expect(FLAG_SEVERITY_TONES[severity]).toMatch(/^flag-/);
    }
  });

  it("labels accountability statuses with paired tones", () => {
    expect(ACCOUNTABILITY_STATUS_LABELS).toEqual({
      clean: "Clean",
      warnings: "Warnings",
      violations: "Violations",
    });
    for (const status of ["clean", "warnings", "violations"] as const) {
      expect(ACCOUNTABILITY_STATUS_TONES[status]).not.toBe("");
    }
  });

  it("labels all six agent roles", () => {
    expect(Object.keys(AGENT_LABELS)).toHaveLength(6);
    expect(agentLabel("verifier")).toBe("Verifier");
    expect(agentLabel("accountability")).toBe("Accountability");
  });
});
