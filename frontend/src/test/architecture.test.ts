/**
 * Architecture guard (Phase 8 PRD §6.11): the UI's boundaries are enforced
 * statically, mirroring the backend's grep-based layer checks.
 */

import { describe, expect, it } from "vitest";

const modules = import.meta.glob("../**/*.{ts,tsx}", {
  query: "?raw",
  import: "default",
  eager: true,
}) as Record<string, string>;

function isTestPath(path: string): boolean {
  return path.includes("/test/") || /\.test\.tsx?$/.test(path);
}

function offenders(pattern: RegExp, allowed: (path: string) => boolean): string[] {
  return Object.entries(modules)
    .filter(([path]) => !allowed(path))
    .filter(([, source]) => pattern.test(source))
    .map(([path]) => path)
    .sort();
}

describe("module boundaries", () => {
  it("fetch( appears only in api.ts and test files", () => {
    expect(
      offenders(/\bfetch\s*\(/, (path) => path.endsWith("/api.ts") || isTestPath(path)),
    ).toEqual([]);
  });

  it("new EventSource( appears only in sse.ts", () => {
    expect(
      offenders(/new\s+EventSource\s*\(/, (path) => path.endsWith("/sse.ts")),
    ).toEqual([]);
  });

  it("the reducer imports no I/O", () => {
    const reducer = modules["../reducer.ts"];
    expect(reducer).toBeDefined();
    expect(reducer).not.toMatch(/fetch\s*\(/);
    expect(reducer).not.toMatch(/EventSource/);
    expect(reducer).not.toMatch(/from\s+"\.\/(api|sse)"/);
  });

  it("evidence.ts is pure (no I/O, no React)", () => {
    // Phase 9 PRD §5: the derivation module joins backend data for display
    // only — it must never fetch, open streams, or pull in React.
    const evidence = modules["../evidence.ts"];
    expect(evidence).toBeDefined();
    expect(evidence).not.toMatch(/fetch\s*\(/);
    expect(evidence).not.toMatch(/EventSource/);
    expect(evidence).not.toMatch(/from\s+"\.\/(api|sse)"/);
    expect(evidence).not.toMatch(/from\s+"react"/);
  });

  it("audits.ts is pure (no I/O, no React)", () => {
    // Phase 11: audit derivation joins the step trace for display only.
    const audits = modules["../audits.ts"];
    expect(audits).toBeDefined();
    expect(audits).not.toMatch(/fetch\s*\(/);
    expect(audits).not.toMatch(/EventSource/);
    expect(audits).not.toMatch(/from\s+"react"/);
    expect(audits).not.toContain('from "./api"');
    expect(audits).not.toContain('from "./sse"');
    expect(audits).not.toContain('from "./reducer"');
    expect(audits).not.toContain('from "./evidence"');
  });

  it("components import no api/sse modules directly", () => {
    const found = Object.entries(modules)
      .filter(([path]) => path.includes("/components/"))
      .filter(([, source]) => /from\s+"\.\.\/(api|sse)"/.test(source))
      .map(([path]) => path)
      .sort();
    expect(found).toEqual([]);
  });
});
