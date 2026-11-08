// Magpie follows AgentCompass's plan templates and status fields (references/plan-templates.json is the authority).
import assert from "node:assert/strict";
import test from "node:test";

import { listQuotaAccounts, normalizeQuotaSnapshot } from "../src/index.js";

const T0 = 1_700_000_000_000;
const HOUR = 3600 * 1000;
const iso = (ms) => new Date(ms).toISOString();
const win = (over = {}) => ({ name: "7d", remaining: 80, used: 20, resetHours: 100, ...over });

const cap = (provider, plan, windowName = "7d") =>
  normalizeQuotaSnapshot(
    { [`${provider}/a`]: { kind: "subscription", plan, quota: { windows: [win({ name: windowName })] } } },
    T0
  )[0].windows[0].capacityUsd;
const entry = (provider, plan) =>
  normalizeQuotaSnapshot({ [`${provider}/a`]: { kind: "subscription", plan, quota: { windows: [win()] } } }, T0)[0];

test("Codex: plus 5h = 10; pro100 (alias prolite) and pro have no 5h window of their own", () => {
  assert.equal(cap("codex", "plus", "five_hour"), 10);
  assert.equal(cap("codex", "team", "five_hour"), 10, "team seats use the Plus prior");
  assert.equal(cap("codex", "pro100"), 175);
  assert.equal(cap("codex", "prolite"), 175);
  assert.equal(cap("codex", "pro"), 325);
  // A 5h window reported for a plan without one is observation-only (null capacity), not sized as weekly.
  assert.equal(cap("codex", "prolite", "five_hour"), null);
  assert.equal(cap("codex", "pro", "five_hour"), null);
});

test("Claude: 5h pro 18 / max5 94 / max20 376; max5 weekly_fable 700; bare max is max5", () => {
  assert.equal(cap("claude", "pro", "five_hour"), 18);
  assert.equal(cap("claude", "max5", "five_hour"), 94);
  assert.equal(cap("claude", "max20", "five_hour"), 376);
  assert.equal(cap("claude", "max5", "weekly_fable"), 700);
  assert.equal(cap("claude", "max20", "weekly_fable"), 1190);
  assert.equal(cap("claude", "max"), 1400);
  assert.equal(entry("claude", "max").planKnown, true);
  assert.equal(entry("claude", "max").planKey, "max5");
});

test("Antigravity: ultra5 and pro use the template values, ultra falls back to ultra20, unknown to pro", () => {
  assert.equal(cap("antigravity", "ultra5", "gemini-weekly"), 156);
  assert.equal(cap("antigravity", "ultra5", "gemini-5h"), 45.5);
  assert.equal(cap("antigravity", "ultra5", "3p-weekly"), 22);
  assert.equal(cap("antigravity", "ultra5", "3p-5h"), 4.4);
  assert.equal(cap("antigravity", "pro", "gemini-weekly"), 31);
  assert.equal(cap("antigravity", "pro", "gemini-5h"), 9.1);
  assert.equal(cap("antigravity", "pro", "3p-weekly"), 4.4);
  assert.equal(cap("antigravity", "pro", "3p-5h"), 0.9);
  assert.equal(cap("antigravity", "ultra20", "gemini-weekly"), 625);
  assert.equal(cap("antigravity", "Google AI Ultra", "gemini-weekly"), 625);
  assert.equal(cap("antigravity", "ultra", "3p-5h"), 17.6);
  const unknown = entry("antigravity", "mystery");
  assert.equal(unknown.planKnown, false);
  assert.equal(unknown.windows[0].capacityUsd, 31, "unknown is the smallest paid tier, never Ultra");
});

test("status rows carry status, reason, last_success_at, and the usage_failing fields", () => {
  const q = {
    "codex/ok": { kind: "subscription", plan: "plus", fetched_at: iso(T0 - 60_000), quota: { windows: [win()] } },
    "codex/bad": {
      kind: "subscription",
      plan: "plus",
      error: "boom",
      last_success_at: iso(T0 - HOUR),
      failing_since: iso(T0 - 30 * 60_000),
      quota: { windows: [win()] },
    },
    "codex/gone": {
      kind: "subscription",
      plan: "plus",
      error: "boom",
      last_success_at: iso(T0 - 7 * HOUR),
      quota: { windows: [win()] },
    },
  };
  const [ok, bad, gone] = listQuotaAccounts(q, T0);
  assert.equal(ok.status, "ready");
  assert.equal(ok.failing_since, undefined);
  assert.equal(ok.exclude_after_hours, undefined);
  assert.equal(ok.usage_excluded, undefined);

  assert.equal(bad.status, "usage_failing");
  assert.match(bad.reason, /boom/);
  assert.equal(bad.last_success_at, iso(T0 - HOUR));
  assert.equal(bad.failing_since, iso(T0 - 30 * 60_000), "an explicit failing_since is kept");
  assert.equal(bad.exclude_after_hours, 6);
  assert.equal(bad.usage_excluded, false);

  assert.equal(gone.failing_since, iso(T0 - 7 * HOUR), "without one, failing starts at the last success");
  assert.equal(gone.usage_excluded, true);
});

test("thresholds: error warns at once; no error but no fresh reading for 2h is failing; 6h excludes", () => {
  const at = (ageH, extra = {}) =>
    listQuotaAccounts(
      { "codex/a": { kind: "subscription", plan: "plus", fetched_at: iso(T0 - ageH * HOUR), quota: { windows: [win()] }, ...extra } },
      T0
    )[0];
  assert.equal(at(1.9).status, "ready");
  const stale = at(2.5);
  assert.equal(stale.status, "usage_failing");
  assert.equal(stale.usage_excluded, false);
  assert.equal(stale.dispatchable, true);
  assert.equal(at(6.5).usage_excluded, true);
  assert.equal(at(0.1, { error: "x", last_success_at: iso(T0 - 0.1 * HOUR) }).status, "usage_failing");
});

test("thresholds are configurable (usage_fresh_hours, usage_max_age_hours)", () => {
  const base = { "codex/a": { kind: "subscription", plan: "plus", fetched_at: iso(T0 - 3 * HOUR), quota: { windows: [win()] } } };
  assert.equal(listQuotaAccounts({ ...base, usage_fresh_hours: 4 }, T0)[0].status, "ready");
  const row = listQuotaAccounts({ ...base, usage_max_age_hours: 3.5, usage_fresh_hours: 1 }, T0)[0];
  assert.equal(row.exclude_after_hours, 3.5);
  assert.equal(row.usage_excluded, false);
  assert.equal(listQuotaAccounts({ ...base, usage_max_age_hours: 2.5, usage_fresh_hours: 1 }, T0)[0].usage_excluded, true);
});

test("plan: the online plan wins and a disagreement is exposed as plan_conflict", () => {
  const row = (item) =>
    listQuotaAccounts({ "codex/a": { kind: "subscription", quota: { windows: [win()] }, ...item } }, T0)[0];
  const conflict = row({ plan: "plus", plan_claim: "pro" });
  assert.deepEqual(conflict.plan_conflict, { online: "plus", credential: "pro" });
  assert.equal(cap("codex", "plus"), 67);
  assert.equal(row({ plan: "plus", plan_claim: "plus" }).plan_conflict, undefined);
  assert.equal(row({ plan: "plus" }).plan_conflict, undefined);

  // No online plan: the credential claim is the fallback, and there is nothing to conflict with.
  const [e] = normalizeQuotaSnapshot(
    { "codex/a": { kind: "subscription", plan_claim: "pro", quota: { windows: [win()] } } },
    T0
  );
  assert.equal(e.windows[0].capacityUsd, 325);
  assert.equal(e.planKnown, true);
  assert.equal(e.planConflict, null);
});
