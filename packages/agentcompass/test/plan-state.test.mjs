// Plan sizing and account state (CONTRACT Q3/Q4): capacity follows the plan, an account that is expired,
// signed out, disabled or failing for too long is never selected, and every non-ready account is listed.
import assert from "node:assert/strict";
import test from "node:test";

import {
  SessionContinuityStore,
  evaluateModelQuota,
  formatQuotaAccountList,
  handleSystemOneRequest,
  healthOnlyQuotas,
  listQuotaAccounts,
  normalizeQuotaSnapshot,
  orderSeatsByQuotaEconomics,
  selectBestRoute,
} from "../src/index.js";
import { onRequest as hookOnRequest } from "../src/hooks.js";
import {
  onAccount as mwOnAccount,
  onProvider as mwOnProvider,
  onRequest as mwOnRequest,
  strategies,
} from "../src/middleware.js";
import { ACCOUNT_STATES, DEFAULT_USAGE_MAX_AGE_HOURS, normalizePlan } from "../src/quota.js";

const T0 = 1_700_000_000_000;
const HOUR = 3600 * 1000;
const iso = (ms) => new Date(ms).toISOString();
const win = (over = {}) => ({ name: "7d", remaining: 80, used: 20, resetHours: 100, ...over });
const acct = (over = {}) => ({ kind: "subscription", quota: { windows: [win()] }, ...over });
const ask = (rawQuotas, model = "gpt-5.3-codex", extra = {}) =>
  evaluateModelQuota(model, "high", { hours: 1 }, rawQuotas, {}, T0 + (extra.later ?? 0));

const cap = (provider, plan, windowName = "7d", hours) => {
  const snap = normalizeQuotaSnapshot(
    { [`${provider}/a`]: acct({ plan, quota: { windows: [win({ name: windowName, hours })] } }) },
    T0
  );
  return snap[0].windows[0].capacityUsd;
};

const implProfile = {
  work_kind: "general_implementation",
  reasoning_depth: 2,
  code_complexity: 2,
  context_scale: 2,
  divergence: 1,
  hallucination_risk: 2,
  error_cost: 2,
  needs_visual: false,
  needs_gui: false,
  local_only: true,
  realtime_loop: false,
  interaction_style: "engineering",
  expected_hours: 1,
  hours_confidence: "medium",
  urgency: "unknown",
  multi_task: false,
  task_count: "1",
};

test("Q4 capacity follows the plan, not one number per provider", () => {
  assert.equal(cap("codex", "plus"), 67);
  assert.equal(cap("codex", "prolite"), 175);
  assert.equal(cap("codex", "pro"), 325);
  assert.equal(cap("claude", "pro"), 156);
  assert.equal(cap("claude", "max5"), 1400);
  assert.equal(cap("claude", "max20"), 2380);
  assert.equal(cap("claude", "max20", "weekly_fable"), 1190);
});

test("Q4 plans match exactly: aliases map, look-alike names do not", () => {
  assert.equal(cap("codex", "Pro Lite"), 175);
  assert.equal(cap("claude", "max_20x"), 2380);
  // `prolite` contains `pro`, `max20-team` contains `max20`: neither may be read as the larger plan.
  assert.equal(normalizeQuotaSnapshot({ "codex/a": acct({ plan: "prolite-beta" }) }, T0)[0].planKnown, false);
  assert.equal(normalizeQuotaSnapshot({ "claude/a": acct({ plan: "max20-team" }) }, T0)[0].planKnown, false);
});

test("Q4 an unknown or unreported plan is sized as the smallest known plan, with a warning", () => {
  for (const plan of ["enterprise", "go", undefined, null]) {
    const [e] = normalizeQuotaSnapshot({ "codex/a": acct({ plan }) }, T0);
    assert.equal(e.windows[0].capacityUsd, 67, `plan=${plan}`);
    assert.equal(e.planKnown, false);
    assert.match(e.warnings.join("\n"), /codex\/a: plan .*smallest known codex plan/);
    assert.equal(e.state, ACCOUNT_STATES.READY, "an unknown plan is still dispatchable");
  }
  const [claude] = normalizeQuotaSnapshot({ "claude/a": acct({ plan: "team" }) }, T0);
  assert.equal(claude.windows[0].capacityUsd, 156);
  assert.equal(claude.windows[0].capacityUsd < 220, true);
});

test("Q4 an explicit capacity_usd still wins over the plan table", () => {
  const [e] = normalizeQuotaSnapshot(
    { "codex/a": acct({ plan: "plus", quota: { windows: [win({ capacity_usd: 999 })] } }) },
    T0
  );
  assert.equal(e.windows[0].capacityUsd, 999);
});

test("Q4 a Plus 5h window is priced with Plus's own size and gates at the small-window floor", () => {
  const five = (plan, remaining) =>
    ask({
      "codex/a": acct({
        plan,
        quota: { windows: [win(), win({ name: "five_hour", remaining, used: 100 - remaining, resetHours: 2 })] },
      }),
    });
  // 20% left of a $13 window is below the 25% small-window floor; the same 20% of a Pro window is not.
  assert.equal(five("plus", 20).eligible, false);
  assert.equal(five("pro", 20).eligible, true);
});

test("Q4 rates measured under another plan are not carried over after a tier change", () => {
  const w = win({ pct_per_hour_24h: 0.46, pct_per_hour_3h: 0 });
  const kept = normalizeQuotaSnapshot({ "codex/a": acct({ plan: "pro", rate_plan: "pro", quota: { windows: [w] } }) }, T0);
  const dropped = normalizeQuotaSnapshot({ "codex/a": acct({ plan: "plus", rate_plan: "pro", quota: { windows: [w] } }) }, T0);
  assert.equal(kept[0].windows[0].pctPerHour24h, 0.46);
  assert.equal(dropped[0].windows[0].pctPerHour24h, null);
});

test("Q2 a credential claim that disagrees with the online plan warns, and the online plan sizes capacity", () => {
  const [e] = normalizeQuotaSnapshot({ "codex/a": acct({ plan: "plus", plan_claim: "pro" }) }, T0);
  assert.equal(e.windows[0].capacityUsd, 67);
  assert.match(e.warnings.join("\n"), /plan sources disagree/);
});

for (const [label, item, state] of [
  ["status expired", { status: "expired" }, "plan_inactive"],
  ["status cancelled", { status: "cancelled" }, "plan_inactive"],
  ["free plan", { plan: "free" }, "plan_inactive"],
  ["payment failed plan", { plan: "payment_failed" }, "plan_inactive"],
  ["HTTP 401", { error: { status: 401, message: "x" } }, "auth_failed"],
  ["HTTP 403 field", { http_status: 403 }, "auth_failed"],
  ["disabled flag", { disabled: true }, "disabled"],
  ["error with no known last success", { error: "boom" }, "usage_failing"],
  ["status error", { status: "error" }, "usage_failing"],
]) {
  test(`Q3 ${label} is excluded from selection, with its state and reason`, () => {
    const q = { "codex/a": acct({ plan: "plus", ...item }) };
    const r = ask(q);
    assert.equal(r.eligible, false);
    assert.match(r.excluded.join("\n"), new RegExp(`codex/a: ${state}`));
    const [row] = listQuotaAccounts(q, T0);
    assert.equal(row.state, state);
    assert.equal(row.dispatchable, false);
    assert.ok(row.reason);
  });
}

test("Q3 an invalid account plus an exhausted one: nothing is selected, the invalid one is not rescued", () => {
  const q = {
    "codex/expired": acct({ plan: "plus", status: "expired", quota: { windows: [win({ remaining: 95, used: 5 })] } }),
    "codex/low": acct({ plan: "plus", quota: { windows: [win({ remaining: 3, used: 97 })] } }),
  };
  const r = ask(q);
  assert.equal(r.eligible, false, "the expired account must not be picked just because the other is exhausted");
});

test("Q3 a healthy account is selected over an invalid one with more quota", () => {
  const q = {
    "codex/expired": acct({ plan: "plus", status: "expired", quota: { windows: [win({ remaining: 99, used: 1 })] } }),
    "codex/ok": acct({ plan: "plus", quota: { windows: [win({ remaining: 40, used: 60 })] } }),
  };
  const r = ask(q);
  assert.equal(r.eligible, true);
  assert.equal(r.account, "codex/ok");
  assert.match(r.warnings.join("\n"), /codex\/expired: plan_inactive/, "the unselected failing account is still reported");
});

test("Q3 usage_failing warns at once, keeps dispatching for 6h, then excludes", () => {
  const lastOk = T0 - 2 * HOUR;
  const q = { "codex/a": acct({ plan: "plus", error: "upstream 500", last_success_at: iso(lastOk) }) };

  const soon = ask(q);
  assert.equal(soon.eligible, true);
  assert.match(soon.warnings.join("\n"), /codex\/a: usage_failing: usage reading failing: upstream 500 \(last success 2023-11-14T20:13:20\.000Z\)/);
  const [row] = listQuotaAccounts(q, T0);
  assert.equal(row.state, "usage_failing");
  assert.equal(row.dispatchable, true);
  assert.equal(row.last_success_at, iso(lastOk));

  assert.equal(DEFAULT_USAGE_MAX_AGE_HOURS, 6);
  const later = ask(q, "gpt-5.3-codex", { later: 5 * HOUR });
  assert.equal(later.eligible, false, "7h without a successful reading");
  assert.match(later.excluded.join("\n"), /usage_failing/);
});

test("Q3 the 6h limit is configurable", () => {
  const lastOk = T0 - 2 * HOUR;
  const base = { "codex/a": acct({ plan: "plus", error: "x", last_success_at: iso(lastOk) }) };
  assert.equal(ask({ ...base, usage_max_age_hours: 1 }).eligible, false);
  assert.equal(ask({ ...base, usage_max_age_hours: 12 }, "gpt-5.3-codex", { later: 5 * HOUR }).eligible, true);
  const viaPolicy = normalizeQuotaSnapshot(base, T0, { usage_max_age_hours: 1 });
  assert.equal(viaPolicy[0].excluded, true);
  const viaOpts = normalizeQuotaSnapshot(base, T0, {}, { usageMaxAgeHours: 1 });
  assert.equal(viaOpts[0].excluded, true);
});

test("Q3 a reading that is simply old counts as failing even without an error", () => {
  const q = { "codex/a": acct({ plan: "plus", fetched_at: iso(T0 - 10 * 24 * HOUR) }) };
  const r = ask(q);
  assert.equal(r.eligible, false);
  const [row] = listQuotaAccounts(q, T0);
  assert.equal(row.state, "usage_failing");
  assert.match(row.reason, /no successful reading for 240\.0h/);
});

test("Q3 an account with no usable reading at all is usage_failing, not silently ready", () => {
  const q = { "codex/a": { kind: "subscription", plan: "plus", quota: { windows: [] } } };
  const [row] = listQuotaAccounts(q, T0);
  assert.equal(row.state, "usage_failing");
  assert.equal(row.last_success_at, null);
  assert.equal(ask(q).eligible, false);
});

test("Q3 a recovered reading returns the account to ready by itself", () => {
  const failing = { "codex/a": acct({ plan: "plus", error: "x", last_success_at: iso(T0 - 8 * HOUR) }) };
  assert.equal(ask(failing).eligible, false);
  const recovered = { "codex/a": acct({ plan: "plus", fetched_at: iso(T0 - 60_000) }) };
  const r = ask(recovered);
  assert.equal(r.eligible, true);
  assert.equal(listQuotaAccounts(recovered, T0)[0].state, "ready");
  assert.deepEqual(r.warnings, []);
});

test("Q3 state is read from structured fields: error text alone never means auth_failed", () => {
  const q = { "codex/a": acct({ plan: "plus", error: "401 Unauthorized", last_success_at: iso(T0 - HOUR) }) };
  assert.equal(listQuotaAccounts(q, T0)[0].state, "usage_failing");
});

test("Q3 state names match AgentCompass", () => {
  assert.deepEqual(
    [ACCOUNT_STATES.READY, ACCOUNT_STATES.PLAN_INACTIVE, ACCOUNT_STATES.USAGE_FAILING, ACCOUNT_STATES.AUTH_FAILED],
    ["ready", "plan_inactive", "usage_failing", "auth_failed"]
  );
});

test("Q3 the account list shows state, reason and last successful reading, in data and text", () => {
  const q = {
    "codex/ok": acct({ plan: "plus", fetched_at: iso(T0 - 60_000) }),
    "claude/out": acct({ plan: "max5", error: "rate", last_success_at: iso(T0 - HOUR) }),
    "claude/dead": acct({ plan: "free" }),
  };
  const rows = listQuotaAccounts(q, T0);
  assert.deepEqual(
    rows.map((r) => [r.key, r.state, r.dispatchable]),
    [
      ["codex/ok", "ready", true],
      ["claude/out", "usage_failing", true],
      ["claude/dead", "plan_inactive", false],
    ]
  );
  const text = formatQuotaAccountList(rows);
  assert.match(text, /claude\/out .*state=usage_failing .*last_success=2023-11-14T21:13:20\.000Z - usage reading failing: rate/);
  assert.match(text, /claude\/dead .*state=plan_inactive \[excluded from dispatch\] .*last_success=never/);
});

test("Q3 plan() output carries the account list and its warnings, selected or not", () => {
  const quotas = {
    "claude/out": acct({ plan: "max5", error: "rate", last_success_at: iso(T0 - HOUR) }),
    "claude/ok": acct({ plan: "max5", fetched_at: iso(T0 - 60_000) }),
  };
  const d = selectBestRoute(implProfile, { candidateModels: ["claude-sonnet-5-5"], now: T0, quotas });
  assert.equal(d.ok, true);
  assert.equal(d.selectedAccount, "claude/ok");
  assert.deepEqual(d.accounts.map((a) => a.state), ["usage_failing", "ready"]);
  assert.ok(d.warnings.some((w) => w.startsWith("claude/out: usage_failing")));
});

test("Q3 a route whose only account is invalid is not rescued by the no-quota fallback", () => {
  const quotas = { "claude/dead": acct({ plan: "max5", status: "expired" }) };
  const primary = selectBestRoute(implProfile, { candidateModels: ["claude-sonnet-5-5"], now: T0, quotas });
  assert.equal(primary.ok, false);
  const fallback = selectBestRoute(implProfile, {
    candidateModels: ["claude-sonnet-5-5"],
    now: T0,
    quotas: healthOnlyQuotas(quotas, T0),
  });
  assert.equal(fallback.ok, false, "expired stays excluded in the fallback");
  assert.ok(fallback.warnings.some((w) => /claude\/dead: plan_inactive, excluded from dispatch/.test(w)));
});

test("Q3 the fallback still ignores exhausted windows on a healthy account", () => {
  const quotas = { "claude/low": acct({ plan: "max5", quota: { windows: [win({ remaining: 2, used: 98 })] } }) };
  const opts = { candidateModels: ["claude-sonnet-5-5"], now: T0 };
  assert.equal(selectBestRoute(implProfile, { ...opts, quotas }).ok, false);
  assert.equal(selectBestRoute(implProfile, { ...opts, quotas: healthOnlyQuotas(quotas, T0) }).ok, true);
});

test("Q3 the decision provider response lists accounts and warnings", async () => {
  const quotas = { "claude/dead": acct({ plan: "max5", status: "expired" }), "codex/ok": acct({ plan: "plus" }) };
  const out = await handleSystemOneRequest(
    {
      model: "agentcompass-systemone",
      state: { task: "Refactor the parser and add tests.", profile: implProfile, quotas },
      questions: { work_kind: { choices: ["general_implementation"] } },
    },
    { candidateModels: ["gpt-6.1-sol", "claude-sonnet-5-5"], decisionProviderDetails: true }
  );
  const ac = out.agentcompass;
  assert.deepEqual(ac.accounts.map((a) => [a.key, a.state]), [["claude/dead", "plan_inactive"], ["codex/ok", "ready"]]);
  assert.ok(ac.warnings.some((w) => w.startsWith("claude/dead: plan_inactive")));
  assert.equal(ac.selectedAccount === "claude/dead", false);
});

test("M1: orderSeatsByQuotaEconomics ranks excluded seats behind exhausted healthy seats, and usage_failing behind ready", () => {
  const seats = [
    {
      key: "claude/expired",
      account: "expired",
      state: "plan_inactive",
      excluded: true,
      status: "expired",
      plan: "free",
      known: true,
      used: 5,
      soon: [T0 + HOUR],
    },
    {
      key: "claude/exhausted",
      account: "exhausted",
      state: "ready",
      excluded: false,
      status: "ok",
      plan: "max20",
      known: true,
      used: 100,
      resting: true,
      soon: [T0 + HOUR],
    },
  ];
  const orderIdx = orderSeatsByQuotaEconomics("claude", seats, { now: T0 });
  assert.deepEqual(orderIdx, [1, 0]);
  assert.equal(seats[orderIdx[0]].key, "claude/exhausted");

  const nativeSeats = [
    { key: "claude/dead", account: "dead", known: true, learns: 1, used: 5, soon: [T0 + 24 * HOUR], pace: 0, weight: 1, resting: false, tokens: 0 },
    { key: "claude/exhausted", account: "exhausted", known: true, learns: 1, used: 100, soon: [T0 + 24 * HOUR], pace: 0, weight: 1, resting: true, tokens: 0 },
  ];
  const stratIdx = strategies.agentcompass("claude", nativeSeats, {
    now: T0,
    quotas: [
      { provider: "claude", account: "dead", plan: "free", updated_at: iso(T0 - 60_000), windows: [win({ remaining: 95 })] },
      { provider: "claude", account: "exhausted", plan: "max20", updated_at: iso(T0 - 60_000), windows: [win({ remaining: 0 })] },
    ],
  });
  assert.deepEqual(stratIdx, [1, 0]);

  const failingVsReady = [
    { key: "claude/failing", account: "failing", state: "usage_failing", excluded: false, known: true, used: 10, soon: [T0 + 12 * HOUR] },
    { key: "claude/ready", account: "ready", state: "ready", excluded: false, known: true, used: 40, soon: [T0 + 100 * HOUR] },
  ];
  assert.deepEqual(orderSeatsByQuotaEconomics("claude", failingVsReady, { now: T0 }), [1, 0]);
});

test("M2: hooks, middleware, and session continuation preserve warnings/accounts and block dispatch when accounts are excluded", async () => {
  const allInactiveQuotas = [
    { provider: "claude", account: "acc1", plan: "free", updated_at: iso(T0 - 60_000), windows: [win()] },
    { provider: "codex", account: "acc2", status: "auth_failed", updated_at: iso(T0 - 60_000), windows: [win()] },
  ];
  const body = { model: "agentcompass", profile: implProfile, messages: [{ role: "user", content: "Write a parser" }] };
  const ctx = {
    options: { quotas: allInactiveQuotas, candidateModels: ["claude-opus-5-5", "gpt-6-astra"], now: T0 },
    state: {},
  };
  await hookOnRequest(body, ctx);
  assert.equal(ctx.state.route.ok, false);
  assert.equal(ctx.state.route.targetModel, null);
  assert.equal(ctx.state.route.accounts.length, 2);
  assert.ok(ctx.state.route.warnings.some((w) => w.includes("claude/acc1: plan_inactive")));

  mwOnRequest({ ...body, profile: implProfile }, ctx);
  assert.equal(body.model, "agentcompass");
  assert.equal(ctx.state.route.ok, false);

  mwOnProvider(body, {
    provider: { id: "claude", name: "claude", model: "claude-opus-5-5" },
    options: ctx.options,
    state: ctx.state,
  });
  assert.equal(body.model, "agentcompass", "onProvider must not rewrite model when all accounts are excluded");
  assert.deepEqual(mwOnAccount({ provider: "claude", name: "acc1" }, ctx), { skip: true });

  // Session continuation hold preserves warnings & accounts and breaks hold if account becomes plan_inactive
  const store = new SessionContinuityStore();
  let activeQuotas = [
    { provider: "claude", account: "acc1", plan: "max20", error: "upstream timeout", last_success_at: iso(T0 - 60_000), windows: [win()] },
  ];
  const holdCtx1 = {
    options: { continuityStore: store, quotas: activeQuotas, candidateModels: ["claude-opus-5-5"], now: T0 },
    state: {},
  };
  await hookOnRequest(
    { model: "agentcompass", session_id: "s-hold", profile: implProfile, estimated_input_tokens: 2048, messages: [{ role: "user", content: "Implement feature A" }] },
    holdCtx1
  );
  assert.equal(holdCtx1.state.route.targetModel, "claude-opus-5-5");

  const holdCtx2 = {
    options: { continuityStore: store, quotas: activeQuotas, candidateModels: ["claude-opus-5-5"], now: T0 + 1000 },
    state: {},
  };
  await hookOnRequest(
    {
      model: "agentcompass",
      session_id: "s-hold",
      profile: implProfile,
      estimated_input_tokens: 2048,
      messages: [
        { role: "user", content: "Implement feature A" },
        { role: "assistant", content: "Part 1 done" },
        { role: "user", content: "继续" },
      ],
    },
    holdCtx2
  );
  assert.equal(holdCtx2.state.route.continuation.held, true);
  assert.ok(holdCtx2.state.route.warnings?.some((w) => w.includes("claude/acc1: usage_failing")));
  assert.equal(holdCtx2.state.route.accounts?.length, 1);

  const sysFollowupInvalid = await handleSystemOneRequest(
    {
      model: "agentcompass-systemone",
      state: {
        turn: "继续",
        profile: implProfile,
        previous_message_kind: "general_implementation",
        previous_message_reasoning: "high",
        quotas: [{ provider: "claude", account: "acc1", plan: "free", status: "expired", updated_at: iso(T0 - 60_000), windows: [win()] }],
      },
    },
    { candidateModels: ["claude-opus-5-5"], decisionProviderDetails: true, now: T0 }
  );
  assert.equal(sysFollowupInvalid.agentcompass.continuation.held, false);
  assert.equal(sysFollowupInvalid.agentcompass.targetModel, null);
  assert.ok(sysFollowupInvalid.agentcompass.warnings.some((w) => w.includes("claude/acc1: plan_inactive")));
});

test("M3b: healthOnlyQuotas preserves meteredOk and exhausted balance ($0) checks", async () => {
  const snapBalance = [
    { provider: "claude", account: "sub-exhausted", plan: "pro", updated_at: iso(T0 - 60_000), windows: [win({ remaining: 0, used: 100 })] },
    { provider: "codex", account: "zero-balance", kind: "balance", balance: { amount: 0, currency: "USD" }, updated_at: iso(T0 - 60_000) },
  ];
  const res = await handleSystemOneRequest(
    { model: "agentcompass", profile: implProfile, messages: [{ role: "user", content: "Hello" }] },
    { quotas: healthOnlyQuotas(snapBalance, T0), meteredOk: false, candidateModels: ["claude-sonnet-5-5", "gpt-6-astra"], decisionProviderDetails: true, now: T0 }
  );
  assert.equal(res.agentcompass.targetModel, "claude-sonnet-5-5");
  assert.equal(res.agentcompass.selectedAccount, "claude/sub-exhausted");
  assert.equal(res.agentcompass.quota.kind, "subscription");

  const bothBlocked = [
    { provider: "claude", account: "dead", plan: "free", status: "expired", updated_at: iso(T0 - 60_000), windows: [win()] },
    { provider: "codex", account: "zero-balance", kind: "balance", balance: { amount: 0, currency: "USD" }, updated_at: iso(T0 - 60_000) },
  ];
  const resBlocked = await handleSystemOneRequest(
    { model: "agentcompass", profile: implProfile, messages: [{ role: "user", content: "Hello" }] },
    { quotas: healthOnlyQuotas(bothBlocked, T0), meteredOk: false, candidateModels: ["claude-sonnet-5-5", "gpt-6-astra"], decisionProviderDetails: true, now: T0 }
  );
  assert.equal(resBlocked.agentcompass.targetModel, null);
});

test("M4: normalizePlan vendor-word stripping, HTTP 402 / INACTIVE_ERROR_CODES as plan_inactive, and failed poll timestamp discipline", () => {
  for (const [prov, label] of [
    ["antigravity", "Google AI Free"],
    ["codex", "ChatGPT Free"],
    ["claude", "Claude Free"],
    ["antigravity", "Antigravity Free Tier"],
  ]) {
    const p = normalizePlan(prov, label);
    assert.equal(p.inactive, true, `${prov}/${label} must be inactive`);
  }

  const rows = listQuotaAccounts(
    [
      { provider: "claude", account: "http402", plan: "pro", error: { status: 402, code: "payment_required" }, last_success_at: iso(T0 - 60_000), windows: [win()] },
      { provider: "claude", account: "http403_sub_exp", plan: "pro", error: { status: 403, code: "subscription_expired" }, last_success_at: iso(T0 - 60_000), windows: [win()] },
      { provider: "claude", account: "poll_reset", plan: "pro", error: "timeout", failing_since: iso(T0 - 10 * HOUR), fetched_at: iso(T0 - 60_000), windows: [win()] },
      { provider: "claude", account: "failing_3h", plan: "pro", error: "timeout", failing_since: iso(T0 - 3 * HOUR), windows: [] },
    ],
    T0
  );
  assert.equal(rows[0].state, "plan_inactive");
  assert.equal(rows[0].dispatchable, false);
  assert.equal(rows[1].state, "plan_inactive");
  assert.equal(rows[1].dispatchable, false);
  assert.equal(rows[2].state, "usage_failing");
  assert.equal(rows[2].last_success_at, null, "failed poll fetched_at must not overwrite last_success_at");
  assert.equal(rows[2].dispatchable, false, "10h failing_since must exclude account");
  assert.equal(rows[3].state, "usage_failing");
  assert.equal(rows[3].dispatchable, true, "3h failing_since without last_success_at is still within 6h grace");
});

test("M5: Codex pro and pro100 treat five_hour as observation-only (block only at <= 0%, no 10% floor, no opportunity cost)", () => {
  const m5a = evaluateModelQuota(
    "gpt-5.4",
    "high",
    { hours: 0.5 },
    {
      "codex/pro-acct": {
        plan: "pro",
        windows: [
          { name: "five_hour", remaining: 8, resetsInHours: 2 },
          { name: "seven_day", remaining: 80, resetsInHours: 100 },
        ],
      },
    },
    {},
    T0
  );
  assert.equal(m5a.eligible, true);
  assert.deepEqual(m5a.excluded, []);

  const m5Exhausted = evaluateModelQuota(
    "gpt-5.4",
    "high",
    { hours: 0.5 },
    {
      "codex/pro-acct": {
        plan: "pro",
        windows: [
          { name: "five_hour", remaining: 0, resetsInHours: 2 },
          { name: "seven_day", remaining: 80, resetsInHours: 100 },
        ],
      },
    },
    {},
    T0
  );
  assert.equal(m5Exhausted.eligible, false);

  const m5b = evaluateModelQuota(
    "gpt-5.4",
    "high",
    { hours: 0.5 },
    {
      "codex/pro-5h-only": {
        plan: "pro",
        windows: [{ name: "five_hour", remaining: 50, resetsInHours: 2 }],
      },
    },
    {},
    T0
  );
  assert.equal(m5b.eligible, true);
  assert.equal(m5b.opportunity_cost, 0);
  assert.equal(m5b.reset_bonus, 0);
});

test("R2-M1: 503 and onAccount global skip only trigger on account-blocked candidates; routeAllModels ability-floor pass-through and modelsByProvider two-stage routing work", async () => {
  const hardMsg = [{ role: "user", content: "Audit the cross-module state machine for deadlocks and prove liveness invariants." }];
  const hardProfile = {
    ...implProfile,
    reasoning_depth: 3,
    code_complexity: 3,
    context_scale: 3,
    hallucination_risk: 3,
    error_cost: 3,
  };

  for (const quotas of [
    undefined,
    [{ provider: "claude", account: "healthy-acc", plan: "max20", updated_at: iso(T0 - 60_000), windows: [win({ remaining: 90 })] }],
  ]) {
    let hookRej = null;
    const hookCtx = {
      protocol: "chat",
      state: {},
      options: { routeAllModels: true, injectProtocolEffort: true, quotas, now: T0 },
      reject(code, msg) {
        hookRej = { code, msg };
      },
    };
    await hookOnRequest({ model: "claude-haiku-5-5", profile: hardProfile, messages: hardMsg }, hookCtx);
    assert.equal(hookRej, null);

    let mwRej = null;
    const mwCtx = {
      protocol: "chat",
      state: {},
      options: { routeAllModels: true, injectProtocolEffort: true, quotas, now: T0 },
      reject(code, msg) {
        mwRej = { code, msg };
      },
    };
    mwOnRequest({ model: "claude-haiku-5-5", profile: hardProfile, messages: hardMsg }, mwCtx);
    assert.equal(mwRej, null);
    assert.equal(mwOnAccount({ provider: "claude", name: "healthy-acc" }, mwCtx), undefined);
  }

  // Two-stage defaultModel + modelsByProvider when defaultModel is below floor
  const req1 = { model: "agentcompass/auto", profile: hardProfile, messages: hardMsg };
  let rej1 = null;
  const state1 = {};
  const opts1 = {
    defaultModel: "claude-haiku-5-5",
    injectProtocolEffort: true,
    modelsByProvider: { anthropic: ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"] },
    now: T0,
  };
  mwOnRequest(req1, {
    protocol: "chat",
    state: state1,
    options: opts1,
    reject(code, msg) {
      rej1 = { code, msg };
    },
  });
  assert.equal(rej1, null);
  const provRes1 = mwOnProvider(req1, {
    protocol: "chat",
    state: state1,
    provider: { id: "anthropic", name: "Anthropic", account: "work", model: req1.model },
    options: opts1,
  });
  assert.ok(provRes1);
  assert.equal(state1.route.ok, true);
  assert.match(req1.model, /^claude-(sonnet|opus)-5-5:/);

  // Two-stage defaultModel (claude inactive) + modelsByProvider (openai healthy)
  const req2 = { model: "agentcompass/auto", profile: implProfile, messages: [{ role: "user", content: "Write a parser" }] };
  let rej2 = null;
  const state2 = {};
  const opts2 = {
    defaultModel: "claude-sonnet-5-5",
    injectProtocolEffort: true,
    quotas: [
      { provider: "claude", account: "dead", plan: "free", status: "expired", updated_at: iso(T0 - 60_000), windows: [win()] },
      { provider: "codex", account: "good", plan: "pro", status: "ok", updated_at: iso(T0 - 60_000), windows: [win()] },
    ],
    modelsByProvider: {
      anthropic: ["claude-sonnet-5-5"],
      openai: ["gpt-6-astra"],
    },
    now: T0,
  };
  mwOnRequest(req2, {
    protocol: "chat",
    state: state2,
    options: opts2,
    reject(code, msg) {
      rej2 = { code, msg };
    },
  });
  assert.equal(rej2, null);
  const provRes2 = mwOnProvider(req2, {
    protocol: "chat",
    state: state2,
    provider: { id: "openai", name: "OpenAI", account: "good", model: "gpt-6-astra" },
    options: opts2,
  });
  assert.ok(provRes2);
  assert.equal(state2.route.ok, true);
  assert.match(req2.model, /^gpt-6-astra:/);
  assert.equal(mwOnAccount({ provider: "codex", name: "good" }, { state: state2, options: opts2 }), undefined);
});

test("R2-M2: cross-provider identical account names validate provider in onAccount and matchSeatQuotaEntry", () => {
  const mixedQuotas = [
    { provider: "claude", account: "acct1", plan: "free", status: "expired", updated_at: iso(T0 - 60_000), windows: [win({ remaining: 95 })] },
    { provider: "codex", account: "acct1", plan: "pro", status: "ok", updated_at: iso(T0 - 60_000), windows: [win({ remaining: 90 })] },
    { provider: "codex", account: "exhausted", plan: "pro", status: "ok", updated_at: iso(T0 - 60_000), windows: [win({ remaining: 0 })] },
  ];
  const body = { model: "agentcompass", profile: implProfile, messages: [{ role: "user", content: "Write a parser in TypeScript" }] };
  const ctx = {
    options: {
      candidateModels: ["claude-sonnet-5-5", "gpt-6.1-sol"],
      quotas: mixedQuotas,
      now: T0,
    },
    state: {},
  };
  mwOnRequest(body, ctx);
  assert.equal(ctx.state.route.selectedAccount, "codex/acct1");
  assert.deepEqual(mwOnAccount({ provider: "claude", name: "acct1" }, ctx), { skip: true });
  assert.equal(mwOnAccount({ provider: "codex", name: "acct1" }, ctx), undefined);

  const codexSeats = [
    { key: "codex/acct1", account: "acct1", known: true, learns: 1, used: 10, soon: [T0 + 24 * HOUR], pace: 0, weight: 1, resting: false, tokens: 0 },
    { key: "codex/exhausted", account: "exhausted", known: true, learns: 1, used: 100, soon: [T0 + 24 * HOUR], pace: 0, weight: 1, resting: true, tokens: 0 },
  ];
  const orderCodex = strategies.agentcompass("codex", codexSeats, { now: T0, quotas: mixedQuotas });
  assert.deepEqual(orderCodex, [0, 1]);
  assert.equal(codexSeats[orderCodex[0]].key, "codex/acct1");
});

test("R2-M3: default catalog candidates are not pruned to quota providers when candidateModels is omitted", () => {
  const onlyClaudeHealthy = [{ provider: "claude", account: "c1", plan: "max20", updated_at: iso(T0 - 60_000), windows: [win()] }];
  const rImpl = selectBestRoute(implProfile, { quotas: onlyClaudeHealthy, now: T0 });
  assert.equal(rImpl.ok, true);
  assert.equal(rImpl.targetModel, "gpt-6.1-sol");

  const rPin = selectBestRoute(
    { ...implProfile, realtime_loop: true },
    { quotas: onlyClaudeHealthy, pinnedModel: "gemini-4-argon-high", now: T0 }
  );
  assert.equal(rPin.ok, true);
  assert.equal(rPin.targetModel, "gemini-4-argon-high");

  const onlyClaudeInactive = [{ provider: "claude", account: "c1", plan: "free", status: "expired", updated_at: iso(T0 - 60_000), windows: [win()] }];
  const rInactive = selectBestRoute(implProfile, { quotas: onlyClaudeInactive, now: T0 });
  assert.equal(rInactive.ok, true);
  assert.equal(rInactive.targetModel, "gpt-6.1-sol");
});

test("tier multipliers match before pool suffix and free ×N is plan_inactive (AgentCompass fix/plan-and-usage-state)", () => {
  for (const [prov, label, expectedPlan, expectedWeekly] of [
    ["antigravity", "Ultra x5", "ultra5", 156],
    ["antigravity", "ultra ×5", "ultra5", 156],
    ["antigravity", "Google AI Ultra x5", "ultra5", 156],
    ["claude", "Max x20", "max20", 2380],
    ["claude", "Claude Max x20", "max20", 2380],
    ["claude", "Claude Max ×20", "max20", 2380],
    ["claude", "Max x5", "max5", 1400],
    ["claude", "Max 20x ×2", "max20", 2380],
    ["codex", "Pro ×2", "pro", 325],
  ]) {
    const p = normalizePlan(prov, label);
    assert.equal(p.known, true, `${prov}/${label} must be known`);
    assert.equal(p.inactive, false, `${prov}/${label} must be active`);
    assert.equal(p.key, expectedPlan, `${prov}/${label} key`);
    assert.equal(cap(prov, label), expectedWeekly, `${prov}/${label} weekly capacity`);
  }

  for (const prov of ["claude", "codex", "antigravity"]) {
    const freePool = normalizePlan(prov, "free ×2");
    assert.equal(freePool.inactive, true, `${prov}/free ×2 must be inactive`);
    const [row] = listQuotaAccounts(
      [{ provider: prov, account: "pooled-free", plan: "free ×2", updated_at: iso(T0 - 60_000), windows: [win()] }],
      T0
    );
    assert.equal(row.state, "plan_inactive");
    assert.equal(row.dispatchable, false, `${prov}/free ×2 must not be dispatchable`);
  }

  assert.equal(normalizePlan("claude", "Max 5x x2").known, false);
});
