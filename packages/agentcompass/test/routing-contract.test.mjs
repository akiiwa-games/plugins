/**
 * One regression test per finding of the routing review (contract R1-R6). Modules are imported as namespaces so that, run
 * against the code before the fix, a missing export fails its own test and not the whole file.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import * as allocator from "../src/allocator.js";
import * as catalog from "../src/catalog.js";
import * as continuity from "../src/continuity.js";
import * as rawHooks from "../src/hooks.js";
import * as rawIndex from "../src/index.js";
import * as rawMiddleware from "../src/middleware.js";
import * as profiler from "../src/profiler-core.js";
import * as quota from "../src/quota.js";
import * as scorer from "../src/scorer-live.js";
import { answerRequest, answerSystemOne, classifierAnswer } from "./classifier-stub.mjs";

const hooks = {
  ...rawHooks,
  onRequest: async (body, ctx = {}) => {
    answerRequest(body, ctx);
    return rawHooks.onRequest(body, ctx);
  },
};
const middleware = {
  ...rawMiddleware,
  onRequest: (body, ctx = {}) => {
    answerRequest(body, ctx);
    return rawMiddleware.onRequest(body, ctx);
  },
};
const index = {
  ...rawIndex,
  handleSystemOneRequest: async (body, options = {}) => rawIndex.handleSystemOneRequest(answerSystemOne(body, options), options),
  systemOneFetch: async (reqInput, init, options = {}) => {
    if (typeof init?.body === "string") {
      try {
        init = { ...init, body: JSON.stringify(answerSystemOne(JSON.parse(init.body), options)) };
      } catch {}
    }
    return rawIndex.systemOneFetch(reqInput, init, options);
  },
};

const { rankCandidates, selectBestRoute } = allocator;
const { SessionContinuityStore } = continuity;
const NOW = 1790000000000;
const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const PROFILE = {
  multi_task: false,
  task_count: "1",
  work_kind: "general_implementation",
  reasoning_depth: 2,
  code_complexity: 2,
  context_scale: 2,
  divergence: 1,
  hallucination_risk: 2,
  error_cost: 2,
  needs_visual: false,
  needs_gui: false,
  local_only: false,
  interaction_style: "engineering",
};
const REALTIME = { ...PROFILE, realtime_loop: true };
const ROUTINE = { ...PROFILE, reasoning_depth: 1, code_complexity: 0, hallucination_risk: 1, error_cost: 1 };

const TASK = "Implement the parser module and its unit tests";
const request = (extra = {}) => ({ model: "agentcompass/auto", messages: [{ role: "user", content: TASK }], ...extra });
const context = (options, extra = {}) => ({
  state: {},
  reject: (status, message) => ({ rejected: status, message }),
  options: { continuityStore: new SessionContinuityStore(), now: NOW, ...options },
  ...extra,
});
const candidate = (ranked, id) => ranked.allCandidates.find((c) => c.id === id);
const SYSTEM_ONE_QUESTIONS = {
  intent: { type: "choice", criteria: { general_implementation: "implementation", bounded_bulk_work: "routine" } },
  effort: { type: "score", criteria: ["low", "medium", "high", "xhigh"] },
};
const systemOne = (state, options) =>
  index.handleSystemOneRequest({ model: "jev-6b-int8", state: { message: TASK, ...state }, questions: SYSTEM_ONE_QUESTIONS }, options);

/** An external classifier answering `answer`, counting its calls. */
function classifier(answer) {
  const calls = { count: 0 };
  const reply = JSON.stringify({ choices: [{ message: { content: JSON.stringify(answer) } }] });
  const options = {
    scorerType: "openai_compatible",
    allowExternalScorer: true,
    scorerBaseUrl: "http://127.0.0.1:9/v1",
    scorerApiKey: "k",
    scorerModel: "m",
    fetch: async () => {
      calls.count += 1;
      return { ok: true, status: 200, text: async () => reply, json: async () => JSON.parse(reply) };
    },
  };
  return { options, calls };
}

// ---------------------------------------------------------------------------- R1: one parser

test("M-1.1 unhealthy models match on the parsed key, and on the raw name for a family or unknown one", () => {
  const dated = rankCandidates(PROFILE, { candidateModels: ["gpt-6.1-sol-20261001:high"], unhealthyModels: ["gpt-6.1-sol"] });
  assert.equal(dated.allCandidates[0].eligible, false);
  const named = rankCandidates(PROFILE, { candidateModels: ["gpt-6.1-sol:high"], unhealthyModels: ["gpt-6.1-sol-20261001"] });
  assert.equal(named.allCandidates[0].eligible, false);
  for (const raw of ["gemini-2.5-pro", "totally-unknown-model"]) {
    const ranked = rankCandidates(PROFILE, { candidateModels: [`${raw}:high`], unhealthyModels: [raw] });
    assert.equal(ranked.allCandidates[0].eligible, false, raw);
  }
});

test("M-1.2 the continuity bonus reaches a dated or preview spelling of the previous model", () => {
  for (const [model, previousModel] of [
    ["claude-opus-5-5-20261001", "claude-opus-5-5-20261001"],
    ["claude-opus-5-5-20261001", "claude-opus-5-5"],
    ["claude-opus-5-5", "claude-opus-5-5-latest"],
  ]) {
    const ranked = rankCandidates(PROFILE, {
      candidateModels: [`${model}:high`],
      continuity: { previousModel, previousEffort: "high", cacheWarm: true, why: "cold" },
    });
    assert.ok(ranked.allCandidates[0].components.continuity > 0, `${model} after ${previousModel}`);
  }
});

test("M-1.3 tier, curves and speed read one parse: own row for an exact or alias name, the shrunk target with default speed for a family", () => {
  for (const name of ["claude-opus-4-6", "claude-opus-5-5-20261001", "gemini-4-argon", "minimax-m4", "o4-mini-2025-04-16", "glm-6-flash"]) {
    assert.equal(catalog.resolveModelSpec(name).canonical, catalog.resolveResponsivenessModelKey(name).key, name);
  }
  assert.notEqual(catalog.resolveModelSpec("minimax-m4").spec.tier, catalog.DEFAULT_MODELS["gpt-6.1-sol"].tier);
  assert.equal(catalog.resolveModelSpec("claude-opus-4-6").canonical, "claude-opus-4-6");
  assert.notEqual(catalog.DEFAULT_MODELS["claude-opus-4-6"].tier, catalog.DEFAULT_MODELS["claude-opus-5-5"].tier);

  const { source, key } = catalog.resolveModel("gemini-2.5-pro");
  assert.equal(source, "family");
  const target = catalog.DEFAULT_MODELS[key];
  assert.ok(Math.abs(catalog.resolveModelSpec("gemini-2.5-pro").spec.tier - (target.tier - 0.3)) < 1e-9);
  const speed = allocator.bundleResponsiveness({ model: "gemini-2.5-pro", effort: "high" });
  assert.equal(speed.value, 0.52);
  assert.equal(speed.source, "default");
  // A `:effort` in the name constrains the effort, it is not part of the key
  assert.equal(catalog.resolveModel("gpt-6-astra:high").key, "gpt-6-astra");
  assert.equal(catalog.resolveModel("gpt-6-astra:high").effort, "high");
});

test("M-1.4 quota: o4 is an OpenAI family, a window binds the raw name, and a flat curve is read as flat", () => {
  const window = { id: "five_hour", used: 50, reset_at: "2030-01-01T00:00:00Z" };
  const codex = { "codex/main": { kind: "subscription", quota: { windows: [window] } } };
  assert.equal(quota.providerMatchesModel("codex", "o4-mini-2025-04-16"), true);
  assert.notEqual(quota.evaluateModelQuota("o4-mini-2025-04-16", "high", { difficulty: "hard", hours: 1 }, codex, {}, NOW), null);

  const bound = { "claude/main": { kind: "subscription", quota: { windows: [{ id: "weekly", used: 99.9, reset_at: "2030-01-01T00:00:00Z", models: ["claude-opus-4-6"] }] } } };
  assert.equal(quota.evaluateModelQuota("claude-opus-4-6", "high", { difficulty: "hard", hours: 1 }, bound, {}, NOW).eligible, false);
  assert.equal(quota.windowBindsModel("weekly", "claude-opus-4-6", ["claude-opus-4-6"], "anthropic/claude-opus-4-6"), true);
  assert.equal(quota.windowBindsModel("weekly", "claude-opus-5-5", ["claude-opus-4-6"], "claude-opus-5-5"), false);

  assert.deepEqual(catalog.EFFORT_CURVES["glm-5.3"].burn, { default: catalog.EFFORT_CURVES["glm-5.3"].burn.default, high: catalog.EFFORT_CURVES["glm-5.3"].burn.high });
});

test("B-1.5 extractModelList keeps object entries and never takes an id for a model name", () => {
  const entry = { id: "seat_a", provider: "codex", account: "codex", model: "gpt-6.1-sol", efforts: ["medium"] };
  assert.deepEqual(hooks.extractModelList([entry]), [entry]);

  const body = request();
  middleware.onRequest(body, context({ candidateModels: [entry] }));
  assert.equal(body.model, "gpt-6.1-sol:medium");

  const ctx = context({ candidateModels: [entry] });
  const viaHooks = request();
  return hooks.onRequest(viaHooks, ctx).then(() => {
    assert.equal(ctx.state.route.targetModel, "gpt-6.1-sol");
    assert.equal(ctx.state.route.targetEffort, "medium");
  });
});

// ---------------------------------------------------------------------------- R2: the efforts a candidate may use

test("B-2.1 `model:effort` in a pool is a fixed effort: too slow is excluded, not stepped down", () => {
  const ranked = rankCandidates(REALTIME, { candidateModels: ["gpt-6-astra:high"] });
  const astra = ranked.allCandidates[0];
  assert.equal(astra.effort, "high");
  assert.equal(astra.realtime_stepped_from, undefined);
  assert.equal(astra.eligible, false);
  assert.match(astra.excluded.join("; "), /gpt-6-astra@high: responsiveness 0\.00 below the realtime_loop floor 0\.5/);
  assert.equal(selectBestRoute(REALTIME, { candidateModels: ["gpt-6-astra:high"] }).targetModel ?? null, null);
});

test("B-2.2 step-down lands only on what the candidate or pool offers, and each effort of an entry has its own id", () => {
  const listed = rankCandidates(REALTIME, { candidateModels: [{ provider: "codex", account: "codex", model: "gpt-6-astra", efforts: ["high"] }] });
  assert.equal(listed.allCandidates[0].effort, "high");
  assert.equal(listed.allCandidates[0].eligible, false);
  assert.ok(listed.allCandidates[0].excluded.length > 0);

  const pool = rankCandidates(REALTIME, { candidateModels: ["gpt-6-astra:low", "gpt-6-astra:high"] });
  assert.equal(pool.selected.effort, "low");
  assert.equal(candidate(pool, "gpt-6-astra:high").effort, "high");

  const split = rankCandidates(REALTIME, { candidateModels: [{ id: "a", provider: "codex", account: "codex", model: "gpt-6-astra", efforts: ["medium", "high"] }] });
  assert.equal(new Set(split.allCandidates.map((c) => c.id)).size, 2);
  assert.equal(split.selected.effort, "medium");

  // Disabled on the same account and model: not a landing
  const opus = { provider: "claude", account: "claude", model: "claude-opus-5-5" };
  const disabled = rankCandidates(REALTIME, { candidateModels: [{ id: "h", ...opus }, { id: "m", ...opus, effort: "medium", disabled: true }] });
  assert.notEqual(candidate(disabled, "h").effort, "medium");
  // On another harness it still is
  const elsewhere = rankCandidates(REALTIME, { candidateModels: [{ id: "h", ...opus, harness: "claude" }, { id: "m", ...opus, harness: "other", effort: "medium", disabled: true }] });
  assert.equal(candidate(elsewhere, "h").effort, "medium");

  // A second account of a provider reads that provider's table; `pi` has one of its own
  const second = rankCandidates(REALTIME, { candidateModels: [{ id: "k", provider: "claude-2", account: "claude-2", model: "claude-haiku-5-5" }] });
  assert.equal(candidate(second, "k").effort, "high", "claude dispatches haiku at high only");
  assert.equal(candidate(second, "k").eligible, false);
  assert.ok(catalog.DISPATCH_EFFORTS.pi);
  const onPi = rankCandidates(REALTIME, { candidateModels: [{ id: "p", provider: "pi", account: "pi", model: "gpt-6-astra" }] });
  assert.ok(catalog.DISPATCH_EFFORTS.pi["gpt-6-astra"].includes(candidate(onPi, "p").effort));
});

// ---------------------------------------------------------------------------- R3: what the user pins

test("B-3.1 a pinned unhealthy or spent model is blocked with the reason, never rerun without the filters", async (t) => {
  t.mock.method(console, "warn", () => {});
  const options = { candidateModels: ["gpt-6-astra", "gpt-6.1-sol"], pinnedModel: "gpt-6-astra", unhealthyModels: ["gpt-6-astra"], realtime_loop: true };

  const decision = selectBestRoute(REALTIME, options);
  assert.equal(decision.blocked, true);
  assert.equal(decision.targetModel ?? null, null);
  assert.ok(decision.warnings.length > 0);

  const viaHooks = context(options);
  await hooks.onRequest(request(), viaHooks);
  assert.equal(viaHooks.state.route, undefined);
  assert.ok(viaHooks.state.routeBlocked.warnings.length > 0);

  const body = request();
  const refused = middleware.onRequest(body, context(options));
  assert.equal(refused.rejected, 503);
  assert.notEqual(body.model, "gpt-6-astra:high");

  const answer = await systemOne({ realtime_loop: true }, { ...options });
  assert.equal(answer.agentcompass.blocked, true);
  assert.equal(answer.agentcompass.targetModel, null);

  // Not pinned either: an ordinary fallback keeps the filters too
  const spent = selectBestRoute(PROFILE, { candidateModels: ["gpt-6-astra"], unhealthyModels: ["gpt-6-astra"] });
  assert.equal(spent.targetModel ?? null, null);
  const small = selectBestRoute(PROFILE, { candidateModels: ["gpt-6-luna"], minContextTokens: 1e12 });
  assert.equal(small.targetModel ?? null, null);
  const unpinned = request();
  assert.equal(middleware.onRequest(unpinned, context({ candidateModels: ["gpt-6-astra"], unhealthyModels: ["gpt-6-astra"] })).rejected, 503);
});

test("B-3.1 a concrete model under routeAllModels, with nothing pinned and no route, goes as asked", () => {
  const body = request({ model: "gpt-6-astra" });
  const result = middleware.onRequest(body, context({ routeAllModels: true, candidateModels: ["gpt-6-astra"], unhealthyModels: ["gpt-6-astra"] }));
  assert.equal(result?.rejected, undefined);
  assert.equal(body.model, "gpt-6-astra");
});

test("M-3.2 a changed pin breaks a warm hold and routes to the new pin", async () => {
  const candidateModels = ["gpt-6-astra", "gpt-6.1-sol"];
  const meta = { metadata: { session_id: "pin-change" } };
  const messages = [{ role: "user", content: TASK }, { role: "assistant", content: "ok" }, { role: "user", content: "Now write the docs for it as well" }];
  for (const run of ["hooks", "middleware"]) {
    const continuityStore = new SessionContinuityStore();
    const first = context({ candidateModels, continuityStore, pinnedModel: "gpt-6.1-sol" });
    await hooks.onRequest(request(meta), first);
    assert.equal(first.state.route.targetModel, "gpt-6.1-sol");
    continuityStore.recordResponse(first.state.sessionKey, { usage: { input_tokens: 2000, output_tokens: 300 } }, {}, NOW);

    const second = context({ candidateModels, continuityStore, pinnedModel: "gpt-6-astra", now: NOW + 1000 });
    if (run === "hooks") await hooks.onRequest({ ...request(meta), messages }, second);
    else middleware.onRequest({ ...request(meta), messages }, second);
    assert.equal(second.state.route.targetModel, "gpt-6-astra", run);
    assert.equal(second.state.route.continuation.why, "user_pin_override", run);
    assert.equal(second.state.route.continuation.held, false, run);
  }
});

test("M-3.3 a pin in the request body or System-One state reaches the allocator, in any mode, and stays out of the upstream body", async () => {
  const candidateModels = ["gpt-6-astra", "gpt-6.1-sol", "gpt-6-luna"];
  for (const pins of [{ pinned_model: "gpt-6.1-sol" }, { constraints: { model: "gpt-6.1-sol" } }, { task: { pinned_model: "gpt-6.1-sol" } }]) {
    const body = request(pins);
    middleware.onRequest(body, context({ candidateModels }));
    assert.match(body.model, /^gpt-6\.1-sol:/, JSON.stringify(pins));
    assert.equal(body.pinned_model, undefined);

    const ctx = context({ candidateModels });
    await hooks.onRequest(request(pins), ctx);
    assert.equal(ctx.state.route.targetModel, "gpt-6.1-sol", JSON.stringify(pins));
  }
  const effort = request({ pinned_model: "gpt-6-astra:low" });
  middleware.onRequest(effort, context({ candidateModels: ["gpt-6-astra"] }));
  assert.equal(effort.model, "gpt-6-astra:low");

  const answer = await systemOne({ pinned_model: "gpt-6.1-sol" }, { candidateModels });
  assert.equal(answer.agentcompass.targetModel, "gpt-6.1-sol");
});

test("M-3.4 a pinned disabled_by_default model is allowed on the default catalog and not routed unpinned; xhigh and max get no exemption", () => {
  for (const model of ["gpt-5.3-codex-spark", "claude-fable-5-1"]) {
    assert.equal(catalog.DEFAULT_MODELS[model].disabled_by_default, true, model);
    assert.equal(selectBestRoute(ROUTINE, { constraintModel: model }).targetModel, model);
  }
  for (const profile of [ROUTINE, PROFILE]) {
    assert.ok(!["gpt-5.3-codex-spark", "claude-fable-5-1"].includes(selectBestRoute(profile).targetModel));
  }

  const astra = (efforts) => ({ constraintModel: "gpt-6-astra", candidateModels: [{ provider: "codex", account: "codex", model: "gpt-6-astra", efforts }] });
  const reaches = selectBestRoute(REALTIME, astra(["xhigh", "medium"]));
  assert.equal(reaches.targetEffort, "medium");
  const none = selectBestRoute(REALTIME, astra(["xhigh", "max"]));
  assert.equal(none.targetEffort, "xhigh", "the fastest of what it offers");
  assert.ok(none.warnings.some((w) => w.includes("realtime_loop")));
  // Model and effort pinned: kept, and only warned about
  const kept = selectBestRoute(REALTIME, { ...astra(["xhigh", "max"]), constraintEffort: "max" });
  assert.equal(kept.targetEffort, "max");
  assert.ok(kept.warnings.length > 0);
  // The final bundle keeps what it was given
  const bundle = { id: "b", provider: "codex", account: "codex", model: "gpt-6-astra", effort: "high", modeId: "full-access", draws: ["x"] };
  const routed = selectBestRoute(PROFILE, { candidateModels: [bundle], pinnedBundle: "b" });
  assert.equal(routed.bundle.effort, "high");
  assert.equal(routed.bundle.modeId, "full-access");
  assert.deepEqual(routed.bundle.draws, ["x"]);
});

test("a pinned family name stays itself: it never folds to the flagship, and with no such row it is blocked", () => {
  const argon = { id: "ag", provider: "antigravity", account: "agy", harness: "antigravity", model: "gemini-4-argon-high", effort: "high" };
  const blocked = selectBestRoute(PROFILE, { constraintModel: "gemini-2.5-pro", candidateModels: [argon] });
  assert.equal(blocked.blocked, true);
  assert.equal(blocked.targetModel ?? null, null);
  assert.ok(blocked.warnings.length > 0);
  const own = { id: "own", provider: "openrouter", account: "or", model: "gemini-2.5-pro", effort: "high" };
  assert.equal(selectBestRoute(PROFILE, { constraintModel: "gemini-2.5-pro", candidateModels: [argon, own] }).targetModel, "gemini-2.5-pro");
});

test("a `-effort` suffix is an effort, in a pool and in a pin", () => {
  const ranked = rankCandidates(REALTIME, { candidateModels: ["gpt-6-astra-high"] });
  assert.equal(ranked.allCandidates[0].effort, "high", "fixed, not stepped");
  assert.equal(ranked.allCandidates[0].eligible, false);
  assert.equal(selectBestRoute(PROFILE, { candidateModels: ["gpt-6-astra-medium"] }).targetEffort, "medium");
  const pinned = selectBestRoute(PROFILE, { constraintModel: "gpt-6-astra-low", candidateModels: ["gpt-6.1-sol", "gpt-6-astra"] });
  assert.equal(pinned.targetModel, "gpt-6-astra");
  assert.equal(pinned.targetEffort, "low");
  // The effort's own field wins over one in the name
  const named = { id: "a", provider: "codex", account: "codex", model: "gpt-6-astra", effort: "high" };
  const both = selectBestRoute(PROFILE, { constraintModel: "gpt-6-astra-medium", constraintEffort: "high", candidateModels: [named] });
  assert.equal(both.targetEffort, "high");
  // A model whose own name ends like an effort stays that model
  assert.equal(catalog.resolveModel("qwen3.8-max").effort, null);
});

test("a provider whose table does not list the model offers no step-down", () => {
  const row = { id: "x", provider: "codex", account: "codex", model: "claude-opus-5-5", effort: "high" };
  const ranked = rankCandidates(REALTIME, { candidateModels: [row] });
  assert.equal(candidate(ranked, "x").effort, "high");
  assert.equal(candidate(ranked, "x").realtime_stepped_from, undefined);
  // A gateway has no table at all: the curve decides
  const gateway = rankCandidates(REALTIME, { candidateModels: [{ ...row, provider: "openrouter", account: "openrouter" }] });
  assert.equal(candidate(gateway, "x").realtime_stepped_from, "high");
});

test("an effort the pool has no row for can be pinned: a row `<carrier>@<effort>` with pin_carrier, on what the provider dispatches", () => {
  const row = { id: "opus", provider: "claude", account: "claude", model: "claude-opus-5-5", effort: "high", modeId: "bypass" };
  const decision = selectBestRoute(PROFILE, { candidateModels: [row], constraintModel: "claude-opus-5-5", constraintEffort: "medium" });
  assert.equal(decision.targetEffort, "medium");
  assert.equal(decision.bundle.pin_carrier, "opus");
  assert.equal(decision.bundle.modeId, "bypass");
  const ranked = rankCandidates(PROFILE, { candidateModels: [row], constraintModel: "claude-opus-5-5", constraintEffort: "medium" });
  assert.ok(ranked.allCandidates.some((c) => c.id === "opus@medium"));
  // Haiku dispatches at high only
  const haiku = { id: "h", provider: "claude", account: "claude", model: "claude-haiku-5-5", effort: "high" };
  assert.equal(selectBestRoute(PROFILE, { candidateModels: [haiku], constraintModel: "claude-haiku-5-5", constraintEffort: "medium" }).blocked, true);
  // A fixed row carries nothing else
  assert.equal(selectBestRoute(PROFILE, { candidateModels: ["claude-opus-5-5:high"], constraintModel: "claude-opus-5-5", constraintEffort: "medium" }).blocked, true);
});

test("a family name scores as auto_bundles says: no curve of its own, the estimated price, the target's route place", () => {
  const family = { id: "f", provider: "claude", account: "claude", model: "claude-opus-9", effort: "low" };
  assert.equal(candidate(rankCandidates(PROFILE, { candidateModels: [family] }), "f").components.ability_value, 3.76);
  assert.equal(catalog.resolveModelSpec("claude-opus-9").spec.usd_per_hour, 3);
  assert.equal(allocator.fitScore("computer_use", "claude-opus-9"), allocator.fitScore("computer_use", "claude-opus-5-5"));
  const gui = { ...REALTIME, needs_gui: true, needs_visual: true, work_kind: "frontend_implementation" };
  const guiFit = (model) => candidate(rankCandidates(gui, { candidateModels: [{ ...family, model, effort: "high" }] }), "f").components.gui_fit;
  assert.equal(guiFit("claude-opus-9"), guiFit("claude-opus-5-5"));
});

test("an unknown name scores on the generic row, whatever its size looks like", () => {
  assert.equal(catalog.resolveModel("totally-unknown-mini").source, "unknown");
  assert.equal(catalog.resolveModelSpec("totally-unknown-mini").spec.tier, catalog.resolveModelSpec("totally-unknown-model").spec.tier);
});

test("dot_text marked unavailable takes dot off the route even with a dot row in the pool", () => {
  const research = { ...PROFILE, work_kind: "web_research" };
  const pool = ["dot", "claude-sonnet-5-5"];
  const fit = (options) => rankCandidates(research, { candidateModels: pool, ...options }).allCandidates.find((c) => c.model === "claude-sonnet-5-5").components.route_fit;
  assert.ok(fit({}) < 1);
  assert.equal(fit({ unavailable: ["dot_text"] }), 1);
});

const turnOne = (text = "Implement the parser module and tests") => ({ model: "agentcompass/auto", estimated_input_tokens: 2048, messages: [{ role: "user", content: text }] });
const followUp = (first) => ({ model: "agentcompass/auto", estimated_input_tokens: 2048, messages: [...first.messages, { role: "assistant", content: "ok" }, { role: "user", content: "继续" }] });
const quietly = (fn) => {
  const warn = console.warn;
  const logged = [];
  console.warn = (line) => logged.push(line);
  try {
    return { value: fn(), logged };
  } finally {
    console.warn = warn;
  }
};
/** Two turns of one session through the middleware; returns what each sent and how the second was decided. */
function twoTurns(optionsOne, optionsTwo) {
  const store = new SessionContinuityStore();
  const first = turnOne();
  const ctxOne = { sessionId: "s", options: { continuityStore: store, now: NOW, ...optionsOne }, state: {} };
  quietly(() => middleware.onRequest(first, ctxOne));
  const second = followUp(first);
  const rejected = [];
  const ctxTwo = { sessionId: "s", options: { continuityStore: store, now: NOW + 1000, ...optionsTwo }, state: {}, reject: (status, message) => rejected.push([status, message]) };
  quietly(() => middleware.onRequest(second, ctxTwo));
  return { first: first.model, second: second.model, route: ctxTwo.state.route, rejected, store };
}

test("a pin changed during a hold takes effect however it is spelled (review M1)", () => {
  for (const pin of ["gpt-6-astra-medium", "gpt-6-astra:medium", "openai/gpt-6-astra-medium"]) {
    const run = twoTurns({ candidateModels: ["gpt-6-astra"], pinnedEffort: "high" }, { candidateModels: ["gpt-6-astra"], constraintModel: pin });
    assert.equal(run.first, "gpt-6-astra:high", pin);
    assert.equal(run.second, "gpt-6-astra:medium", pin);
    assert.equal(run.route.continuation.held, false, pin);
    assert.equal(run.route.continuation.why, "user_pin_override", pin);
  }
  // A family name never passes for the flagship the session is on...
  const pool = ["claude-opus-5-5", "claude-opus-9"];
  const family = twoTurns({ candidateModels: ["claude-opus-5-5"] }, { candidateModels: pool, constraintModel: "claude-opus-9" });
  assert.equal(family.route.targetModel, "claude-opus-9");
  assert.equal(family.route.continuation.held, false);
  // ...with no row of that name the route says so rather than keep the old model...
  const noRow = twoTurns({ candidateModels: ["claude-opus-5-5"] }, { candidateModels: ["claude-opus-5-5"], constraintModel: "claude-opus-9" });
  assert.equal(noRow.rejected[0]?.[0], 503);
  assert.match(noRow.rejected[0][1], /claude-opus-9/);
  // ...and a session already on that name stays on it.
  const stay = twoTurns({ candidateModels: pool, constraintModel: "claude-opus-9" }, { candidateModels: pool, constraintModel: "claude-opus-9" });
  assert.equal(stay.route.targetModel, "claude-opus-9");
  assert.equal(stay.route.continuation.held, true);
});

test("a hold steps down to what the pool still offers, and never sends the effort it took away (review M2)", () => {
  const sol = "gpt-6.1-sol";
  for (const [name, one, two, effort] of [
    ["`model:effort`", [`${sol}:high`], [`${sol}:medium`], "medium"],
    ["efforts[]", [{ model: sol, efforts: ["high"] }], [{ model: sol, efforts: ["medium"] }], "medium"],
    ["another model in the pool", [`${sol}:high`], [`${sol}:medium`, "claude-sonnet-5-5:high"], "medium"],
  ]) {
    const run = twoTurns({ candidateModels: one }, { candidateModels: two });
    assert.equal(run.first, `${sol}:high`, name);
    assert.equal(run.second, `${sol}:${effort}`, name);
    assert.equal(run.route.continuation.held, true, name);
    assert.match(run.route.warnings[0], /no longer offered/, name);
  }
  // The dispatch table counts too: Haiku dispatches at high only, so nothing is below it and the hold breaks
  const haiku = { id: "h", provider: "claude", account: "claude", model: "claude-haiku-5-5" };
  const noLower = twoTurns({ candidateModels: [`${sol}:high`] }, { candidateModels: [`${sol}:xhigh`, haiku] });
  assert.equal(noLower.route.continuation.held, false);
  assert.equal(noLower.route.continuation.why, "effort_gone");
  assert.notEqual(noLower.second, `${sol}:high`);
  // Unchecking the effort is taking it away
  const disabled = twoTurns(
    { candidateModels: [{ id: "a", provider: "codex", account: "codex", model: sol, effort: "high" }] },
    { candidateModels: [{ id: "a", provider: "codex", account: "codex", model: sol, effort: "high", disabled: true }, { id: "b", provider: "codex", account: "codex", model: sol, effort: "medium" }] },
  );
  assert.notEqual(disabled.second, `${sol}:high`);
  // A pin names the effort, so nothing may step it: the route says what is wrong
  const pinned = twoTurns({ candidateModels: [`${sol}:high`], pinnedEffort: "high" }, { candidateModels: [`${sol}:medium`], pinnedEffort: "high" });
  assert.equal(pinned.route?.targetEffort === "high", false);
  assert.equal(pinned.rejected[0]?.[0], 503);
  assert.equal(allocator.heldEffortOn([`${sol}:medium`], sol, "high"), "medium");
  assert.equal(allocator.heldEffortOn([`${sol}:xhigh`], sol, "high"), null);
  assert.equal(allocator.candidateOffers([{ model: sol, effort: "high" }], sol, "low"), true, "one bundle at its scoring effort steps within dispatch");
  assert.equal(allocator.candidateOffers(["gpt-6-astra-medium"], "gpt-6-astra-medium", "high"), false);
});

test("the hooks hold as the middleware does, on the name routed and what the pool offers (review M1, M2)", async () => {
  const run = async (one, two, extra = {}) => {
    const store = new SessionContinuityStore();
    const first = turnOne();
    await hooks.onRequest(first, { protocol: "chat", sessionId: "h", options: { candidateModels: one, continuityStore: store, now: NOW }, state: {} });
    const ctx = { protocol: "chat", sessionId: "h", options: { candidateModels: two, continuityStore: store, now: NOW + 1000, ...extra }, state: {} };
    await hooks.onRequest(followUp(first), ctx);
    return ctx.state.route;
  };
  const stepped = await run(["gpt-6.1-sol:high"], ["gpt-6.1-sol:medium"]);
  assert.equal(stepped.continuation.held, true);
  assert.equal(stepped.targetEffort, "medium");
  assert.match(stepped.warnings[0], /no longer offered/);
  const gone = await run(["gpt-6.1-sol:high"], ["gpt-6.1-sol:xhigh", "claude-sonnet-5-5"]);
  assert.equal(gone.continuation.held, false);
  assert.equal(gone.continuation.why, "effort_gone");
  assert.notEqual(gone.targetEffort, "high");
  const pinned = await run(["gpt-6-astra"], ["gpt-6-astra"], { constraintModel: "gpt-6-astra-medium" });
  assert.equal(pinned.continuation.why, "user_pin_override");
  assert.equal(pinned.targetEffort, "medium");
  const row = { id: "x", provider: "codex", account: "codex", model: "gpt-6-astra", effort: "medium", modeId: "m", draws: 2 };
  const held = await run([row], [row]);
  assert.equal(held.continuation.held, true);
  assert.equal(held.bundle.modeId, "m", "review m7");
  assert.equal(held.bundle.draws, 2, "review m7");
});

test("a pool name with no row of its own says so, in the route and in the log (review M4)", () => {
  const ok = selectBestRoute(PROFILE, { candidateModels: ["claude-sonnet-5-5", "gpt-5.4", "gemini-2.5-pro"] });
  assert.equal(ok.ok, true);
  assert.equal(ok.warnings.length, 2);
  assert.match(ok.warnings[0], /gpt-5\.4 is not in the catalog: scored on the generic row \(tier 2\.8\); use gpt-6-astra/);
  assert.match(ok.warnings[1], /gemini-2\.5-pro is not in the catalog: scored as an estimate of gemini-4-argon-high \(tier 3\.7\); use gemini-4-argon-high/);
  assert.deepEqual(selectBestRoute(PROFILE, { candidateModels: ["claude-sonnet-5-5", "claude-opus-4-6"] }).warnings, []);

  const rejected = [];
  const refuse = (pool, content, extra = {}) =>
    quietly(() =>
      middleware.onRequest(
        { model: "agentcompass/auto", messages: [{ role: "user", content }] },
        { options: { candidateModels: pool, continuityStore: new SessionContinuityStore(), now: NOW, ...extra }, state: {}, reject: (s, m) => rejected.push([s, m]) },
      ),
    );
  const { logged } = refuse(["gpt-5.4"], "Prove the correctness of this distributed consensus algorithm and redesign the architecture across all modules");
  assert.deepEqual(rejected[0], [503, "agentcompass: no_route: no candidate can serve this request"]);
  assert.match(logged.join("\n"), /no candidate is available: gpt-5\.4@low: ability/);
  assert.match(logged.join("\n"), /candidate name gpt-5\.4 is not in the catalog/);
  refuse(["gpt-5.4"], "hello", { constraintModel: "claude-opus-9" });
  assert.deepEqual(rejected[1], [503, "agentcompass: pin_not_in_pool: pinned model claude-opus-9 is not among the candidates"]);

  // The log carries a name warning once per runtime, a route's other warnings every time
  const name = `gpt-5.4-${NOW}-unlisted`;
  const once = quietly(() => {
    allocator.logRouteWarnings([`candidate name ${name} is not in the catalog`, "slow"]);
    allocator.logRouteWarnings([`candidate name ${name} is not in the catalog`, "slow"]);
  });
  assert.equal(once.logged.filter((line) => line.includes(name)).length, 1);
  assert.equal(once.logged.filter((line) => line.endsWith("slow")).length, 2);

  // The changelog lists every retired name with its replacement
  const names = Object.values(catalog.RETIRED_NAMES).flat();
  assert.equal(names.length, 57);
  const notes = fs.readFileSync(path.join(REPO, "CHANGELOG.md"), "utf8");
  for (const [target, retired] of Object.entries(catalog.RETIRED_NAMES)) {
    assert.ok(catalog.DEFAULT_MODELS[target], target);
    for (const old of retired) {
      assert.ok(notes.includes(`\`${old}\``), old);
      if (catalog.resolveModel(old).source === "exact") continue;
      assert.match(catalog.catalogNameWarning(old), new RegExp(`use ${target}$`), old);
    }
  }
});

test("a failure of a family name stays on that name; the flagship keeps serving (review M5)", () => {
  const store = new SessionContinuityStore();
  const pool = ["gpt-6.2-sol-preview", "gpt-6.1-sol"];
  const first = turnOne();
  const ctx = { sessionId: "f", options: { candidateModels: ["gpt-6.2-sol-preview"], continuityStore: store, now: NOW }, state: {} };
  quietly(() => middleware.onRequest(first, ctx));
  assert.equal(ctx.state.route.targetModel, "gpt-6.2-sol-preview");
  middleware.onResponse({ error: { message: "bad gateway" } }, { ...ctx, status: 502 });
  assert.deepEqual(store.getUnhealthyModels(NOW), ["gpt-6.2-sol-preview"]);
  assert.equal(store.isModelHealthy("gpt-6.1-sol", NOW), true);
  assert.equal(store.isModelHealthy("gpt-6.2-sol-preview", NOW), false);

  // The next turn leaves the failed name and lands on the flagship, in this session and in a new one
  const next = followUp(first);
  const ctxNext = { sessionId: "f", options: { candidateModels: pool, continuityStore: store, now: NOW + 1000 }, state: {} };
  quietly(() => middleware.onRequest(next, ctxNext));
  assert.equal(ctxNext.state.route.targetModel, "gpt-6.1-sol");
  assert.equal(ctxNext.state.route.continuation.why, "provider_unhealthy");
  const fresh = selectBestRoute(PROFILE, { candidateModels: ["gpt-6.1-sol"], unhealthyModels: store.getUnhealthyModels(NOW) });
  assert.equal(fresh.targetModel, "gpt-6.1-sol");
  const allocated = rankCandidates(PROFILE, { candidateModels: ["gpt-6.1-sol:high"], unhealthyModels: ["gpt-6.2-sol-preview"] });
  assert.equal(allocated.allCandidates[0].eligible, true);

  // An exact or alias name is its row: any spelling takes the row down
  const exact = new SessionContinuityStore();
  exact.markModelUnhealthy("gpt-6.1-sol-20261001", "http_502", NOW);
  assert.deepEqual(exact.getUnhealthyModels(NOW), ["gpt-6.1-sol"]);
  assert.equal(rankCandidates(PROFILE, { candidateModels: ["gpt-6.1-sol:high"], unhealthyModels: exact.getUnhealthyModels(NOW) }).allCandidates[0].eligible, false);
});

test("a pin that names a fixed-effort id makes no row of another effort (review m2)", () => {
  const argon = { id: "a", provider: "antigravity", account: "agy", harness: "antigravity", model: "gemini-4-argon-high", effort: "high" };
  assert.equal(selectBestRoute(PROFILE, { candidateModels: [argon], constraintModel: "gemini-4-argon-high", constraintEffort: "medium" }).blocked, true);
  assert.equal(catalog.nameFixesEffort("gemini-4-argon-high"), true);
  assert.equal(catalog.nameFixesEffort("gpt-6.1-sol"), false);
  assert.equal(catalog.nameFixesEffort("qwen3.8-max"), false, "-max is a size unless the base is a row");
  assert.equal(catalog.nameFixesEffort("gpt-6.1-sol-max"), true);
});

test("scores round as Python rounds them (review m3)", () => {
  assert.equal(catalog.pyRound(0.7625, 3), 0.762);
  assert.equal(allocator.round3(0.7625), 0.762);
  assert.equal(catalog.pyRound(0.0625, 3), 0.062, "an exact tie goes to the even digit");
  assert.equal(catalog.pyRound(0.1875, 3), 0.188, "an exact tie goes to the even digit");
  assert.equal(catalog.pyRound(2.5, 0), 2);
  assert.equal(catalog.pyRound(0.12345, 4), 0.1235);
  assert.equal(catalog.pyRound(-0.7625, 3), -0.762);
  assert.ok(Object.is(catalog.pyRound(-0.00001, 3), 0));
});

test("`:auto` and `:off` constrain nothing; `-off` is not an effort (review m4)", () => {
  assert.deepEqual(catalog.splitEffortSuffix("gpt-6.1-sol:auto"), { name: "gpt-6.1-sol:auto", effort: null });
  assert.equal(catalog.resolveModel("gpt-6.1-sol:auto").key, "gpt-6.1-sol");
  assert.equal(catalog.resolveModel("gpt-6.1-sol:off").effort, null);
  assert.equal(catalog.resolveModel("gpt-6.1-sol-off").source, "family");
  const pool = selectBestRoute(PROFILE, { candidateModels: ["gpt-6.1-sol:auto"] });
  assert.equal(pool.targetModel, "gpt-6.1-sol");
  assert.doesNotMatch(pool.modelWithEffort, /auto/);
  const pin = selectBestRoute(PROFILE, { candidateModels: ["gpt-6.1-sol"], constraintModel: "gpt-6.1-sol:auto" });
  assert.equal(pin.ok, true);
  assert.equal(pin.targetModel, "gpt-6.1-sol");
});

test("task-level multi_task and task_count are checked as stated (review m5)", () => {
  for (const bad of [{ multi_task: "false" }, { task_count: 0 }, { task_count: NaN }, { task_count: "" }, { multi_task: true, task_count: "1" }, { multi_task: false, task_count: "3" }]) {
    const decision = selectBestRoute({ objective: "Fix a bug", ...bad }, { candidateModels: ["gpt-6.1-sol"] });
    assert.equal(decision.blocked, true, JSON.stringify(bad));
    assert.equal(decision.error, "invalid_caller_profile", JSON.stringify(bad));
  }
  assert.equal(profiler.validateProfile({ ...PROFILE, task_count: 3, multi_task: true }).multi_task, true);
  assert.equal(profiler.validateProfile({ ...PROFILE, multi_task: false, task_count: "1" }).task_count, "1");
  assert.equal(profiler.validateProfile({ ...PROFILE, multi_task: true, task_count: 2 }).task_count, "2");
});

test("a request this plugin does not route keeps its pin fields from the upstream too (review m6)", () => {
  const body = { model: "gpt-6-luna", pinned_model: "x", constraints: { model: "x" }, constraint_effort: "low", messages: [{ role: "user", content: "hi" }] };
  assert.equal(middleware.onRequest(body, { options: {}, state: {} }), body);
  for (const field of ["pinned_model", "constraints", "constraint_effort"]) assert.equal(field in body, false, field);
});

test("a held turn carries the pool's bundle, with its modeId and draws (review m7)", () => {
  const row = { id: "x", provider: "codex", account: "codex", model: "gpt-6-astra", effort: "medium", modeId: "m", draws: 2 };
  const run = twoTurns({ candidateModels: [row] }, { candidateModels: [row] });
  assert.equal(run.route.continuation.held, true);
  assert.equal(run.route.bundle.modeId, "m");
  assert.equal(run.route.bundle.draws, 2);
  assert.equal(run.route.bundle.effort, run.route.targetEffort);
});

test("a route its `when` rules block says why (review m8)", () => {
  const decision = selectBestRoute(PROFILE, { candidateModels: [{ id: "w", provider: "codex", model: "gpt-6.1-sol", effort: "high", when: { max_reasoning_depth: 1 } }] });
  assert.equal(decision.blocked, true);
  assert.match(decision.warnings[0], /gpt-6\.1-sol@high: its when rule allows reasoning depth up to 1, the task has 2/);
  assert.match(decision.excluded[0], /when rule/);
});

test("onProvider cannot turn a request away: nothing usable on this provider warns, and the request goes as routed (review 2 M2)", () => {
  const rejected = [];
  const body = turnOne();
  const ctx = {
    sessionId: "p",
    options: { candidateModels: ["gpt-6.1-sol"], continuityStore: new SessionContinuityStore(), now: NOW, pinnedModel: "gpt-6.1-sol", modelsByProvider: { other: ["claude-sonnet-5-5"] } },
    state: {},
    provider: { id: "other" },
    reject: (status, message) => rejected.push([status, message]),
  };
  quietly(() => middleware.onRequest(body, ctx));
  const sent = body.model;
  const { value, logged } = quietly(() => middleware.onProvider(body, ctx));
  assert.equal(value, undefined);
  assert.equal(body.model, sent);
  assert.deepEqual(rejected, []);
  assert.match(logged.join("\n"), /pinned model gpt-6\.1-sol is not among the candidates/);
});

test("a pin names its own model's rows: a family row is another model (contract R1, X1)", async () => {
  const flagship = "claude-opus-5-5";
  const family = "claude-opus-9";
  for (const pool of [[flagship, family], [family, flagship]]) {
    for (const profile of [PROFILE, ROUTINE]) {
      assert.equal(selectBestRoute(profile, { candidateModels: pool, constraintModel: flagship }).targetModel, flagship);
      assert.equal(selectBestRoute(profile, { candidateModels: pool, constraintModel: `${flagship}:high` }).modelWithEffort, `${flagship}:high`);
      assert.equal(selectBestRoute(profile, { candidateModels: pool, constraintModel: `${flagship}-20261001` }).targetModel, flagship);
      assert.equal(selectBestRoute(profile, { candidateModels: pool, constraintModel: family }).targetModel, family);
    }
  }
  // Alone in the pool, a family row does not answer to the flagship, and no row of it is made up
  const alone = selectBestRoute(PROFILE, { candidateModels: [family], constraintModel: flagship });
  assert.equal(alone.blocked, true);
  assert.equal(alone.refusal, `pin_not_in_pool: pinned model ${flagship} is not among the candidates`);
  assert.equal(selectBestRoute(PROFILE, { candidateModels: [family], constraintModel: `${flagship}:medium` }).blocked, true);
  // An alias spelling of the row still is the row
  assert.equal(selectBestRoute(PROFILE, { candidateModels: [`openai/${flagship}`], constraintModel: flagship }).ok, true);
  // A session on the family row moves to the pinned flagship, in both hosts
  const pool = [flagship, family];
  const moved = twoTurns({ candidateModels: [family, flagship], constraintModel: family }, { candidateModels: pool, constraintModel: flagship });
  assert.match(moved.first, /^claude-opus-9:/);
  assert.match(moved.second, /^claude-opus-5-5:/);
  assert.equal(moved.route.continuation.why, "user_pin_override");
  const store = new SessionContinuityStore();
  const first = turnOne();
  await hooks.onRequest(first, { protocol: "chat", sessionId: "x1", options: { candidateModels: pool, constraintModel: family, continuityStore: store, now: NOW }, state: {} });
  const ctx = { protocol: "chat", sessionId: "x1", options: { candidateModels: pool, constraintModel: flagship, continuityStore: store, now: NOW + 1000 }, state: {} };
  await hooks.onRequest(followUp(first), ctx);
  assert.equal(ctx.state.route.targetModel, flagship);
  assert.equal(ctx.state.route.continuation.why, "user_pin_override");
});

test("a pinned model that is down is refused, with no family row standing in for it (review 2 M1)", async () => {
  const flagship = "claude-opus-5-5";
  const pool = [flagship, "claude-opus-9"];
  const options = { candidateModels: pool, unhealthyModels: [flagship] };

  for (const pin of [flagship, `${flagship}:low`, `${flagship}-20261001`, `openai/${flagship}`]) {
    const decision = selectBestRoute(PROFILE, { ...options, constraintModel: pin });
    assert.equal(decision.blocked, true, pin);
    assert.match(decision.warnings[0], /is unavailable: model is currently marked unhealthy/, pin);
  }

  const rejected = [];
  const body = turnOne();
  quietly(() => middleware.onRequest(body, { options: { ...options, constraintModel: flagship, continuityStore: new SessionContinuityStore(), now: NOW }, state: {}, reject: (s, m) => rejected.push([s, m]) }));
  assert.deepEqual(rejected, [[503, `agentcompass: model_unhealthy: pinned model ${flagship} is marked unhealthy`]]);
  assert.equal(body.model, "agentcompass/auto");

  // The hooks hand the same refusal to the middleware
  const hooked = { protocol: "chat", options: { ...options, constraintModel: flagship, continuityStore: new SessionContinuityStore(), now: NOW }, state: {}, reject: (s, m) => [s, m] };
  await hooks.onRequest(turnOne(), hooked);
  assert.deepEqual(quietly(() => middleware.onRequest(turnOne(), hooked)).value, [503, `agentcompass: model_unhealthy: pinned model ${flagship} is marked unhealthy`]);

  // A session on the flagship, or on a family row with the pin arriving now, does not hold through it
  for (const one of [{ candidateModels: pool, constraintModel: flagship }, { candidateModels: ["claude-opus-9"] }]) {
    const held = twoTurns(one, { ...options, constraintModel: flagship });
    assert.equal(held.rejected[0]?.[0], 503);
    assert.match(held.rejected[0][1], /^agentcompass: model_unhealthy: /);
  }

  // With no pin the family row serves (review M5), and a pin on the family name itself is that name's to answer
  assert.equal(selectBestRoute(PROFILE, options).targetModel, "claude-opus-9");
  assert.equal(selectBestRoute(PROFILE, { ...options, constraintModel: "claude-opus-9" }).targetModel, "claude-opus-9");
  const familyDown = selectBestRoute(PROFILE, { candidateModels: pool, unhealthyModels: ["claude-opus-9"], constraintModel: "claude-opus-9" });
  assert.equal(familyDown.blocked, true);
});

test("a pinned effort off a bare name's table holds like any other, with no extra classifier call (review 2 r1)", async () => {
  const model = "grok-4.7";
  const pin = { candidateModels: [model], constraintModel: `${model}:medium` };
  assert.equal(allocator.candidateOffers([model], model, "medium"), false);
  assert.equal(allocator.candidateOffers([model], model, "medium", undefined, { constraintModel: `${model}:medium` }), true);
  // A pool entry that fixes another effort is still not offering it (review M2)
  assert.equal(allocator.candidateOffers([`${model}:high`], model, "medium", undefined, { constraintModel: `${model}:medium` }), false);

  const run = twoTurns(pin, pin);
  assert.equal(run.second, `${model}:medium`);
  assert.equal(run.route.continuation.held, true);

  let calls = 0;
  const fetch = async () => {
    calls++;
    const content = JSON.stringify({ ...PROFILE, multi_task: false, task_count: "1" });
    return { ok: true, status: 200, json: async () => ({ choices: [{ message: { content } }] }) };
  };
  const store = new SessionContinuityStore();
  let history = turnOne();
  for (let turn = 0; turn < 3; turn++) {
    const body = { ...history };
    const ctx = {
      protocol: "chat",
      sessionId: "r1",
      options: { ...pin, continuityStore: store, now: NOW + turn, allowExternalScorer: true, scorerType: "openai_compatible", scorerBaseUrl: "http://127.0.0.1:9/v1", scorerApiKey: "fake", scorerModel: "fake", useCache: false, fetch },
      state: {},
      reject: (status, message) => assert.fail(`${status} ${message}`),
    };
    await hooks.onRequest(body, ctx);
    quietly(() => middleware.onRequest(body, ctx));
    assert.equal(body.model, `${model}:medium`);
    if (turn > 0) assert.equal(ctx.state.route.continuation.held, true);
    history = followUp(history);
  }
  assert.equal(calls, 1);
});

test("a hold steps down no lower than a new route would go (review 2 r2)", async () => {
  const sol = "gpt-6.1-sol";
  // This task's floor is above gpt-6.1-sol at low: a new route refuses it, so the hold does too
  const fresh = selectBestRoute(classifierAnswer({ objective: "Implement the parser module and tests" }), { candidateModels: [`${sol}:low`] });
  assert.equal(fresh.blocked, true);
  const middlewareRun = twoTurns({ candidateModels: [`${sol}:high`] }, { candidateModels: [`${sol}:low`] });
  assert.equal(middlewareRun.first, `${sol}:high`);
  assert.equal(middlewareRun.second, followUp(turnOne()).model);
  assert.equal(middlewareRun.rejected[0]?.[0], 503);

  const store = new SessionContinuityStore();
  const first = turnOne();
  await hooks.onRequest(first, { protocol: "chat", sessionId: "h2", options: { candidateModels: [`${sol}:high`], continuityStore: store, now: NOW }, state: {} });
  const ctx = { protocol: "chat", sessionId: "h2", options: { candidateModels: [`${sol}:low`], continuityStore: store, now: NOW + 1000 }, state: {} };
  await hooks.onRequest(followUp(first), ctx);
  assert.equal(ctx.state.route, undefined);
  assert.equal(ctx.state.routeBlocked.error, "no_route");

  const easy = { work_kind: "tool_execution", difficulty: "routine" };
  assert.equal(allocator.heldEffortOn([`${sol}:low`], sol, "high", undefined, { profile: PROFILE }), null);
  assert.equal(allocator.heldEffortOn([`${sol}:medium`], sol, "high", undefined, { profile: PROFILE }), "medium");
  assert.equal(allocator.heldEffortOn([`${sol}:low`], sol, "high", undefined, { profile: easy }), "low");
  assert.equal(allocator.heldEffortOn([`${sol}:low`], sol, "high"), "low", "with no task to judge, what the pool offers");
});

test("the name-warning log is bounded, and names stay on one line and short (review 2 r3, r4)", () => {
  const warning = (i) => `candidate name lru-${NOW}-${i} is not in the catalog`;
  const count = (fn) => quietly(fn).logged.length;
  assert.equal(count(() => allocator.logRouteWarnings([warning("a")])), 1);
  for (let i = 0; i < 1023; i++) quietly(() => allocator.logRouteWarnings([warning(i)]));
  assert.equal(count(() => allocator.logRouteWarnings([warning("a")])), 0, "seen, and now the most recent");
  quietly(() => allocator.logRouteWarnings([warning("new")]));
  assert.equal(count(() => allocator.logRouteWarnings([warning(0)])), 1, "the least recently seen went first");
  for (let i = 1; i <= 1024; i++) quietly(() => allocator.logRouteWarnings([warning(`more-${i}`)]));
  assert.equal(count(() => allocator.logRouteWarnings([warning("a")])), 1, "at 1024 names a new one pushes the oldest out");

  assert.equal(catalog.safeName("a\nb\u2028c"), "a\\u000ab\\u2028c");
  assert.equal(catalog.safeName("x".repeat(2000)).length, 131);
  assert.equal(catalog.safeName("gpt-6.1-sol"), "gpt-6.1-sol");

  const forged = "zz\n2026/11/07 00:00:00 FAKE LOG LINE";
  const rejected = [];
  const body = { ...turnOne(), pinned_model: forged };
  const { logged } = quietly(() =>
    middleware.onRequest(body, { options: { candidateModels: [forged, "y".repeat(5000)], continuityStore: new SessionContinuityStore(), now: NOW }, state: {}, reject: (s, m) => rejected.push([s, m]) }),
  );
  assert.ok(logged.length > 0);
  for (const line of logged) {
    assert.doesNotMatch(line, /[\r\n]/);
    assert.ok(line.length < 16500, String(line.length));
  }
  assert.doesNotMatch(logged.join(""), /y{129}/);
  assert.doesNotMatch(String(rejected[0]?.[1]), /[\r\n]/);
  assert.equal(rejected[0]?.[0], 503);

  const longReason = quietly(() => allocator.logRouteWarnings([`${"z".repeat(4000)} of\nnote`])).logged[0];
  assert.doesNotMatch(longReason, /z{129}|\n/);
});

test("a refusal tells the client what it pinned and a code; the pool and quotas stay in the log (review 2 r5)", () => {
  const refusal = (pool, extra) => selectBestRoute(PROFILE, { candidateModels: pool, ...extra }).refusal;
  assert.equal(refusal(["gpt-6-luna"], { unhealthyModels: ["gpt-6-luna"] }), "no_route: no candidate can serve this request");
  assert.equal(refusal(["gpt-6-luna", "gpt-6-astra"], { constraintModel: "gpt-6.1-sol" }), "pin_not_in_pool: pinned model gpt-6.1-sol is not among the candidates");
  assert.equal(refusal(["gpt-6.1-sol"], { constraintModel: "gpt-6.1-sol", unhealthyModels: ["gpt-6.1-sol"] }), "model_unhealthy: pinned model gpt-6.1-sol is marked unhealthy");
  assert.equal(refusal(["gpt-6.1-sol:high"], { constraintModel: "gpt-6.1-sol:low" }), "pin_unavailable: pinned model gpt-6.1-sol:low cannot serve this request");
  assert.equal(refusal(["gpt-6.1-sol"]), null);
  assert.equal(selectBestRoute({ objective: "Fix a bug", multi_task: "false" }, { candidateModels: ["gpt-6.1-sol"] }).refusal, "invalid_caller_profile: the task profile is invalid");
  // The detail is still in the decision, for the local log
  assert.match(selectBestRoute(PROFILE, { candidateModels: ["gpt-6-luna"], unhealthyModels: ["gpt-6-luna"] }).warnings[0], /gpt-6-luna@\w+: model is currently marked unhealthy/);
  assert.equal(allocator.blockedRefusal({ error: "classifier_invalid_output" }), "classifier_invalid_output: this request could not be routed");
});

test("`:default` constrains nothing on a candidate; a pin that says it names an effort no row has (review 2 r6)", () => {
  assert.equal(selectBestRoute(PROFILE, { candidateModels: ["gpt-6.1-sol:default"] }).targetModel, "gpt-6.1-sol");
  const pinned = selectBestRoute(PROFILE, { candidateModels: ["gpt-6.1-sol"], constraintModel: "gpt-6.1-sol:default" });
  assert.equal(pinned.blocked, true);
  assert.equal(pinned.refusal, "pin_unavailable: pinned model gpt-6.1-sol:default cannot serve this request");
});

const SCORED = (over = {}) => ({
  work_kind: "general_implementation",
  reasoning_depth: 2,
  code_complexity: 2,
  context_scale: 1,
  divergence: 1,
  hallucination_risk: 1,
  error_cost: 1,
  needs_visual: false,
  needs_gui: false,
  local_only: false,
  expected_hours: 1,
  hours_confidence: "medium",
  urgency: 1,
  multi_task: false,
  task_count: "1",
  interaction_style: "engineering",
  realtime_loop: false,
  ...over,
});

test("an effort the user disabled is never dispatched, by any row of that model (review 3 M1)", () => {
  const profiles = [
    SCORED({ work_kind: "tool_execution", reasoning_depth: 1, code_complexity: 0 }),
    SCORED(),
    SCORED({ reasoning_depth: 3, code_complexity: 3, context_scale: 3, error_cost: 3 }),
    SCORED({ realtime_loop: true }),
  ];
  const all = ["low", "medium", "high", "xhigh", "max"];
  let won = 0;
  for (const m of Object.keys(catalog.DEFAULT_MODELS)) {
    for (const effort of all) {
      for (const pool of [
        [m, { model: m, effort, disabled: true }],
        [{ model: m, efforts: all }, { model: m, effort, disabled: true }],
        [{ id: "a", account: "x", provider: "codex", model: m, efforts: all }, { id: "b", account: "x", provider: "codex", model: m, effort, disabled: true }],
        [{ model: m, account: "x" }, { id: "b", account: "x", model: m, effort, disabled: true }],
      ]) {
        for (const p of profiles) {
          const route = selectBestRoute(p, { candidateModels: pool, realtime_loop: p.realtime_loop });
          if (route.ok && route.targetEffort === effort) won++;
        }
      }
    }
  }
  assert.equal(won, 0);

  // Pinned, it is refused as AgentCompass refuses it, and every turn asks no more than the first
  const sol = "gpt-6.1-sol";
  for (const [pool, pin] of [
    [[sol, { model: sol, effort: "medium", disabled: true }], { constraintModel: `${sol}:medium` }],
    [[sol, { model: sol, effort: "medium", disabled: true }], { constraintModel: sol, constraintEffort: "medium" }],
    [[{ model: sol, efforts: all }, { model: sol, effort: "medium", disabled: true }], { constraintModel: `${sol}:medium` }],
    [["grok-4.7", { model: "grok-4.7", effort: "medium", disabled: true }], { constraintModel: "grok-4.7:medium" }],
  ]) {
    const route = selectBestRoute(SCORED(), { candidateModels: pool, ...pin });
    assert.equal(route.blocked, true);
    assert.match(route.refusal, /^pin_unavailable: /);
    assert.match(route.warnings[0], /unchecked in model management/);
    const run = twoTurns({ candidateModels: pool, ...pin }, { candidateModels: pool, ...pin });
    assert.equal(run.first, "agentcompass/auto");
    assert.equal(run.second, "agentcompass/auto");
    assert.match(run.rejected[0][1], /^agentcompass: pin_unavailable: /);
  }
  // The other efforts stay on offer, and a session on the disabled one leaves it
  assert.notEqual(selectBestRoute(SCORED(), { candidateModels: [sol, { model: sol, effort: "high", disabled: true }] }).targetEffort, "high");
  const left = twoTurns({ candidateModels: [`${sol}:high`] }, { candidateModels: [sol, { model: sol, effort: "high", disabled: true }] });
  assert.equal(left.first, `${sol}:high`);
  assert.notEqual(left.second, `${sol}:high`);
});

test("a route after a hold refused a step down judges by the session's task, and says why (review 3 r1)", async () => {
  const sonnet = "claude-sonnet-5-5";
  const hard = "Prove the correctness of this distributed consensus algorithm and redesign the architecture across all modules";
  const asked = { candidateModels: [`${sonnet}:xhigh`] };
  const down = { candidateModels: [`${sonnet}:low`] };

  const store = new SessionContinuityStore();
  const first = { model: "agentcompass/auto", estimated_input_tokens: 2048, messages: [{ role: "user", content: hard }] };
  const ctxOne = { sessionId: "k", options: { ...asked, continuityStore: store, now: NOW }, state: {} };
  quietly(() => middleware.onRequest(first, ctxOne));
  assert.equal(first.model, `${sonnet}:xhigh`);
  const profile = ctxOne.state.route.profile;
  assert.equal(selectBestRoute(profile, down).blocked, true, "a new route for this task refuses the low row");
  const rejected = [];
  const second = followUp(first);
  const { logged } = quietly(() =>
    middleware.onRequest(second, { sessionId: "k", options: { ...down, continuityStore: store, now: NOW + 1000 }, state: {}, reject: (s, m) => rejected.push([s, m]) }),
  );
  assert.equal(second.model, "agentcompass/auto");
  assert.equal(rejected[0]?.[0], 503);
  assert.match(logged[0], new RegExp(`${sonnet}: effort xhigh is no longer offered; routed again`));

  // The hooks do the same, with no second classifier call
  let calls = 0;
  const fetch = async () => {
    calls++;
    const content = JSON.stringify(calls === 1 ? SCORED({ reasoning_depth: 3, code_complexity: 3, context_scale: 3, divergence: 2, error_cost: 3, hallucination_risk: 2 }) : SCORED({ reasoning_depth: 1, code_complexity: 0 }));
    return { ok: true, status: 200, json: async () => ({ choices: [{ message: { content } }] }) };
  };
  const scored = { allowExternalScorer: true, scorerType: "openai_compatible", scorerBaseUrl: "http://127.0.0.1:9/v1", scorerApiKey: "fake", scorerModel: "fake", useCache: false, fetch };
  const hookStore = new SessionContinuityStore();
  const hookFirst = turnOne();
  await hooks.onRequest(hookFirst, { protocol: "chat", sessionId: "k2", options: { ...asked, ...scored, continuityStore: hookStore, now: NOW }, state: {} });
  assert.equal(calls, 1);
  const hookCtx = { protocol: "chat", sessionId: "k2", options: { ...down, ...scored, continuityStore: hookStore, now: NOW + 1000 }, state: {} };
  await hooks.onRequest(followUp(hookFirst), hookCtx);
  assert.equal(calls, 1);
  assert.equal(hookCtx.state.route, undefined);
  assert.match(hookCtx.state.routeBlocked.warnings[0], /effort xhigh is no longer offered; routed again/);

  // Routed somewhere new, the warning goes with the route
  const sol = "gpt-6.1-sol";
  const up = twoTurns({ candidateModels: [`${sol}:medium`] }, { candidateModels: [`${sol}:high`] });
  assert.equal(up.second, `${sol}:high`);
  assert.equal(up.route.continuation.why, "effort_gone");
  assert.match(up.route.warnings[0], new RegExp(`${sol}: effort medium is no longer offered; routed again`));
  const hookUp = new SessionContinuityStore();
  const hookUpFirst = turnOne();
  await hooks.onRequest(hookUpFirst, { protocol: "chat", sessionId: "k3", options: { candidateModels: [`${sol}:medium`], continuityStore: hookUp, now: NOW }, state: {} });
  const hookUpCtx = { protocol: "chat", sessionId: "k3", options: { candidateModels: [`${sol}:high`], continuityStore: hookUp, now: NOW + 1000 }, state: {} };
  await hooks.onRequest(followUp(hookUpFirst), hookUpCtx);
  assert.equal(hookUpCtx.state.route.targetEffort, "high");
  assert.match(hookUpCtx.state.route.warnings[0], /effort medium is no longer offered; routed again/);
});

test("/v1/systemone gives any caller the conclusion; the details take the operator's option (review 3 r2)", async () => {
  const account = "claude/secret-account@example.com";
  const windows = (remaining) => [{ name: "7d", used: 100 - remaining, remaining, resetsAt: new Date(Date.now() + 48 * 3600 * 1000).toISOString() }];
  const quotaOf = (remaining) => ({ [account]: { kind: "subscription", quota: { windows: windows(remaining) } } });
  const ok = { candidateModels: ["claude-sonnet-5-5"], quotas: quotaOf(80) };
  const spent = { candidateModels: ["claude-sonnet-5-5"], quotas: quotaOf(0) };
  const sick = { candidateModels: ["claude-sonnet-5-5"], unhealthyModels: ["claude-sonnet-5-5"] };
  const leaks = (answer) => /secret-account|remaining|unhealthy|ability|floor|selectedAccount|"quota"|"warnings"|"profile"|"scorer"/.test(JSON.stringify(answer));
  const silently = async (fn) => {
    const warn = console.warn;
    const logged = [];
    console.warn = (line) => logged.push(line);
    try {
      return { answer: await fn(), logged };
    } finally {
      console.warn = warn;
    }
  };

  // A route: model and effort, and the answers Magpie reads
  const routed = await systemOne({}, ok);
  assert.deepEqual(Object.keys(routed.agentcompass).sort(), ["targetEffort", "targetModel"]);
  assert.equal(routed.agentcompass.targetModel, "claude-sonnet-5-5");
  assert.ok(routed.answers.effort);
  assert.ok(routed.usage);
  assert.equal(leaks(routed), false);

  // A block: a code and a sentence, whatever blocked it; the detail is in the local log
  for (const [options, state] of [[spent, {}], [sick, {}]]) {
    const { answer, logged } = await silently(() => systemOne(state, options));
    assert.deepEqual(answer.agentcompass, { targetModel: null, blocked: true, error: "no_route", message: "no candidate can serve this request" });
    assert.equal(leaks(answer), false);
    assert.ok(logged.length > 0);
  }
  assert.match((await silently(() => systemOne({}, spent))).logged.join("\n"), /secret-account/);
  // A pin learns no more: not whether the name is in the pool, nor whether it is healthy
  const pinAnswers = [];
  for (const [options, pin] of [[sick, "claude-sonnet-5-5"], [ok, "not-in-the-pool"], [{ candidateModels: ["claude-sonnet-5-5:high"] }, "claude-sonnet-5-5:low"]]) {
    pinAnswers.push((await silently(() => systemOne({ constraintModel: pin }, options))).answer.agentcompass);
  }
  for (const answer of pinAnswers) {
    assert.deepEqual(answer, { targetModel: null, blocked: true, error: "pin_unavailable", message: "the pinned model cannot serve this request" });
  }
  // A short follow-up is a conclusion too
  const nudge = await index.handleSystemOneRequest(
    { model: "jev-6b-int8", state: { message: "继续", previous_message_kind: "general_implementation", previous_message_reasoning: "high" }, questions: SYSTEM_ONE_QUESTIONS },
    ok,
  );
  assert.deepEqual(Object.keys(nudge.agentcompass).sort(), ["targetEffort", "targetModel"]);
  // Through the provider's fetch too
  const fetched = await index.systemOneFetch("http://plugin/v1/systemone", { method: "POST", body: JSON.stringify({ model: "m", state: { message: TASK }, questions: SYSTEM_ONE_QUESTIONS }) }, spent);
  const fetchedBody = await (await silently(async () => fetched)).answer.json();
  assert.equal(leaks(fetchedBody), false);

  // Only the operator's literal `true` opens the details
  assert.equal(leaks(await silently(() => systemOne({ decisionProviderDetails: true }, spent)).then((r) => r.answer)), false);
  assert.equal(leaks(await silently(() => systemOne({}, { ...spent, decisionProviderDetails: "true" })).then((r) => r.answer)), false);
  const details = { decisionProviderDetails: true };
  const blocked = (await silently(() => systemOne({}, { ...spent, ...details }))).answer.agentcompass;
  assert.equal(blocked.error, "no_route");
  assert.equal(blocked.refusal, "no_route: no candidate can serve this request");
  assert.match(blocked.warnings.join(" "), /secret-account/);
  const open = (await systemOne({}, { ...ok, ...details })).agentcompass;
  assert.equal(open.selectedAccount, account);
  assert.equal(open.quota.account, account);
  assert.ok(open.profile);
  assert.ok(Array.isArray(open.warnings));
  const pinned = (await silently(() => systemOne({ constraintModel: "claude-sonnet-5-5" }, { ...sick, ...details }))).answer.agentcompass;
  assert.equal(pinned.refusal, "model_unhealthy: pinned model claude-sonnet-5-5 is marked unhealthy");
});

test("model_unhealthy says whose health it is (review 3 r3)", () => {
  const flagship = "claude-opus-5-5";
  const family = "claude-opus-9";
  const refuse = (pool, unhealthyModels, pin) => selectBestRoute(PROFILE, { candidateModels: pool, unhealthyModels, ...pin });
  // The pinned model itself
  assert.equal(refuse([flagship, family], [flagship], { constraintModel: flagship }).refusal, `model_unhealthy: pinned model ${flagship} is marked unhealthy`);
  assert.equal(refuse([family], [family], { constraintModel: family }).refusal, `model_unhealthy: pinned model ${family} is marked unhealthy`);
  // A family row down says nothing of the flagship pinned: it is another model
  assert.equal(refuse([family], [family], { constraintModel: flagship }).refusal, `pin_not_in_pool: pinned model ${flagship} is not among the candidates`);
  assert.equal(refuse([flagship, family], [family], { constraintModel: flagship }).targetModel, flagship);
  // An effort pin has no model to blame, and a row left out by its effort is not the cause
  assert.match(refuse(["gpt-6.1-sol:high", "gpt-6-astra:high"], ["gpt-6.1-sol"], { constraintEffort: "low" }).refusal, /^pin_unavailable: pinned effort low /);
  assert.match(refuse(["gpt-6.1-sol:low"], ["gpt-6.1-sol"], { constraintEffort: "low" }).refusal, /^pin_unavailable: pinned effort low /);
  // A bundle's model is named as the bundle's
  const bundle = { id: "row", provider: "codex", account: "codex", model: "gpt-6.1-sol", effort: "high" };
  assert.equal(refuse([bundle], ["gpt-6.1-sol"], { pinnedBundle: "row" }).refusal, "model_unhealthy: the model of pinned bundle row is marked unhealthy");
  // Blocked for another reason, with nothing down: no health in the wording
  assert.match(refuse([`${flagship}:high`], [], { constraintModel: `${flagship}:low` }).refusal, /^pin_unavailable: /);
});

test("onProvider's route leaves out what the user disabled, though the provider's list has no such row (review 4 M1)", () => {
  const sol = "gpt-6.1-sol";
  const astra = "gpt-6-astra";
  const off = (model, effort) => ({ model, effort, disabled: true });
  const send = (options) => {
    const body = turnOne("rename a variable in the README");
    const ctx = { sessionId: "p", options: { continuityStore: new SessionContinuityStore(), now: NOW, ...options }, state: {}, provider: { id: "p" } };
    quietly(() => middleware.onRequest(body, ctx));
    quietly(() => middleware.onProvider(body, ctx));
    return body.model;
  };
  assert.equal(send({ candidateModels: [sol, off(sol, "low")], routePerProvider: true, modelsByProvider: { p: [sol] } }), `${sol}:medium`);
  assert.equal(send({ candidateModels: [sol, off(sol, "low")], routePerProvider: true, modelsByProvider: { p: [sol, off(sol, "low")] } }), `${sol}:medium`);
  assert.equal(send({ candidateModels: [sol, astra, off(astra, "low")], modelsByProvider: { p: [astra] } }), `${astra}:medium`);
  // The list keeps its own say: a disabled row stays out of what it offers
  assert.equal(allocator.candidateOffers([sol, off(sol, "low")], sol, "low"), false);
});

test("a turn with no text still routes when the user set a pool or a pin (review 5 M1)", () => {
  const sol = "gpt-6.1-sol";
  const astra = "gpt-6-astra";
  const pool = [sol, { model: sol, effort: "high", disabled: true }];
  const image = { role: "user", content: [{ type: "image_url", image_url: { url: "data:image/png;base64,AAAA" } }] };
  const send = (options, extra = {}) => {
    const body = { model: "agentcompass/auto", messages: [image], ...extra };
    const ctx = context(options, { sessionId: "i", provider: { id: "p", model: sol } });
    const { value } = quietly(() => middleware.onRequest(body, ctx));
    if (value?.rejected) return value;
    quietly(() => middleware.onProvider(body, ctx));
    return body;
  };
  // The first turn is an image, so the effort the user disabled must not reach the provider
  assert.equal(send({ candidateModels: pool }).model, `${sol}:low`);
  assert.equal(send({ candidateModels: pool, routeModelTriggers: [sol] }, { model: sol }).model, `${sol}:low`);
  assert.equal(send({ candidateModels: pool, routeAllModels: true }, { model: sol, reasoning_effort: "high" }).model, `${sol}:low`);
  // A pin holds without text, and an unhealthy one is refused
  assert.equal(send({ candidateModels: [sol, astra], routeAllModels: true, pinnedModel: astra }, { model: sol }).model, `${astra}:low`);
  assert.equal(send({ candidateModels: [sol, astra], routeAllModels: true, pinnedModel: sol, unhealthyModels: [sol] }, { model: sol }).rejected, 503);
  assert.equal(send({ candidateModels: [sol, astra], pinnedModel: sol, unhealthyModels: [sol] }).rejected, 503);
  // With neither a pool nor a pin there is nothing to route on
  const bare = send({ routeAllModels: true }, { model: sol, reasoning_effort: "high" });
  assert.equal(bare.model, sol);
  assert.equal(bare.reasoning_effort, "high");
  // Under the default install (no decision provider), a virtual model name with no text is routed in onRequest (status 200, no longer 404)
  const defaultInstallGateway = (options, messages = [image]) => {
    const body = { model: "agentcompass/auto", messages };
    const ctx = context(options, { sessionId: "i" });
    const { value } = quietly(() => middleware.onRequest(body, ctx));
    if (value?.rejected) return { status: value.rejected, model: body.model };
    if (body.model === "agentcompass/auto") return { status: 404, model: body.model };
    return { status: 200, model: body.model };
  };
  assert.deepEqual(defaultInstallGateway({ candidateModels: pool }), { status: 200, model: `${sol}:low` });
  assert.deepEqual(defaultInstallGateway({ candidateModels: pool }, [{ role: "user", content: "" }]), { status: 200, model: `${sol}:low` });
  assert.deepEqual(defaultInstallGateway({ candidateModels: ["auto"] }), { status: 503, model: "agentcompass/auto" });
});

test("the hooks route a turn with no text too when the user set a pool or a pin (review 5 M1)", async () => {
  const sol = "gpt-6.1-sol";
  const astra = "gpt-6-astra";
  const pool = [sol, { model: sol, effort: "high", disabled: true }];
  const image = { role: "user", content: [{ type: "image_url", image_url: { url: "data:image/png;base64,AAAA" } }] };
  const send = async (options, extra = {}) => {
    const ctx = context(options, { sessionId: "i" });
    await hooks.onRequest({ model: "agentcompass/auto", messages: [image], ...extra }, ctx);
    return ctx.state;
  };
  const picked = ({ route }) => `${route?.targetModel}:${route?.targetEffort}`;
  // The first turn is an image, so the effort the user disabled must not be routed to
  assert.equal(picked(await send({ candidateModels: pool })), `${sol}:low`);
  assert.equal(picked(await send({ candidateModels: pool, routeModelTriggers: [sol] }, { model: sol })), `${sol}:low`);
  assert.equal(picked(await send({ candidateModels: pool, routeAllModels: true }, { model: sol, reasoning_effort: "high" })), `${sol}:low`);
  // A pin holds without text, and an unhealthy one is refused
  assert.equal(picked(await send({ candidateModels: [sol, astra], routeAllModels: true, pinnedModel: astra }, { model: sol })), `${astra}:low`);
  for (const [options, extra] of [[{ routeAllModels: true }, { model: sol }], [{}, {}]]) {
    const refused = await send({ candidateModels: [sol, astra], pinnedModel: sol, unhealthyModels: [sol], ...options }, extra);
    assert.equal(refused.route, undefined);
    assert.ok(refused.routeBlocked);
  }
  // With neither a pool nor a pin there is nothing to route on
  const bare = await send({ routeAllModels: true }, { model: sol, reasoning_effort: "high" });
  assert.equal(bare.route, undefined);
  assert.equal(bare.routeBlocked, undefined);
});

test("an effort the user disabled stays out for a row off by default, pinned too (review 4 r1)", () => {
  const sol = "gpt-6.1-sol";
  const dflt = (extra) => ({ model: sol, disabled: true, disabled_by_default: true, ...extra });
  const off = { id: "b", model: sol, effort: "medium", disabled: true };
  for (const [pool, pin] of [
    [[dflt({ id: "a", efforts: ["low", "medium", "high"] }), off], { constraintModel: `${sol}:medium` }],
    [[dflt({ id: "a", efforts: ["low", "medium", "high"] }), off], { pinnedBundle: "a", constraintEffort: "medium" }],
    [[dflt({ id: "a", effort: "medium" }), off], { constraintModel: `${sol}:medium` }],
    [[dflt({ id: "a" }), off], { constraintModel: `${sol}:medium` }],
  ]) {
    const route = selectBestRoute(PROFILE, { candidateModels: pool, ...pin });
    assert.equal(route.blocked, true, JSON.stringify(pin));
    assert.match(route.refusal, /^pin_unavailable: /);
  }
  // Named, the row off by default still runs at the efforts left
  assert.equal(selectBestRoute(PROFILE, { candidateModels: [dflt({ id: "a", efforts: ["medium", "high"] }), off], constraintModel: `${sol}:high` }).modelWithEffort, `${sol}:high`);
});

test("a disabled effort is that model's: a flagship and its family row keep their own (review 4 r2)", () => {
  const flagship = "claude-opus-5-5";
  const family = "claude-opus-9";
  const off = (model, effort) => ({ model, effort, disabled: true });
  assert.equal(selectBestRoute(PROFILE, { candidateModels: [flagship, family, off(flagship, "high")], constraintModel: `${family}:high` }).modelWithEffort, `${family}:high`);
  assert.equal(selectBestRoute(PROFILE, { candidateModels: [flagship, family, off(family, "high")], constraintModel: `${flagship}:high` }).modelWithEffort, `${flagship}:high`);
  // Its own row stays refused, and a spelling of the row is still the row
  assert.equal(selectBestRoute(PROFILE, { candidateModels: [flagship, family, off(flagship, "high")], constraintModel: `${flagship}:high` }).blocked, true);
  assert.equal(selectBestRoute(PROFILE, { candidateModels: [flagship, off(`${flagship}-20261001`, "high")], constraintModel: `${flagship}:high` }).blocked, true);
  assert.equal(allocator.candidateOffers([flagship, family, off(flagship, "high")], family, "high"), true);
  assert.equal(allocator.candidateOffers([flagship, family, off(flagship, "high")], flagship, "high"), false);
});

test("a short follow-up on /v1/systemone checks the effort locally, and only a disabled one routes (review 4 r3)", async () => {
  const sol = "gpt-6.1-sol";
  const questions = { effort: { type: "score", criteria: ["low", "medium", "high", "xhigh"] } };
  let calls = 0;
  const fetch = async () => {
    calls++;
    const content = JSON.stringify(SCORED());
    return { ok: true, status: 200, json: async () => ({ choices: [{ message: { content } }] }) };
  };
  const scored = { allowExternalScorer: true, scorerType: "openai_compatible", scorerBaseUrl: "http://127.0.0.1:9/v1", scorerApiKey: "fake", scorerModel: "fake", useCache: false, fetch };
  const follow = (pool) =>
    index.handleSystemOneRequest(
      { model: "m", state: { message: "继续", previous_message_kind: "general_implementation", previous_message_reasoning: "high" }, questions },
      { candidateModels: pool, ...scored },
    );
  // Without the removed text heuristic, follow-ups on /v1/systemone ask the classifier and check the pool
  assert.equal((await follow([sol])).agentcompass.targetEffort, "high");
  assert.equal((await follow([sol, { model: sol, effort: "low", disabled: true }])).agentcompass.targetEffort, "high");
  assert.equal(calls, 2);
  // That effort disabled: a route, the classifier allowed, and it is not sent
  const answer = await follow([sol, { model: sol, effort: "high", disabled: true }]);
  assert.notEqual(answer.agentcompass.targetEffort, "high");
  assert.notEqual(answer.answers.effort.score, 2);
  assert.equal(allocator.effortUnchecked([sol, { model: sol, effort: "high", disabled: true }], "high"), true);
  assert.equal(allocator.effortUnchecked([sol], "high"), false);
  assert.equal(allocator.effortUnchecked(null, "high"), false);
});

test("a held bundle is a row that offers the effort, on an account that has it (review 4 r4)", async () => {
  const sol = "gpt-6.1-sol";
  const all = ["low", "medium", "high"];
  const x = (extra) => ({ id: "xa", provider: "codex", account: "x", model: sol, ...extra });
  const y = (extra) => ({ id: "ya", provider: "codex", account: "y", model: sol, ...extra });
  const off = { id: "xb", provider: "codex", account: "x", model: sol, effort: "medium", disabled: true };
  for (const pool of [[x({ efforts: all }), off, y({ efforts: all })], [off, x({ efforts: all }), y({ efforts: all })]]) {
    const run = twoTurns({ candidateModels: [x({ effort: "medium" })] }, { candidateModels: pool });
    assert.equal(run.second, `${sol}:medium`);
    assert.equal(run.route.continuation.held, true);
    assert.equal(run.route.bundle.account, "y");
    assert.equal(run.route.bundle.disabled, undefined);
    assert.equal(allocator.heldBundle(pool, sol, "medium").account, "y");

    const store = new SessionContinuityStore();
    const first = turnOne();
    await hooks.onRequest(first, { protocol: "chat", sessionId: "b", options: { candidateModels: [x({ effort: "medium" })], continuityStore: store, now: NOW }, state: {} });
    const ctx = { protocol: "chat", sessionId: "b", options: { candidateModels: pool, continuityStore: store, now: NOW + 1000 }, state: {} };
    await hooks.onRequest(followUp(first), ctx);
    assert.equal(ctx.state.route.continuation.held, true);
    assert.equal(ctx.state.route.bundle.account, "y");
  }
});

test("the docs say what a disabled row covers and what the log carries (review 4 r5)", () => {
  const notes = fs.readFileSync(path.join(REPO, "CHANGELOG.md"), "utf8");
  assert.doesNotMatch(notes, /账号和配额不写。/);
  assert.match(notes, /when no candidate can serve a request, `warnings` in the local log may include account names and quotas/);
  const readme = fs.readFileSync(path.join(REPO, "README.md"), "utf8");
  assert.match(readme, /A disabled row with an `account` applies to that account only/);
  assert.match(readme, /禁用行带 `account` 时只对该账户生效/);
});

test("M-3.5 onProvider reroutes on a held turn too, and keeps the unhealthy models out", async (t) => {
  t.mock.method(console, "warn", () => {});
  const continuityStore = new SessionContinuityStore();
  const options = { candidateModels: ["claude-sonnet-5-5"], modelsByProvider: { codex: ["gpt-6.1-sol"] }, continuityStore };
  const first = context(options);
  const firstBody = request({ metadata: { session_id: "held" } });
  middleware.onRequest(firstBody, first);
  continuityStore.recordResponse(first.state.route.continuation.sessionKey, { usage: { input_tokens: 2000, output_tokens: 300 } }, {}, NOW);

  const held = context({ ...options, now: NOW + 1000 });
  const body = {
    ...request({ metadata: { session_id: "held" } }),
    messages: [{ role: "user", content: TASK }, { role: "assistant", content: "ok" }, { role: "user", content: "继续" }],
  };
  middleware.onRequest(body, held);
  assert.equal(held.state.route.continuation.held, true);
  assert.match(body.model, /^claude-sonnet-5-5:/);
  middleware.onProvider(body, { ...held, provider: { id: "codex" } });
  assert.match(body.model, /^gpt-6\.1-sol:/);

  const pool = ["claude-sonnet-5-5", "gpt-6.1-sol"];
  const unhealthy = context({ candidateModels: pool, modelsByProvider: { p: pool }, routePerProvider: true, unhealthyModels: ["claude-sonnet-5-5"] });
  const other = request();
  middleware.onRequest(other, unhealthy);
  middleware.onProvider(other, { ...unhealthy, provider: { id: "p" } });
  assert.match(other.model, /^gpt-6\.1-sol:/);
});

// ---------------------------------------------------------------------------- R4: profiles

test("M-4.1 task_count takes a bucket from a positive integer, contradictions and anything else are illegal, and validation is idempotent", async () => {
  const { validateProfile } = profiler;
  for (const [value, bucket, multi] of [[1, "1", false], [2, "2", true], [3, "3", true], [4, ">3", true], [9, ">3", true], ["2", "2", true], [">3", ">3", true]]) {
    assert.equal(validateProfile({ ...PROFILE, multi_task: multi, task_count: value }).task_count, bucket, String(value));
  }
  for (const value of [0, -1, 1.5, NaN, Infinity, "many", null, undefined]) {
    assert.throws(() => validateProfile({ ...PROFILE, task_count: value }), /task_count/, String(value));
  }
  assert.throws(() => validateProfile({ ...PROFILE, multi_task: true, task_count: "1" }), /task_count/);
  assert.throws(() => validateProfile({ ...PROFILE, multi_task: false, task_count: "3" }), /task_count/);
  const once = validateProfile(PROFILE);
  assert.deepEqual(validateProfile(once), once);

  // An illegal classifier answer is not repaired: one retry, then blocked
  const { options, calls } = classifier({ reasoning_depth: 2, code_complexity: 2, context_scale: 2, divergence: 1, hallucination_risk: 2, error_cost: 2, interaction_style: "engineering", multi_task: true, task_count: "1" });
  const scored = await scorer.scoreTaskLive({ objective: TASK }, { ...options, useCache: false });
  assert.equal(scored.profile, null);
  assert.equal(scored.meta.error, "classifier_invalid_output");
  assert.equal(calls.count, 2);

  // An illegal caller profile says so
  const decision = selectBestRoute({ ...PROFILE, multi_task: true, task_count: "1" }, { candidateModels: ["gpt-6.1-sol"] });
  assert.equal(decision.error, "invalid_caller_profile");
  assert.throws(() => rankCandidates({ ...PROFILE, task_count: "many" }), (err) => err.code === "invalid_caller_profile");
});

test("M-4.2 the profile cache keys on every structured field, copies both ways, and a bad entry is a miss", async () => {
  const answer = { ...PROFILE };
  delete answer.work_kind;
  const { options, calls } = classifier(answer);
  scorer.clearProfileCache();

  await scorer.scoreTaskLive({ objective: TASK, work_kind: "general_implementation" }, options);
  await scorer.scoreTaskLive({ objective: TASK, work_kind: "code_review" }, options);
  await scorer.scoreTaskLive({ objective: TASK, needs: { gui: true } }, options);
  assert.equal(calls.count, 3, "different structured fields, different entries");

  const task = { objective: `${TASK} (isolation)` };
  const first = await scorer.scoreTaskLive(task, options);
  const before = calls.count;
  const depth = first.profile.reasoning_depth;
  first.profile.reasoning_depth = 99;
  const second = await scorer.scoreTaskLive(task, options);
  assert.equal(calls.count, before, "served from the cache");
  assert.equal(second.profile.reasoning_depth, depth, "a caller's change to the result does not reach the cache");
  second.profile.reasoning_depth = 98;
  assert.equal((await scorer.scoreTaskLive(task, options)).profile.reasoning_depth, depth);

  // Written by an older version: revalidated on read, a failure is a miss
  const stale = { objective: `${TASK} (stale)` };
  scorer.putCachedProfile(scorer.cacheKeyOf(stale, options), { reasoning_depth: 99 }, {});
  const reread = await scorer.scoreTaskLive(stale, options);
  assert.ok(reread.profile.reasoning_depth <= 4);
  assert.equal(calls.count, before + 1);
  scorer.clearProfileCache();
});

// ---------------------------------------------------------------------------- R5 / R6: pinned data, scoring as AgentCompass

test("7.2 no routine effort overhead: on a routine task astra:high does not lose to astra:xhigh", () => {
  const computerUse = { ...ROUTINE, work_kind: "computer_use", needs_visual: true, needs_gui: true };
  const ranked = rankCandidates(computerUse, { candidateModels: [{ model: "gpt-6-astra", efforts: ["high", "xhigh"] }] });
  const byEffort = Object.fromEntries(ranked.allCandidates.map((c) => [c.effort, c]));
  assert.equal(ranked.task.difficulty, "routine");
  assert.ok(byEffort.high.score > byEffort.xhigh.score, `high ${byEffort.high.score}, xhigh ${byEffort.xhigh.score}`);
  assert.equal(byEffort.high.components.effort_penalty, 0);
});

test("7.3 a bundle's `when` is checked in every mode", () => {
  const bundle = { id: "easy_only", provider: "codex", account: "codex", model: "gpt-6.1-sol", effort: "high", when: { max_reasoning_depth: 1 } };
  const fallback = { id: "any", provider: "codex", account: "codex", model: "gpt-6.1-terra", effort: "high" };
  assert.deepEqual(rankCandidates(PROFILE, { candidateModels: [bundle, fallback] }).ranking.map((c) => c.id), ["any"]);
  assert.equal(rankCandidates(ROUTINE, { candidateModels: [bundle, fallback] }).ranking.length, 2);
});

test("7.4 the harness factor discounts a harness's score as accounts.json says", () => {
  const model = "claude-sonnet-5-5";
  const direct = { id: "direct", provider: "claude", account: "claude", harness: "claude", model, effort: "high" };
  const resilient = { id: "agy", provider: "antigravity-resilient", account: "agy", harness: "antigravity-resilient", model, effort: "high" };
  const ranked = rankCandidates(PROFILE, { candidateModels: [resilient, direct] });
  const a = candidate(ranked, "direct");
  const b = candidate(ranked, "agy");
  assert.ok(a.eligible && b.eligible);
  const factor = catalog.DEFAULT_SCORING_POLICY.harness_factors.rules.find((r) => r.harness === "antigravity-resilient" && r.model_prefix === "claude").factor;
  assert.ok(factor < 1);
  assert.ok(b.score < a.score, `${b.score} < ${a.score}`);
  assert.equal(b.components.harness_factor, factor);
});

// ---------------------------------------------------------------------------- warnings and plumbing

test("m-7.1 a route's warnings survive the hand-off between hooks and middleware", async (t) => {
  const warn = t.mock.method(console, "warn", () => {});
  const options = { candidateModels: ["gemini-4-argon-high"], pinnedModel: "gemini-4-argon-high", realtime_loop: true };
  const ctx = context(options);
  await hooks.onRequest(request(), ctx);
  assert.ok(ctx.state.route.warnings.length > 0);

  middleware.onRequest(request(), context({ candidateModels: ["gemini-4-argon-high"] }, { state: { route: ctx.state.route } }));
  assert.ok(warn.mock.calls.some((c) => String(c.arguments[0]).includes(ctx.state.route.warnings[0])));
});

test("m-7.2 a blocked route says why, pinned or not", () => {
  for (const options of [
    { candidateModels: ["gpt-6-astra"], constraintModel: "gpt-6-astra", unhealthyModels: ["gpt-6-astra"] },
    { candidateModels: ["gpt-6-luna"], constraintModel: "gpt-6-luna", minContextTokens: 1e12 },
    { candidateModels: ["gpt-6-astra"], unhealthyModels: ["gpt-6-astra"] },
  ]) {
    const decision = selectBestRoute(PROFILE, options);
    assert.equal(decision.blocked, true, JSON.stringify(options));
    assert.ok(decision.warnings.length > 0, JSON.stringify(options));
    assert.ok(decision.excluded.length > 0, JSON.stringify(options));
  }
});

test("m-7.3 handleSystemOneRequest reads every candidate list the plugin may be given", async () => {
  const entry = { id: "seat", provider: "codex", account: "codex", model: "gpt-6.1-sol", efforts: ["medium"] };
  for (const key of ["availableModels", "allowedModels", "models"]) {
    const answer = await systemOne({}, { [key]: [entry] });
    assert.equal(answer.agentcompass.targetModel, "gpt-6.1-sol", key);
    assert.equal(answer.agentcompass.targetEffort, "medium", key);
  }
  const mapped = await systemOne({}, { modelMap: { "gpt-6.1-sol": "upstream/gpt-6.1-sol" } });
  assert.equal(mapped.agentcompass.targetModel, "gpt-6.1-sol");
});

// ---------------------------------------------------------------------------- realtime_loop responsiveness

const RT_PROFILE = { ...PROFILE, work_kind: "general_implementation", reasoning_depth: 1, code_complexity: 1, needs_visual: false, needs_gui: false, realtime_loop: true };
const REALTIME_FIRST_TURN = "Stay in the live match and react each turn";

test("m1-R3: named aliases and family fallbacks land where model_resolution.py puts them", () => {
  const key = (m) => catalog.resolveResponsivenessModelKey(m).key;
  assert.equal(key("zai-org-glm-5-2"), "glm-5.2");
  assert.equal(key("venice-api/zai-org-glm-5-2"), "glm-5.2");
  assert.equal(key("deepseek-flash"), "deepseek-v4.1-flash-free");
  assert.equal(key("deepseek-v4-flash"), "deepseek-v4.1-flash-free");
  assert.equal(key("claude-haiku-4-20250514"), "claude-haiku-5-5");
  assert.equal(key("grok-4-fast"), "grok-4.7-build-fast");
  assert.equal(key("gpt-6.10-terra"), "gpt-6.1-terra");
});

test("M1-R3: slow models with a flat ability curve are excluded, models without one step down", () => {
  const stay = (model, provider) => allocator.resolveRealtimeEffort({ model, effort: "high", provider }, { realtime_loop: true }, {}, { gate: 3.0 });
  for (const model of ["deepseek-v4-pro", "deepseek-v4-pro-20260101", "deepseek-r1", "glm-5.3", "glm-5.3-latest", "mistral-large-4", "mistral-large-4-20260101"]) {
    const r = stay(model, "openrouter");
    assert.equal(r.effort, "high", model);
    assert.equal(r.steppedFrom, null, model);
  }
  assert.deepEqual([stay("grok-4.6", "xai").effort, stay("grok-4.6", "xai").responsiveness], ["low", 0.675]);
  assert.deepEqual([stay("xai/grok-4.6", "xai").effort, stay("xai/grok-4.6", "xai").steppedFrom], ["low", "high"]);
  assert.deepEqual([stay("claude-opus-4-7", "openrouter").effort, stay("claude-opus-4-7", "openrouter").responsiveness], ["none", 0.929]);
  assert.equal(stay("anthropic/claude-opus-4-7-20260101", "openrouter").effort, "none");

  const pool = ["deepseek-v4-pro:high", "glm-5.3:high", "mistral-large-4:high", { id: "spark", provider: "openrouter", model: "gpt-5.3-codex-spark", effort: "medium" }];
  const ranked = rankCandidates(RT_PROFILE, { candidateModels: pool, now: NOW });
  assert.equal(ranked.selected?.id, "spark");
  for (const model of ["deepseek-v4-pro", "glm-5.3", "mistral-large-4"]) {
    const c = ranked.allCandidates.find((x) => x.model === model);
    assert.equal(c.eligible, false, model);
    assert.ok(c.excluded.some((e) => e.includes("below the realtime_loop floor")), model);
  }
});

test("M2-R3: a 5-5 candidate is not a sibling of an older opus, so the step-down is not blocked", () => {
  assert.equal(catalog.canonicalModel("claude-opus-4-7"), "claude-opus-4-7");
  assert.equal(catalog.resolveModelSpec("claude-opus-4-7").canonical, "claude-opus-4-7");
  const bundles = [
    { id: "o47", account: "acct", provider: "gw", model: "claude-opus-4-7", effort: "high" },
    { id: "o55", account: "acct", provider: "gw", model: "claude-opus-5-5", effort: "medium" },
  ];
  const r = allocator.resolveRealtimeEffort(bundles[0], { realtime_loop: true, bundles }, {}, { gate: 0 });
  assert.equal(r.effort, "none");
  assert.equal(r.steppedFrom, "high");
  const withSibling = [...bundles, { id: "o47none", account: "acct", provider: "gw", model: "anthropic/claude-opus-4-7", effort: "none" }];
  const held = allocator.resolveRealtimeEffort(withSibling[0], { realtime_loop: true, bundles: withSibling }, {}, { gate: 0 });
  assert.equal(held.effort, "high");
  assert.equal(held.steppedFrom, null);
});

test("R3: provider-less candidates stay on dispatchable efforts, with or without a known family", () => {
  for (const [entry, allowed] of [
    ["claude-haiku-5-5", ["high"]],
    ["gpt-6-luna", ["high"]],
    ["claude-opus-4-6", ["high"]],
    ["grok-4.7", ["low", "high", "xhigh"]],
  ]) {
    const ranked = rankCandidates(RT_PROFILE, { candidateModels: [entry], now: NOW });
    for (const c of ranked.allCandidates) {
      assert.ok(allowed.includes(c.effort), `${entry}: ${c.effort}`);
    }
  }
});

function rtScorerOptions(store, pool, extra = {}) {
  let calls = 0;
  const options = {
    continuityStore: store,
    candidateModels: pool,
    allowExternalScorer: true,
    scorerType: "openai_compatible",
    scorerBaseUrl: "http://127.0.0.1:9/v1",
    scorerApiKey: "k",
    scorerModel: "m",
    useCache: false,
    now: NOW,
    fetch: async () => {
      calls += 1;
      const content = JSON.stringify({ ...SCORED(), realtime_loop: calls === 1 });
      return new Response(JSON.stringify({ choices: [{ message: { content } }] }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    },
    ...extra,
  };
  return { options, calls: () => calls };
}

async function rtHooksTurn(options, messages, extraBody = {}) {
  const ctx = { state: {}, options };
  await hooks.onRequest({ model: "agentcompass/auto", messages, ...extraBody }, ctx);
  return ctx;
}

const rtTwoTurns = (second) => [
  { role: "user", content: REALTIME_FIRST_TURN },
  { role: "assistant", content: "ok" },
  { role: "user", content: second },
];

test("M3-R3: a short realtime session holds on turn 2, without rerouting or ratcheting the effort back up", async () => {
  const store = new SessionContinuityStore();
  const { options, calls } = rtScorerOptions(store, ["gpt-6-astra", "gpt-5.3-codex-spark"]);
  const first = await rtHooksTurn(options, [{ role: "user", content: REALTIME_FIRST_TURN }]);
  assert.equal(first.state.route.realtimeLoop, true);
  assert.equal(store.get(first.state.sessionKey, NOW).realtimeLoop, true);
  const firstEffort = first.state.route.targetEffort;
  const firstModel = first.state.route.targetModel;

  const second = await rtHooksTurn(options, rtTwoTurns("Here is the stack trace from the handler when a socket closes."));
  assert.equal(second.state.route.continuation.held, true);
  assert.equal(second.state.route.continuation.why, "realtime_loop");
  assert.equal(second.state.route.targetEffort, firstEffort);
  assert.equal(second.state.route.targetModel, firstModel);
  assert.equal(second.state.route.realtimeLoop, true);
  assert.equal(calls(), 1, "a hold does not call the scorer");
  assert.equal(store.get(first.state.sessionKey, NOW).realtimeLoop, true);
});

test("M3-R3: an explicit realtime_loop:false ends the realtime hold; a non-realtime session is untouched", async () => {
  const store = new SessionContinuityStore();
  const { options } = rtScorerOptions(store, ["gpt-6-astra"]);
  const first = await rtHooksTurn(options, [{ role: "user", content: REALTIME_FIRST_TURN }]);
  assert.equal(first.state.route.realtimeLoop, true);
  const off = await rtHooksTurn(options, rtTwoTurns("Now write up the summary."), { realtime_loop: false });
  assert.equal(off.state.route.continuation.held, false);
  assert.equal(off.state.route.realtimeLoop, false);
  assert.equal(store.get(first.state.sessionKey, NOW).realtimeLoop, false);
});

test("M3-R3: a realtime session is not moved off its fast model by tier escalation", () => {
  const session = { model: "gpt-6-luna", effort: "high", tokens: 5000, cached: true, realtimeLoop: true, updatedAt: NOW - 1000 };
  const facts = { demandsFrontierEscalation: true };
  const held = continuity.evaluateProviderContinuation(facts, session, { now: NOW });
  assert.equal(held.shouldHold, true);
  assert.equal(held.why, "cache");
  const notRealtime = continuity.evaluateProviderContinuation(facts, { ...session, realtimeLoop: false }, { now: NOW });
  assert.equal(notRealtime.shouldHold, false);
  assert.equal(notRealtime.reasonDetail, "tier_escalation_required");
  assert.equal(
    continuity.evaluateProviderContinuation({ turnText: "Redesign the distributed architecture and find the root cause of this deadlock" }, { ...session, realtimeLoop: false }, { now: NOW }).why,
    "cache",
  );
  assert.equal(continuity.evaluateProviderContinuation({ ...facts, realtime_loop: false }, session, { now: NOW }).reasonDetail, "tier_escalation_required");
  assert.equal(continuity.evaluateProviderContinuation({ hasCurrentTurnVisual: true }, { ...session, model: "grok-4.7" }, { now: NOW }).why, "capability_outgrown");
  assert.equal(continuity.evaluateProviderContinuation({}, session, { now: NOW, unhealthyModels: ["gpt-6-luna"] }).why, "provider_unhealthy");
  assert.equal(continuity.evaluateProviderContinuation({}, { ...session, cached: false, tokens: 10, updatedAt: NOW - 3 * 3600 * 1000 }, { now: NOW }).why, "cold");
});

test("M3-R3: keepsEffort reads the realtime flag off the session", () => {
  const session = { model: "gpt-6-astra", effort: "high", tokens: 5000, cached: true, realtimeLoop: true, updatedAt: NOW - 1000 };
  assert.equal(continuity.keepsEffort(session, "low", 2048, NOW), "low");
  assert.equal(continuity.keepsEffort(session, undefined, 2048, NOW), "high");
  assert.equal(continuity.keepsEffort(session, "low", 2048, NOW, { realtime_loop: false }), "high");
  assert.equal(continuity.keepsEffort({ ...session, realtimeLoop: false }, "low", 2048, NOW), "high");
});

test("M3-R3: a hold keeps the routed name instead of rewriting it to the canonical model", async () => {
  const store = new SessionContinuityStore();
  const options = { continuityStore: store, candidateModels: ["gpt-6.1-sol-20261001:medium"], now: NOW };
  const first = await rtHooksTurn(options, [{ role: "user", content: REALTIME_FIRST_TURN }], { realtime_loop: true });
  assert.equal(first.state.route.targetModel, "gpt-6.1-sol-20261001");
  store.recordResponse(first.state.sessionKey, { usage: { input_tokens: 2000, output_tokens: 300 } }, {}, NOW);
  const second = await rtHooksTurn(options, rtTwoTurns("Redesign the distributed architecture and find the root cause of this deadlock."), {});
  assert.equal(second.state.route.continuation.held, true);
  assert.equal(second.state.route.targetModel, "gpt-6.1-sol-20261001");
  assert.equal(second.state.route.modelWithEffort, "gpt-6.1-sol-20261001:medium");
  assert.equal(second.state.route.realtimeLoop, true);
});

test("M3-R3: rerouting a realtime session keeps it realtime, so the effort is not ratcheted back up", async () => {
  const store = new SessionContinuityStore();
  const pool = ["gpt-6-astra", "claude-sonnet-5-5"];
  const { options } = rtScorerOptions(store, pool, { affinityMode: "turn" });
  const first = await rtHooksTurn(options, [{ role: "user", content: REALTIME_FIRST_TURN }]);
  assert.equal(first.state.route.realtimeLoop, true);
  const second = await rtHooksTurn(options, rtTwoTurns("Here is what the live page shows now."));
  assert.equal(second.state.route.continuation.why, "turn_mode");
  assert.equal(second.state.route.continuation.held, false);
  assert.equal(second.state.route.realtimeLoop, true);
  assert.notEqual(second.state.route.targetEffort, "xhigh");
  assert.equal(store.get(first.state.sessionKey, NOW).realtimeLoop, true);

  const failed = new SessionContinuityStore();
  const again = rtScorerOptions(failed, pool);
  const a = await rtHooksTurn(again.options, [{ role: "user", content: REALTIME_FIRST_TURN }]);
  const model = a.state.route.targetModel;
  const b = await rtHooksTurn({ ...again.options, unhealthyModels: [model] }, rtTwoTurns("Here is what the live page shows now."));
  assert.equal(b.state.route.continuation.why, "provider_unhealthy");
  assert.equal(b.state.route.realtimeLoop, true);
  assert.notEqual(b.state.route.targetModel, model);
  assert.notEqual(b.state.route.targetEffort, "xhigh");
});

test("M3-R3: middleware keeps realtimeLoop through holds and across stateless replay", async () => {
  const pool = ["gpt-5.3-codex-spark:medium", "claude-sonnet-5-5:medium"];
  const turn = async (store, messages, extraBody = {}) => {
    const ctx = { state: {}, options: { continuityStore: store, candidateModels: pool, now: NOW } };
    const body = { model: "agentcompass/auto", messages, ...extraBody };
    await middleware.onRequest(body, ctx);
    return { route: ctx.state.route, sessionKey: ctx.state.sessionKey, body };
  };
  const m1 = [{ role: "user", content: REALTIME_FIRST_TURN }];
  const m2 = [...m1, { role: "assistant", content: "ok" }, { role: "user", content: "here is the live state now" }];
  const m3 = [...m2, { role: "assistant", content: "ok" }, { role: "user", content: "and here is the next state" }];
  const store = new SessionContinuityStore();
  const first = await turn(store, m1, { realtime_loop: true });
  assert.equal(first.route.realtimeLoop, true);
  const second = await turn(store, m2);
  assert.equal(second.route.continuation.held, true);
  assert.equal(second.route.realtimeLoop, true);
  assert.equal(second.route.targetModel, first.route.targetModel);
  assert.equal(second.route.targetEffort, first.route.targetEffort);

  const replay = await turn(new SessionContinuityStore(), m3, { realtime_loop: true });
  assert.equal(replay.route.realtimeLoop, true);
  assert.equal(replay.route.continuation.held, false);
  assert.equal(replay.route.continuation.why, "cold");
  assert.equal(replay.route.continuation.replayed, true);
  assert.equal(replay.route.targetModel, first.route.targetModel);
});

test("Sandbox: realtime_loop stops at the gateway and never reaches the upstream", async () => {
  const pool = ["gpt-6-astra:medium", "claude-sonnet-5-5:medium"];
  const run = async (extra) => {
    const ctx = { state: {}, options: { continuityStore: new SessionContinuityStore(), candidateModels: pool, now: NOW } };
    const body = { messages: [{ role: "user", content: "Tune the live loop" }], ...extra };
    const result = await middleware.onRequest(body, ctx);
    return { body, result, route: ctx.state.route };
  };
  for (const flag of [true, false]) {
    const claimed = await run({ model: "agentcompass/auto", realtime_loop: flag });
    assert.equal(claimed.route.realtimeLoop, flag, "the router read the flag first");
    assert.ok(!("realtime_loop" in claimed.body));
    const named = await run({ model: "upstream/gpt-6-luna", realtime_loop: flag });
    assert.equal(named.result, named.body, "a hook only passes on a body it returns");
    assert.ok(!("realtime_loop" in named.body));
  }
  const needs = await run({ model: "agentcompass/auto", needs: { realtime_loop: true } });
  assert.equal(needs.route.realtimeLoop, true);
  assert.ok(!("needs" in needs.body));
  const otherNeeds = await run({ model: "agentcompass/auto", needs: { realtime_loop: true, visual: false } });
  assert.deepEqual(otherNeeds.body.needs, { visual: false });
  assert.equal((await run({ model: "upstream/gpt-6-luna" })).result, undefined, "a request without the flag is left alone");
  const notAFlag = await run({ model: "upstream/gpt-6-luna", realtime_loop: "yes" });
  assert.equal(notAFlag.result, undefined);
  assert.equal(notAFlag.body.realtime_loop, "yes");
});

test("Sandbox: pin options reach routing through the middleware and warn in the log", async (t) => {
  const warn = t.mock.method(console, "warn", () => {});
  const pool = ["upstream/gemini-4-argon-high", "upstream/gpt-6-astra"];
  const run = async (extra, pins) => {
    const ctx = { state: {}, options: { continuityStore: new SessionContinuityStore(), candidateModels: pool, now: NOW, ...pins } };
    const body = { model: "agentcompass/auto", messages: [{ role: "user", content: "Tune the live loop" }], ...extra };
    await middleware.onRequest(body, ctx);
    return body.model;
  };
  const pin = { pinnedModel: "upstream/gemini-4-argon-high" };
  assert.match(await run({ realtime_loop: true }, {}), /^upstream\/gpt-6-astra:/);
  assert.equal(warn.mock.callCount(), 0);
  assert.equal(await run({ realtime_loop: true }, pin), "upstream/gemini-4-argon-high:high");
  assert.equal(warn.mock.callCount(), 1);
  assert.match(warn.mock.calls[0].arguments[0], /^agentcompass: pinned .*below the realtime_loop floor/);
  assert.equal(await run({}, pin), await run({}, {}), "a pin does nothing without realtime_loop");
  assert.equal(warn.mock.callCount(), 1);
});

test("m1-R4: a middleware with its own store keeps the realtime flag of the hooks' route", async () => {
  const pool = ["gpt-6-astra", "claude-sonnet-5-5"];
  const hooksStore = new SessionContinuityStore();
  const middlewareStore = new SessionContinuityStore();
  const messages = [{ role: "user", content: REALTIME_FIRST_TURN }];
  const ctx = { state: {}, options: { continuityStore: hooksStore, candidateModels: pool, now: NOW } };
  await hooks.onRequest({ model: "agentcompass/auto", messages, realtime_loop: true }, ctx);
  const route = ctx.state.route;
  assert.equal(route.realtimeLoop, true);
  middleware.onRequest({ model: "agentcompass/auto", messages }, { state: { route }, options: { continuityStore: middlewareStore, candidateModels: pool, now: NOW } });
  assert.equal(middlewareStore.get(route.continuation.sessionKey, NOW).realtimeLoop, true);
  assert.equal(hooksStore.get(route.continuation.sessionKey, NOW).realtimeLoop, true);
});

const rtClassifierSays = (realtime_loop) => {
  const answer = JSON.stringify({ multi_task: false, task_count: "1", reasoning_depth: 2, code_complexity: 2, context_scale: 2, divergence: 1, hallucination_risk: 2, error_cost: 2, interaction_style: "engineering", needs_visual: false, needs_gui: false, realtime_loop, work_kind: "debug_review", confidence: 0.9 });
  const reply = JSON.stringify({ choices: [{ message: { content: answer } }] });
  const fetch = async () => ({ ok: true, status: 200, text: async () => reply, json: async () => JSON.parse(reply) });
  return { scorerType: "openai_compatible", allowExternalScorer: true, scorerBaseUrl: "http://127.0.0.1:9/v1", scorerApiKey: "k", scorerModel: "m", fetch, useCache: false };
};

test("P1: a Tandem middleware with its own store follows a classifier-only realtime route on an effort-suffixed pool", async () => {
  const pools = {
    suffixed: ["claude-opus-5-5:medium", "gemini-4-argon-high:high", "gpt-6-astra:high", "claude-sonnet-5-5:high"],
    bare: ["claude-opus-5-5", "gemini-4-argon-high", "gpt-6-astra", "claude-sonnet-5-5"],
  };
  const tandem = async (pool, shared, realtime) => {
    const hooksStore = new SessionContinuityStore();
    const store = shared ? hooksStore : new SessionContinuityStore();
    const options = { candidateModels: pool, now: NOW, ...rtClassifierSays(realtime) };
    const body = { model: "agentcompass/auto", messages: [{ role: "user", content: "Find the root cause of the race in the scheduler across modules" }] };
    const ctx = { state: {}, options: { ...options, continuityStore: hooksStore } };
    await hooks.onRequest(body, ctx);
    const route = ctx.state.route;
    middleware.onRequest(body, { state: { route }, options: { ...options, continuityStore: store } });
    return { route, final: body.model, stored: store.get(route.continuation.sessionKey, NOW)?.realtimeLoop };
  };
  for (const [name, pool] of Object.entries(pools)) {
    const own = await tandem(pool, false, true);
    assert.equal(own.route.realtimeLoop, true, name);
    assert.equal(own.final, `${own.route.targetModel}:${own.route.targetEffort}`, name);
    assert.equal(own.final, (await tandem(pool, true, true)).final, name);
    assert.equal(own.stored, true, name);
    assert.doesNotMatch(own.final, /^gemini-4-argon-high/, name);
  }
  const plain = await tandem(pools.suffixed, false, false);
  assert.equal(plain.final, `${plain.route.targetModel}:${plain.route.targetEffort}`);
});

test("m2-R4: onProvider and the decision provider honor pins, with or without realtime_loop", async (t) => {
  t.mock.method(console, "warn", () => {});
  const pool = ["gemini-4-argon-high", "gpt-6-astra"];
  const pin = { pinnedModel: "gemini-4-argon-high" };
  const provider = async (extra, pins) => {
    const options = { continuityStore: new SessionContinuityStore(), candidateModels: pool, modelsByProvider: { p: pool }, routePerProvider: true, now: NOW, ...pins };
    const ctx = { state: {}, options };
    const body = { model: "agentcompass/auto", messages: [{ role: "user", content: "Tune the live loop" }], ...extra };
    await middleware.onRequest(body, ctx);
    middleware.onProvider(body, { ...ctx, provider: { id: "p" } });
    return body.model;
  };
  assert.equal(await provider({ realtime_loop: true }, pin), "gemini-4-argon-high:high");
  assert.match(await provider({ realtime_loop: true }, {}), /^gpt-6-astra:/);
  assert.equal(await provider({}, pin), "gemini-4-argon-high:high");

  const questions = { effort: { type: "level" }, intent: { criteria: { "gemini-4-argon-high": "Argon", "gpt-6-astra": "Astra", none: "none of these" } } };
  const effort = async (state, pins) =>
    (await index.handleSystemOneRequest({ model: "jev-6b-int8", state: { message: "Tune the live loop", ...state }, questions }, { candidateModels: pool, ...pins })).answers.effort;
  assert.deepEqual(await effort({ realtime_loop: true }, { ...pin, pinnedEffort: "high" }), await effort({ realtime_loop: true }, { candidateModels: ["gemini-4-argon-high"], ...pin }));
  assert.notDeepEqual(await effort({ realtime_loop: true }, pin), await effort({ realtime_loop: true }, {}));

  const spent = { kind: "subscription", quota: { windows: [{ name: "7d", used: 100, remaining: 0, resetsAt: new Date(Date.now() + 48 * 3600 * 1000).toISOString() }] } };
  const quotas = Object.fromEntries(["codex", "gemini", "antigravity", "openai", "google"].map((p) => [`${p}/a@example.com`, spent]));
  const spentRoute = async (pins) =>
    (await index.handleSystemOneRequest({ model: "jev-6b-int8", state: { message: "Tune the live loop", realtime_loop: true, quotas }, questions }, { candidateModels: pool, decisionProviderDetails: true, ...pins })).agentcompass;
  for (const pins of [pin, {}]) {
    const route = await spentRoute(pins);
    assert.equal(route.blocked, true);
    assert.equal(route.targetModel, null);
    assert.ok(route.warnings.length > 0);
  }
});

test("m3-R4: the slow pin warning names the model the speed was read for", () => {
  const warningOf = (candidate, pinnedModel) => selectBestRoute(RT_PROFILE, { candidateModels: [candidate], pinnedModel, now: NOW }).warnings[0];
  const row = (m) => allocator.bundleResponsiveness({ model: m, effort: "high" }).value.toFixed(2);
  assert.match(warningOf("claude-opus-4-6:high", "claude-opus-4-6"), new RegExp(`\\(claude-opus-4-6@high\\) responsiveness ${row("claude-opus-4-6")}`));
  assert.match(warningOf("gemini-4-argon-high:high", "gemini-4-argon-high"), /\(gemini-4-argon-high@high\)/);
  assert.match(warningOf("grok-bot-20260101", "grok-bot-20260101"), /\(grok-bot@high\) responsiveness is not_applicable/);
});

const SPARK = "gpt-5.3-codex-spark";

test("Spark: every spelling has its own conservative entry and none resolves to gpt-6.1-sol", () => {
  for (const name of [SPARK, `openai/${SPARK}`, `codex/${SPARK}:medium`, "OpenAI/GPT-5.3-Codex-Spark", `${SPARK}-20260212`, `${SPARK}@20260212`, `${SPARK}-preview`, `${SPARK}-latest`]) {
    const { canonical, spec } = catalog.resolveModelSpec(name);
    assert.equal(canonical, SPARK, name);
    assert.equal(spec.tier, 3.0, name);
  }
  const spark = catalog.resolveModelSpec(SPARK).spec;
  assert.match(spark.basis, /Not in AgentCompass accounts\.json/);
  assert.equal(catalog.resolveModel("gpt-5.3-codex").source, "unknown");
  for (const name of ["spark", "gemini-spark", "grok-spark", "gpt-5.3-spark", "gpt-5.3-codex_spark"]) assert.notEqual(catalog.resolveModelSpec(name).canonical, SPARK, name);
  assert.ok(rankCandidates(RT_PROFILE, { now: NOW }).allCandidates.every((c) => c.canonicalModel !== SPARK));
  const bundle = allocator.bundleResponsiveness({ model: SPARK, effort: "high" });
  assert.deepEqual([bundle.value, bundle.source], [0.52, "default"]);
});
