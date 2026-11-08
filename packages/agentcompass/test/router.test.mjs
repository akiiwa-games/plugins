import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";

import {
  bundleHarness,
  bundleResponsiveness,
  burnScale,
  computeTurnSeconds,
  effortAbility,
  effortValue,
  evaluateCandidate,
  isCandidatePinned,
  lookupCurveResponsiveness,
  namedRoute,
  rankCandidates as rankProfile,
  resolveRealtimeEffort,
  responsivenessConfig,
  selectBestRoute as selectForProfile,
  splitEffortFromModelId,
  turnSecondsToResponsiveness,
} from "../src/allocator.js";
import {
  CANONICAL_ROUTES,
  DEFAULT_MODELS,
  DEFAULT_SCORING_POLICY,
  DISPATCH_EFFORTS,
  EFFORT_CURVES,
  EFFORT_RESPONSIVENESS,
  MODEL_ALIASES,
  canonicalModel,
  resolveModelSpec,
  resolveResponsivenessModelKey,
} from "../src/catalog.js";
import {
  CACHE_COLD_MS,
  CHATGPT_CODEX_COLD_MS,
  SessionContinuityStore,
  WORKER_WORN_TOKENS,
  analyzeTurnState,
  coldAfterMs,
  computeSessionKey,
  evaluateProviderContinuation,
  firstWords,
  keepsEffort,
  sha256Hex,
  userSaidFromText,
} from "../src/continuity.js";
import {
  ROUTE_CARRIER_KEY,
  onRequest as hooksOnRequest,
  onResponse as hookOnResponse,
} from "../src/hooks.js";
import {
  AGENTCOMPASS_ICON_DATA_URI,
  AgentCompassRouterPlugin,
  DECISION_MODEL_ID,
  PROVIDER_ID,
  clearQuotasCache,
  evaluateModelQuota,
  handleSystemOneRequest as systemOneForState,
  loadLiveQuotas,
  normalizeQuotaSnapshot,
  orderSeatsByQuotaEconomics,
} from "../src/index.js";
import {
  onAccount as mwOnAccount,
  onProvider as mwOnProvider,
  onRequest as middlewareOnRequest,
  onResponse as mwOnResponse,
  strategies,
} from "../src/middleware.js";
import {
  PROFILE_FLAGS,
  REALTIME_LOOP_QUESTION,
  TASK_PROFILE_SCHEMA,
  extractRequestFacts,
  validateProfile,
} from "../src/profiler-core.js";
import {
  SCORER_SYSTEM_PROMPT,
  buildJevScorerPayload,
  clearProfileCache,
  parseJevScorerAnswer,
  scoreTaskLive,
} from "../src/scorer-live.js";

import { answerRequest, answerSystemOne, classifierAnswer } from "./classifier-stub.mjs";

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

// Routing starts from a task profile: these send a text the way a client does, with the classifier's answer for it.
const answered = (task) => (task?.reasoning_depth || task?.profile ? task : { ...task, profile: classifierAnswer(task) });
const rankCandidates = (task, options) => rankProfile(answered(task ?? { objective: "" }), options);
const selectBestRoute = (task, options) => selectForProfile(answered(task ?? { objective: "" }), options);
const hookOnRequest = async (body, ctx = {}) => {
  answerRequest(body, ctx);
  return hooksOnRequest(body, ctx);
};
const mwOnRequest = (body, ctx = {}) => {
  answerRequest(body, ctx);
  return middlewareOnRequest(body, ctx);
};
const handleSystemOneRequest = async (body, options = {}) => systemOneForState(answerSystemOne(body, options), options);

test("1.1 Routine bounded bulk work selects lightweight model (gpt-6-luna) and penalizes expensive overqualification", () => {
    const profile = classifierAnswer({
      objective: "Perform a mechanical batch rename of `foo_bar` to `fooBar` across the JSON fixtures using a uniform rule.",
    });
    assert.equal(profile.work_kind, "bounded_bulk_work");
    assert.equal(profile.reasoning_depth, 1);

    const decision = selectBestRoute(profile);
    assert.equal(decision.ok, true);
    assert.equal(decision.difficulty, "routine");
    assert.equal(decision.canonicalModel, "gpt-6-luna");
    assert.equal(decision.targetEffort, "high");

    // Verify that expensive frontier models like gpt-6-astra or claude-fable-5-1 received overqualification penalty
    const ranked = rankCandidates(profile);
    const astraHigh = ranked.ranking.find((c) => c.canonicalModel === "gpt-6-astra" && c.effort === "high");
    assert.ok(astraHigh, "gpt-6-astra:high should be in eligible ranking");
    assert.ok(astraHigh.components.overqualification > 0, "gpt-6-astra:high should pay overqualification penalty on routine work");
  });

test("1.2 Convergent single-module production coding selects gpt-6.1-sol via module_precision DNA", () => {
    const profile = classifierAnswer({
      objective: "Implement a standalone self-contained LRU cache algorithm module in TypeScript with unit tests.",
    });
    assert.equal(profile.work_kind, "self_contained_code");
    const decision = selectBestRoute(profile);
    assert.equal(decision.canonicalModel, "gpt-6.1-sol");
    assert.equal(decision.targetEffort, "high");
  });

test("1.3 Coupled cross-module deadlock & invariant audit selects claude-opus-5-5 via global_rigor + anti_hallucination", () => {
    const profile = classifierAnswer({
      objective:
        "Conduct a read-only security audit and cross-module deadlock / race condition root-cause review across the gateway subsystem.",
    });
    assert.equal(profile.work_kind, "debug_review");
    assert.equal(profile.code_complexity, 0, "read-only review must force code_complexity=0");
    assert.equal(profile.reasoning_depth, 3);

    const decision = selectBestRoute(profile);
    assert.equal(decision.difficulty, "coupled");
    assert.equal(decision.canonicalModel, "claude-opus-5-5");
    assert.ok(["high", "xhigh", "max"].includes(decision.targetEffort));
  });

test("1.4 Open-ended divergent architecture ideation selects gemini-4-argon-high via divergence DNA (0.96)", () => {
    const profile = classifierAnswer({
      objective:
        "Brainstorm and formulate a novel creative gameplay scheme and first-principles roadmap for an AI agent simulation game.",
      work_kind: "direction_reasoning",
    });
    assert.equal(profile.divergence, 3);
    assert.equal(profile.reasoning_depth, 3);

    const decision = selectBestRoute(profile);
    assert.equal(decision.workKind, "direction_reasoning");
    assert.equal(decision.canonicalModel, "gemini-4-argon-high");
  });

test("1.5 Frontier mathematical proof (reasoning_depth=4, floor=4.2) selects claude-opus-5-5", () => {
    const profile = classifierAnswer({
      objective: "Provide a formal mathematical proof or refute the optimal-stopping theorem bound for this Markov chain.",
    });
    assert.equal(profile.reasoning_depth, 4);
    const decision = selectBestRoute(profile);
    assert.equal(decision.difficulty, "frontier");
    assert.equal(decision.canonicalModel, "claude-opus-5-5");
  });

test("1.6 GUI computer-use task routes to gpt-6-astra (#1 GUI grounding) and excludes text-only models", () => {
    const profile = classifierAnswer({
      objective: "Operate the desktop app GUI: click the Settings button, take a screenshot, and toggle the proxy switch.",
    });
    assert.equal(profile.work_kind, "computer_use");
    assert.equal(profile.needs_gui, true);
    assert.equal(profile.needs_visual, true);

    const ranked = rankCandidates(profile);
    const grok = ranked.allCandidates.find((c) => c.canonicalModel === "grok-4.7");
    assert.ok(grok.excluded.includes("task needs visual input"), "text-only grok-4.7 must be excluded on visual/GUI work");
    assert.equal(ranked.selected.canonicalModel, "gpt-6-astra");

    const visualReview = classifierAnswer({
      objective: "Compare screenshots of the login dialog and inspect the screenshot for layout misalignment.",
    });
    assert.equal(visualReview.work_kind, "visual_analysis");
    assert.equal(visualReview.needs_visual, true);
    assert.equal(visualReview.needs_gui, false);
});

test("2. Gateway Middleware (`magpie.middleware` [`moejs` 250ms]) and optional `hooks.js` state routing rewrite model & effort without polluting `body`", async () => {
  const originalBody = {
    model: "agentcompass/auto",
    stream: true,
    messages: [
      {
        role: "user",
        content: "Review the diff and diagnose the cross-module race condition in internal/gateway/decide.go (read-only review).",
      },
    ],
    tools: [{ type: "function", function: { name: "read_file" } }],
  };

  let reportedOutcome = null;
  let reportedBytes = 0;
  const hookCtx = {
    protocol: "chat",
    agent: "claude-code",
    options: {
      gatewayTimeoutMs: 8000,
      candidateModels: ["claude-opus-5-5", "gpt-6.1-sol", "gemini-3.8-flash-high"],
    },
    state: {},
    report(code, bytes) {
      reportedOutcome = code;
      reportedBytes = bytes;
    },
  };

  // Stage 1: Run Async Bun Hook (`src/hooks.js`)
  const bodyClone = structuredClone(originalBody);
  const afterHook = await hookOnRequest(bodyClone, hookCtx);
  assert.equal(afterHook, undefined, "hookOnRequest must not mutate or replace body by default");
  assert.equal(ROUTE_CARRIER_KEY in bodyClone, false, "hookOnRequest must never inject __agentcompass_route by default");
  assert.equal(reportedOutcome, "ready");
  assert.ok(reportedBytes > 0);
  assert.ok(hookCtx.state.route, "route decision must be stored in ctx.state.route");
  assert.equal(hookCtx.state.route.targetModel, "claude-opus-5-5");

  // Stage 2: Run Sync `moejs` Middleware (`src/middleware.js`)
  const mwCtx = {
    protocol: "chat",
    agent: "claude-code",
    options: {
      injectProtocolEffort: true,
      candidateModels: ["claude-opus-5-5", "gpt-6.1-sol", "gemini-3.8-flash-high"],
    },
    state: {},
  };
  const afterMw = mwOnRequest(bodyClone, mwCtx);
  assert.ok(afterMw, "mwOnRequest should return mutated body");
  assert.equal(ROUTE_CARRIER_KEY in afterMw, false, "carrier must never be present on body");
  assert.match(afterMw.model, /^claude-opus-5-5:(high|xhigh|max)$/);
  assert.ok(["high", "xhigh"].includes(afterMw.reasoning_effort));

  // Verify `onAccount` works with Magpie's real `{ref, provider, name, kind, model}` account shape
  const skipRes = mwOnAccount(
    {
      ref: "a1b2c3d4e5f60718",
      provider: "openai",
      name: "OpenAI Plus",
      kind: "codex",
      model: "gpt-6.1-sol",
    },
    {
      state: mwCtx.state,
      options: { accountsByWorkKind: { debug_review: ["anthropic"] } },
    },
  );
  assert.deepEqual(skipRes, { skip: true });
});

test("3.1 Live OpenAI-compatible scorer parses 10-field JSON profile and caches it when explicitly opted in", async () => {
  clearProfileCache();
    let fetchCalls = 0;
    const mockFetch = async (_url, init) => {
      fetchCalls++;
      assert.equal("User-Agent" in (init.headers || {}), false, "Must not send fake Magpie-Router/1.0 User-Agent");
      return new Response(
        JSON.stringify({
          choices: [
            {
              message: {
                content: JSON.stringify({
                  work_kind: "frontend_implementation",
                  reasoning_depth: 2,
                  code_complexity: 2,
                  context_scale: 1,
                  divergence: 2,
                  hallucination_risk: 2,
                  error_cost: 2,
                  needs_visual: true,
                  needs_gui: false,
                  local_only: true,
                  interaction_style: "engineering",
                  expected_hours: 1.0,
                  hours_confidence: "high",
                  urgency: "unknown",
                  multi_task: false,
                  task_count: "1",
                }),
              },
            },
          ],
        }),
        { status: 200 },
      );
    };

    const task = { objective: "Build the interactive React analytics dashboard component from the Figma spec." };
    const first = await scoreTaskLive(task, {
      allowExternalScorer: true,
      scorerType: "openai_compatible",
      scorerBaseUrl: "http://127.0.0.1:4096",
      scorerModel: "gemini-3.8-flash-high",
      fetch: mockFetch,
    });
    assert.equal(first.meta.source, "system_one_live");
    assert.equal(first.profile.work_kind, "frontend_implementation");
    assert.equal(fetchCalls, 1);

    // Second call with identical task hits LRU cache in 0ms without calling fetch
    const second = await scoreTaskLive(task, {
      allowExternalScorer: true,
      scorerType: "openai_compatible",
      scorerBaseUrl: "http://127.0.0.1:4096",
      scorerModel: "gemini-3.8-flash-high",
      fetch: mockFetch,
    });
    assert.equal(second.meta.cached, true);
    assert.equal(fetchCalls, 1);
  });

test("3.2 Timeout or HTTP failure leaves no profile, and says why", async () => {
    clearProfileCache();
    const failingFetch = async () => {
      throw new Error("Simulated timeout");
    };
    const res = await scoreTaskLive(
      { objective: "Fix the memory leak in the WebSocket connection pool." },
      {
        allowExternalScorer: true,
        scorerType: "openai_compatible",
        scorerBaseUrl: "http://127.0.0.1:4096",
        fetch: failingFetch,
      },
    );
    assert.equal(res.meta.source, "unavailable");
    assert.equal(res.meta.reason, "Simulated timeout");
    assert.equal(res.profile, null);
});

test("4. Native Magpie `/v1/systemone` Decision Handler (`handleSystemOneRequest`)", async () => {
  // 4a. Test `jevLevelled` probe (`decide.go:310`): whether the labels are a scale is the classifier's to say
  const probe = {
    model: DECISION_MODEL_ID,
    state: { kinds: ["简单日常", "常规开发", "复杂跨模块架构"] },
    questions: {
      levels: { type: "noul" },
      work0: { type: "score" },
      work1: { type: "score" },
      work2: { type: "score" },
    },
  };
  const noClassifier = await handleSystemOneRequest(probe);
  assert.ok(noClassifier.answers.levels.noul < 0.5, "without a classifier the labels are plain kinds");
  assert.equal(noClassifier.answers.work0, undefined);
  assert.equal(noClassifier.agentcompass.routed, false);
  const levelProbe = await handleSystemOneRequest(probe, {
    allowExternalScorer: true,
    scorerType: "jev",
    jevBaseUrl: "http://127.0.0.1:9999",
    fetch: async (_url, init) => {
      assert.deepEqual(JSON.parse(init.body).state.kinds, probe.state.kinds);
      return new Response(
        JSON.stringify({ answers: { levels: { noul: 0.9 }, work0: { score: 0 }, work1: { score: 1 }, work2: { score: 2 } } }),
        { status: 200 },
      );
    },
  });
  assert.ok(levelProbe.answers.levels.noul >= 0.5, "difficulty scale intents must be recognized as levels");
  assert.ok(levelProbe.answers.work0.score < levelProbe.answers.work2.score);

  // 4b. Test `jevBody` turn classification (`decide.go:167`)
  const turnDecision = await handleSystemOneRequest({
    model: DECISION_MODEL_ID,
    state: {
      turn: "Please conduct a read-only security audit and race condition review across the authentication middleware.",
    },
    questions: {
      intent: {
        type: "choice",
        criteria: {
          frontend_implementation: "Frontend UI work",
          debug_review: "Bug diagnosis and code review",
          bounded_bulk_work: "Simple batch edits",
          none: "None of the above",
        },
      },
      effort: {
        type: "score",
        criteria: ["low", "medium", "high", "xhigh"],
      },
    },
  });
  assert.equal(turnDecision.answers.intent.choice, "debug_review");
  assert.ok(turnDecision.answers.intent.confidence >= 0.6, "confidence must clear Magpie's jevSure (0.6)");
  assert.ok(turnDecision.answers.effort.score >= 2.0, "coupled review must select high/xhigh effort (>= 2.0)");

  // 4c. Default plugin export does NOT inject a fake auth/provider entry unless explicitly opted in
  assert.equal(PROVIDER_ID, "agentcompass");
  const defaultPlugin = await AgentCompassRouterPlugin({}, {});
  const emptyCfg = {};
  await defaultPlugin.config(emptyCfg);
  assert.equal(emptyCfg.provider, undefined, "Default gateway plugin must not register a fake provider");
  assert.equal(defaultPlugin.auth, undefined, "Default gateway plugin must not register a fake auth hook");

  // Opt-in decision provider mode
  const optInPlugin = await AgentCompassRouterPlugin({}, { enableDecisionProvider: true });
  const cfg = {};
  await optInPlugin.config(cfg);
  assert.equal(cfg.provider[PROVIDER_ID].name, "AgentCompass");
  assert.equal(cfg.provider[PROVIDER_ID].decide, true);
  assert.equal(cfg.provider[PROVIDER_ID].models[DECISION_MODEL_ID].decides, true);

  const { fetch: decisionFetch, baseURL } = await optInPlugin.auth.loader(async () => ({}), {});
  const httpRes = await decisionFetch(`${baseURL}/systemone`, {
    method: "POST",
    body: JSON.stringify({
      model: DECISION_MODEL_ID,
      state: {
        turn: "Batch rename all `old_id` fields to `new_id` across fixtures.",
        profile: classifierAnswer("Batch rename all `old_id` fields to `new_id` across fixtures."),
      },
      questions: {
        intent: {
          type: "choice",
          criteria: {
            bounded_bulk_work: "Bulk edits",
            debug_review: "Debugging",
            none: "None",
          },
        },
        effort: { type: "score", criteria: ["low", "medium", "high", "xhigh"] },
      },
    }),
  });
  assert.equal(httpRes.status, 200);
  const json = await httpRes.json();
  assert.equal(json.answers.intent.choice, "bounded_bulk_work");
});

/**
 * What Magpie's plugin host (0.1.2804 through 0.1.2891, `internal/plugin/host.js`) needs before a
 * decision provider exists, emulated: the server function is called with plugins.json's options
 * (`fn(input, p.options)`); a provider is listed, signed in and served only when a hook's
 * `auth.provider` names it (`auths()` / `providers()`); `decide` and the model's `decides` come from
 * the config entry (`info()`); `magpie plugin login` takes a keyless `api` method's `authorize()`;
 * System One is posted to the loader's `baseURL` + `/systemone` through the loader's `fetch` (`load()`).
 * A `config` entry with no auth hook passed the old test and was never listed (issue #1).
 */
async function magpieHostView(serverFn, options) {
  const hooks = (await serverFn({}, options)) ?? {};
  const config = {};
  if (typeof hooks.config === "function") await hooks.config(config);
  const auth = hooks.auth?.provider ? hooks.auth : null;
  if (!auth) return { providers: [] };
  const cfg = config.provider?.[auth.provider] ?? {};
  const keyless = (auth.methods ?? []).filter(
    (m) => m.type === "api" && m.keyless === true && typeof m.authorize === "function"
  );
  const signedIn = keyless.length ? await keyless[0].authorize() : null;
  const loaded = signedIn?.type === "success" && auth.loader ? await auth.loader(async () => signedIn, cfg) : {};
  return {
    providers: [
      {
        id: auth.provider,
        name: cfg.name ?? auth.provider,
        decide: cfg.decide === true,
        decisionModels: Object.entries(cfg.models ?? {}).filter(([, m]) => m.decides === true).map(([id]) => id),
        keyless: keyless.length > 0,
        signedIn: signedIn?.type === "success",
        baseURL: typeof loaded.baseURL === "string" ? loaded.baseURL : "",
        fetch: typeof loaded.fetch === "function" ? loaded.fetch : null,
      },
    ],
  };
}

test("4d. Decision provider registers the way Magpie's host lists, signs in and serves it (issue #1)", async () => {
  const pluginEntry = (await import("../src/index.js")).default;
  assert.equal(pluginEntry.id, "agentcompass");

  for (const options of [{}, { enableDecisionProvider: false }, { enableDecisionProvider: "true" }]) {
    assert.deepEqual((await magpieHostView(pluginEntry.server, options)).providers, [], JSON.stringify(options));
  }

  const { providers } = await magpieHostView(pluginEntry.server, {
    enableDecisionProvider: true,
    injectProtocolEffort: true,
    routeAllModels: true,
  });
  assert.equal(providers.length, 1, "magpie plugin list must show the agentcompass provider");
  const [p] = providers;
  assert.equal(p.id, PROVIDER_ID);
  assert.equal(p.name, "AgentCompass");
  assert.equal(p.decide, true);
  assert.deepEqual(p.decisionModels, [DECISION_MODEL_ID], "agentcompass/agentcompass-systemone is a Classifier option");
  assert.ok(p.keyless && p.signedIn, "magpie plugin login agentcompass signs in without a key");
  assert.match(p.baseURL, /^https?:\/\//);
  assert.equal(typeof p.fetch, "function");

  const answered = await p.fetch(`${p.baseURL}/systemone`, {
    method: "POST",
    body: JSON.stringify({
      model: DECISION_MODEL_ID,
      state: {
        task: "Find the root cause of a deadlock across three services",
        profile: classifierAnswer("Find the root cause of a deadlock across three services"),
      },
      questions: { work_kind: { type: "choice" }, reasoning_depth: { type: "score" } },
    }),
  });
  assert.equal(answered.status, 200);
  const body = await answered.json();
  assert.equal(body.model, DECISION_MODEL_ID);
  assert.equal(typeof body.answers.work_kind.choice, "string");
  assert.equal(typeof body.answers.reasoning_depth.score, "number");

  // Only System One is served: a chat request sent to the provider is refused, never answered as a model.
  const chat = await p.fetch(`${p.baseURL}/chat/completions`, { method: "POST", body: "{}" });
  assert.equal(chat.status, 404);
  const viaRequest = await p.fetch(new Request(`${p.baseURL}/systemone`, { method: "POST", body: "{}" }));
  assert.equal(viaRequest.status, 200);
});

test("5. Pool strategy `agentcompass` / `agentcompass-pure` orders non-resting lowest-used seat indexes first", () => {
  const seats = [
    { key: "seat-resting", used: 10, resting: true },
    { key: "seat-busy", used: 80, resting: false },
    { key: "seat-fresh", used: 20, resting: false },
  ];
  const order1 = strategies.agentcompass("anthropic", seats, {});
  const order2 = strategies["agentcompass-pure"]("anthropic", seats, {});
  assert.deepEqual(order1, [2, 1, 0]);
  assert.deepEqual(order2, [2, 1, 0]);
  assert.deepEqual(order1.map((i) => seats[i].key), ["seat-fresh", "seat-busy", "seat-resting"]);
});

test("6. Magpie Plugin Market & Registry Contract (`package.json`, `icon`, `options`, `registry.json`)", async () => {
  const pkg = JSON.parse(fs.readFileSync(path.join(REPO_ROOT, "package.json"), "utf8"));
  assert.ok(["agentcompass", "@magpie-community/gateway-agentcompass"].includes(pkg.name));
  assert.equal(pkg.magpie.name, "AgentCompass");
  assert.equal(pkg.magpie.hooks, undefined, "magpie.hooks must not be declared alongside magpie.middleware");
  assert.equal(pkg.magpie.middleware, "./src/middleware.js");
  assert.ok(pkg.magpie.icon.startsWith("data:image/svg+xml;base64,"));
  assert.equal(pkg.magpie.icon, AGENTCOMPASS_ICON_DATA_URI);

  const decodedSvg = Buffer.from(pkg.magpie.icon.split(",")[1], "base64").toString("utf8");
  assert.ok(decodedSvg.includes("<svg"), "Decoded icon must be valid SVG for Magpie StoreIcon()");

  for (const kw of ["magpie", "magpie-middleware", "magpie-routing", "routing", "load-balancing", "llm-gateway", "agentcompass"]) {
    assert.ok(pkg.keywords.includes(kw), `package.json keywords must include "${kw}"`);
  }
  assert.equal(pkg.keywords.includes("provider"), false, "Middleware plugin must not claim 'provider' keyword");
  assert.equal(pkg.keywords.includes("auth"), false, "Middleware plugin must not claim 'auth' keyword");
  assert.equal(
    pkg.keywords.includes("pi-package"),
    false,
    "Plugins must not include 'pi-package' in keywords"
  );

  const standaloneListingPath = path.join(REPO_ROOT, "docs", "magpie-community-registry-entry.json");
  const monorepoRegistryPath = path.resolve(REPO_ROOT, "..", "..", "registry.json");
  if (fs.existsSync(standaloneListingPath)) {
    const listing = JSON.parse(fs.readFileSync(standaloneListingPath, "utf8"));
    assert.ok(["agentcompass", "@magpie-community/gateway-agentcompass"].includes(listing.package));
    assert.equal(listing.name, "AgentCompass");
    assert.equal(listing.kind, "middleware");
    assert.equal("providers" in listing, false);
    assert.ok(listing.summary?.en && listing.summary?.zh);
  }
  if (fs.existsSync(monorepoRegistryPath)) {
    const reg = JSON.parse(fs.readFileSync(monorepoRegistryPath, "utf8"));
    const entry = reg.plugins.find((p) => p.package === pkg.name);
    assert.ok(entry, `registry.json must include an entry for ${pkg.name}`);
    assert.equal(entry.name, "AgentCompass");
    assert.equal(entry.kind, "middleware");
    assert.equal(entry.community, true);
    assert.ok(entry.summary?.en && entry.summary?.zh);
  }
});

test("7.1 Scenario 1 (`why: first` & subagent isolation): New session runs intent routing and isolates `x-claude-code-agent-id` subagents", async () => {
    const store = new SessionContinuityStore();
    const t0 = 1_700_000_000_000;
    let scorerCalls = 0;

    // Verify pure-JS SHA-256 against NIST test vector ("abc")
    assert.equal(
      sha256Hex("abc"),
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );

    const req1 = {
      model: "agentcompass/auto",
      messages: [
        {
          role: "user",
          content:
            "Audit our distributed Raft consensus state machine across 4 modules to find the split-brain deadlock root cause.",
        },
      ],
    };

    const ctx1 = {
      protocol: "chat",
      headers: { "x-session-id": "sess-main-1" },
      state: {},
      options: {
        continuityStore: store,
        now: t0,
        allowExternalScorer: true,
        scorerType: "openai_compatible",
        scorerBaseUrl: "http://127.0.0.1:9999/v1",
        useCache: false,
        fetch: async () => {
          scorerCalls += 1;
          return new Response(
            JSON.stringify({
              choices: [
                {
                  message: {
                    content: JSON.stringify({
                      work_kind: "debug_review",
                      reasoning_depth: 3,
                      code_complexity: 3,
                      context_scale: 3,
                      divergence: 1,
                      hallucination_risk: 3,
                      error_cost: 3,
                      needs_visual: false,
                      needs_gui: false,
                      local_only: true,
                      interaction_style: "engineering",
                      expected_hours: 2.0,
                      hours_confidence: "high",
                      urgency: "normal",
                      multi_task: false,
                      task_count: "1",
                    }),
                  },
                },
              ],
            }),
            { status: 200 }
          );
        },
      },
    };
    await hookOnRequest(req1, ctx1);

    assert.equal(scorerCalls, 1, "First turn of a new session must invoke intent scorer once");
    const carrier1 = ctx1.state.route;
    assert.equal(carrier1.continuation.held, false);
    assert.equal(carrier1.continuation.why, "first");
    assert.equal(carrier1.canonicalModel, "claude-opus-5-5");

    // Simulate subagent sharing the same `x-session-id` but with `x-claude-code-agent-id`
    const subagentReq = {
      model: "agentcompass/auto",
      messages: [
        {
          role: "user",
          content: "Batch rename `old_var` to `new_var` across JSON fixtures.",
        },
      ],
    };
    const mainKey = computeSessionKey(req1, { headers: { "x-session-id": "sess-main-1" } });
    const subKey = computeSessionKey(subagentReq, {
      headers: {
        "x-session-id": "sess-main-1",
        "x-claude-code-agent-id": "subagent-haiku-worker",
      },
    });
    assert.notEqual(mainKey, subKey, "Subagent must have an isolated session key");

    await hookOnRequest(subagentReq, {
      protocol: "chat",
      headers: {
        "x-session-id": "sess-main-1",
        "x-claude-code-agent-id": "subagent-haiku-worker",
      },
      options: { continuityStore: store, now: t0 + 1000 },
    });

    // Main session must still hold `claude-opus-5-5`, unaffected by subagent
    assert.equal(store.get(mainKey, t0 + 1000).model, "claude-opus-5-5");
  });

test("7.2 Scenario 2 (`why: turn` mid-turn lock): Tool results & agent-injected `<system-reminder>` lock Provider/Model with 0 scorer calls", async () => {
    const store = new SessionContinuityStore();
    const t0 = 1_700_000_000_000;
    let scorerCalls = 0;
    const countingFetch = async () => {
      scorerCalls += 1;
      throw new Error("Scorer should NOT be called during mid-turn hold!");
    };

    // Turn 1: User starts a complex task on `claude-opus-5-5:high`
    const firstTurn = {
      model: "agentcompass/auto",
      messages: [
        {
          role: "user",
          content:
            "Refactor the distributed transaction coordinator across 4 microservices to fix the two-phase commit deadlock.",
        },
      ],
    };
    const ctx1 = {
      protocol: "chat",
      headers: { "x-session-id": "sess-turn-lock" },
      state: {},
      options: {
        continuityStore: store,
        now: t0,
        allowUnconstrainedCatalog: true,
      },
    };
    await hookOnRequest(firstTurn, ctx1);
    const routed1 = mwOnRequest(structuredClone(firstTurn), {
      protocol: "chat",
      headers: { "x-session-id": "sess-turn-lock" },
      state: ctx1.state,
      options: { continuityStore: store, now: t0, allowUnconstrainedCatalog: true },
    });
    assert.equal(routed1.model, "claude-opus-5-5:xhigh");

    // Step 2a: Assistant calls a tool, and the next request ends with `role: "tool"`
    const toolResultReq = {
      model: "agentcompass/auto",
      messages: [
        ...firstTurn.messages,
        {
          role: "assistant",
          content: null,
          tool_calls: [{ id: "call_1", type: "function", function: { name: "read_file", arguments: "{}" } }],
        },
        {
          role: "tool",
          tool_call_id: "call_1",
          content: "package coordinator // simple 10-line file",
        },
      ],
    };

    const ctx2a = {
      protocol: "chat",
      headers: { "x-session-id": "sess-turn-lock" },
      state: {},
      options: {
        continuityStore: store,
        now: t0 + 10 * 60 * 1000, // Even after 10 minutes (> 5m cache TTL), mid-turn `within: true` MUST still hold!
        allowExternalScorer: true,
        scorerType: "openai_compatible",
        scorerBaseUrl: "http://127.0.0.1:9999/v1",
        fetch: countingFetch,
      },
    };
    await hookOnRequest(toolResultReq, ctx2a);
    assert.equal(scorerCalls, 0, "Mid-turn tool result must make 0 scorer calls");
    assert.equal(ctx2a.state.route.continuation.held, true);
    assert.equal(ctx2a.state.route.continuation.why, "turn");
    assert.equal(ctx2a.state.route.continuation.within, true);
    assert.equal(ctx2a.state.route.modelWithEffort, "claude-opus-5-5:xhigh");

    // Step 2b: Agent injects a `role: "user"` message containing only `<system-reminder>` and `TodoWrite` noise
    const agentNoiseReq = {
      model: "agentcompass/auto",
      messages: [
        ...toolResultReq.messages,
        {
          role: "user",
          content:
            "<system-reminder>Remember to run tests before finishing.</system-reminder>\n\nThe TodoWrite tool hasn't been used recently.",
        },
      ],
    };
    assert.equal(userSaidFromText(agentNoiseReq.messages[3].content), "");
    const turnState2b = analyzeTurnState(agentNoiseReq);
    assert.equal(turnState2b.within, true, "Agent-injected reminder must keep within === true");
    assert.equal(turnState2b.turn, 1, "Agent-injected reminder must not increment user turn count");

    const ctx2b = {
      protocol: "chat",
      headers: { "x-session-id": "sess-turn-lock" },
      state: {},
      options: {
        continuityStore: store,
        now: t0 + 12 * 60 * 1000,
        allowExternalScorer: true,
        scorerType: "openai_compatible",
        scorerBaseUrl: "http://127.0.0.1:9999/v1",
        fetch: countingFetch,
      },
    };
    await hookOnRequest(agentNoiseReq, ctx2b);
    assert.equal(scorerCalls, 0);
    assert.equal(ctx2b.state.route.continuation.held, true);
    assert.equal(ctx2b.state.route.continuation.why, "turn");
  });

test("7.3 Scenario 3 (`why: cache` + `keepsEffort`): Warm KV cache holds model and ratchets effort, and a short turn is classified like any other", async () => {
    const store = new SessionContinuityStore();
    const t0 = 1_700_000_000_000;
    let scorerCalls = 0;

    // Turn 1: Complex task routes to `claude-opus-5-5:high`
    const turn1 = {
      model: "agentcompass/auto",
      messages: [
        {
          role: "user",
          content:
            "Audit the cross-module state synchronization protocol across gateway, worker, and store for race conditions.",
        },
      ],
    };
    const ctx1 = {
      protocol: "anthropic",
      headers: { "x-session-id": "sess-warm-cache" },
      state: {},
      options: { continuityStore: store, now: t0 },
    };
    await hookOnRequest(turn1, ctx1);
    assert.equal(ctx1.state.route.modelWithEffort, "claude-opus-5-5:xhigh");

    // Simulate response returning 4,200 tokens (>= 1024 CACHE_WORTH_TOKENS) and cache creation
    await hookOnResponse(
      {
        id: "msg_1",
        model: "claude-opus-5-5",
        usage: {
          input_tokens: 1800,
          cache_creation_input_tokens: 2100,
          output_tokens: 300,
        },
      },
      ctx1
    );

    // Turn 2 (2 minutes later, within 5m Anthropic cache TTL): User asks a trivial routine question
    // Because the 4,200-token KV cache is warm, it MUST hold `claude-opus-5-5:xhigh` (`why: "cache"`) without calling scorer!
    const turn2 = {
      model: "agentcompass/auto",
      messages: [
        ...turn1.messages,
        { role: "assistant", content: "Here is the protocol audit and synchronization state machine." },
        { role: "user", content: "Also rename the variable `sync_flag` to `isSynced` in that file." },
      ],
    };
    const ctx2 = {
      protocol: "anthropic",
      headers: { "x-session-id": "sess-warm-cache" },
      state: {},
      options: {
        continuityStore: store,
        now: t0 + 2 * 60 * 1000,
        allowExternalScorer: true,
        scorerType: "openai_compatible",
        scorerBaseUrl: "http://127.0.0.1:9999/v1",
        fetch: async () => {
          scorerCalls += 1;
          throw new Error("Should not call scorer while KV cache is warm!");
        },
      },
    };
    await hookOnRequest(turn2, ctx2);
    assert.equal(scorerCalls, 0);
    assert.equal(ctx2.state.route.continuation.held, true);
    assert.equal(ctx2.state.route.continuation.why, "cache");
    assert.equal(ctx2.state.route.modelWithEffort, "claude-opus-5-5:xhigh");

    // `/v1/systemone`: nothing in the text says "follow-up". The classifier sees the previous turn and answers, and
    // the effort still ratchets on the same kind.
    let classified = null;
    const sysOneFollowup = await handleSystemOneRequest(
      {
        model: DECISION_MODEL_ID,
        state: {
          message: "继续",
          previous_message_kind: "debug_review",
          previous_message_reasoning: "high",
          previous_user_message: "Audit the distributed lock manager.",
          previous_assistant_reply: "I found two lock ordering inversions. Shall I continue?",
        },
        questions: {
          intent: {
            type: "choice",
            criteria: {
              debug_review: "Debugging and review",
              bounded_bulk_work: "Routine bulk edits",
              none: "None",
            },
          },
          effort: {
            type: "score",
            criteria: ["low", "medium", "high", "xhigh"],
          },
        },
      },
      {
        decisionProviderDetails: true,
        allowExternalScorer: true,
        scorerType: "openai_compatible",
        scorerBaseUrl: "http://127.0.0.1:9999/v1",
        useCache: false,
        fetch: async (_url, init) => {
          scorerCalls += 1;
          classified = JSON.parse(JSON.parse(init.body).messages[1].content);
          return new Response(
            JSON.stringify({ choices: [{ message: { content: JSON.stringify({ ...classifierAnswer("Audit the distributed lock manager."), work_kind: "debug_review" }) } }] }),
            { status: 200 },
          );
        },
      }
    );
    assert.equal(scorerCalls, 1);
    assert.equal(classified.previous_user, "Audit the distributed lock manager.");
    assert.equal(sysOneFollowup.answers.intent.choice, "debug_review");
    assert.ok(sysOneFollowup.answers.effort.score >= 2.0, "Effort must stay ratcheted at high or above");
    assert.equal(sysOneFollowup.agentcompass.continuation.held, false);
    assert.equal(sysOneFollowup.agentcompass.continuation.why, "reclassified");

    // Verify `keepsEffort` directly: warm session on `xhigh` never drops to `low`
    const sessionSnap = store.get(ctx2.state.route.continuation.sessionKey, t0 + 2 * 60 * 1000);
    assert.equal(keepsEffort(sessionSnap, "low", 4200, t0 + 2 * 60 * 1000), "xhigh");
    // Verify ChatGPT/Codex & DeepSeek 1h cache TTL vs default 5m TTL
    assert.equal(coldAfterMs("claude-opus-5-5"), CACHE_COLD_MS);
    assert.equal(coldAfterMs("gpt-6.1-sol"), CHATGPT_CODEX_COLD_MS);
    assert.equal(coldAfterMs("deepseek-v4.1-flash-free"), CHATGPT_CODEX_COLD_MS);
  });

test("7.4 Scenario 4 (`why: capability_outgrown` / `outgrown` / `worn`): Modality, tier escalation, 95% context window, and 2M wear trigger re-routing", async () => {
    const store = new SessionContinuityStore();
    const t0 = 1_700_000_000_000;

    // 4a. Visual modality outgrowth: session starts on text-only `grok-4.7`, next turn sends an image
    const sessKeyVis = "sess-vis-outgrow";
    store.set(
      sessKeyVis,
      {
        model: "grok-4.7",
        effort: "medium",
        tokens: 4000,
        cached: true,
        healthy: true,
        turn: 1,
      },
      t0
    );
    const visReq = {
      model: "agentcompass/auto",
      session_id: sessKeyVis,
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "Inspect this UI screenshot and diagnose the overlapping flex layout bug." },
            { type: "image_url", image_url: { url: "https://example.com/bug.png" } },
          ],
        },
      ],
    };
    const ctxVis = {
      protocol: "chat",
      session_id: sessKeyVis,
      state: {},
      options: { continuityStore: store, now: t0 + 30_000 },
    };
    await hookOnRequest(visReq, ctxVis);
    assert.equal(ctxVis.state.route.continuation.held, false);
    assert.equal(ctxVis.state.route.continuation.why, "capability_outgrown");
    assert.equal(ctxVis.state.route.continuation.reasonDetail, "visual_modality_required");
    assert.notEqual(ctxVis.state.route.canonicalModel, "grok-4.7");

    // 4b. Tier escalation outgrowth: session starts on lightweight `claude-haiku-5-5` (tier 3.2),
    // next turn asks for a complex cross-module architecture & deadlock refactoring
    const sessKeyTier = "sess-tier-outgrow";
    store.set(
      sessKeyTier,
      {
        model: "claude-haiku-5-5",
        effort: "low",
        tokens: 3000,
        cached: true,
        healthy: true,
        turn: 1,
      },
      t0
    );
    const tierReq = {
      model: "agentcompass/auto",
      messages: [
        { role: "user", content: "Rename `foo` to `bar`." },
        { role: "assistant", content: "Done." },
        {
          role: "user",
          content:
            "Now perform a cross-module architecture refactoring and root cause deadlock audit across our consensus engine.",
        },
      ],
    };
    const tierKey = computeSessionKey(tierReq, { headers: { "x-session-id": sessKeyTier } });
    store.set(tierKey, store.get(sessKeyTier, t0), t0);
    const ctxTier = {
      protocol: "chat",
      headers: { "x-session-id": sessKeyTier },
      state: {},
      options: { continuityStore: store, now: t0 + 20_000 },
    };
    await hookOnRequest(tierReq, ctxTier);
    assert.equal(ctxTier.state.route.continuation.held, false);
    assert.equal(ctxTier.state.route.continuation.why, "capability_outgrown");
    assert.equal(ctxTier.state.route.canonicalModel, "claude-opus-5-5");

    // 4c. Context window 95% outgrowth (`claude-opus-5-5` has 200k limit; 95% = 190k; pass 195k tokens)
    const outgrownReq = {
      model: "agentcompass/auto",
      estimated_input_tokens: 195_000,
      messages: [
        {
          role: "user",
          content: "Analyze the entire multi-module repository dump and architecture trade-offs.",
        },
      ],
    };
    const outgrownKey = computeSessionKey(outgrownReq, { headers: { "x-session-id": "sess-ctx-outgrow" } });
    store.set(
      outgrownKey,
      {
        model: "claude-opus-5-5",
        effort: "high",
        tokens: 195_000,
        cached: true,
        healthy: true,
        turn: 2,
      },
      t0
    );
    const ctxOutgrown = {
      protocol: "chat",
      headers: { "x-session-id": "sess-ctx-outgrow" },
      state: {},
      options: { continuityStore: store, now: t0 + 10_000 },
    };
    await hookOnRequest(outgrownReq, ctxOutgrown);
    assert.equal(ctxOutgrown.state.route.continuation.held, false);
    assert.equal(ctxOutgrown.state.route.continuation.why, "outgrown");
    // Must switch to a model whose 95% context window >= 195k (e.g. `gemini-4-argon-high` 1M or `gpt-6-astra` 400k)
    assert.ok(
      ["gemini-4-argon-high", "gpt-6-astra", "gpt-6.1-terra", "gpt-6.1-sol"].includes(
        ctxOutgrown.state.route.canonicalModel
      ),
      `Expected >205k context model, got ${ctxOutgrown.state.route.canonicalModel}`
    );

    // 4d. 2M cumulative token wear (`compactedTokens + tokens >= 2,000,000`)
    const wornCheck = evaluateProviderContinuation(
      { within: false, turn: 15, estimatedInputTokens: 150_000 },
      {
        model: "claude-opus-5-5",
        effort: "high",
        tokens: 150_000,
        compactedTokens: WORKER_WORN_TOKENS - 100_000,
        cached: true,
        healthy: true,
        updatedAt: t0,
      },
      { now: t0 + 5_000 }
    );
    assert.equal(wornCheck.shouldHold, false);
    assert.equal(wornCheck.why, "worn");
  });

test("7.5 Scenario 5 (`why: cold` & `why: compacted`): Expired KV cache or a token drop re-evaluates and cleanly downshifts on routine work", async () => {
    const store = new SessionContinuityStore();
    const t0 = 1_700_000_000_000;

    const routineFollowupReq = {
      model: "agentcompass/auto",
      messages: [
        {
          role: "user",
          content: "Audit the distributed consensus engine across modules.",
        },
        {
          role: "assistant",
          content: "Audit complete.",
        },
        {
          role: "user",
          content: "Perform a mechanical batch rename of `old_key` to `newKey` across the JSON fixtures.",
        },
      ],
    };
    const sessKey = computeSessionKey(routineFollowupReq, { headers: { "x-session-id": "sess-cold-downshift" } });
    store.set(
      sessKey,
      {
        model: "claude-opus-5-5",
        effort: "high",
        tokens: 8000,
        cached: true,
        healthy: true,
        turn: 1,
      },
      t0
    );

    // 5a. Advance clock by 6 minutes (> 5m Anthropic cacheCold): cache is now COLD
    const ctxCold = {
      protocol: "anthropic",
      headers: { "x-session-id": "sess-cold-downshift" },
      state: {},
      options: {
        continuityStore: store,
        now: t0 + 6 * 60 * 1000,
      },
    };
    await hookOnRequest(routineFollowupReq, ctxCold);
    assert.equal(ctxCold.state.route.continuation.held, false);
    assert.equal(ctxCold.state.route.continuation.why, "cold");
    // Must downshift from `claude-opus-5-5` to lightweight `gpt-6-luna`!
    assert.equal(ctxCold.state.route.canonicalModel, "gpt-6-luna");

    // 5b. Reset to warm `claude-opus-5-5` with 20,000 tokens, then send a compacted context (2,000 tokens < 50% of 20,000)
    store.set(
      sessKey,
      {
        model: "claude-opus-5-5",
        effort: "high",
        tokens: 20_000,
        cached: true,
        healthy: true,
        turn: 2,
      },
      t0 + 10 * 60 * 1000
    );
    const compactedReq = {
      ...routineFollowupReq,
      estimated_input_tokens: 2_000,
    };
    const ctxCompacted = {
      protocol: "anthropic",
      headers: { "x-session-id": "sess-cold-downshift" },
      state: {},
      options: {
        continuityStore: store,
        now: t0 + 11 * 60 * 1000, // Only 1 minute later, but tokens dropped from 20k -> 2k!
      },
    };
    await hookOnRequest(compactedReq, ctxCompacted);
    assert.equal(ctxCompacted.state.route.continuation.held, false);
    assert.equal(ctxCompacted.state.route.continuation.why, "compacted");
    assert.equal(ctxCompacted.state.route.canonicalModel, "gpt-6-luna");
    // Verify `compactedTokens` accumulated the pre-compaction 20,000 tokens
    assert.equal(store.get(sessKey, t0 + 11 * 60 * 1000).compactedTokens, 20_000);
  });

test("7.6 Scenario 6 (`why: provider_unhealthy`): Upstream 429/502 error in `onResponse` or `onUsage` marks model unhealthy and fails over", async () => {
    const store = new SessionContinuityStore();
    const t0 = 1_700_000_000_000;

    const req = {
      model: "agentcompass/auto",
      messages: [
        {
          role: "user",
          content: "Audit the cross-module concurrency lock graph and race condition invariants.",
        },
      ],
    };
    const ctx = {
      protocol: "chat",
      headers: { "x-session-id": "sess-failover" },
      state: {},
      options: { continuityStore: store, now: t0 },
    };

    await hookOnRequest(req, ctx);
    assert.equal(ctx.state.route.canonicalModel, "claude-opus-5-5");

    // Simulate upstream 429 rate-limit failure reported to `hookOnResponse` (or `mwOnUsage`)
    await hookOnResponse(
      { error: { message: "Rate limit exceeded", status: 429 } },
      { ...ctx, status: 429 }
    );
    assert.equal(store.isModelHealthy("claude-opus-5-5", t0 + 1000), false);

    // Retry the exact same request (or even a mid-turn tool loop): because `claude-opus-5-5` is unhealthy,
    // it MUST break the lock (`why: "provider_unhealthy"`) and switch to a healthy fallback model!
    const ctx2 = {
      ...ctx,
      state: {},
      options: { ...ctx.options, now: t0 + 2000 },
    };
    await hookOnRequest(req, ctx2);
    assert.equal(ctx2.state.route.continuation.held, false);
    assert.equal(ctx2.state.route.continuation.why, "provider_unhealthy");
    assert.notEqual(
      ctx2.state.route.canonicalModel,
      "claude-opus-5-5",
      "Unhealthy model must be excluded on failover"
    );

    // Also verify `mwOnResponse` marking a model unhealthy on `{ status: 503 }` in the shared pooled request VM
    const fallbackModel = ctx2.state.route.canonicalModel;
    mwOnResponse(
      {
        error: { message: "Service unavailable", status: 503 },
      },
      {
        protocol: "chat",
        status: 503,
        model: fallbackModel,
        state: {
          agentcompass: {
            sessionKey: ctx2.state.route.continuation.sessionKey,
            targetModel: fallbackModel,
            targetEffort: ctx2.state.route.targetEffort,
          },
        },
        options: { continuityStore: store, now: t0 + 3000 },
      }
    );
    assert.equal(store.isModelHealthy(fallbackModel, t0 + 3500), false);
  });

test("8.1 Exhausted subscription window (< 5% safety margin or < 10% short-window floor) excludes provider and routes to healthy subscription", () => {
    const t0 = 1_700_000_000_000;
    const implProfile = {
      work_kind: "general_implementation",
      multi_task: false,
      task_count: "1",
      reasoning_depth: 2,
      code_complexity: 2,
      context_scale: 1,
      divergence: 1,
      hallucination_risk: 2,
      error_cost: 2,
      needs_visual: false,
      needs_gui: false,
      local_only: true,
      interaction_style: "engineering",
      expected_hours: 0.5,
    };

    // Without quota constraints, `gpt-6.1-sol` (`gpt-6.1-sol`) wins `general_implementation`
    const baseline = selectBestRoute(implProfile, {
      candidateModels: ["gpt-6.1-sol", "claude-sonnet-5-5"],
      now: t0,
    });
    assert.equal(baseline.targetModel, "gpt-6.1-sol");

    // When `codex` has only 3% remaining on its 7d window (< 5% safety_margin_pct), it is excluded
    // and `claude-sonnet-5-5` (with 80% remaining) wins automatically!
    const withExhaustedCodex = selectBestRoute(implProfile, {
      candidateModels: ["gpt-6.1-sol", "claude-sonnet-5-5"],
      now: t0,
      quotas: {
        "codex/work@example.com": {
          kind: "subscription",
          quota: {
            windows: [
              { name: "7d", used: 97, remaining: 3, resetsAt: new Date(t0 + 48 * 3600 * 1000).toISOString() },
            ],
          },
        },
        "claude/pro@example.com": {
          kind: "subscription",
          quota: {
            windows: [
              { name: "7d", used: 20, remaining: 80, resetsAt: new Date(t0 + 72 * 3600 * 1000).toISOString() },
            ],
          },
        },
      },
    });
    assert.equal(withExhaustedCodex.targetModel, "claude-sonnet-5-5");
    assert.equal(withExhaustedCodex.selectedAccount, "claude/pro@example.com");
  });

test("8.2 High opportunity cost / scarcity on over-burned subscription shifts routing to well-funded subscription before hitting 0%", () => {
    const t0 = 1_700_000_000_000;
    const implProfile = {
      work_kind: "general_implementation",
      multi_task: false,
      task_count: "1",
      reasoning_depth: 2,
      code_complexity: 2,
      context_scale: 2,
      divergence: 1,
      hallucination_risk: 2,
      error_cost: 2,
      needs_visual: false,
      needs_gui: false,
      local_only: true,
      interaction_style: "engineering",
      expected_hours: 1.5,
    };

    // Codex has 14% left with 130h (5.4 days) remaining in its 168h week (burning 2.3%/h, high shortage probability & scarcity).
    // Claude has 92% left with 48h remaining in its week.
    const decision = selectBestRoute(implProfile, {
      candidateModels: ["gpt-6.1-sol", "claude-sonnet-5-5"],
      now: t0,
      quotas: {
        "codex/scarce@example.com": {
          kind: "subscription",
          quota: {
            windows: [
              {
                name: "7d",
                used: 86,
                remaining: 14,
                resetsAt: new Date(t0 + 130 * 3600 * 1000).toISOString(),
                pct_per_hour_24h: 2.2,
              },
            ],
          },
        },
        "claude/plentiful@example.com": {
          kind: "subscription",
          quota: {
            windows: [
              {
                name: "7d",
                used: 8,
                remaining: 92,
                resetsAt: new Date(t0 + 48 * 3600 * 1000).toISOString(),
              },
            ],
          },
        },
      },
    });
    assert.equal(
      decision.targetModel,
      "claude-sonnet-5-5",
      "High opportunity cost + scarcity on Codex should shift routing to plentiful Claude subscription"
    );
    assert.ok(decision.components.reset_bonus >= 0);
  });

test("8.3 Pre-reset surplus sprint (`reset_bonus` & `sprint_sunk`): harvests expiring subscription quota and unlocks higher effort", () => {
    const t0 = 1_700_000_000_000;
    const hardProfile = {
      work_kind: "general_implementation",
      multi_task: false,
      task_count: "1",
      reasoning_depth: 2,
      code_complexity: 2,
      context_scale: 2,
      divergence: 1,
      hallucination_risk: 2,
      error_cost: 2,
      needs_visual: false,
      needs_gui: false,
      local_only: true,
      interaction_style: "engineering",
      expected_hours: 1.0,
    };

    // Normally on `general_implementation` (depth 2), `claude-opus-5-5` pays an overqualification penalty
    // and `gpt-6.1-sol:high` wins.
    // Now give `claude` 65% remaining quota resetting in just 3 hours (`resetHours = 3 <= 36h horizon`)!
    const sprintDecision = selectBestRoute(hardProfile, {
      candidateModels: ["gpt-6.1-sol", "claude-opus-5-5"],
      now: t0,
      quotas: {
        "codex/normal@example.com": {
          kind: "subscription",
          quota: {
            windows: [
              {
                name: "7d",
                used: 45,
                remaining: 55,
                resetsAt: new Date(t0 + 100 * 3600 * 1000).toISOString(),
              },
            ],
          },
        },
        "claude/expiring-soon@example.com": {
          kind: "subscription",
          quota: {
            windows: [
              {
                name: "7d",
                used: 35,
                remaining: 65,
                resetsAt: new Date(t0 + 3 * 3600 * 1000).toISOString(),
              },
            ],
          },
        },
      },
    });

    assert.equal(
      sprintDecision.targetModel,
      "claude-opus-5-5",
      "Expiring Claude subscription with 65% surplus resetting in 3h must win via reset_bonus"
    );
    assert.ok(
      sprintDecision.components.reset_bonus > 0.8,
      `Expected reset_bonus > 0.8, got ${sprintDecision.components.reset_bonus}`
    );
    assert.ok(
      ["xhigh", "max"].includes(sprintDecision.targetEffort),
      `Pre-reset surplus sprint on non-routine work should waive effort penalty and unlock xhigh/max effort (got ${sprintDecision.targetEffort})`
    );
  });

test("8.4 Sticky session failover on quota exhaustion (`why: quota_exhausted`) & explicit `magpieConfigDir` quota loading", async () => {
    const store = new SessionContinuityStore();
    const t0 = 1_700_000_000_000;

    // Create a temporary Magpie config directory with `quotas.json` and `providers.json`
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "magpie-quotas-test-"));
    const quotasFile = path.join(tmpDir, "quotas.json");
    fs.writeFileSync(path.join(tmpDir, "providers.json"), "{}", "utf8");
    fs.writeFileSync(
      quotasFile,
      JSON.stringify({
        "codex/acct1": {
          kind: "subscription",
          quota: {
            windows: [
              {
                name: "7d",
                used: 98,
                remaining: 2,
                resetsAt: new Date(t0 + 48 * 3600 * 1000).toISOString(),
              },
            ],
          },
        },
        "claude/acct1": {
          kind: "subscription",
          quota: {
            windows: [
              {
                name: "7d",
                used: 15,
                remaining: 85,
                resetsAt: new Date(t0 + 72 * 3600 * 1000).toISOString(),
              },
            ],
          },
        },
      }),
      "utf8"
    );
    clearQuotasCache();

    try {
      const loaded = loadLiveQuotas({ magpieConfigDir: tmpDir }, t0);
      assert.ok(loaded && loaded["codex/acct1"]);

      const req = {
        model: "agentcompass/auto",
        messages: [
          {
            role: "user",
            content: "Implement the next pagination helper method in the user service.",
          },
        ],
      };
      const sessKey = computeSessionKey(req, { headers: { "x-session-id": "sess-quota-spent" } });
      // Session was previously holding `gpt-6.1-sol` with warm KV cache
      store.set(
        sessKey,
        {
          model: "gpt-6.1-sol",
          effort: "high",
          provider: "codex",
          tokens: 12_000,
          cached: true,
          healthy: true,
          turn: 2,
        },
        t0
      );

      const ctxQuota = {
        protocol: "chat",
        headers: { "x-session-id": "sess-quota-spent" },
        state: {},
        options: {
          continuityStore: store,
          magpieConfigDir: tmpDir,
          candidateModels: ["gpt-6.1-sol", "claude-sonnet-5-5"],
          now: t0 + 15_000,
        },
      };
      await hookOnRequest(req, ctxQuota);

      assert.equal(ctxQuota.state.route.continuation.held, false);
      assert.equal(ctxQuota.state.route.continuation.why, "quota_exhausted");
      assert.equal(ctxQuota.state.route.targetModel, "claude-sonnet-5-5");
      assert.equal(ctxQuota.state.route.selectedAccount, "claude/acct1");
    } finally {
      clearQuotasCache();
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });

test("8.5 `strategies.agentcompass` pool seat ordering: `learns` -> pre-reset sprint -> healthy subscription -> API key -> low reserve -> resting/spent", () => {
    const t0 = 1_700_000_000_000;
    const seats = [
      // Index 0: Resting seat
      { key: "claude/resting", account: true, known: true, learns: false, used: 20, soon: [t0 + 48 * 3600_000], pace: 1.6, resting: true, tokens: 0 },
      // Index 1: Low-reserve seat (94% used >= 90% keepPercent)
      { key: "claude/low-reserve", account: true, known: true, learns: false, used: 94, soon: [t0 + 24 * 3600_000], pace: 0.25, resting: false, tokens: 100 },
      // Index 2: Unshared pay-as-you-go API key (`account: false`, `soon: []`)
      { key: "claude/api-key", account: false, known: false, learns: false, used: 0, soon: [], pace: 0, resting: false, tokens: 0 },
      // Index 3: Normal healthy subscription (50% used, 100h left)
      { key: "claude/normal-sub", account: true, known: true, learns: false, used: 50, soon: [t0 + 100 * 3600_000], pace: 0.5, resting: false, tokens: 500 },
      // Index 4: Pre-reset surplus sprint subscription (30% used = 70% left, renewing in 3 hours!)
      { key: "claude/sprint-sub", account: true, known: true, learns: false, used: 30, soon: [t0 + 3 * 3600_000], pace: 23.3, resting: false, tokens: 1000 },
      // Index 5: Unknown subscription that `learns` on its first response
      { key: "claude/learns-sub", account: true, known: false, learns: true, used: 0, soon: [], pace: 0, resting: false, tokens: 0 },
    ];

    const order = strategies.agentcompass("claude", seats, { now: t0 });
    assert.deepEqual(
      order,
      [5, 4, 3, 2, 1, 0],
      `Expected [learns(5), sprint(4), normalSub(3), apiKey(2), lowReserve(1), resting(0)], got [${order.join(", ")}]`
    );
  });

test("8.6 Strict AgentCompass Main-Repo Parity (`allocator.py` + `reference-accounts.json` invariants)", () => {
    // Verify calibrated subjective blend weights (0.49 / 0.34 / 0.17) and default_hours (0.4 / 1.0 / 2.0 / 3.5)
    assert.deepEqual(DEFAULT_SCORING_POLICY.profile.subjective, {
      weight: 0.49,
      vector_weight: 0.34,
      route_weight: 0.17,
    });
    assert.deepEqual(DEFAULT_SCORING_POLICY.default_hours, {
      routine: 0.4,
      hard: 1.0,
      coupled: 2.0,
      frontier: 3.5,
    });

    // Verify structured-only profile (no free-text `reason`)
    assert.equal(SCORER_SYSTEM_PROMPT.includes("- reason:"), false, "SCORER_SYSTEM_PROMPT must not request free-text reason");

    // Verify `parseJevScorerAnswer` produces structured-only profile without `reason`
    const jevParsed = parseJevScorerAnswer({
      model: "jev-latest",
      answers: {
        work_kind: { choice: "debug_review" },
        interaction_style: { choice: "exploration" },
        multi_task: { noul: 0.05 },
        task_count: { choice: "1" },
        reasoning_depth: { score: 2 },
        code_complexity: { score: 0 },
        context_scale: { score: 1 },
        divergence: { score: 1 },
        hallucination_risk: { score: 2 },
        error_cost: { score: 1 },
        needs_visual: { noul: 0.1 },
        needs_gui: { noul: 0.1 },
        local_only: { noul: 0.9 },
      },
    });
    assert.equal("reason" in jevParsed, false);
    assert.equal(jevParsed.work_kind, "debug_review");
    assert.equal(jevParsed.local_only, true);

    // Verify `namedRoute` trusts classifier's `work_kind` directly without secondary `code_complexity === 0` override (89f66ab)
    assert.equal(
      namedRoute({ work_kind: "general_implementation", code_complexity: 0 }),
      "general_implementation"
    );

    // Verify `free_usage` (+0.18 `unlimited_bonus`) on unlimited subscription accounts
    const unlimitedDec = selectBestRoute(
      {
        work_kind: "bounded_bulk_work",
        multi_task: false,
        task_count: "1",
        reasoning_depth: 1,
        code_complexity: 1,
        context_scale: 2,
        divergence: 1,
        hallucination_risk: 1,
        error_cost: 1,
        needs_visual: false,
        needs_gui: false,
        local_only: true,
        interaction_style: "engineering",
        expected_hours: 0.4,
      },
      {
        candidateModels: ["gpt-6.1-sol", "gemini-3.8-flash-high"],
        quotas: {
          "codex/unlimited@example.com": {
            kind: "subscription",
            unlimited: true,
          },
          "gemini/limited@example.com": {
            kind: "subscription",
            quota: {
              windows: [{ name: "7d", used: 40, remaining: 60, resetHours: 100 }],
            },
          },
        },
      }
    );
    assert.equal(unlimitedDec.targetModel, "gpt-6.1-sol");
    assert.equal(unlimitedDec.components.free_usage, 0.18);

    // Verify v0.3.7 + 54bb51e calibration parity:
    assert.deepEqual(DEFAULT_SCORING_POLICY.ability_floor, {
      routine: 2.3,
      hard: 3.1,
      coupled: 3.6,
      frontier: 4.2,
    });
    assert.deepEqual(DEFAULT_SCORING_POLICY.ability_score_origin, {
      routine: 3.0,
      hard: 3.3,
      coupled: 3.6,
      frontier: 4.2,
    });
    assert.deepEqual(DEFAULT_SCORING_POLICY.critical_headroom_by_difficulty, {
      frontier: 0.1,
    });

    // A3 (54bb51e): origin = max(floor, ability_score_origin[d]) so raising floor above origin does not hand out free ability score
    const raisedFloorRanked = rankCandidates(
      {
        work_kind: "debug_review",
        multi_task: false,
        task_count: "1",
        reasoning_depth: 2,
        code_complexity: 2,
        context_scale: 1,
        divergence: 1,
        hallucination_risk: 2,
        error_cost: 3, // critical -> uses origin-based gradient
        needs_visual: false,
        needs_gui: false,
        local_only: true,
        interaction_style: "engineering",
      },
      {
        candidateModels: ["claude-opus-5-5", "gpt-6.1-sol"],
        catalog: {
          policy: {
            ...DEFAULT_SCORING_POLICY,
            ability_floor: { ...DEFAULT_SCORING_POLICY.ability_floor, hard: 3.6 },
            critical_headroom: 0.0,
          },
        },
      }
    );
    const solHigh = raisedFloorRanked.ranking.find((c) => c.canonicalModel === "gpt-6.1-sol" && c.effort === "high");
    assert.ok(solHigh, "gpt-6.1-sol:high (ability 3.6) clears floor=3.6 when critical_headroom=0");
    assert.equal(solHigh.components.ability, 0, "Model right at raised floor (3.6 > origin 3.3) must not get free ability score");

    // gpt-6-luna's effort curve is AgentCompass v0.4.1 `effort_curves["gpt-6-luna"]`
    assert.deepEqual(EFFORT_CURVES["gpt-6-luna"], {
      ability: { low: -0.54, medium: -0.15, high: 0, xhigh: 0.1, max: 0.35 },
      burn: { low: 0.1, medium: 0.4, high: 0.7, xhigh: 1.0, max: 1.5 },
      responsiveness: { none: 0.959, low: 0.926, medium: 0.777, high: 0.664, xhigh: 0.374, max: 0 },
    });
  });

const GATE_PROFILE = {
  multi_task: false,
  task_count: "1",
  code_complexity: 2,
  context_scale: 1,
  divergence: 1,
  hallucination_risk: 2,
  needs_visual: false,
  needs_gui: false,
  local_only: true,
  interaction_style: "engineering",
};

test("8.7 Critical frontier gate is 4.3 (frontier floor 4.2 + `critical_headroom_by_difficulty.frontier` 0.1): Astra Max is admitted, Astra High is not", () => {
  const ranked = rankCandidates(
    { ...GATE_PROFILE, work_kind: "general_implementation", reasoning_depth: 4, code_complexity: 3, error_cost: 3 },
    { candidateModels: [{ model: "gpt-6-astra", efforts: ["high", "max"] }] },
  );
  assert.equal(ranked.task.difficulty, "frontier");
  assert.equal(ranked.task.critical, true);
  const byEffort = Object.fromEntries(ranked.allCandidates.map((c) => [c.effort, c]));
  assert.equal(byEffort.max.eligible, true, `Astra Max (4.3) clears the 4.3 gate: ${byEffort.max.excluded.join("; ")}`);
  assert.equal(byEffort.high.eligible, false, "Astra High (4.2) is below the 4.3 gate");
  assert.match(byEffort.high.excluded.join("; "), /ability 4\.20 below the frontier floor 4\.30/);
});

test("8.8 tool_execution/hard uses its route floor 2.7, not the policy's hard floor 3.1", () => {
  // gpt-6.1-sol:low is 2.86: above the route's 2.7, below the policy's 3.1
  const rank = (workKind) =>
    rankCandidates(
      { ...GATE_PROFILE, work_kind: workKind, reasoning_depth: 2, error_cost: 2 },
      { candidateModels: [{ model: "gpt-6.1-sol", efforts: ["low"] }] },
    );
  const tool = rank("tool_execution");
  assert.equal(tool.task.kind, "tool_execution");
  assert.equal(tool.task.difficulty, "hard");
  const toolSol = tool.allCandidates[0];
  assert.equal(toolSol.eligible, true, `sol:low (2.86) clears the 2.7 route floor: ${toolSol.excluded.join("; ")}`);

  const impl = rank("general_implementation");
  assert.equal(impl.task.kind, "general_implementation");
  assert.equal(impl.task.difficulty, "hard");
  assert.match(impl.allCandidates[0].excluded.join("; "), /ability 2\.86 below the hard floor 3\.10/);
});

test("8.9 Effort fallback (plugin-side, not in AgentCompass): `:minimal` and `:ultra` parse as efforts, `ultra` reads as the model's Max and `minimal` sits below its Low", () => {
  const astra = DEFAULT_MODELS["gpt-6-astra"];
  const curves = EFFORT_CURVES["gpt-6-astra"];
  const policy = DEFAULT_SCORING_POLICY;

  // Parsing: each suffix is one candidate at that effort, not the model's whole effort list
  const ranked = rankCandidates(
    { ...GATE_PROFILE, work_kind: "general_implementation", reasoning_depth: 1, code_complexity: 1, error_cost: 1 },
    { candidateModels: ["gpt-6-astra:minimal", "gpt-6-astra:ultra"] },
  );
  assert.deepEqual(ranked.allCandidates.map((c) => c.id).sort(), ["gpt-6-astra:minimal", "gpt-6-astra:ultra"]);

  // Scoring: Astra has no `ultra` row, so ultra is its Max (4.2 + 0.1), not the policy table's generic Max
  assert.equal(effortAbility(policy, curves, astra, "ultra"), effortAbility(policy, curves, astra, "max"));
  assert.equal(burnScale(policy, curves, "ultra"), burnScale(policy, curves, "max"));
  // minimal is Astra's own Low less the policy's Low->minimal step (-1.0 - -0.54), and burns less than Low
  const low = effortAbility(policy, curves, astra, "low");
  assert.ok(Math.abs(effortAbility(policy, curves, astra, "minimal") - (low - 0.46)) < 1e-9);
  assert.ok(burnScale(policy, curves, "minimal") < burnScale(policy, curves, "low"));
  // A model whose curve has no Low row takes the policy table's minimal step
  const deepseek = DEFAULT_MODELS["deepseek-v4.1-flash-free"];
  const deepseekCurves = EFFORT_CURVES["deepseek-v4.1-flash-free"];
  assert.ok(Math.abs(effortAbility(policy, deepseekCurves, deepseek, "minimal") - (deepseek.tier - 1.0)) < 1e-9);

  // Continuity: on a warm cache, minimal after low is a downgrade and keeps only a quarter of the continuity bonus
  const held = rankCandidates(
    { ...GATE_PROFILE, work_kind: "general_implementation", reasoning_depth: 1, code_complexity: 1, error_cost: 1, context_scale: 2 },
    {
      candidateModels: ["gpt-6-astra:low", "gpt-6-astra:minimal"],
      previousModel: "gpt-6-astra",
      previousEffort: "low",
      cacheWarm: true,
    },
  );
  const cont = Object.fromEntries(held.allCandidates.map((c) => [c.effort, c.components.continuity]));
  assert.equal(cont.low, 0.25);
  assert.equal(cont.minimal, 0.0625);
});

test("8.10 canonicalModel: a multi-prefix id resolves by its last segment; an unknown one keeps only its first prefix stripped", () => {
  assert.equal(canonicalModel("openrouter/openai/gpt-5-mini"), "openai/gpt-5-mini");
  assert.equal(canonicalModel("OpenRouter/OpenAI/GPT-5-Mini"), "OpenAI/GPT-5-Mini");
  // Alias keys with capitals (`MiniMax-M3`) need the exact-case last-segment lookup
  assert.equal(canonicalModel("openrouter/minimax/MiniMax-M3"), "minimax-m3");
  assert.equal(canonicalModel("openrouter/qwen/foo"), "qwen/foo");
  assert.equal(canonicalModel("qwen/foo"), "foo");
});

test("8.11 `ability_score_origin` above the floor decides the pick (hard: origin 3.3, floor 3.1)", () => {
  // Critical hard task (error_cost 3): abilityScore = (credited - origin - (gate - floor)) / span, gate 3.4, span 0.6.
  const ranked = rankCandidates(
    { ...GATE_PROFILE, work_kind: "general_implementation", reasoning_depth: 2, error_cost: 3 },
    { candidateModels: [{ model: "claude-sonnet-5-5", efforts: ["xhigh"] }, { model: "gpt-6-astra", efforts: ["medium"] }] },
  );
  assert.equal(ranked.task.difficulty, "hard");
  assert.equal(ranked.task.minimum_sufficient, false);
  const sonnet = ranked.allCandidates.find((c) => c.canonicalModel === "claude-sonnet-5-5");
  assert.equal(sonnet.components.ability_value, 4.3);
  // Measured against origin 3.3; with origin = floor 3.1 the score is 0.333 higher and Sonnet xhigh overtakes Astra medium
  assert.equal(`${ranked.selected.canonicalModel}:${ranked.selected.effort}`, "gpt-6-astra:medium");
});

test("8.12 hasComputerUse comes from structured fields, never from the turn's words", () => {
  const facts = (text, extra = {}, ctx = { protocol: "chat" }) =>
    extractRequestFacts({ messages: [{ role: "user", content: text }], ...extra }, ctx);
  for (const text of ["Click the Save button in the desktop app and confirm the dialog closes", "用鼠标点击设置里的保存"]) {
    assert.equal(facts(text).hasComputerUse, false, text);
  }
  assert.equal(facts("x", { needs: { gui: true } }).hasComputerUse, true);
  assert.equal(facts("x", { profile: { needs_gui: true } }).hasComputerUse, true);
  assert.equal(facts("x", {}, { profile: { needs_gui: true } }).hasComputerUse, true);
});

test("8.13 Continuity: only a computer-use turn outgrows a model without GUI (`capability_outgrown` / `gui_capability_required`)", async () => {
  const t0 = 1_700_000_000_000;
  const nextTurn = async (text) => {
    const store = new SessionContinuityStore();
    store.set("sess-gui", { model: "grok-4.7", effort: "medium", tokens: 4000, cached: true, healthy: true, turn: 1 }, t0);
    const ctx = {
      protocol: "chat",
      session_id: "sess-gui",
      state: {},
      options: { continuityStore: store, now: t0 + 30_000 },
    };
    await hookOnRequest({ model: "agentcompass/auto", session_id: "sess-gui", messages: [{ role: "user", content: text }] }, ctx);
    return ctx.state.route;
  };
  const gui = await nextTurn("Click the Save button in the desktop app and confirm the dialog closes");
  assert.equal(gui.continuation.held, false);
  assert.equal(gui.continuation.why, "capability_outgrown");
  assert.equal(gui.continuation.reasonDetail, "gui_capability_required");
  assert.notEqual(gui.canonicalModel, "grok-4.7");
  // Before 0.1.2 the broader GUI words (" gui", "desktop app", "keyboard", "桌面", "键盘") also moved the session; now they hold it
  for (const text of [
    "Add keyboard shortcuts to the desktop app",
    "Open the gui and check the desktop app",
    "修改桌面版的键盘快捷键",
    "Click the Save button handler in settings.tsx is broken",
  ]) {
    const held = await nextTurn(text);
    assert.notEqual(held.continuation.why, "capability_outgrown", text);
    assert.equal(held.canonicalModel, "grok-4.7", text);
  }
});

// AgentCompass v0.3.8 skills/agent-compass/references/accounts.json: `effort_curves` for the 13 models the plugin carries
const V038_EFFORT_CURVES = {
  "deepseek-v4.1-flash-free": { ability: { high: 0, max: 0.2 }, burn: { high: 0.7, max: 0.91 } },
  "claude-opus-5-5": {
    ability: { low: -0.84, medium: -0.2, high: 0, xhigh: 0.25, max: 0.35 },
    burn: { low: 0.2, medium: 0.5, high: 0.7, xhigh: 1.22, max: 2.03 },
  },
  "claude-sonnet-5-5": {
    ability: { low: -0.79, medium: -0.54, high: 0, xhigh: 0.4, max: 0.64 },
    burn: { low: 0.22, medium: 0.33, high: 0.7, xhigh: 1.75, max: 4.28 },
  },
  "gpt-6-astra": {
    ability: { low: -0.59, medium: -0.15, high: 0, xhigh: 0, max: 0.1 },
    burn: { low: 0.31, medium: 0.59, high: 0.7, xhigh: 0.79, max: 1.35 },
  },
  "gpt-6.1-sol": {
    ability: { low: -0.74, medium: -0.2, high: 0, xhigh: 0.1, max: 0.25 },
    burn: { low: 0.19, medium: 0.43, high: 0.7, xhigh: 0.9, max: 1.51 },
  },
  "grok-4.7": { ability: { low: -0.2, high: 0, xhigh: 0 }, burn: { low: 0.33, high: 0.7, xhigh: 0.88 } },
  "gemini-3.8-flash-high": { ability: { low: -0.4, medium: -0.05, high: 0 }, burn: { medium: 0.56, high: 0.7 } },
  "gemini-4-argon-high": { ability: { low: -0.3, medium: -0.1, high: 0 }, burn: { low: 0.63, medium: 0.7, high: 0.7 } },
  "claude-haiku-5-5": {
    ability: { low: -0.54, medium: -0.25, high: 0, xhigh: 0.25, max: 0.3 },
    burn: { low: 0.17, medium: 0.39, high: 0.7, xhigh: 1.09, max: 1.79 },
  },
  "gpt-6.1-terra": {
    ability: { low: -0.64, medium: -0.2, high: 0, xhigh: 0.15, max: 0.45 },
    burn: { low: 0.19, medium: 0.46, high: 0.7, xhigh: 1.08, max: 2.58 },
  },
  "gpt-6-luna": {
    ability: { low: -0.54, medium: -0.15, high: 0, xhigh: 0.1, max: 0.35 },
    burn: { low: 0.1, medium: 0.4, high: 0.7, xhigh: 1.0, max: 1.5 },
  },
  "claude-fable-5-1": {
    ability: { low: -0.4, medium: -0.2, high: 0, xhigh: 0.15, max: 0.15 },
    burn: { low: 0.41, medium: 0.52, high: 0.7, xhigh: 1.01, max: 1.21 },
  },
  "gemini-4-carbon": { ability: { low: -0.3, medium: -0.1, high: 0 }, burn: { low: 0.65, medium: 0.59, high: 0.7 } },
};

test("8.14 All 13 effort curves and `effort_ability` / `effort_burn` equal AgentCompass v0.3.8", () => {
  for (const [model, curve] of Object.entries(V038_EFFORT_CURVES)) {
    assert.deepEqual(EFFORT_CURVES[model].ability, curve.ability, `${model} ability`);
    assert.deepEqual(EFFORT_CURVES[model].burn, curve.burn, `${model} burn`);
  }
  // scoring.effort_ability / scoring.effort_burn (low..max; none/off/minimal are the plugin's own rows, 8.9 and 8.15)
  const { none, off, minimal, ...upstreamAbility } = DEFAULT_SCORING_POLICY.effort_ability;
  assert.deepEqual(upstreamAbility, { low: -0.54, medium: -0.2, high: 0, xhigh: 0.15, max: 0.3 });
  assert.deepEqual({ none, off, minimal }, { none: -1.1, off: -1.1, minimal: -1.0 });
  assert.deepEqual(DEFAULT_SCORING_POLICY.effort_burn, { low: 0.2, medium: 0.4, high: 0.7, xhigh: 1.0, max: 1.3 });
});

test("8.15 Effort fallback branches the default policy can't reach: policy Max for `ultra`, the 0.2 step floor, the 0.75 burn clamp, none/off", () => {
  const profile = {
    work_kind: "general_implementation",
    multi_task: false,
    task_count: "1",
    reasoning_depth: 1,
    code_complexity: 1,
    context_scale: 2,
    divergence: 1,
    hallucination_risk: 2,
    error_cost: 1,
    needs_visual: false,
    needs_gui: false,
    local_only: true,
    interaction_style: "engineering",
  };
  const policy = DEFAULT_SCORING_POLICY;
  const round = (value) => Math.round(value * 1e4) / 1e4;
  const abilities = (candidateModels) =>
    Object.fromEntries(rankCandidates(profile, { candidateModels }).allCandidates.map((c) => [c.effort, c.components.ability_value]));

  // ultra on a model with no curve of its own reads as the policy's Max
  assert.equal(EFFORT_CURVES["qwen3-max"], undefined);
  const qwen = abilities(["qwen3-max:high", "qwen3-max:max", "qwen3-max:ultra"]);
  assert.equal(qwen.ultra, qwen.max);
  assert.ok(Math.abs(qwen.ultra - (qwen.high + policy.effort_ability.max)) < 0.001, `ultra ${qwen.ultra}, high ${qwen.high}`);
  assert.equal(effortValue(policy, undefined, "ability", "ultra", 0), policy.effort_ability.max);
  assert.equal(effortValue(policy, undefined, "burn", "ultra", 0.7), policy.effort_burn.max);

  // Step floor: a policy whose Low->minimal step is smaller than 0.2 still puts minimal 0.2 below the model's Low
  const shallow = { ...policy, effort_ability: { ...policy.effort_ability, low: -0.54, minimal: -0.6 } };
  assert.equal(round(effortValue(shallow, { ability: { low: -0.5, high: 0 } }, "ability", "minimal", 0)), -0.7);
  // ...and a deeper step is used as given
  const deep = { ...policy, effort_ability: { ...policy.effort_ability, low: -0.54, minimal: -1.0 } };
  assert.equal(round(effortValue(deep, { ability: { low: -0.5, high: 0 } }, "ability", "minimal", 0)), -0.96);

  // Burn clamp: a policy with effort_burn.minimal close to Low still burns at most 0.75 of the model's Low
  const curve = { burn: { low: 0.4, high: 0.7 } };
  const nearLow = { ...policy, effort_burn: { ...policy.effort_burn, low: 0.2, minimal: 0.19 } };
  assert.equal(effortValue(nearLow, curve, "burn", "minimal", 0.7), 0.3);
  const cheap = { ...policy, effort_burn: { ...policy.effort_burn, low: 0.2, minimal: 0.1 } };
  assert.equal(effortValue(cheap, curve, "burn", "minimal", 0.7), 0.2);

  // none: parsed as an effort and scored from its own policy row (below minimal), on and off a model's curve
  const astra = abilities(["gpt-6-astra:none", "gpt-6-astra:minimal", "gpt-6-astra:low"]);
  assert.deepEqual(Object.keys(astra).sort(), ["low", "minimal", "none"]);
  const lowToNone = Math.min(policy.effort_ability.none - policy.effort_ability.low, -0.2);
  assert.ok(Math.abs(astra.none - (astra.low + lowToNone)) < 0.001, `none ${astra.none}, low ${astra.low}`);
  assert.ok(astra.none < astra.minimal, `none ${astra.none} below minimal ${astra.minimal}`);
  const glmNone = abilities(["glm-5.3:none", "glm-5.3:high"]);
  assert.ok(Math.abs(glmNone.none - (glmNone.high + policy.effort_ability.none)) < 0.001);
});

test("9.1 PR #670 Review Guard: Default triggers ONLY claim `agentcompass` / `agentcompass/*` and never hijack `auto` or `group/intent`", async () => {
  for (const nonOwnedModel of ["auto", "intent", "group/intent", "default", "claude-sonnet-4-6"]) {
    const req = {
      model: nonOwnedModel,
      messages: [{ role: "user", content: "Explain how HTTP/3 QUIC connection migration works." }],
    };
    const hookCtx = { protocol: "chat", state: {}, options: {} };
    const hookRes = await hookOnRequest(req, hookCtx);
    assert.equal(hookRes, undefined, `hookOnRequest must skip non-owned model "${nonOwnedModel}" by default`);
    assert.equal(hookCtx.state.route, undefined);

    const mwCtx = { protocol: "chat", state: {}, options: {} };
    const mwRes = mwOnRequest(req, mwCtx);
    assert.equal(mwRes, undefined, `mwOnRequest must skip non-owned model "${nonOwnedModel}" by default`);
    assert.equal(req.model, nonOwnedModel, `req.model "${nonOwnedModel}" must remain untouched`);
  }
});

test("9.2 PR #670 Review Guard: `availableModels` / `candidateModels` / `ctx.models` & safe fallback when candidate list is omitted", async () => {
  // 1. When `availableModels: ["claude-haiku-5-5", "claude-sonnet-5-5"]` is passed, `agentcompass/auto` ONLY selects from those two models
  const reqLimited = {
    model: "agentcompass/auto",
    messages: [
      {
        role: "user",
        content:
          "Conduct a deep cross-module deadlock and race condition root-cause audit across the distributed transaction engine.",
      },
    ],
  };
  const mwCtxLimited = {
    protocol: "chat",
    state: {},
    options: {
      availableModels: ["claude-haiku-5-5", "claude-sonnet-5-5"],
      injectProtocolEffort: true,
    },
  };
  const routedLimited = mwOnRequest(structuredClone(reqLimited), mwCtxLimited);
  assert.ok(routedLimited, "mwOnRequest must route within availableModels");
  assert.match(
    routedLimited.model,
    /^claude-sonnet-5-5:(high|xhigh|max)$/,
    `Must pick claude-sonnet-5-5 from availableModels, got ${routedLimited.model}`
  );

  // 2. When NO candidate list is configured and `routeAllModels: true` is used on a concrete model (`claude-sonnet-5-5`),
  // it MUST keep `claude-sonnet-5-5` and only inject effort — never rewriting to an unverified catalog model like `claude-opus-5-5`!
  const reqConcrete = {
    model: "claude-sonnet-5-5",
    messages: [
      {
        role: "user",
        content: "Audit the cross-module state machine for deadlocks and prove liveness invariants.",
      },
    ],
  };
  const routedConcrete = mwOnRequest(structuredClone(reqConcrete), {
    protocol: "chat",
    state: {},
    options: { routeAllModels: true, injectProtocolEffort: true },
  });
  assert.ok(routedConcrete);
  assert.match(
    routedConcrete.model,
    /^claude-sonnet-5-5:(high|xhigh|max)$/,
    `Concrete model must not be rewritten to an unverified catalog model (got ${routedConcrete.model})`
  );

  // 3. When `model: "agentcompass/auto"` is sent WITHOUT `candidateModels` or `defaultModel`, `onRequest` calls
  // `ctx.reject(400, "agentcompass: a virtual model needs candidateModels or defaultModel in the plugin's options")` instead of guessing from a fixed list.
  const reqVirtual = {
    model: "agentcompass/auto",
    messages: [
      {
        role: "user",
        content: "Implement a standalone LRU cache module in TypeScript with unit tests.",
      },
    ],
  };
  let rejectedStatus = null;
  let rejectedMessage = null;
  const unconfiguredRes = mwOnRequest(structuredClone(reqVirtual), {
    protocol: "chat",
    state: {},
    options: { injectProtocolEffort: true },
    reject(status, message) {
      rejectedStatus = status;
      rejectedMessage = message;
      return undefined;
    },
  });
  assert.equal(unconfiguredRes, undefined);
  assert.equal(rejectedStatus, 400);
  assert.equal(rejectedMessage, "agentcompass: a virtual model needs candidateModels or defaultModel in the plugin's options");

  // With `defaultModel` configured, `onRequest` rewrites the virtual model to `defaultModel:<effort>` and `onProvider`
  // refines within `options.modelsByProvider` using Magpie's real `{id, name, account, model}` `ctx.provider` shape!
  const sharedState = {};
  const afterReq = mwOnRequest(reqVirtual, {
    protocol: "chat",
    state: sharedState,
    options: { defaultModel: "claude-sonnet-5-5", injectProtocolEffort: true },
  });
  assert.ok(afterReq, "Virtual model must be rewritten in onRequest when defaultModel is set");
  assert.match(afterReq.model, /^claude-sonnet-5-5:(medium|high|xhigh)$/);
  assert.ok(sharedState.agentcompass?.profile, "Task profile must be saved in ctx.state for onProvider/onAccount");

  const afterProv = mwOnProvider(afterReq, {
    protocol: "chat",
    state: sharedState,
    provider: {
      id: "anthropic",
      name: "Anthropic",
      account: "work",
      model: afterReq.model,
    },
    options: {
      injectProtocolEffort: true,
      modelsByProvider: {
        anthropic: ["claude-haiku-5-5", "claude-sonnet-5-5"],
      },
    },
  });
  assert.ok(afterProv);
  assert.match(afterProv.model, /^claude-sonnet-5-5:(medium|high|xhigh)$/);
});

test("9.3 PR #670 Review Guard: `hooks.onRequest` never injects `__agentcompass_route` and `mwOnRequest` strips it on claimed models", async () => {
  // 1. `hooks.onRequest` leaves zero `__agentcompass_route` property on `body` even if `useRouteCarrier: true` is passed
  const req = {
    model: "agentcompass/auto",
    messages: [{ role: "user", content: "Fix the bug in the parser." }],
  };
  const hookCtx = {
    protocol: "chat",
    state: {},
    options: { candidateModels: ["claude-sonnet-5-5"], useRouteCarrier: true },
  };
  const hookOut = await hookOnRequest(req, hookCtx);
  assert.equal(hookOut, undefined);
  assert.equal(ROUTE_CARRIER_KEY in req, false);

  // 2. On a claimed request (`model: "agentcompass"`), `mwOnRequest` strips `__agentcompass_route` from the returned body
  const claimedWithCarrier = {
    model: "agentcompass",
    messages: [{ role: "user", content: "Fix the bug in the parser." }],
    [ROUTE_CARRIER_KEY]: { targetModel: "claude-opus-5-5" },
  };
  const routed = mwOnRequest(claimedWithCarrier, {
    protocol: "chat",
    state: {},
    options: { candidateModels: ["claude-sonnet-5-5"] },
  });
  assert.ok(routed);
  assert.equal(ROUTE_CARRIER_KEY in routed, false);
});

test("9.4 PR #670 Review Guard: `onAccount` natively matches Magpie's `{ref, provider, name, kind, model}` context", () => {
  const state = {
    agentcompass: {
      workKind: "debug_review",
      targetModel: "claude-sonnet-5-5",
      targetEffort: "high",
    },
  };

  // Match by `ref`
  assert.equal(
    mwOnAccount(
      { ref: "deadbeefcafe1234", provider: "anthropic", name: "Team Seat", kind: "claude", model: "claude-sonnet-5-5" },
      { state, options: { accountsByWorkKind: { debug_review: ["deadbeefcafe1234"] } } }
    ),
    undefined
  );

  // Match by `provider/name`
  assert.equal(
    mwOnAccount(
      { ref: "deadbeefcafe1234", provider: "anthropic", name: "work", kind: "claude", model: "claude-sonnet-5-5" },
      { state, options: { accountsByWorkKind: { debug_review: ["anthropic/work"] } } }
    ),
    undefined
  );

  // Skip when neither `ref`, `name`, `provider`, `kind`, nor `${provider}/${name}` matches
  assert.deepEqual(
    mwOnAccount(
      { ref: "deadbeefcafe1234", provider: "openai", name: "personal", kind: "codex", model: "gpt-5.3-codex" },
      { state, options: { accountsByWorkKind: { debug_review: ["anthropic/work", "claude"] } } }
    ),
    { skip: true }
  );
});

test("9.5 PR #670 Review Guard: Zero implicit external scorer calls and zero parent-directory `quotas.json` walking", async () => {
  clearProfileCache();
  clearQuotasCache();

  // 1. `loadLiveQuotas({})` returns `null` without walking parent directories or reading disk
  assert.equal(loadLiveQuotas({}), null);

  // 2. `scoreTaskLive` with `scorerType: "openai_compatible"` or `"jev"` and `scorerBaseUrl` set,
  // but WITHOUT `allowExternalScorer: true` (and without injected test `fetch`), makes 0 network calls!
  const origFetch = globalThis.fetch;
  let globalFetchCalled = false;
  globalThis.fetch = async () => {
    globalFetchCalled = true;
    throw new Error("globalThis.fetch must not be called without allowExternalScorer: true");
  };
  try {
    const res1 = await scoreTaskLive(
      { objective: "Design a distributed consensus protocol." },
      {
        scorerType: "openai_compatible",
        scorerBaseUrl: "http://127.0.0.1:4096",
        useCache: false,
      }
    );
    assert.equal(globalFetchCalled, false);
    assert.equal(res1.meta.source, "unavailable");

    const res2 = await scoreTaskLive(
      { objective: "Design a distributed consensus protocol." },
      {
        scorerType: "jev",
        jevBaseUrl: "https://api.typesafe.ai",
        useCache: false,
      }
    );
    assert.equal(globalFetchCalled, false);
    assert.equal(res2.meta.source, "unavailable");
  } finally {
    globalThis.fetch = origFetch;
  }
});

test("9.6 PR #670 Round 2 & 3 Guard: Client body injection (`__agentcompass_route`, `useRouteCarrier`, `available_models`, `candidate_models`) is rejected", async () => {
  // 1. Unclaimed model (`"claude-sonnet-5-5"`) is untouched and returns `undefined`, even if `__agentcompass_route` and `useRouteCarrier: true` are passed
  const nonClaimedBody = {
    model: "claude-sonnet-5-5",
    messages: [{ role: "user", content: "hello" }],
    [ROUTE_CARRIER_KEY]: { targetModel: "evil", modelWithEffort: "evil:max" },
  };
  const nonClaimedRes = mwOnRequest(nonClaimedBody, {
    protocol: "chat",
    state: {},
    options: { useRouteCarrier: true },
  });
  assert.equal(nonClaimedRes, undefined);
  assert.equal(nonClaimedBody.model, "claude-sonnet-5-5");

  // 2a. Claimed model (`"agentcompass"`) WITH `candidateModels` where client `__agentcompass_route` names a model
  // INSIDE `candidateModels` (`"claude-opus-5-5"`) AND passes `{ useRouteCarrier: true }` + a malformed `profile.work_kind`:
  // MUST ignore the client carrier completely, never pass client `profile` to `selectBestRoute`, and route the rename to `"claude-haiku-5-5:high"`!
  const claimedInCandidatesBody = {
    model: "agentcompass",
    messages: [{ role: "user", content: "rename the variable foo to bar in utils.go" }],
    [ROUTE_CARRIER_KEY]: {
      targetModel: "claude-opus-5-5",
      targetEffort: "max",
      modelWithEffort: "claude-opus-5-5:max",
      profile: { work_kind: "invalid_evil_work_kind" },
    },
  };
  const claimedInCandidatesRes = mwOnRequest(claimedInCandidatesBody, {
    protocol: "chat",
    state: {},
    options: {
      useRouteCarrier: true,
      candidateModels: ["claude-opus-5-5", "claude-haiku-5-5", "gpt-6.1-sol"],
    },
  });
  assert.ok(claimedInCandidatesRes);
  assert.equal(ROUTE_CARRIER_KEY in claimedInCandidatesRes, false);
  assert.equal(claimedInCandidatesRes.model, "claude-haiku-5-5:high");

  // 2b. Claimed model (`"agentcompass"`) where client `__agentcompass_route` names a model OUTSIDE `candidateModels`
  // and supplies a hostile `profile.work_kind`: MUST NOT pass `profile` to `selectBestRoute` or throw!
  const claimedOutsideCandidatesBody = {
    model: "agentcompass",
    messages: [{ role: "user", content: "rename the variable foo to bar in utils.go" }],
    [ROUTE_CARRIER_KEY]: {
      targetModel: "evil",
      targetEffort: "max",
      modelWithEffort: "evil:max",
      profile: { work_kind: "unknown_hostile_work_kind" },
    },
  };
  const claimedOutsideCandidatesRes = mwOnRequest(claimedOutsideCandidatesBody, {
    protocol: "chat",
    state: {},
    options: {
      useRouteCarrier: true,
      candidateModels: ["claude-opus-5-5", "claude-haiku-5-5", "gpt-6.1-sol"],
    },
  });
  assert.ok(claimedOutsideCandidatesRes);
  assert.equal(claimedOutsideCandidatesRes.model, "claude-haiku-5-5:high");

  // 2c. Claimed model (`"agentcompass"`) with NO `candidateModels` (`allowUnconstrainedCatalog: true`) and `{ useRouteCarrier: true }`:
  // client `__agentcompass_route` (`targetModel: "evil"`) MUST be ignored and never steer routing to `"evil:max"`!
  const claimedNoCandidatesBody = {
    model: "agentcompass",
    messages: [{ role: "user", content: "rename the variable foo to bar in utils.go" }],
    [ROUTE_CARRIER_KEY]: { targetModel: "evil", targetEffort: "max", modelWithEffort: "evil:max" },
  };
  const claimedNoCandidatesRes = mwOnRequest(claimedNoCandidatesBody, {
    protocol: "chat",
    state: {},
    options: { useRouteCarrier: true, allowUnconstrainedCatalog: true },
  });
  assert.ok(claimedNoCandidatesRes);
  assert.equal(ROUTE_CARRIER_KEY in claimedNoCandidatesRes, false);
  assert.equal(claimedNoCandidatesRes.model, "gpt-6-luna:high");

  // 3. Client-supplied `body.available_models` and `body.candidate_models` MUST NOT influence candidate model selection
  const bodySpoofCandidates = {
    model: "claude-sonnet-5-5",
    available_models: ["evil-model-1", "evil-model-2"],
    candidate_models: ["evil-model-1", "evil-model-2"],
    messages: [{ role: "user", content: "rename the variable foo to bar in utils.go" }],
  };
  const spoofHookCtx = { protocol: "chat", state: {}, options: { routeAllModels: true } };
  await hookOnRequest(bodySpoofCandidates, spoofHookCtx);
  assert.equal(spoofHookCtx.state.route.targetModel, "claude-sonnet-5-5");

  const spoofMwRes = mwOnRequest(structuredClone(bodySpoofCandidates), {
    protocol: "chat",
    state: {},
    options: { routeAllModels: true },
  });
  assert.ok(spoofMwRes);
  assert.match(spoofMwRes.model, /^claude-sonnet-5-5:/);
});

test("9.7 PR #670 Round 2 Guard: Mainstream real model IDs & trivial single-variable rename routing across all candidate orders", () => {
  // 1. Built-in catalog (`allowUnconstrainedCatalog: true`) routes "rename the variable foo to bar in utils.go" to `gemini-3.8-flash-high:high`
  const builtInBody = {
    model: "agentcompass",
    messages: [{ role: "user", content: "rename the variable foo to bar in utils.go" }],
  };
  const builtInRes = mwOnRequest(builtInBody, {
    protocol: "chat",
    state: {},
    options: { allowUnconstrainedCatalog: true },
  });
  assert.ok(builtInRes);
  assert.equal(builtInRes.model, "gpt-6-luna:high");

  // 2. Across all 6 permutations of `["claude-opus-5-5", "claude-haiku-5-5", "gpt-6.1-sol"]`:
  //    - Trivial single-variable rename routes to `"claude-haiku-5-5:high"`
  //    - Complex multi-module concurrency/deadlock architecture audit routes to `"claude-opus-5-5:high"` or `"claude-opus-5-5:xhigh"`
  const permutations = [
    ["claude-opus-5-5", "claude-haiku-5-5", "gpt-6.1-sol"],
    ["claude-opus-5-5", "gpt-6.1-sol", "claude-haiku-5-5"],
    ["claude-haiku-5-5", "claude-opus-5-5", "gpt-6.1-sol"],
    ["claude-haiku-5-5", "gpt-6.1-sol", "claude-opus-5-5"],
    ["gpt-6.1-sol", "claude-opus-5-5", "claude-haiku-5-5"],
    ["gpt-6.1-sol", "claude-haiku-5-5", "claude-opus-5-5"],
  ];

  for (const candidateModels of permutations) {
    const renameReq = {
      model: "agentcompass",
      messages: [{ role: "user", content: "rename the variable foo to bar in utils.go" }],
    };
    const renameOut = mwOnRequest(renameReq, {
      protocol: "chat",
      state: {},
      options: { candidateModels, continuityStore: new SessionContinuityStore() },
    });
    assert.equal(
      renameOut.model,
      "claude-haiku-5-5:high",
      `Expected claude-haiku-5-5:low for permutation [${candidateModels.join(", ")}], got ${renameOut.model}`
    );

    const auditReq = {
      model: "agentcompass",
      messages: [
        {
          role: "user",
          content:
            "Conduct a deep cross-module root-cause audit of the intermittent deadlock and lock-ordering race condition across internal/gateway/affinity.go and internal/gateway/decide.go.",
        },
      ],
    };
    const auditOut = mwOnRequest(auditReq, {
      protocol: "chat",
      state: {},
      options: { candidateModels, continuityStore: new SessionContinuityStore() },
    });
    assert.match(
      auditOut.model,
      /^claude-opus-5-5:(high|xhigh|max)$/,
      `Expected claude-opus-5-5:(high|xhigh|max) for permutation [${candidateModels.join(", ")}], got ${auditOut.model}`
    );
  }
});

test("9.8 PR #670 Round 2 Guard: `options.fetch` alone never enables external scorer unless `allowExternalScorer === true`", async () => {
  clearProfileCache();
  let fetchCalled = false;
  const customFetch = async () => {
    fetchCalled = true;
    throw new Error("options.fetch must not be called when allowExternalScorer is not true");
  };

  const scoreRes = await scoreTaskLive(
    { objective: "Refactor the authentication service." },
    {
      scorerType: "openai_compatible",
      scorerBaseUrl: "http://127.0.0.1:4096",
      fetch: customFetch,
      useCache: false,
    }
  );
  assert.equal(fetchCalled, false);
  assert.equal(scoreRes.meta.source, "unavailable");

  const hookReq = {
    model: "agentcompass",
    messages: [{ role: "user", content: "Refactor the authentication service." }],
  };
  await hookOnRequest(hookReq, {
    protocol: "chat",
    state: {},
    options: {
      scorerType: "openai_compatible",
      scorerBaseUrl: "http://127.0.0.1:4096",
      fetch: customFetch,
      candidateModels: ["claude-sonnet-4-6"],
    },
  });
  assert.equal(fetchCalled, false);
});

test("9.9 PR #670 Round 2 & 3 Guard: Single-pass bounded work (`scannedChars`, `extractedTurns`) as body grows", () => {
  const messages = [];
  const hugeLogChunk = "x".repeat(50_000) + "\n<system-reminder>hook context reminder</system-reminder>\n";
  for (let i = 0; i < 25; i++) {
    messages.push({
      role: "user",
      content: [
        {
          type: "text",
          text:
            i === 0
              ? `<system-reminder>Session initialized</system-reminder>\nImplement the distributed rate limiter middleware.\n${hugeLogChunk}`
              : `Tool output batch ${i}:\n${"z".repeat(4000)}`,
        },
      ],
    });
    messages.push({
      role: "assistant",
      content: [
        {
          type: "text",
          text: `Completed step ${i}. Reading next file section:\n` + "y".repeat(4000),
        },
      ],
    });
  }
  messages.push({
    role: "user",
    content:
      "Now audit the cross-module lock ordering invariants and fix the race condition in internal/gateway/affinity.go.\n" +
      hugeLogChunk +
      "\nEnsure deadlock freedom across all goroutines.",
  });

  const rawSize = JSON.stringify(messages).length;
  assert.ok(rawSize >= 250_000, `Expected synthetic Claude Code payload >= 250KB, got ${rawSize} bytes`);

  const req = {
    model: "agentcompass",
    system: "You are Claude Code, Anthropic's official CLI for Claude.\n" + "s".repeat(12_000),
    messages,
  };

  // 1. Verify work-boundedness invariants directly (raising MAX_NOISE_STRIP_CHARS or MAX_USER_TURN_CHARS turns these red)
  const facts = extractRequestFacts(req, { protocol: "messages" });
  assert.ok(
    facts.extractedTurns <= 4,
    `Expected at most 4 full-text user turns extracted across ${messages.length} messages, got ${facts.extractedTurns}`
  );
  assert.ok(
    facts.scannedChars <= 36_000,
    `Expected total scannedChars bounded by <= 36,000 on ${(rawSize / 1024).toFixed(1)}KB body, got ${facts.scannedChars}`
  );
  assert.ok(
    facts.objective.length <= 4_001,
    `Expected extracted objective capped at <= 4,001 chars (MAX_USER_TURN_CHARS), got ${facts.objective.length}`
  );

  // 2. Generous smoke ceiling (200ms) for end-to-end `mwOnRequest`
  const tStart = performance.now();
  const out = mwOnRequest(req, {
    protocol: "messages",
    headers: { "x-session-id": "perf-bench-run" },
    state: {},
    // Affinity off: the last turn's capped text alone decides the route. Under the default `cache` affinity
    // the 25 turns before it are a session, decided again from the history and held (checked below).
    options: {
      candidateModels: ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"],
      affinityMode: "off",
      continuityStore: new SessionContinuityStore(),
    },
  });
  const elapsedMs = performance.now() - tStart;

  assert.ok(out);
  assert.match(out.model, /^claude-opus-5-5:(high|xhigh|max)$/);
  assert.ok(
    elapsedMs < 200,
    `Expected single-pass mwOnRequest smoke check on ${(rawSize / 1024).toFixed(1)}KB body to finish in < 200ms, took ${elapsedMs.toFixed(2)}ms`
  );

  // 3. The same request on a runtime that never saw the session decides its earlier turns again from the
  // history (`MAX_REPLAY_DEPTH` of them): bounded too
  const tReplay = performance.now();
  const replayed = mwOnRequest({ ...structuredClone(req), model: "agentcompass" }, {
    protocol: "messages",
    state: {},
    options: {
      candidateModels: ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"],
      continuityStore: new SessionContinuityStore(),
    },
  });
  const replayMs = performance.now() - tReplay;
  assert.match(replayed.model, /^claude-(opus|sonnet|haiku)-5-5:/);
  assert.ok(replayMs < 200, `Expected replayed mwOnRequest on ${(rawSize / 1024).toFixed(1)}KB body in < 200ms, took ${replayMs.toFixed(2)}ms`);
});





test("7.7 Pooled runtimes: two runtimes with their own stores agree on the turn-1 route, the tool loop, the next turn and past the cache TTL", () => {
  // Magpie hands each request a moejs runtime from a sync.Pool, and middleware has no storage across requests
  // (internal/middleware/middleware.go). Each store below is one runtime's module state; Magpie's ctx carries no
  // headers, so the session is known only by its body.
  const candidateModels = [
    "claude-opus-5-5",
    "claude-sonnet-5-5",
    "claude-haiku-5-5",
    "gpt-6-astra",
    "gpt-6.1-sol",
    "gpt-6.1-terra",
    "gpt-6-sol",
    "gpt-6-luna",
    "gemini-4-argon-high",
    "gemini-3.8-flash-high",
    "grok-4.7",
    "deepseek-v4.1-flash-free",
  ].map((m) => `upstream/${m}`);
  const t0 = 1_700_000_000_000;
  const runtime = (store, now, profile, affinityMode) => ({
    protocol: "chat",
    state: {},
    ...(profile ? { profile } : {}),
    options: { candidateModels, continuityStore: store, now, ...(affinityMode ? { affinityMode } : {}) },
  });
  const first = {
    role: "user",
    content: "Audit our multi-module distributed lock manager to find the root cause of the deadlock and race condition.",
  };
  const toolLoop = [
    first,
    { role: "assistant", content: null, tool_calls: [{ id: "c1", type: "function", function: { name: "read_file", arguments: "{}" } }] },
    { role: "tool", tool_call_id: "c1", content: "package lock // simple utility" },
    { role: "user", content: "<system-reminder>The TodoWrite tool hasn't been used recently.</system-reminder>" },
  ];
  const followup = [
    first,
    { role: "assistant", content: "I found the lock inversion between ShardLock and TxLock. Shall I continue with the fix?" },
    { role: "user", content: "继续" },
  ];

  const storeA = new SessionContinuityStore();
  const turn1 = mwOnRequest({ model: "agentcompass/auto", messages: [first] }, runtime(storeA, t0));
  assert.ok(turn1?.model, "turn 1 is routed");

  // The tool loop and the follow-up each land on a runtime that never saw the session
  const loopB = mwOnRequest({ model: "agentcompass/auto", messages: toolLoop }, runtime(new SessionContinuityStore(), t0 + 1_000));
  // A short turn is a turn like any other: the classifier, which sees the previous turn, answers for it
  const sameTask = classifierAnswer(first.content);
  const followB = mwOnRequest({ model: "agentcompass/auto", messages: followup }, runtime(new SessionContinuityStore(), t0 + 2_000, sameTask));
  // ... and on the runtime that did
  const followA = mwOnRequest({ model: "agentcompass/auto", messages: followup }, runtime(storeA, t0 + 2_000, sameTask));

  assert.equal(loopB.model, turn1.model, "a fresh runtime holds the turn's model and effort in the tool loop");
  assert.equal(followB.model, turn1.model, "a fresh runtime routes the next turn the same for the same answer");
  assert.equal(followA.model, turn1.model, "the runtime that routed turn 1 agrees on the next turn");

  // A runtime whose record is of an earlier turn doesn't take its record for the current turn's
  const storeC = new SessionContinuityStore();
  const bulk = { role: "user", content: "Perform a mechanical batch rename of foo_bar to fooBar across JSON fixtures." };
  const turn2 = [first, { role: "assistant", content: "Fixed the lock inversion." }, bulk];
  mwOnRequest({ model: "agentcompass/auto", messages: [first] }, runtime(storeC, t0));
  const turn2B = mwOnRequest({ model: "agentcompass/auto", messages: turn2 }, runtime(new SessionContinuityStore(), t0 + 3_000));
  const turn2Loop = [
    ...turn2,
    { role: "assistant", content: null, tool_calls: [{ id: "c2", type: "function", function: { name: "grep", arguments: "{}" } }] },
    { role: "tool", tool_call_id: "c2", content: "fixtures/a.json" },
  ];
  const loopC = mwOnRequest({ model: "agentcompass/auto", messages: turn2Loop }, runtime(storeC, t0 + 4_000));
  const loopD = mwOnRequest({ model: "agentcompass/auto", messages: turn2Loop }, runtime(new SessionContinuityStore(), t0 + 4_000));
  assert.equal(loopC.model, turn2B.model, "a record of turn 1 doesn't decide turn 2's tool loop");
  assert.equal(loopD.model, turn2B.model, "turn 2's tool loop is held on a fresh runtime too");

  // Past the cache TTL. The history carries no time, so a fresh runtime can't know the age of the turn it decides
  // again: its state holds only in the tool loop, and the runtime with the record, which sees the real age, goes
  // cold as well. A turn 1 large enough that its cache is worth holding (>= 1024 tokens).
  const dump = Array.from({ length: 300 }, (_, i) => `shard-${i} acquire ShardLock then TxLock; waiter=${i + 1} holds TxLock, wants ShardLock`).join("\n");
  const big = { role: "user", content: `${first.content}\n\n${dump}` };
  const auditTask = classifierAnswer(first.content);
  const nudge = classifierAnswer("继续");
  const bulkTask = classifierAnswer(bulk.content);
  const storeE = new SessionContinuityStore();
  const bigTurn1 = mwOnRequest({ model: "agentcompass/auto", messages: [big] }, runtime(storeE, t0, auditTask));
  assert.ok(bigTurn1?.model, "the large turn 1 is routed");
  const bigFollow = [big, { role: "assistant", content: "Found the inversion. Continue with the fix?" }, { role: "user", content: "继续" }];
  const bigBulk = [big, { role: "assistant", content: "Fixed the lock inversion." }, bulk];
  const bigLoop = [
    big,
    { role: "assistant", content: null, tool_calls: [{ id: "c3", type: "function", function: { name: "read_file", arguments: "{}" } }] },
    { role: "tool", tool_call_id: "c3", content: "package lock" },
  ];
  const decide = (store, messages, at, profile, affinityMode) => {
    const ctx = runtime(store, at, profile, affinityMode);
    const out = mwOnRequest({ model: "agentcompass/auto", messages }, ctx);
    return { model: out.model, held: ctx.state.agentcompass?.continuation?.held, why: ctx.state.agentcompass?.continuation?.why };
  };
  const rows = [
    ["\"继续\" 30 min later", bigFollow, t0 + 30 * 60_000, nudge],
    ["\"继续\" 3 h later", bigFollow, t0 + 3 * 3_600_000, nudge],
    ["a new mechanical turn 30 min later", bigBulk, t0 + 30 * 60_000, bulkTask],
  ];
  for (const [label, messages, at, profile] of rows) {
    // Each row on its own copy of the store that routed turn 1, so one row's record doesn't become the next one's
    const withRecord = new SessionContinuityStore();
    mwOnRequest({ model: "agentcompass/auto", messages: [big] }, runtime(withRecord, t0, auditTask));
    const a = decide(withRecord, messages, at, profile);
    const b = decide(new SessionContinuityStore(), messages, at, profile);
    assert.equal(a.held, false, `${label}: the runtime with the record lets the session go cold (${a.why})`);
    assert.deepEqual(b, a, `${label}: a fresh runtime agrees with the runtime that has the record`);
  }
  // Inside the TTL the tool loop still holds on a fresh runtime: time doesn't matter there
  const loopE = decide(new SessionContinuityStore(), bigLoop, t0 + 30 * 60_000, auditTask);
  assert.equal(loopE.model, bigTurn1.model, "a tool loop holds turn 1's route on a fresh runtime whatever the pause");
  assert.equal(loopE.why, "turn");

  // In `session` and `strict` affinity modes, within 24h (`STICK_KEEP_MS`), both the runtime holding
  // the record and a fresh runtime (whose state is replayed from history) hold the session's model across turns.
  // Past 24h (`t0 + 25h`), the runtime holding the record expires the session directly (`why: "cold"`,
  // `reasonDetail: "session_expired"`, `replayed: false`) without falling back to `replayedState`.
  const decideCont = (store, messages, at, profile, affinityMode) => {
    const ctx = runtime(store, at, profile, affinityMode);
    const out = mwOnRequest({ model: "agentcompass/auto", messages }, ctx);
    const cont = ctx.state.agentcompass?.continuation;
    return {
      model: out.model,
      held: cont?.held,
      why: cont?.why,
      reasonDetail: cont?.reasonDetail,
      replayed: cont?.replayed,
    };
  };
  for (const mode of ["session", "strict"]) {
    const within24hStore = new SessionContinuityStore();
    mwOnRequest({ model: "agentcompass/auto", messages: [big] }, runtime(within24hStore, t0, auditTask, mode));
    const heldWithRecord = decideCont(within24hStore, bigBulk, t0 + 30 * 60_000, bulkTask, mode);
    const heldOnFresh = decideCont(new SessionContinuityStore(), bigFollow, t0 + 30 * 60_000, auditTask, mode);
    assert.equal(heldWithRecord.held, true, `${mode}: runtime with record holds within 24h`);
    assert.equal(heldWithRecord.why, mode);
    assert.equal(heldWithRecord.replayed, false);
    assert.equal(heldWithRecord.model, bigTurn1.model);
    assert.equal(heldOnFresh.held, true, `${mode}: fresh runtime holds replayed state within 24h`);
    assert.equal(heldOnFresh.why, mode);
    assert.equal(heldOnFresh.replayed, true);
    assert.equal(heldOnFresh.model, bigTurn1.model);

    for (const [label, messages, profile] of [
      [`${mode} mode: "继续" 25 h later`, bigFollow, nudge],
      [`${mode} mode: a new mechanical turn 25 h later`, bigBulk, bulkTask],
    ]) {
      const withRecord = new SessionContinuityStore();
      mwOnRequest({ model: "agentcompass/auto", messages: [big] }, runtime(withRecord, t0, auditTask, mode));
      const expired = decideCont(withRecord, messages, t0 + 25 * 3_600_000, profile, mode);
      assert.equal(expired.held, false, `${label}: the runtime with the record expires the session past 24h`);
      assert.equal(expired.why, "cold");
      assert.equal(expired.reasonDetail, "session_expired");
      assert.equal(expired.replayed, false);
    }

    const loopMode = decide(new SessionContinuityStore(), bigLoop, t0 + 25 * 3_600_000, auditTask, mode);
    assert.equal(loopMode.model, bigTurn1.model, `${mode}: a tool loop still holds turn 1's route on a fresh runtime`);
    assert.equal(loopMode.why, "turn");
  }
});

test("7.8 All candidate accounts unavailable returns accounts_unavailable refusal code, while capability or effort exclusion stays no_route", async () => {
  const nowMs = Date.parse("2026-04-12T12:00:00Z");
  const allUnavailableQuotas = {
    updated_at: "2026-04-12T11:55:00Z",
    accounts: {
      "claude:main": {
        provider: "claude",
        label: "main",
        status: "auth_failed",
        plan_type: "max_20x",
        five_hour: { remaining: 80, reset_at: "2026-04-12T16:00:00Z" },
        seven_day: { remaining: 80, reset_at: "2026-04-18T00:00:00Z" },
      },
      "codex:main": {
        provider: "codex",
        label: "main",
        status: "plan_inactive",
        plan_type: "free",
        five_hour: { remaining: 80, reset_at: "2026-04-12T16:00:00Z" },
        seven_day: { remaining: 80, reset_at: "2026-04-18T00:00:00Z" },
      },
      "grok:main": {
        provider: "grok",
        label: "main",
        status: "usage_failing",
        disabled: true,
        five_hour: { remaining: 80, reset_at: "2026-04-12T16:00:00Z" },
      },
    },
  };

  const scenario13CPool = [
    "claude/claude-opus-5-5",
    "claude/claude-sonnet-5-5",
    "codex/gpt-6.1-sol",
    "codex/gpt-6-astra",
    "grok/grok-4.7-build-fast",
  ];

  const codingObjective = "Perform a mechanical batch rename of `foo_bar` to `fooBar` across the JSON fixtures using a uniform rule.";
  const coupledObjective = "Audit our multi-module distributed lock manager to find the root cause of the deadlock and race condition.";

  // 1. selectBestRoute & middleware 503 when all candidates are excluded due to account unavailability
  const sel = selectBestRoute(codingObjective, {
    candidateModels: scenario13CPool,
    quotas: allUnavailableQuotas,
    nowMs,
  });
  assert.equal(sel.ok, false);
  assert.equal(sel.blocked, true);
  assert.equal(sel.error, "accounts_unavailable");
  assert.equal(sel.refusal, "accounts_unavailable: every candidate account is unavailable");

  // Also on a coupled task with bare model names (where lower efforts fall below the ability floor
  // but capable efforts clear the floor and are all excluded due to account unavailability)
  const selCoupled = selectBestRoute(coupledObjective, {
    candidateModels: scenario13CPool,
    quotas: allUnavailableQuotas,
    nowMs,
  });
  assert.equal(selCoupled.ok, false);
  assert.equal(selCoupled.error, "accounts_unavailable");
  assert.equal(selCoupled.refusal, "accounts_unavailable: every candidate account is unavailable");

  let rejected = null;
  const warnings = [];
  const origWarn = console.warn;
  console.warn = (...args) => warnings.push(args.join(" "));
  try {
    mwOnRequest(
      {
        model: "agentcompass/auto",
        messages: [{ role: "user", content: codingObjective }],
      },
      {
        protocol: "chat",
        state: {},
        options: {
          candidateModels: scenario13CPool,
          quotas: allUnavailableQuotas,
          nowMs,
        },
        reject(status, message) {
          rejected = { status, message };
        },
      }
    );
  } finally {
    console.warn = origWarn;
  }
  assert.deepEqual(rejected, {
    status: 503,
    message: "agentcompass: accounts_unavailable: every candidate account is unavailable",
  });
  assert.ok(!rejected.message.includes("claude:main") && !rejected.message.includes("codex:main"));
  assert.ok(warnings.some((w) => w.includes("claude:main: auth_failed") && w.includes("codex:main: plan_inactive")));

  // 2. Hook and System-One synthetic refusal honor accounts_unavailable without leaking pool details
  let hookRejected = null;
  const hookBody = {
    model: "agentcompass/auto",
    messages: [{ role: "user", content: codingObjective }],
  };
  const hookCtx = {
    state: {},
    options: {
      candidateModels: scenario13CPool,
      quotas: allUnavailableQuotas,
      nowMs,
    },
    reject(status, message) {
      hookRejected = { status, message };
    },
  };
  await hookOnRequest(hookBody, hookCtx);
  assert.deepEqual(hookRejected, {
    status: 503,
    message: "agentcompass: accounts_unavailable: every candidate account is unavailable",
  });
  const sys1 = await handleSystemOneRequest(
    {
      model: "agentcompass-systemone",
      state: { turn: codingObjective, quotas: allUnavailableQuotas },
      questions: { intent: { criteria: {} } },
    },
    {
      candidateModels: scenario13CPool,
      now: nowMs,
    }
  );
  assert.deepEqual(sys1.agentcompass, {
    targetModel: null,
    blocked: true,
    error: "accounts_unavailable",
    message: "every candidate account is unavailable",
  });

  // 3. When all candidates are excluded by non-quota reasons (or a capable candidate is excluded by
  // window quota rather than account unavailability), refusal stays no_route
  const baseProfile = classifierAnswer(coupledObjective);
  // 3a. Capability exclusion: grok-4.7-build-fast cannot serve a visual task (`needs_visual: true`)
  const visualSel = selectBestRoute(
    { ...baseProfile, needs_visual: true },
    {
      candidateModels: ["grok/grok-4.7-build-fast"],
      quotas: allUnavailableQuotas,
      nowMs,
    }
  );
  assert.equal(visualSel.ok, false);
  assert.equal(visualSel.refusal, "no_route: no candidate can serve this request");

  // 3b. Ability floor exclusion when every model in the pool is below the coupled floor (3.60)
  const floorModelSel = selectBestRoute(coupledObjective, {
    candidateModels: ["claude/claude-haiku-5-5", "grok/grok-4.7-build-fast"],
    quotas: allUnavailableQuotas,
    nowMs,
  });
  assert.equal(floorModelSel.ok, false);
  assert.equal(floorModelSel.refusal, "no_route: no candidate can serve this request");

  // 3c. Explicit effort exclusion below ability floor (`codex/gpt-6.1-sol:low` and `claude/claude-sonnet-5-5:low` on a coupled task)
  const explicitLowSel = selectBestRoute(baseProfile, {
    candidateModels: ["claude/claude-sonnet-5-5:low", "codex/gpt-6.1-sol:low"],
    quotas: allUnavailableQuotas,
    nowMs,
  });
  assert.equal(explicitLowSel.ok, false);
  assert.equal(explicitLowSel.refusal, "no_route: no candidate can serve this request");

  // 3d. Explicit effort exclusion due to realtime_loop responsiveness (`codex/gpt-6-astra:high` on realtime_loop)
  const rtSel = selectBestRoute(
    { ...classifierAnswer(codingObjective), realtime_loop: true },
    {
      candidateModels: ["codex/gpt-6-astra:high"],
      quotas: allUnavailableQuotas,
      nowMs,
    }
  );
  assert.equal(rtSel.ok, false);
  assert.equal(rtSel.refusal, "no_route: no candidate can serve this request");
});

const REALTIME_BASE_PROFILE = {
  work_kind: "computer_use",
  multi_task: false,
  task_count: "1",
  reasoning_depth: 2,
  code_complexity: 0,
  context_scale: 1,
  divergence: 1,
  hallucination_risk: 2,
  error_cost: 2,
  needs_visual: true,
  needs_gui: true,
  local_only: true,
  interaction_style: "engineering",
};

test("10.1 v0.4.1 Schema & validateProfile: `realtime_loop` is in schema, scorer prompt, Jev payload, and validated strictly as boolean", async () => {
  assert.ok(PROFILE_FLAGS.includes("realtime_loop"), "PROFILE_FLAGS must include realtime_loop");
  assert.ok(REALTIME_LOOP_QUESTION.includes("continuous observe-act loop"));
  assert.deepEqual(TASK_PROFILE_SCHEMA.properties.realtime_loop, {
    type: "boolean",
    description: REALTIME_LOOP_QUESTION,
  });
  assert.ok(TASK_PROFILE_SCHEMA.required.includes("realtime_loop"));
  assert.match(SCORER_SYSTEM_PROMPT, /- realtime_loop:\s*boolean/);
  assert.match(SCORER_SYSTEM_PROMPT, /continuous observe-act loop/);
  assert.match(SCORER_SYSTEM_PROMPT, /CRITICAL RULES/);

  const jevPayload = buildJevScorerPayload({ objective: "Play the daily dungeon in the Android emulator." });
  assert.deepEqual(jevPayload.questions.realtime_loop, {
    type: "noul",
    instructions: REALTIME_LOOP_QUESTION,
  });

  // validateProfile: defaults to false when omitted, accepts true/false, rejects non-boolean
  const omitted = validateProfile({ ...REALTIME_BASE_PROFILE });
  assert.equal(omitted.realtime_loop, false);
  const explicitTrue = validateProfile({ ...REALTIME_BASE_PROFILE, realtime_loop: true });
  assert.equal(explicitTrue.realtime_loop, true);
  const explicitFalse = validateProfile({ ...REALTIME_BASE_PROFILE, realtime_loop: false });
  assert.equal(explicitFalse.realtime_loop, false);

  for (const invalid of ["true", "false", 1, 0, null, {}, []]) {
    assert.throws(
      () => validateProfile({ ...REALTIME_BASE_PROFILE, realtime_loop: invalid }),
      /task\.profile\.realtime_loop must be true or false/,
    );
  }

  // Shape C `/v1/systemone` handler answers `questions.realtime_loop`
  const sysOneRes = await handleSystemOneRequest(
    {
      state: { task: { objective: "Monitor the live order book and place orders", realtime_loop: true } },
      questions: {
        realtime_loop: { type: "noul", instructions: REALTIME_LOOP_QUESTION },
      },
    },
  );
  assert.equal(sysOneRes.answers.realtime_loop.noul, 0.95);
});

test("10.2 v0.4.1 Classifier missing `realtime_loop` field defaults to `false` (both OpenAI-compatible and Jev)", async () => {
  clearProfileCache();
  const mockFetchMissingRealtime = async () =>
    new Response(
      JSON.stringify({
        choices: [
          {
            message: {
              content: JSON.stringify({
                work_kind: "general_implementation",
                reasoning_depth: 2,
                code_complexity: 2,
                context_scale: 1,
                divergence: 1,
                hallucination_risk: 2,
                error_cost: 2,
                needs_visual: false,
                needs_gui: false,
                local_only: true,
                interaction_style: "engineering",
                expected_hours: 1.0,
                hours_confidence: "high",
                urgency: "unknown",
                multi_task: false,
                task_count: "1",
              }),
            },
          },
        ],
      }),
      { status: 200 },
    );

  const missingRes = await scoreTaskLive(
    { objective: "Write a ticket sniping script in Python with Playwright." },
    {
      allowExternalScorer: true,
      scorerType: "openai_compatible",
      scorerBaseUrl: "http://127.0.0.1:4096",
      fetch: mockFetchMissingRealtime,
      useCache: false,
    },
  );
  assert.equal(missingRes.meta.source, "system_one_live");
  assert.equal(missingRes.profile.realtime_loop, false);

  const mockFetchTrueRealtime = async () =>
    new Response(
      JSON.stringify({
        choices: [
          {
            message: {
              content: JSON.stringify({
                work_kind: "computer_use",
                reasoning_depth: 2,
                code_complexity: 0,
                context_scale: 1,
                divergence: 1,
                hallucination_risk: 2,
                error_cost: 2,
                needs_visual: true,
                needs_gui: true,
                local_only: true,
                realtime_loop: true,
                interaction_style: "engineering",
                expected_hours: 1.0,
                hours_confidence: "high",
                urgency: "high",
                multi_task: false,
                task_count: "1",
              }),
            },
          },
        ],
      }),
      { status: 200 },
    );

  const trueRes = await scoreTaskLive(
    { objective: "Watch the 20:00 flash sale page in Chrome and click Buy immediately when inventory opens." },
    {
      allowExternalScorer: true,
      scorerType: "openai_compatible",
      scorerBaseUrl: "http://127.0.0.1:4096",
      fetch: mockFetchTrueRealtime,
      useCache: false,
    },
  );
  assert.equal(trueRes.meta.source, "system_one_live");
  assert.equal(trueRes.profile.realtime_loop, true);

  // Jev parser: missing `realtime_loop` defaults to false; explicit `noul >= 0.5` parses as true
  const fallback = {};
  const baseJevAnswers = {
    work_kind: { choice: "general_implementation" },
    interaction_style: { choice: "engineering" },
    multi_task: { noul: 0.05 },
    task_count: { choice: "1" },
    reasoning_depth: { score: 1 },
    code_complexity: { score: 2 },
    context_scale: { score: 0 },
    divergence: { score: 0 },
    hallucination_risk: { score: 1 },
    error_cost: { score: 1 },
    needs_visual: { noul: 0.1 },
    needs_gui: { noul: 0.1 },
    local_only: { noul: 0.9 },
  };
  const jevMissing = parseJevScorerAnswer({ answers: baseJevAnswers }, fallback);
  assert.equal(jevMissing.realtime_loop, false);
  const jevTrue = parseJevScorerAnswer(
    { answers: { ...baseJevAnswers, work_kind: { choice: "computer_use" }, realtime_loop: { noul: 0.85 } } },
    fallback,
  );
  assert.equal(jevTrue.realtime_loop, true);
});

test("10.3 v0.4.1 `realtime_loop` is the classifier's or the caller's, and `realtime_loop=true` never auto-promotes `needs_gui`/`needs_visual`/`local_only`", () => {
  const explicitTask = rankCandidates({
    profile: { ...REALTIME_BASE_PROFILE, needs_gui: false, needs_visual: false, local_only: false },
    realtime_loop: true,
  }).task.profile;
  assert.equal(explicitTask.realtime_loop, true);
  // realtime_loop=true must NOT auto-promote needs_gui, needs_visual, or local_only
  assert.equal(explicitTask.needs_gui, false, "realtime_loop=true must not auto-set needs_gui");
  assert.equal(explicitTask.needs_visual, false, "realtime_loop=true must not auto-set needs_visual");
  assert.equal(explicitTask.local_only, false, "realtime_loop=true must not auto-set local_only");

  // Explicit caller `realtime_loop: false` overrides `profile.realtime_loop: true`
  const overridden = rankCandidates({
    profile: { ...REALTIME_BASE_PROFILE, realtime_loop: true },
    realtime_loop: false,
  });
  assert.equal(overridden.task.realtime_loop, false);
  assert.equal(overridden.task.profile.realtime_loop, false);
});

test("10.4 v0.4.1 Slow models (`gemini-4-argon-high`, `grok-bot`, `dot`) are excluded on `realtime_loop` with 'too slow per turn for a live loop' while fast models remain eligible", () => {
  const profile = validateProfile({
    ...REALTIME_BASE_PROFILE,
    work_kind: "tool_execution", // route floor = 2.7
    needs_visual: false,
    needs_gui: false,
    realtime_loop: true,
  });
  const ranked = rankCandidates(profile, {
    candidateModels: [
      "gemini-4-argon-high:high",
      "grok-bot:high",
      "dot:high",
      "claude-sonnet-5-5:high",
      { id: "sol", provider: "codex", account: "codex", model: "gpt-6.1-sol" },
      "gpt-6-luna:high",
      "gemini-3.8-flash-high:medium",
    ],
  });

  const byModel = Object.fromEntries(ranked.allCandidates.map((c) => [c.model, c]));

  // Argon High (responsiveness 0.00 < 0.50) is excluded
  assert.equal(byModel["gemini-4-argon-high"].eligible, false);
  assert.ok(
    byModel["gemini-4-argon-high"].excluded.some(
      (r) => r.includes("below the realtime_loop floor 0.5"),
    ),
    `Expected Argon exclusion reason to mention realtime_loop floor and 'too slow per turn for a live loop', got: ${byModel["gemini-4-argon-high"].excluded.join("; ")}`,
  );

  // grok-bot and dot (`responsiveness: "not_applicable"`) are excluded
  for (const ch of ["grok-bot", "dot"]) {
    assert.equal(byModel[ch].eligible, false);
    assert.ok(
      byModel[ch].excluded.some(
        (r) => r.includes("not_applicable"),
      ),
      `Expected ${ch} exclusion reason to mention not_applicable and 'too slow per turn for a live loop', got: ${byModel[ch].excluded.join("; ")}`,
    );
  }

  // Fast models remain eligible (bare gpt-6.1-sol steps down from high to medium with resp 0.754 >= 0.50) and carry measured responsiveness >= 0.50
  for (const fastModel of ["claude-sonnet-5-5", "gpt-6.1-sol", "gemini-3.8-flash-high"]) {
    assert.equal(byModel[fastModel].eligible, true, `${fastModel} should be eligible on realtime_loop`);
    assert.ok(byModel[fastModel].components.responsiveness >= 0.5);
    assert.equal(byModel[fastModel].components.responsiveness_source, "measured");
  }
  assert.equal(byModel["gpt-6.1-sol"].effort, "medium");
  assert.equal(byModel["gpt-6.1-sol"].realtime_stepped_from, "high");
});

test("10.5 v0.4.1 Effort step-down rules: dispatchable efforts, encoded model IDs, antigravity lock, ability+responsiveness gate, sibling deference, and stepped-down cost", () => {
  const rtProfile = validateProfile({
    ...REALTIME_BASE_PROFILE,
    work_kind: "general_implementation",
    multi_task: false,
    task_count: "1",
    reasoning_depth: 2, // hard floor = 3.1
    needs_visual: false,
    needs_gui: false,
    realtime_loop: true,
    expected_hours: 2.0,
  });

  // 1. claude-opus-5-5:high (resp 0.107 < 0.50) steps down to medium (resp 0.53 >= 0.50, ability 4.2 >= 3.1)
  const opusRanked = rankCandidates(rtProfile, {
    candidateModels: [{ id: "claude_opus_high", provider: "claude", account: "claude", model: "claude-opus-5-5" }],
  });
  const opusCand = opusRanked.allCandidates[0];
  assert.equal(opusCand.eligible, true);
  assert.equal(opusCand.effort, "medium");
  assert.equal(opusCand.realtime_stepped_from, "high");
  assert.equal(opusCand.components.effort_stepped_down_from, "high");
  assert.equal(opusCand.components.responsiveness, 0.53);
  assert.ok(opusCand.notes.includes("effort stepped down from high to medium for realtime_loop"));

  // Cost / burn is computed at stepped-down effort (medium burn 0.50 / 0.70 vs high 1.0)
  const mediumScale = burnScale(DEFAULT_SCORING_POLICY, EFFORT_CURVES["claude-opus-5-5"], "medium");
  const expectedMediumUsd = Math.round(2.0 * DEFAULT_MODELS["claude-opus-5-5"].usd_per_hour * mediumScale * 10000) / 10000;
  assert.equal(opusCand.expected_usd, expectedMediumUsd);
  assert.ok(opusCand.expected_usd < 2.0 * DEFAULT_MODELS["claude-opus-5-5"].usd_per_hour);

  // 2. Effort encoded in model ID (`gemini-3.8-flash-high`, `gemini-4-argon-high`) or on `antigravity` never steps down
  assert.equal(splitEffortFromModelId("gemini-3.8-flash-high").effort, "high");
  assert.equal(splitEffortFromModelId("gemini-4-argon-high").effort, "high");
  assert.equal(splitEffortFromModelId("qwen3.8-max").effort, null, "qwen3.8-max is a base model name, not -max effort");

  const flashEncoded = rankCandidates(rtProfile, {
    candidateModels: [{ id: "oc_flash_high", provider: "opencode", model: "gemini-3.8-flash-high", effort: "high" }],
  }).allCandidates[0];
  assert.equal(flashEncoded.effort, "high");
  assert.equal(flashEncoded.realtime_stepped_from, undefined);
  assert.equal(flashEncoded.eligible, false, "gemini-3.8-flash-high at high (resp 0.486 < 0.50) has -high in model ID so it cannot step down");

  const agyBundle = rankCandidates(rtProfile, {
    candidateModels: [{ id: "agy_opus", provider: "antigravity-resilient", harness: "antigravity-resilient", model: "claude-opus-5-5", effort: "high" }],
  }).allCandidates[0];
  assert.equal(agyBundle.effort, "high");
  assert.equal(agyBundle.realtime_stepped_from, undefined);
  assert.equal(agyBundle.eligible, false);

  // 3. Only steps down to efforts allowed in DISPATCH_EFFORTS for the provider
  // For provider "claude", claude-haiku-5-5 is only dispatchable at ["high"] in DISPATCH_EFFORTS.claude (resp 0.439 < 0.50 at high)
  assert.deepEqual(DISPATCH_EFFORTS.claude["claude-haiku-5-5"], ["high"]);
  const haikuClaude = rankCandidates(rtProfile, {
    candidateModels: [{ id: "claude_haiku", provider: "claude", account: "claude", model: "claude-haiku-5-5", effort: "high" }],
  }).allCandidates[0];
  assert.equal(haikuClaude.effort, "high", "claude-haiku-5-5 on claude cannot step down to medium/low because DISPATCH_EFFORTS.claude only allows [high]");
  assert.equal(haikuClaude.eligible, false);

  // For provider "codex", gpt-6.1-terra is only dispatchable at ["high"] in DISPATCH_EFFORTS.codex (0.65), so a bare entry stays there
  const terraCodex = rankCandidates(rtProfile, {
    candidateModels: [{ id: "codex_terra_xhigh", provider: "codex", account: "codex", model: "gpt-6.1-terra" }],
  }).allCandidates[0];
  assert.equal(terraCodex.effort, "high");
  assert.equal(terraCodex.realtime_stepped_from, undefined);
  assert.equal(terraCodex.eligible, true);
  // A row's own effort steps within dispatch_efforts; `model:effort` fixes it (contract R2): too slow is excluded, not stepped
  const terraRow = rankCandidates(rtProfile, {
    candidateModels: [{ id: "codex_terra_xhigh", provider: "codex", account: "codex", model: "gpt-6.1-terra", effort: "xhigh" }],
  }).allCandidates[0];
  assert.equal(terraRow.effort, "high");
  assert.equal(terraRow.realtime_stepped_from, "xhigh");
  assert.equal(terraRow.eligible, true);
  const terraFixed = rankCandidates(rtProfile, {
    candidateModels: [{ id: "codex_terra_xhigh", provider: "codex", account: "codex", model: "gpt-6.1-terra:xhigh" }],
  }).allCandidates[0];
  assert.equal(terraFixed.effort, "xhigh");
  assert.equal(terraFixed.realtime_stepped_from, undefined);
  assert.equal(terraFixed.eligible, false);

  // 4. Ability gate vs responsiveness floor interaction on grok-4.7 (tier 3.4, ability: low=-0.2 -> 3.2, high=0 -> 3.4; responsiveness: low=0.744, high=0.201):
  // On hard (floor 3.1), low (3.2 >= 3.1) clears BOTH gate and resp_floor -> steps down to low and is eligible!
  const grokHard = rankCandidates(rtProfile, {
    candidateModels: [{ id: "grok_high", provider: "grok", account: "grok", model: "grok-4.7" }],
  }).allCandidates[0];
  assert.equal(grokHard.effort, "low");
  assert.equal(grokHard.realtime_stepped_from, "high");
  assert.equal(grokHard.eligible, true);

  // On frontier (floor 4.2), low (3.2 < 4.2) is the only effort clearing resp_floor, so it steps down to low (bestClearingResp) and is excluded by ability floor!
  const frontierProfile = validateProfile({ ...rtProfile, reasoning_depth: 4 });
  const grokFrontier = rankCandidates(frontierProfile, {
    candidateModels: [{ id: "grok_high", provider: "grok", account: "grok", model: "grok-4.7" }],
  }).allCandidates[0];
  assert.equal(grokFrontier.effort, "low");
  assert.equal(grokFrontier.eligible, false);
  assert.match(grokFrontier.excluded.join("; "), /ability 3\.20 at low is below the frontier floor 4\.20/);

  // 5. Sibling deference:
  // (a) When a healthy, applicable sibling (`claude_opus_med`) clears both gate and resp_floor, `claude_opus_high` defers (does not step down)
  const withHealthySibling = rankCandidates(rtProfile, {
    candidateModels: [
      { id: "claude_opus_high", provider: "claude", account: "claude", model: "claude-opus-5-5" },
      { id: "claude_opus_med", provider: "claude", account: "claude", model: "claude-opus-5-5", effort: "medium" },
    ],
  });
  const byIdHealthy = Object.fromEntries(withHealthySibling.allCandidates.map((c) => [c.id, c]));
  assert.equal(byIdHealthy.claude_opus_high.effort, "high", "must defer to healthy medium sibling instead of duplicating it");
  assert.equal(byIdHealthy.claude_opus_high.eligible, false);
  assert.equal(byIdHealthy.claude_opus_med.effort, "medium");
  assert.equal(byIdHealthy.claude_opus_med.eligible, true);

  // (b) When the medium sibling has `when: { max_reasoning_depth: 1 }` on a depth-2 task, it does not apply -> `claude_opus_high` steps down!
  const withInapplicableSibling = rankCandidates(rtProfile, {
    candidateModels: [
      { id: "claude_opus_high", provider: "claude", account: "claude", model: "claude-opus-5-5" },
      { id: "claude_opus_med", provider: "claude", account: "claude", model: "claude-opus-5-5", effort: "medium", when: { max_reasoning_depth: 1 } },
    ],
  });
  assert.deepEqual(withInapplicableSibling.ranking.map((c) => c.id), ["claude_opus_high"]);
  assert.equal(withInapplicableSibling.ranking[0].effort, "medium");
  assert.equal(withInapplicableSibling.ranking[0].realtime_stepped_from, "high");

  // (c) When the medium sibling is disabled, unavailable, or in launch_failures cooldown -> `claude_opus_high` does NOT defer and steps down, past a disabled effort
  const withDisabledSibling = rankCandidates(rtProfile, {
    candidateModels: [
      { id: "claude_opus_high", provider: "claude", account: "claude", model: "claude-opus-5-5" },
      { id: "claude_opus_med", provider: "claude", account: "claude", model: "claude-opus-5-5", effort: "medium", disabled: true },
    ],
  });
  const highWhenSibDisabled = withDisabledSibling.allCandidates.find((c) => c.id === "claude_opus_high");
  assert.equal(highWhenSibDisabled.effort, "low", "the user disabled medium on this account and model, so it is not a landing (contract R2)");
  assert.equal(highWhenSibDisabled.realtime_stepped_from, "high");
  assert.equal(highWhenSibDisabled.eligible, true);

  const nowMs = Date.parse("2026-11-06T12:00:00Z");
  const withCoolingSibling = rankCandidates(rtProfile, {
    now: nowMs,
    unavailable: ["claude_opus_med"],
    candidateModels: [
      { id: "claude_opus_high", provider: "claude", account: "claude", model: "claude-opus-5-5" },
      { id: "claude_opus_med", provider: "claude", account: "claude", model: "claude-opus-5-5", effort: "medium" },
    ],
  });
  const highWhenSibUnavailable = withCoolingSibling.allCandidates.find((c) => c.id === "claude_opus_high");
  assert.equal(highWhenSibUnavailable.effort, "medium");
  assert.equal(highWhenSibUnavailable.realtime_stepped_from, "high");

  // (d) When a faster sibling fails the ability gate (e.g. frontier gate 4.2 where opus:low is 4.6 - 0.84 = 3.76 < 4.2), opus:high does NOT defer to opus:low and steps down to opus:medium (4.40 >= 4.2)!
  const frontierRtProfile = validateProfile({
    ...rtProfile,
    multi_task: false,
    task_count: "1",
    reasoning_depth: 4, // frontier gate = 4.2
  });
  const withSubGateLowSibling = rankCandidates(frontierRtProfile, {
    candidateModels: [
      { id: "claude_opus_high", provider: "claude", account: "claude", model: "claude-opus-5-5" },
      { id: "claude_opus_low", provider: "claude", account: "claude", model: "claude-opus-5-5", effort: "low" },
    ],
  });
  const highWithLowSib = withSubGateLowSibling.allCandidates.find((c) => c.id === "claude_opus_high");
  assert.equal(highWithLowSib.effort, "medium", "must not defer to a low sibling (3.76) that fails the 4.2 frontier gate");
  assert.equal(highWithLowSib.realtime_stepped_from, "high");
  assert.equal(highWithLowSib.eligible, true);
});

test("10.6 v0.4.1 User-pinned model or effort on `realtime_loop` is neither stepped down nor excluded, only emits warning", () => {
  const rtProfile = validateProfile({
    ...REALTIME_BASE_PROFILE,
    work_kind: "general_implementation",
    needs_visual: false,
    needs_gui: false,
    realtime_loop: true,
  });

  // 1. Pinned bundle (`claude_opus_high` at high, resp 0.107 < 0.50): keeps high, stays eligible, emits warning
  const pinnedBundleRes = rankCandidates(rtProfile, {
    pinnedBundle: "claude_opus_high",
    candidateModels: [
      { id: "claude_opus_high", provider: "claude", account: "claude", model: "claude-opus-5-5", effort: "high" },
    ],
  });
  const pinnedOpus = pinnedBundleRes.allCandidates[0];
  assert.equal(pinnedOpus.eligible, true);
  assert.equal(pinnedOpus.effort, "high", "pinned bundle must not step down");
  assert.equal(pinnedOpus.realtime_stepped_from, undefined);
  assert.match(pinnedOpus.components.realtime_slow_pin_warning, /below the realtime_loop floor 0\.5.*kept per explicit user selection/);
  assert.ok(pinnedOpus.notes.some((n) => n.includes("kept per explicit user selection")));

  // 2. Pinned slow model (`gemini-4-argon-high` at high, resp 0.00 < 0.50): stays eligible, emits warning
  const pinnedArgonRes = rankCandidates(rtProfile, {
    constraintModel: "gemini-4-argon-high",
    candidateModels: ["gemini-4-argon-high:high"],
  });
  const pinnedArgon = pinnedArgonRes.allCandidates[0];
  assert.equal(pinnedArgon.eligible, true);
  assert.equal(pinnedArgon.effort, "high");
  assert.match(pinnedArgon.components.realtime_slow_pin_warning, /responsiveness 0\.00 is below the realtime_loop floor 0\.5/);

  // 3. Pinned not_applicable channel (`grok-bot`): stays eligible, emits not_applicable pin warning
  const pinnedGrokBot = rankCandidates({ ...rtProfile, local_only: false }, {
    constraintModel: "grok-bot",
    candidateModels: ["grok-bot"],
  }).allCandidates[0];
  assert.equal(pinnedGrokBot.eligible, true);
  assert.match(
    pinnedGrokBot.components.realtime_slow_pin_warning,
    /responsiveness is not_applicable for realtime_loop \(too slow per turn for a live loop\); kept per explicit user selection/,
  );
});

test("10.7 v0.4.1 Unmeasured models get conservative `0.52` (`responsiveness: default`), while date/`-latest`/`-preview` variants of measured models use measured curves", () => {
  const rtProfile = validateProfile({
    ...REALTIME_BASE_PROFILE,
    work_kind: "general_implementation",
    needs_visual: false,
    needs_gui: false,
    realtime_loop: true,
  });

  // 1. Models that have no measured responsiveness in AgentCompass (Python also returns 0.52 / 'default')
  for (const unmeasuredModel of [
    "gpt-6-pro",
    "doubao-seed-2-1-pro-260628",
    "qwen3-max",
    "qwen3-flash-2025-07-28",
    "llama-4-maverick",
    "custom-unmeasured-model-v1",
  ]) {
    const resolved = resolveModelSpec(unmeasuredModel);
    const [respVal, respSrc] = bundleResponsiveness({ model: unmeasuredModel, effort: "high" }, resolved.spec);
    assert.equal(respVal, 0.52, `${unmeasuredModel} must use conservative default 0.52`);
    assert.equal(respSrc, "default", `${unmeasuredModel} must have source 'default'`);

    const ranked = rankCandidates(rtProfile, { candidateModels: [`${unmeasuredModel}:high`] });
    const cand = ranked.allCandidates[0];
    assert.equal(cand.components.responsiveness, 0.52);
    assert.equal(cand.components.responsiveness_source, "default");
    assert.ok(cand.notes.includes("responsiveness: default"), `${unmeasuredModel} notes must include 'responsiveness: default'`);
  }

  // 2. Date / -latest / -preview variants of known measured models resolve to their measured responsiveness (matching Python canonical)
  const measuredVariants = [
    ["claude-opus-5-5-20260901", "high", 0.107],
    ["gpt-6.1-sol-20260901", "high", 0],
    ["gpt-6-astra-preview", "high", 0],
    ["grok-4.7-latest", "high", 0.201],
  ];
  for (const [variantModel, eff, expectedResp] of measuredVariants) {
    const [respVal, respSrc] = bundleResponsiveness({ model: variantModel, effort: eff });
    assert.equal(respVal, expectedResp, `${variantModel}@${eff} must resolve to measured ${expectedResp}`);
    assert.equal(respSrc, "measured", `${variantModel}@${eff} must have source 'measured'`);
  }

  // Verify computeTurnSeconds and turnSecondsToResponsiveness math matches AgentCompass allocator.py
  assert.equal(responsivenessConfig().floor, 0.5);
  assert.equal(responsivenessConfig().default, 0.52);
  assert.equal(turnSecondsToResponsiveness(4.0), 1.0);
  assert.equal(turnSecondsToResponsiveness(45.0), 0.0);
  const turnS = computeTurnSeconds(2.5, 100, "high", 9999, 400);
  assert.equal(turnS, 6.5, "reasoningTokens must not be added to ttft_s");
});

test("10.8 v0.4.1 Non-realtime routing invariance: `realtime_loop: false` leaves routing decisions and scores completely unchanged", () => {
  const testObjectives = [
    { objective: "Perform a mechanical batch rename of `foo_bar` to `fooBar` across the JSON fixtures using a uniform rule." },
    { objective: "Implement a standalone self-contained LRU cache algorithm module in TypeScript with unit tests." },
    { objective: "Conduct a read-only security audit and cross-module deadlock / race condition root-cause review across the gateway subsystem." },
    { objective: "Brainstorm and formulate a novel creative gameplay scheme and first-principles roadmap for an AI agent simulation game.", work_kind: "direction_reasoning" },
    { objective: "Provide a formal mathematical proof or refute the optimal-stopping theorem bound for this Markov chain." },
    { objective: "Operate the desktop app GUI: click the Settings button, take a screenshot, and toggle the proxy switch." },
    { objective: "Compare screenshots of the login dialog and inspect the screenshot for layout misalignment." },
    { objective: "Build the interactive React analytics dashboard component from the Figma spec.", work_kind: "frontend_implementation" },
    { objective: "Run the deployment script and report its exit status.", work_kind: "tool_execution" },
  ];

  // Legacy-equivalent evaluation: strip all responsiveness fields from policy/models/curves and verify identical results
  const strippedModels = Object.fromEntries(
    Object.entries(DEFAULT_MODELS).map(([k, v]) => {
      const { responsiveness, responsiveness_status, score_estimated, ...rest } = v;
      return [k, rest];
    }),
  );
  const strippedCurves = Object.fromEntries(
    Object.entries(EFFORT_CURVES).map(([k, v]) => {
      const { responsiveness, ...rest } = v;
      return [k, rest];
    }),
  );
  const { responsiveness: _unusedRespPolicy, ...strippedPolicy } = DEFAULT_SCORING_POLICY;

  for (const input of testObjectives) {
    const profile = classifierAnswer(input);
    assert.equal(profile.realtime_loop, false);

    const currentRanked = rankCandidates(profile);
    const legacyRanked = rankCandidates(profile, {
      catalog: {
        models: strippedModels,
        effortCurves: strippedCurves,
        effortResponsiveness: {},
        policy: strippedPolicy,
      },
    });

    assert.equal(currentRanked.selected.id, legacyRanked.selected.id, `Selected route changed for: ${input.objective}`);
    assert.equal(currentRanked.selected.score, legacyRanked.selected.score, `Selected score changed for: ${input.objective}`);
    assert.deepEqual(
      currentRanked.ranking.map((c) => ({ id: c.id, score: c.score, components: c.components })),
      legacyRanked.ranking.map((c) => ({ id: c.id, score: c.score, components: c.components })),
      `Full candidate ranking or components changed on non-realtime task: ${input.objective}`,
    );
    assert.equal(
      "responsiveness" in currentRanked.selected.components,
      false,
      "Non-realtime candidate components must not include responsiveness",
    );
  }
});

test("10.9 Finding 1: `lookupCurveResponsiveness` single-key & tier-aligned fallbacks (0 diffs across all 31 models x 9 efforts vs Python `bundle_responsiveness`)", () => {
  // 1. Direct lookupCurveResponsiveness checks: single-key fallback and tier-aligned fallbacks
  assert.equal(lookupCurveResponsiveness({ high: 0 }, "low"), 0);
  assert.equal(lookupCurveResponsiveness({ high: 0 }, "medium"), 0);
  assert.equal(lookupCurveResponsiveness({ high: 0 }, "max"), 0);
  assert.equal(lookupCurveResponsiveness({ default: 0.964 }, "high"), 0.964);
  assert.equal(lookupCurveResponsiveness({ low: 0.744, high: 0.201, xhigh: 0 }, "max"), 0.201);
  assert.equal(lookupCurveResponsiveness({ low: 0.744, high: 0.201, xhigh: 0 }, "medium"), 0.744);
  assert.equal(lookupCurveResponsiveness({ none: 1, high: 0.46, max: 0 }, "low"), 1);
  assert.equal(lookupCurveResponsiveness({ default: 0, xhigh: 0 }, "low"), 0);
  assert.equal(lookupCurveResponsiveness({ default: 0, xhigh: 0 }, "high"), 0);

  // 2. Exhaustive 31 models x 9 efforts parity against Python `allocator._lookup_curve_responsiveness` + `bundle_responsiveness`
  const pyLookupCurve = (curvesResp, effort, defaultEffort) => {
    if (!curvesResp || typeof curvesResp !== "object") return null;
    const eff = String(effort || defaultEffort || "high");
    if (eff in curvesResp && typeof curvesResp[eff] === "number") {
      return Math.max(0.0, Math.min(1.0, Number(curvesResp[eff])));
    }
    const vals = Object.values(curvesResp);
    if (vals.length === 1 && typeof vals[0] === "number") {
      return Math.max(0.0, Math.min(1.0, Number(vals[0])));
    }
    const fallbacks = ["high", "xhigh", "max", "ultra"].includes(eff)
      ? ["high", "default", "max", "xhigh"]
      : ["none", "minimal", "low", "default"].includes(eff)
        ? ["low", "none", "default", "medium"]
        : ["medium", "low", "default", "none", "high"];
    for (const fb of fallbacks) {
      if (fb in curvesResp && typeof curvesResp[fb] === "number") {
        return Math.max(0.0, Math.min(1.0, Number(curvesResp[fb])));
      }
    }
    return null;
  };

  const efforts = ["none", "off", "minimal", "low", "default", "medium", "high", "xhigh", "max"];
  let checkedPairs = 0;
  for (const [modelName, info] of Object.entries(DEFAULT_MODELS)) {
    for (const eff of efforts) {
      checkedPairs += 1;
      const status = info.responsiveness_status;
      const rawResp = info.responsiveness;
      let pyVal = null;
      let pySrc = "measured";
      if (rawResp === "not_applicable" || status === "not_applicable") {
        pyVal = null;
        pySrc = "not_applicable";
      } else {
        const defaultEff = String(info.default_effort || info.fixed_effort || "high");
        const curvesResp = EFFORT_CURVES[modelName]?.responsiveness || EFFORT_RESPONSIVENESS[modelName] || null;
        let base = pyLookupCurve(curvesResp, eff, defaultEff);
        if (base === null) {
          if (rawResp && typeof rawResp === "object") {
            base = pyLookupCurve(rawResp, eff, defaultEff);
          } else if (typeof rawResp === "number" && !info.score_estimated && eff === defaultEff) {
            base = Math.max(0.0, Math.min(1.0, Number(rawResp)));
          }
        }
        if (base === null) {
          base = 0.52;
          pySrc = "default";
        }
        pyVal = Math.round(base * 10000) / 10000;
      }
      const [jsVal, jsSrc] = bundleResponsiveness({ model: modelName, effort: eff }, info, DEFAULT_SCORING_POLICY);
      assert.equal(jsVal, pyVal, `bundleResponsiveness value mismatch on ${modelName}@${eff}`);
      assert.equal(jsSrc, pySrc, `bundleResponsiveness source mismatch on ${modelName}@${eff}`);
    }
  }
  assert.equal(checkedPairs, Object.keys(DEFAULT_MODELS).length * efforts.length);

  // 3. Slow models at non-default efforts cannot escape exclusion, and grok-4.7:max steps down to low
  const rtProfile = validateProfile({
    ...REALTIME_BASE_PROFILE,
    work_kind: "general_implementation",
    needs_visual: false,
    needs_gui: false,
    realtime_loop: true,
  });
  const ranked = rankCandidates(rtProfile, {
    candidateModels: [
      { id: "argon_med", provider: "antigravity", account: "agy", model: "gemini-4-argon-high", effort: "medium" },
      { id: "argon_low", provider: "antigravity", account: "agy", model: "gemini-4-argon-high", effort: "low" },
      { id: "qwen_max_low", provider: "opencode", account: "oc", model: "qwen3.8-max", effort: "low" },
      { id: "grok_max", provider: "grok", account: "grok", model: "grok-4.7" },
    ],
  });
  const byId = Object.fromEntries(ranked.allCandidates.map((c) => [c.id, c]));
  assert.equal(byId.argon_med.eligible, false);
  assert.equal(byId.argon_med.components.responsiveness, 0);
  assert.equal(byId.argon_med.components.responsiveness_source, "measured");
  assert.equal(byId.argon_low.eligible, false);
  assert.equal(byId.argon_low.components.responsiveness, 0);
  assert.equal(byId.qwen_max_low.eligible, false);
  assert.equal(byId.qwen_max_low.components.responsiveness, 0);
  assert.equal(byId.grok_max.eligible, true);
  assert.equal(byId.grok_max.effort, "low", "grok-4.7 must step down to low (0.744)");
  assert.equal(byId.grok_max.realtime_stepped_from, "high");
});

test("10.10 Finding 2: `isCandidatePinned` only matches the user-pinned bundle/model; pinned slow model wins selection with warnings while unpinned slow models are excluded", () => {
  // 1. Direct isCandidatePinned checks
  assert.equal(
    isCandidatePinned(
      { id: "ag_argon", model: "gemini-4-argon-high", effort: "high" },
      "gemini-4-argon-high",
      { constraint_model: "claude-sonnet-5-5", constraint_effort: "high" },
    ),
    false,
    "constraint_effort alone must not pin a different model",
  );
  assert.equal(
    isCandidatePinned(
      { id: "opus_high", model: "claude-opus-5-5", effort: "high" },
      "claude-opus-5-5",
      { constraint_model: "claude-opus-5-5", constraint_effort: "low" },
    ),
    false,
    "wrong effort on constrained model must not be treated as pinned",
  );
  assert.equal(
    isCandidatePinned(
      { id: "opus_low", model: "claude-opus-5-5", effort: "low" },
      "claude-opus-5-5",
      { constraint_model: "claude-opus-5-5", constraint_effort: "low" },
    ),
    true,
  );

  const rtProfile = validateProfile({
    ...REALTIME_BASE_PROFILE,
    work_kind: "general_implementation",
    needs_visual: false,
    needs_gui: false,
    realtime_loop: true,
  });

  // 2. When user pins claude-sonnet-5-5 at high, unpinned slow models at high are excluded without pin warnings
  const sonnetPinnedRes = rankCandidates(rtProfile, {
    constraintModel: "claude-sonnet-5-5",
    constraintEffort: "high",
    candidateModels: [
      { id: "ag_argon", provider: "antigravity", account: "agy", model: "gemini-4-argon-high", effort: "high" },
      { id: "claude_opus_high", provider: "claude", account: "claude", model: "claude-opus-5-5", effort: "high" },
      { id: "claude_sonnet_high", provider: "claude", account: "claude", model: "claude-sonnet-5-5", effort: "high" },
    ],
  });
  assert.equal(sonnetPinnedRes.selected.id, "claude_sonnet_high");
  const argonCand = sonnetPinnedRes.allCandidates.find((c) => c.id === "ag_argon");
  const opusCand = sonnetPinnedRes.allCandidates.find((c) => c.id === "claude_opus_high");
  assert.equal(argonCand.eligible, false);
  assert.equal(argonCand.components.realtime_slow_pin_warning, undefined);
  assert.equal(opusCand.eligible, false);
  assert.equal(opusCand.components.realtime_slow_pin_warning, undefined);

  // 3. When user pins a slow model in a multi-candidate pool, the pinned slow model wins selection and selectBestRoute exposes top-level warnings
  const slowPinnedDecision = selectBestRoute(rtProfile, {
    pinnedBundle: "ag_argon",
    constraintModel: "gemini-4-argon-high",
    candidateModels: [
      { id: "ag_argon", provider: "antigravity", account: "agy", model: "gemini-4-argon-high", effort: "high" },
      { id: "claude_sonnet_high", provider: "claude", account: "claude", model: "claude-sonnet-5-5", effort: "high" },
    ],
  });
  assert.equal(slowPinnedDecision.ok, true);
  assert.equal(slowPinnedDecision.targetModel, "gemini-4-argon-high");
  assert.equal(slowPinnedDecision.targetEffort, "high");
  assert.equal(slowPinnedDecision.warnings.length, 1);
  assert.match(slowPinnedDecision.warnings[0], /too slow per turn for a live loop.*kept per explicit user selection/);
});

test("10.11 Finding 3: `scoreTaskLive` caller-explicit `realtime_loop` overrides external scorer and isolates LRU cache key", async () => {
  clearProfileCache();
  const objective = "Monitor live flash-sale inventory and submit order as soon as button turns active.";

  const mockOpenAiRealtimeTrue = async () => ({
    ok: true,
    status: 200,
    async json() {
      return {
        choices: [
          {
            message: {
              content: JSON.stringify({
                work_kind: "computer_use",
                multi_task: false,
                task_count: "1",
                reasoning_depth: 2,
                code_complexity: 0,
                context_scale: 1,
                divergence: 1,
                hallucination_risk: 2,
                error_cost: 2,
                needs_visual: true,
                needs_gui: true,
                local_only: true,
                realtime_loop: true,
                interaction_style: "engineering",
              }),
            },
          },
        ],
      };
    },
  });

  // 1. Caller omitted realtime_loop -> external scorer's realtime_loop: true is used and cached under (objective, null)
  const resFromLlm = await scoreTaskLive(
    { objective },
    {
      allowExternalScorer: true,
      scorerType: "openai_compatible",
      scorerBaseUrl: "http://127.0.0.1:4096/v1",
      fetch: mockOpenAiRealtimeTrue,
      useCache: true,
    },
  );
  assert.equal(resFromLlm.profile.realtime_loop, true);
  assert.equal(resFromLlm.meta.cached, undefined);

  // 2. Caller explicitly passes realtime_loop: false on the SAME objective with useCache: true -> must NOT hit the true cache entry and must override LLM output to false
  const resCallerFalse = await scoreTaskLive(
    { objective, realtime_loop: false },
    {
      allowExternalScorer: true,
      scorerType: "openai_compatible",
      scorerBaseUrl: "http://127.0.0.1:4096/v1",
      fetch: mockOpenAiRealtimeTrue,
      useCache: true,
    },
  );
  assert.equal(resCallerFalse.profile.realtime_loop, false, "caller explicit realtime_loop: false must override LLM scorer");
  assert.equal(resCallerFalse.meta.cached, undefined, "cache key must isolate explicit realtime_loop: false from omitted");

  // 3. Jev scorer with caller explicit needs.realtime_loop: false also honors caller override
  const mockJevRealtimeTrue = async () => ({
    ok: true,
    status: 200,
    async json() {
      return {
        answers: {
          work_kind: { choice: "computer_use" },
          interaction_style: { choice: "engineering" },
          multi_task: { noul: 0.05 },
          task_count: { choice: "1" },
          reasoning_depth: { score: 1 },
          code_complexity: { score: 0 },
          context_scale: { score: 0 },
          divergence: { score: 0 },
          hallucination_risk: { score: 1 },
          error_cost: { score: 1 },
          needs_visual: { noul: 0.9 },
          needs_gui: { noul: 0.9 },
          local_only: { noul: 0.9 },
          realtime_loop: { noul: 0.95 },
        },
      };
    },
  });
  const resJevCallerFalse = await scoreTaskLive(
    { objective, needs: { realtime_loop: false } },
    {
      allowExternalScorer: true,
      scorerType: "jev",
      jevBaseUrl: "http://127.0.0.1:4096",
      fetch: mockJevRealtimeTrue,
      useCache: false,
    },
  );
  assert.equal(resJevCallerFalse.profile.realtime_loop, false, "caller needs.realtime_loop: false must override Jev scorer");
});

test("10.12 Finding 4: Continuation effort ratchet (`keepsEffort`, `handleSystemOneRequest`, `hooks.onRequest`, `middleware.onRequest`) never undoes `realtime_loop` step-down", async () => {
  const now = 1_760_000_000_000;

  // 1. Direct keepsEffort check: warm session at high does not ratchet stepped-down medium when realtimeLoop is true
  const warmSession = {
    model: "claude-opus-5-5",
    effort: "high",
    provider: "claude",
    tokens: 5000,
    requestTokens: 5000,
    updatedAt: now - 10_000,
  };
  assert.equal(keepsEffort(warmSession, "medium", 5200, now, { realtimeLoop: false }), "high");
  assert.equal(keepsEffort(warmSession, "medium", 5200, now, { realtimeLoop: true }), "medium");

  // 2. handleSystemOneRequest with previous_message_reasoning: "high" and realtime_loop: true keeps stepped-down medium
  const sysOneRes = await handleSystemOneRequest(
    {
      model: DECISION_MODEL_ID,
      state: {
        message: "Continue monitoring the live order book and placing limit orders on spread triggers.",
        previous_message_kind: "general_implementation",
        previous_message_reasoning: "high",
        realtime_loop: true,
      },
      questions: {
        intent: {
          type: "choice",
          criteria: {
            general_implementation: "General software implementation",
            bounded_bulk_work: "Routine work",
          },
        },
        effort: {
          type: "score",
          criteria: ["low", "medium", "high", "xhigh"],
        },
      },
    },
    {
      decisionProviderDetails: true,
      candidateModels: [
        { id: "claude_opus_high", provider: "claude", account: "claude", model: "claude-opus-5-5" },
      ],
    },
  );
  assert.equal(sysOneRes.agentcompass.realtimeLoop, true);
  assert.equal(sysOneRes.agentcompass.targetEffort, "medium", "handleSystemOneRequest must not ratchet stepped-down medium back to high");
  assert.equal(sysOneRes.answers.effort.score, 1, "effort answer score must be 1 (medium), not 2 (high)");

  // 3. hooks.onRequest & middleware.onRequest on a warm session at high re-evaluate when realtime_loop: true and keep stepped-down medium
  const store = new SessionContinuityStore();
  const turn1Body = {
    model: "agentcompass/auto",
    session_id: "sess_rt_ratchet",
    estimated_input_tokens: 4000,
    messages: [{ role: "user", content: "Implement the order book parser module in TypeScript." }],
  };
  const turn1Ctx = {
    now,
    state: {},
    options: {
      continuityStore: store,
      candidateModels: ["claude-opus-5-5:high"],
    },
  };
  await hookOnRequest(turn1Body, turn1Ctx);
  assert.equal(turn1Ctx.state.route.targetModel, "claude-opus-5-5");
  assert.equal(turn1Ctx.state.route.targetEffort, "high");

  const turn2Body = {
    model: "agentcompass/auto",
    session_id: "sess_rt_ratchet",
    estimated_input_tokens: 4500,
    realtime_loop: true,
    messages: [
      { role: "user", content: "Implement the order book parser module in TypeScript." },
      { role: "assistant", content: "Implemented." },
      { role: "user", content: "Now stay in the live loop watching the order book and firing orders." },
    ],
  };
  const turn2Ctx = {
    now: now + 5_000,
    state: {},
    options: {
      continuityStore: store,
      candidateModels: [{ id: "claude_opus", provider: "claude", account: "claude", model: "claude-opus-5-5" }],
    },
  };
  await hookOnRequest(turn2Body, turn2Ctx);
  assert.equal(turn2Ctx.state.route.realtimeLoop, true);
  assert.equal(turn2Ctx.state.route.targetEffort, "medium", "hooks.onRequest must not ratchet realtime_loop medium back to high");

  const mwStore = new SessionContinuityStore();
  mwOnRequest(
    {
      model: "agentcompass/auto",
      session_id: "sess_mw_rt",
      estimated_input_tokens: 4000,
      messages: [{ role: "user", content: "Implement the order book parser module in TypeScript." }],
    },
    {
      now,
      options: {
        continuityStore: mwStore,
        candidateModels: [{ id: "claude_opus", provider: "claude", account: "claude", model: "claude-opus-5-5" }],
      },
    },
  );
  const mwCtx2 = {
    now: now + 5_000,
    state: {},
    options: {
      continuityStore: mwStore,
      candidateModels: [{ id: "claude_opus", provider: "claude", account: "claude", model: "claude-opus-5-5" }],
    },
  };
  const mwTurn2 = mwOnRequest(
    {
      model: "agentcompass/auto",
      session_id: "sess_mw_rt",
      estimated_input_tokens: 4500,
      realtime_loop: true,
      messages: [
        { role: "user", content: "Implement the order book parser module in TypeScript." },
        { role: "assistant", content: "Implemented." },
        { role: "user", content: "Now stay in the live loop watching the order book and firing orders." },
      ],
    },
    mwCtx2,
  );
  assert.equal(mwTurn2.model, "claude-opus-5-5:medium", "middleware.onRequest must not ratchet realtime_loop medium back to high");
  assert.equal(mwCtx2.state.route.targetEffort, "medium");
});

test("10.13 Finding 5: String candidates and provider-less object candidates honor `spec.efforts` and `DISPATCH_EFFORTS` without fabricating step-down", () => {
  const rtProfile = validateProfile({
    ...REALTIME_BASE_PROFILE,
    multi_task: false,
    task_count: "1",
    reasoning_depth: 1,
    code_complexity: 1,
    work_kind: "bounded_bulk_work",
    needs_visual: false,
    needs_gui: false,
    realtime_loop: true,
  });

  // 1. String candidate "claude-haiku-5-5:high": DISPATCH_EFFORTS.claude["claude-haiku-5-5"] is ["high"], so it cannot step down to medium and is excluded (0.439 < 0.50)
  const haikuRes = rankCandidates(rtProfile, {
    candidateModels: ["claude-haiku-5-5:high"],
  });
  const haikuCand = haikuRes.allCandidates[0];
  assert.equal(haikuCand.effort, "high", "string candidate claude-haiku-5-5:high must respect DISPATCH_EFFORTS.claude=['high']");
  assert.equal(haikuCand.realtime_stepped_from, undefined);
  assert.equal(haikuCand.eligible, false);
  assert.ok(haikuCand.excluded.some((e) => e.includes("below the realtime_loop floor")));

  // 2. String candidate "gpt-6-luna:xhigh" fixes its effort, and DISPATCH_EFFORTS.codex["gpt-6-luna"] is ["high"]: it does not step, it is excluded
  const lunaRes = rankCandidates(rtProfile, {
    candidateModels: ["gpt-6-luna:xhigh"],
  });
  const lunaCand = lunaRes.allCandidates[0];
  assert.equal(lunaCand.effort, "xhigh");
  assert.equal(lunaCand.realtime_stepped_from, undefined);
  assert.equal(lunaCand.eligible, false);

  // 3. Provider-less object candidates `{ model: "gpt-6.1-terra" }`, `{ model: "gemini-4-argon-high" }`, and `{ model: "grok-bot" }` honor DISPATCH_EFFORTS / splitEffortFromModelId / scoringEffort on realtime_loop
  const objRes = rankCandidates(rtProfile, {
    candidateModels: [{ model: "gpt-6.1-terra" }, { model: "gemini-4-argon-high" }, { model: "grok-bot" }],
  });
  assert.deepEqual(
    objRes.allCandidates.map((c) => `${c.model}:${c.effort}`),
    [
      "gpt-6.1-terra:high",
      "gemini-4-argon-high:high",
      "grok-bot:high",
    ],
  );
  assert.equal(objRes.allCandidates.find((c) => c.model === "gpt-6.1-terra").eligible, true);
  assert.ok(
    objRes.allCandidates
      .filter((c) => c.model !== "gpt-6.1-terra")
      .every((c) => c.eligible === false),
  );
});

test("10.14 Finding 6: `MODEL_ALIASES` does not shadow dedicated `EFFORT_RESPONSIVENESS` / `DISPATCH_EFFORTS` for `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-haiku-4-5`, and `validateProfile` preserves `realtime_loop`", () => {
  // 1. Legacy Claude 4.6 / 4.5 models use their own dedicated curves in EFFORT_RESPONSIVENESS and DISPATCH_EFFORTS
  assert.deepEqual(
    bundleResponsiveness({ model: "claude-opus-4-6", effort: "none" }),
    { value: 0.853, source: "measured" },
  );
  assert.deepEqual(
    bundleResponsiveness({ model: "claude-opus-4-6", effort: "high" }),
    { value: 0, source: "measured" },
  );
  assert.deepEqual(
    bundleResponsiveness({ model: "claude-sonnet-4-6", effort: "low" }),
    { value: 0.874, source: "measured" },
  );
  assert.deepEqual(
    bundleResponsiveness({ model: "claude-sonnet-4-6", effort: "high" }),
    { value: 0, source: "measured" },
  );
  assert.deepEqual(
    bundleResponsiveness({ model: "claude-haiku-4-5", effort: "none" }),
    { value: 0.978, source: "measured" },
  );
  assert.deepEqual(
    bundleResponsiveness({ model: "claude-haiku-4-5", effort: "high" }),
    { value: 0, source: "measured" },
  );

  const rtProfile = validateProfile({
    ...REALTIME_BASE_PROFILE,
    work_kind: "general_implementation",
    needs_visual: false,
    needs_gui: false,
    realtime_loop: true,
  });
  const legacyClaudeRanked = rankCandidates(rtProfile, {
    candidateModels: ["claude-opus-4-6:high", "claude-sonnet-4-6:high", "claude-haiku-4-5:high"],
  });
  for (const cand of legacyClaudeRanked.allCandidates) {
    assert.equal(cand.effort, "high", `${cand.model} must not step down since DISPATCH_EFFORTS.claude only allows ['high']`);
    assert.equal(cand.eligible, false, `${cand.model}@high has responsiveness 0.0 and must be excluded`);
    assert.equal(cand.components.responsiveness, 0);
  }

  // 2. grok-bot keeps fixed_effort unset in DEFAULT_MODELS so non-realtime effort remains 'high' (m6)
  assert.equal(DEFAULT_MODELS["grok-bot"].fixed_effort, "medium");

  // 3. validateProfile preserves realtime_loop (true when true, false when false or omitted)
  const validatedTrue = validateProfile({ ...REALTIME_BASE_PROFILE, realtime_loop: true });
  assert.equal(validatedTrue.realtime_loop, true);
  assert.equal("realtime_loop" in validatedTrue, true);
  const { realtime_loop: _omitted, ...withoutRt } = REALTIME_BASE_PROFILE;
  const validatedOmitted = validateProfile(withoutRt);
  assert.equal(validatedOmitted.realtime_loop, false);
  assert.equal("realtime_loop" in validatedOmitted, true);
});

test("10.15 Round-2 B1, M2, m6", async () => {
  const nonRtProfile = {
    work_kind: "general_implementation",
    multi_task: false,
    task_count: "1",
    reasoning_depth: 3,
    code_complexity: 2,
    context_scale: 3,
    divergence: 2,
    hallucination_risk: 2,
    error_cost: 2,
    interaction_style: "exploration",
    needs_visual: false,
    needs_gui: false,
    local_only: false,
    realtime_loop: false,
  };

  // B1: Legacy Claude names read their own rows (contract R1), not the 5.5 flagship's
  const opus46NonRt = rankCandidates(nonRtProfile, { candidateModels: ["claude-opus-4-6"] });
  assert.deepEqual(opus46NonRt.allCandidates.map((c) => c.effort), ["high"]);
  assert.equal(opus46NonRt.allCandidates[0].components.ability_value, 2.74);

  // M2: A candidate without effort/efforts offers what its provider dispatches; a bundle is one candidate, as in AgentCompass
  for (const [candObj, efforts] of [
    [{ model: "claude-opus-5-5" }, ["low", "medium", "high", "xhigh", "max"]],
    [{ id: "o", provider: "claude", model: "claude-opus-5-5" }, ["high"]],
    [{ model: "gpt-6.1-terra" }, ["high"]],
  ]) {
    const res = rankCandidates(nonRtProfile, { candidateModels: [candObj] });
    assert.deepEqual(res.allCandidates.map((c) => c.effort), efforts, JSON.stringify(candObj));
  }

  // m6: grok-bot non-realtime effort remains "high" (not "medium")
  const grokBotNonRt = rankCandidates(nonRtProfile, { candidateModels: ["grok-bot"] });
  assert.equal(grokBotNonRt.selected.id, "grok-bot:high");
  assert.equal(grokBotNonRt.selected.effort, "high");
});

test("10.16 Round-2 M1 & m5: `SessionContinuityStore.recordRequest` persists `realtimeLoop` so consecutive realtime turns hold on cache without re-scoring, and plugin-level `options.realtime_loop` reaches continuation evaluation", async () => {
  const NOW = 1790000000000;
  const pad = "\n" + "context ".repeat(800);
  const msg = (t) => ({ role: "user", content: t });

  // 1. M1: Consecutive realtime_loop turns hold on cache ("why: cache", held: true) and call external scorer only on turn 1
  const store = new SessionContinuityStore();
  let scorerCalls = 0;
  const mockFetch = async () => {
    scorerCalls++;
    return {
      ok: true,
      status: 200,
      json: async () => ({
        choices: [
          {
            message: {
              content: JSON.stringify({
                reasoning_depth: 1,
                code_complexity: 0,
                context_scale: 1,
                divergence: 1,
                hallucination_risk: 1,
                error_cost: 1,
                needs_visual: false,
                needs_gui: false,
                local_only: false,
                realtime_loop: true,
                interaction_style: "engineering",
                expected_hours: 0.2,
                hours_confidence: "high",
                urgency: "unknown",
                multi_task: false,
                task_count: "1",
                work_kind: "general_implementation",
              }),
            },
          },
        ],
      }),
    };
  };

  clearProfileCache();
  const msgs = [];
  const routes = [];
  let t = NOW;
  for (const turnText of ["Keep driving the running browser game", "next action", "next action again"]) {
    msgs.push(msg(turnText + pad));
    const ctx = {
      options: {
        continuityStore: store,
        candidateModels: ["claude-opus-5-5:high", "gpt-6.1-terra:high"],
        allowExternalScorer: true,
        scorerType: "openai_compatible",
        scorerBaseUrl: "http://127.0.0.1:9999/v1",
        fetch: mockFetch,
        now: t,
      },
      state: {},
    };
    await hookOnRequest({ model: "agentcompass", messages: [...msgs], metadata: { session_id: "rt-session" } }, ctx);
    routes.push({
      modelWithEffort: ctx.state.route.modelWithEffort,
      why: ctx.state.route.continuation.why,
      held: ctx.state.route.continuation.held,
    });
    const sessionState = store.get(ctx.state.sessionKey, t);
    assert.equal(sessionState.realtimeLoop, true, "SessionContinuityStore must persist realtimeLoop: true");
    store.recordResponse(ctx.state.sessionKey, { usage: { input_tokens: 4000, output_tokens: 50 } }, { status: 200 }, t + 1);
    t += 20000;
    msgs.push({ role: "assistant", content: "done" });
  }

  assert.deepEqual(routes, [
    { modelWithEffort: "gpt-6.1-terra:high", why: "first", held: false },
    { modelWithEffort: "gpt-6.1-terra:high", why: "cache", held: true },
    { modelWithEffort: "gpt-6.1-terra:high", why: "cache", held: true },
  ]);
  assert.equal(scorerCalls, 1, "External scorer must be called only once on turn 1 when warm cache holds");

  // 2. m5: Plugin-level options.realtime_loop=true on a warm non-realtime session triggers capability_outgrown (realtime_loop_required)
  for (const mode of ["hooks", "middleware"]) {
    const m5Store = new SessionContinuityStore();
    const msgs = [msg("Implement the parser module and unit tests" + pad)];
    const ctx1 = {
      options: { continuityStore: m5Store, candidateModels: ["claude-opus-5-5"], now: NOW },
      state: {},
    };
    if (mode === "hooks") {
      await hookOnRequest({ model: "agentcompass", messages: [...msgs], metadata: { session_id: "m5" } }, ctx1);
    } else {
      mwOnRequest({ model: "agentcompass", messages: [...msgs], metadata: { session_id: "m5" } }, ctx1);
    }
    assert.equal(ctx1.state.route.modelWithEffort, "claude-opus-5-5:medium");
    assert.equal(ctx1.state.route.continuation.why, "first");

    msgs.push({ role: "assistant", content: "ok" }, msg("Now enter the live game loop and click on spawn" + pad));
    const ctx2 = {
      options: {
        continuityStore: m5Store,
        candidateModels: ["claude-opus-5-5"],
        realtime_loop: true,
        now: NOW + 10000,
      },
      state: {},
    };
    if (mode === "hooks") {
      await hookOnRequest({ model: "agentcompass", messages: [...msgs], metadata: { session_id: "m5" } }, ctx2);
    } else {
      mwOnRequest({ model: "agentcompass", messages: [...msgs], metadata: { session_id: "m5" } }, ctx2);
    }
    assert.equal(ctx2.state.route.continuation.why, "capability_outgrown", `${mode} must break hold when options.realtime_loop=true`);
    assert.equal(ctx2.state.route.continuation.held, false);
    assert.equal(ctx2.state.route.modelWithEffort, "claude-opus-5-5:low", `${mode} must pick a fast effort of claude-opus-5-5`);
  }
});

test("10.17 Round-2 M3: Bare string candidates and bare `{ model }` candidates on `realtime_loop` only expand to `DISPATCH_EFFORTS`-supported efforts", () => {
  const rtProfile = validateProfile({
    ...REALTIME_BASE_PROFILE,
    work_kind: "general_implementation",
    multi_task: false,
    task_count: "1",
    reasoning_depth: 1,
    code_complexity: 1,
    needs_visual: false,
    needs_gui: false,
    realtime_loop: true,
  });

  // Bare "claude-sonnet-4-6" and "claude-opus-4-6" only have ["high"] in DISPATCH_EFFORTS.claude, so they cannot expand to "medium" or "low"
  const claudeBare = rankCandidates(rtProfile, {
    candidateModels: ["claude-opus-4-6", "claude-sonnet-4-6", { model: "claude-sonnet-4-6" }],
  });
  assert.deepEqual(
    claudeBare.allCandidates.map((c) => `${c.model}:${c.effort}`),
    ["claude-opus-4-6:high", "claude-sonnet-4-6:high", "claude-sonnet-4-6:high"],
  );
  assert.ok(claudeBare.allCandidates.every((c) => c.eligible === false));

  // On the README pool, realtime_loop never selects an undispatchable claude-sonnet-4-6:low/medium
  const readmePool = ["claude-opus-5-5", "claude-sonnet-5-5", "gpt-6.1-sol", "gemini-4-argon-high", "gemini-3.8-flash-high"];
  const route = selectBestRoute(rtProfile, { candidateModels: readmePool });
  assert.equal(route.ok, true);
  assert.equal(route.targetModel, "claude-sonnet-5-5");
  assert.ok(["low", "medium"].includes(route.targetEffort));
});

test("10.18 Round-2 M4: Model name normalization before responsiveness lookup matches Python AgentCompass across spellings, suffixes, and prefixes", () => {
  const rtProfile = validateProfile({
    ...REALTIME_BASE_PROFILE,
    work_kind: "general_implementation",
    multi_task: false,
    task_count: "1",
    reasoning_depth: 1,
    code_complexity: 1,
    needs_visual: false,
    needs_gui: false,
    realtime_loop: true,
  });

  // Contract R1: each spelling reads the row of its own key, a family name the default 0.52
  const expectedSpellings = [
    ["claude-sonnet-4-6", 0, "measured", false],
    ["claude-sonnet-4-6[1m]", 0, "measured", false],
    ["anthropic/claude-sonnet-4-6", 0, "measured", false],
    ["CLAUDE-SONNET-4-6", 0, "measured", false],
    ["claude-haiku-4-5@20251001", 0, "measured", false],
    ["claude-opus-5-5", 0.107, "measured", false],
    ["claude-opus-5-5[1m]", 0.107, "measured", false],
    ["openrouter/anthropic/claude-opus-5-5", 0.107, "measured", false],
    ["CLAUDE-OPUS-5-5", 0.107, "measured", false],
    ["claude-opus-5-5-20260101", 0.107, "measured", false],
    ["claude-opus-5-5-latest", 0.107, "measured", false],
    ["claude-opus-5-5@20260101", 0.107, "measured", false],
    ["claude-opus-4-8", 0.051, "measured", false],
    ["gemini-4-carbon", 0, "measured", false],
    ["grok-4.6", 0.129, "measured", false],
    ["xai/grok-4.6", 0.129, "measured", false],
    ["grok-4.7-latest", 0.201, "measured", false],
    ["openrouter/openai/gpt-6-astra", 0, "measured", false],
    ["gemini-4-argon", 0, "measured", false],
    ["argon", 0, "measured", false],
    ["gemini-2.5-pro", 0.52, "default", true],
  ];

  for (const [model, expectedRawResp, expectedSrc, expectedEligible] of expectedSpellings) {
    const [val, src] = bundleResponsiveness({ model, effort: "high" });
    assert.equal(val, expectedRawResp, `${model}:high raw responsiveness must be ${expectedRawResp}`);
    assert.equal(src, expectedSrc, `${model}:high source must be ${expectedSrc}`);

    // `:high` fixes the effort, so it never steps
    const cand = rankCandidates(rtProfile, { candidateModels: [`${model}:high`] }).allCandidates[0];
    assert.equal(cand.effort, "high", model);
    assert.equal(cand.eligible, expectedEligible, `${model}:high eligibility mismatch`);
  }
});

test("10.19 Round-2 M5 & m2: User pin canonicalization matches aliases/prefixes/computed IDs, and candidate `pinned:true` or profile-level fields never spoof user pins", () => {
  const rtProfile = validateProfile({
    ...REALTIME_BASE_PROFILE,
    work_kind: "general_implementation",
    multi_task: false,
    task_count: "1",
    reasoning_depth: 1,
    code_complexity: 1,
    needs_visual: false,
    needs_gui: false,
    realtime_loop: true,
  });

  const argon = { id: "ag", provider: "antigravity-resilient", model: "gemini-4-argon-high", effort: "high" };
  const opus = { id: "op", provider: "claude", model: "claude-opus-5-5", effort: "high" };
  const fast = { id: "fast", provider: "claude", model: "claude-sonnet-5-5", effort: "medium" };

  // M5: Pinning by alias, provider prefix, or [1m] selects the pinned slow model with warning
  for (const [pin, cands, expectedSel] of [
    ["gemini-4-argon-high", [argon, fast], "gemini-4-argon-high:high"],
    ["gemini-4-argon", [argon, fast], "gemini-4-argon-high:high"],
    ["google/gemini-4-argon-high", [argon, fast], "gemini-4-argon-high:high"],
    ["argon", [argon, fast], "gemini-4-argon-high:high"],
  ]) {
    const r = selectBestRoute(rtProfile, { constraintModel: pin, candidateModels: cands });
    assert.equal(r.ok, true, `pin=${pin} must succeed`);
    assert.equal(r.modelWithEffort, expectedSel, `pin=${pin} must select ${expectedSel}`);
    assert.equal(r.warnings.length, 1, `pin=${pin} must emit slow-pin warning`);
  }
  // A model named without an effort takes the effort it can run live at, so there is nothing to warn about
  for (const pin of ["claude-opus-5", "anthropic/claude-opus-5-5", "claude-opus-5-5[1m]"]) {
    const r = selectBestRoute(rtProfile, { constraintModel: pin, candidateModels: [opus, fast] });
    assert.equal(r.modelWithEffort, "claude-opus-5-5:low", `pin=${pin}`);
    assert.deepEqual(r.warnings, [], `pin=${pin}`);
  }

  // Computed ID pin on bare string candidates
  const computedIdPin = selectBestRoute(rtProfile, {
    pinnedBundle: "gemini-4-argon-high:high",
    candidateModels: ["gemini-4-argon-high:high", "claude-sonnet-5-5:medium"],
  });
  assert.equal(computedIdPin.ok, true);
  assert.equal(computedIdPin.modelWithEffort, "gemini-4-argon-high:high");
  assert.equal(computedIdPin.warnings.length, 1);

  // m2: Candidate with `pinned: true` on the candidate object itself is NOT a user pin
  const spoofedCandidate = rankCandidates(rtProfile, {
    candidateModels: [{ id: "ag", model: "gemini-4-argon-high", effort: "high", pinned: true }, fast],
  });
  const spoofedArgon = spoofedCandidate.allCandidates.find((c) => c.id === "ag");
  assert.equal(spoofedArgon.eligible, false, "Candidate-level pinned:true must not bypass floor");
  assert.equal(spoofedArgon.components.realtime_slow_pin_warning, undefined);
  assert.equal(spoofedCandidate.selected.id, "fast");

  // Pin fields attached to a flat profile object (when options has no pin) are ignored
  const profileWithSmuggledPin = {
    ...rtProfile,
    pinned_bundle: "ag",
    constraint_model: "gemini-4-argon-high",
  };
  const smuggledRes = selectBestRoute(profileWithSmuggledPin, { candidateModels: [argon, fast] });
  assert.equal(smuggledRes.modelWithEffort, "claude-sonnet-5-5:medium");
});

test("10.20 Round-2 m1, m3, m4: Strict boolean `realtime_loop` & non-object `body.needs`, prefix-invariant `bundleHarness`, and case-insensitive effort", async () => {
  // m1: Non-object body.needs does not throw; non-boolean realtime_loop ("false", 0, null) is never coerced to true
  for (const badNeeds of ["visual", 1, null, true, []]) {
    const facts = extractRequestFacts({
      model: "agentcompass",
      messages: [{ role: "user", content: "Fix the login button" }],
      needs: badNeeds,
    });
    assert.equal(facts.realtime_loop, undefined);

    const store = new SessionContinuityStore();
    const hookBody = {
      model: "agentcompass",
      messages: [{ role: "user", content: "Fix the login button" }],
      needs: badNeeds,
    };
    const hookCtx = { options: { continuityStore: store, candidateModels: ["claude-opus-5-5:high"] }, state: {} };
    await hookOnRequest(hookBody, hookCtx);
    assert.equal(hookCtx.state.route.modelWithEffort, "claude-opus-5-5:high");

    const mwBody = {
      model: "agentcompass",
      messages: [{ role: "user", content: "Fix the login button" }],
      needs: badNeeds,
    };
    mwOnRequest(mwBody, { options: { continuityStore: store, candidateModels: ["claude-opus-5-5:high"] }, state: {} });
    assert.equal(mwBody.model, "claude-opus-5-5:high");
  }

  for (const falsyNonBool of ["false", "0", 0, null, undefined, {}, []]) {
    assert.equal(extractRequestFacts({ messages: [], realtime_loop: falsyNonBool }).realtime_loop, undefined);
    const liveRes = await scoreTaskLive({ objective: "test", realtime_loop: falsyNonBool }, { allowExternalScorer: false });
    assert.equal(liveRes.profile, null);
  }

  // m3: bundleHarness inspects leaf model name so provider prefix never changes harness inference
  for (const [plain, prefixed, expectedHarness] of [
    ["gemini-3.8-flash-high", "google/gemini-3.8-flash-high", "antigravity-resilient"],
    ["gemini-4-argon-high", "openrouter/google/gemini-4-argon-high", "antigravity-resilient"],
    ["grok-bot", "xai/grok-bot", "grok-bot"],
    ["grok-4.7", "xai/grok-4.7", "grok"],
    ["glm-5.2", "zhipu/glm-5.2", "omp"],
    ["dot", "openai/dot", "dot"],
    ["claude-opus-5-5", "anthropic/claude-opus-5-5", "claude"],
  ]) {
    assert.equal(bundleHarness({ model: plain }), expectedHarness);
    assert.equal(bundleHarness({ model: prefixed }), expectedHarness);
  }

  const rtProfile = validateProfile({
    ...REALTIME_BASE_PROFILE,
    work_kind: "general_implementation",
    multi_task: false,
    task_count: "1",
    reasoning_depth: 1,
    code_complexity: 1,
    needs_visual: false,
    needs_gui: false,
    realtime_loop: true,
  });
  const prefixedFlash = rankCandidates(rtProfile, { candidateModels: ["google/gemini-3.8-flash:high"] }).allCandidates[0];
  assert.equal(prefixedFlash.effort, "high", "google/gemini-3.8-flash:high must infer antigravity-resilient and not step down");
  assert.equal(prefixedFlash.eligible, false);

  // m4: Object candidate effort is normalized to lowercase and trimmed
  for (const rawEff of ["HIGH", " high ", "Max", "XHIGH"]) {
    const r = rankCandidates(rtProfile, {
      candidateModels: [{ id: "ag", provider: "antigravity-resilient", model: "claude-opus-5-5", effort: rawEff }],
    });
    const c = r.allCandidates[0];
    assert.equal(c.effort, rawEff.trim().toLowerCase());
    assert.equal(c.eligible, false, `claude-opus-5-5 with effort=${rawEff} on antigravity-resilient must be < 0.50 and excluded`);
  }
});
