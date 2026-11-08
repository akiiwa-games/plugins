/**
 * AgentCompass Subscription Quota, Opportunity Cost & Pre-Reset Sprint Economics
 *
 * Exact port of the subscription window state, Normal demand forecast (`demand_forecast`),
 * expected shortage (`expected_shortage`), shortage probability (`shortage_probability`),
 * window opportunity cost (`window_cost`), scarcity penalty, pre-reset surplus sprint
 * (`reset_sunk_weight`, `reset_bonus`), and subscription-vs-metered billing discipline from
 * `skills/agent-compass/scripts/allocator.py` (lines 604-797, 1655-2170).
 *
 * Also integrates natively with Magpie's quota telemetry:
 * 1. `quotas.json` (`{ "<provider>/<account>": { kind, quota: { windows: [...] }, balance } }`)
 * 2. `GET /v1/magpie/quotas` (`[{ provider, name, kind, windows: [...], balance }]`)
 * 3. Magpie `Seat` objects in `strategies.agentcompass(pool, seats, options)`
 *    (`{ key, account, known, learns, used, soon, pace, weight, resting, tokens }`)
 *
 * Pure ES2022 module with zero external or Node-only dependencies so it runs in both
 * Bun (`magpie.hooks` / `/v1/systemone`) and Magpie's in-process `moejs` (`magpie.middleware`).
 */

import {
  DEFAULT_MODELS,
  DEFAULT_SCORING_POLICY,
  EFFORT_CURVES,
  MODEL_ALIASES,
  canonicalModel,
  resolveModelSpec,
  pyRound,
} from "./catalog.js";
import { loadLiveQuotas } from "./quota-loader.js";

const SQRT2 = Math.SQRT2;
const SQRT_2PI = Math.sqrt(2 * Math.PI);

/**
 * Default API-equivalent USD capacity priors for subscription windows when `capacity_usd`
 * is not explicitly specified in `quotas.json` or `options.windowCapacityUsd`.
 */
export const DEFAULT_WINDOW_CAPACITY_USD = {
  claude: {
    short: 45.0,
    weekly: 220.0,
    monthly: 800.0,
  },
  codex: {
    short: 50.0,
    weekly: 250.0,
    monthly: 900.0,
  },
  antigravity: {
    "gemini-5h": 182.0,
    "gemini-weekly": 625.0,
    "3p-5h": 17.6,
    "3p-weekly": 88.0,
    short: 182.0,
    weekly: 625.0,
    monthly: 1800.0,
  },
  gemini: {
    short: 120.0,
    weekly: 450.0,
    monthly: 1500.0,
  },
  default: {
    short: 40.0,
    weekly: 180.0,
    monthly: 650.0,
  },
};

/**
 * Per-plan API-equivalent USD capacity (`weekly` = long window, `short` = 5h window), taken from
 * AgentCompass's `references/plan-templates.json` (the authority; keep in step with it).
 * A plan without a `short` entry has no 5h window (Codex pro100 / pro); a 5h window reported for it is sized
 * as the plan's weekly figure, never as a separate smaller budget.
 * Plans are matched exactly (`PLAN_ALIASES`), never by substring.
 */
const ANTIGRAVITY_ULTRA20 = DEFAULT_WINDOW_CAPACITY_USD.antigravity;
const antigravityTier = (gWeekly, g5h, pWeekly, p5h) => ({
  "gemini-weekly": gWeekly,
  "gemini-5h": g5h,
  "3p-weekly": pWeekly,
  "3p-5h": p5h,
  weekly: gWeekly,
  short: g5h,
});

export const PLAN_CAPACITY_USD = {
  codex: {
    plus: { weekly: 67, short: 10 },
    pro100: { weekly: 175 },
    pro: { weekly: 325 },
  },
  claude: {
    pro: { weekly: 156, short: 18 },
    max5: { weekly: 1400, short: 94, weekly_fable: 700 },
    max20: { weekly: 2380, short: 376, weekly_fable: 1190 },
  },
  antigravity: {
    pro: antigravityTier(31, 9.1, 4.4, 0.9),
    ultra5: antigravityTier(156, 45.5, 22, 4.4),
    ultra20: ANTIGRAVITY_ULTRA20,
  },
};

const PLAN_ALIASES = {
  codex: {
    plus: "plus",
    team: "plus",
    pro: "pro",
    pro100: "pro100",
    pro_100: "pro100",
    100: "pro100",
    prolite: "pro100",
    pro_lite: "pro100",
  },
  claude: {
    pro: "pro",
    max: "max5",
    max5: "max5",
    max_5x: "max5",
    max_x5: "max5",
    max5x: "max5",
    "5x": "max5",
    x5: "max5",
    max20: "max20",
    max_20x: "max20",
    max_x20: "max20",
    max20x: "max20",
    "20x": "max20",
    x20: "max20",
  },
  antigravity: {
    pro: "pro",
    google_ai_pro: "pro",
    ultra: "ultra20",
    google_ai_ultra: "ultra20",
    ultra20: "ultra20",
    ultra_20x: "ultra20",
    ultra_x20: "ultra20",
    ultra20x: "ultra20",
    ultra5: "ultra5",
    ultra_5x: "ultra5",
    ultra_x5: "ultra5",
    ultra5x: "ultra5",
  },
};

/** Plan values that mean the subscription is not paying any more (not a capacity tier). */
const INACTIVE_PLANS = new Set([
  "free",
  "free_tier",
  "antigravity_free_tier",
  "expired",
  "cancelled",
  "canceled",
  "inactive",
  "plan_inactive",
  "payment_failed",
  "past_due",
  "unpaid",
  "lapsed",
]);

/** `status` values by state; AgentCompass names (`ready` / `plan_inactive` / `usage_failing` / `auth_failed`). */
export const ACCOUNT_STATES = Object.freeze({
  READY: "ready",
  PLAN_INACTIVE: "plan_inactive",
  USAGE_FAILING: "usage_failing",
  AUTH_FAILED: "auth_failed",
  // Operator switch (`disabled: true`), not a health state; listed so it is never silent either.
  DISABLED: "disabled",
});

const STATUS_PLAN_INACTIVE = new Set([
  "expired",
  "cancelled",
  "canceled",
  "inactive",
  "plan_inactive",
  "payment_failed",
  "past_due",
  "unpaid",
  "lapsed",
]);
const INACTIVE_ERROR_CODES = new Set([
  "subscription_inactive",
  "subscription_expired",
  "subscription_canceled",
  "subscription_cancelled",
  "plan_expired",
  "plan_inactive",
  "payment_required",
  "payment_failed",
]);
const STATUS_AUTH_FAILED = new Set(["auth_failed", "unauthorized", "forbidden"]);
const STATUS_USAGE_FAILING = new Set(["error", "failed", "usage_failing", "no_data"]);
const PLAN_VENDOR_WORDS = new Set(["antigravity", "chatgpt", "claude", "google", "ai", "openai", "plan"]);

/** No successful reading for this long takes the account out of dispatch. */
export const DEFAULT_USAGE_MAX_AGE_HOURS = 6;
/** An error warns at once; with no error, a reading older than this counts as failing (still dispatchable). */
export const DEFAULT_USAGE_FRESH_HOURS = 2;

const plansProviders = Object.keys(PLAN_CAPACITY_USD);

const POOL_SUFFIX_RE = /\s*×\s*\d+\s*$/;

function stripPoolCount(raw) {
  return String(raw ?? "").replace(POOL_SUFFIX_RE, "");
}

function planSlug(raw) {
  return String(raw ?? "")
    .replace(/×/g, "x")
    .trim()
    .toLowerCase()
    .replace(/[\s-]+/g, "_");
}

function stripVendorWords(slug) {
  if (!slug) return "";
  return slug
    .split("_")
    .filter((w) => w && !PLAN_VENDOR_WORDS.has(w))
    .join("_");
}

/**
 * Map a provider's reported plan label to a known plan key. `inactive` means the subscription is no
 * longer paying; `known: false` means an unrecognised (or absent) plan, which is sized conservatively.
 * Matches the full name first, and only strips a trailing `×N` pool count if the full name did not match.
 */
export function normalizePlan(provider, rawPlan) {
  const rawStr = String(rawPlan ?? "");
  const slug = planSlug(rawStr);
  const prov = String(provider || "").toLowerCase();
  if (!slug || slug === "none" || slug === "null" || slug === "unknown") {
    return { raw: slug ? rawStr : null, key: null, known: false, inactive: false };
  }
  const withoutPool = stripPoolCount(rawStr);
  const candidates = withoutPool !== rawStr ? [rawStr, withoutPool] : [rawStr];
  for (const candidate of candidates) {
    const candSlug = planSlug(candidate);
    if (!candSlug) continue;
    const stripped = stripVendorWords(candSlug);
    if (INACTIVE_PLANS.has(candSlug) || (stripped && INACTIVE_PLANS.has(stripped))) {
      return { raw: rawStr, key: null, known: true, inactive: true };
    }
    const key = PLAN_ALIASES[prov]?.[candSlug] ?? (stripped ? PLAN_ALIASES[prov]?.[stripped] : null) ?? null;
    if (key !== null) {
      return { raw: rawStr, key, known: true, inactive: false };
    }
  }
  return { raw: rawStr, key: null, known: false, inactive: false };
}

/** Smallest capacity table across a provider's known plans: what an unknown plan is sized as. */
function conservativePlanTable(provider) {
  const plans = Object.values(PLAN_CAPACITY_USD[provider] || {});
  if (plans.length === 0) return null;
  const out = {};
  for (const name of new Set(plans.flatMap((t) => Object.keys(t)))) {
    const vals = plans.map((t) => t[name]).filter((v) => typeof v === "number");
    // `short` is the smallest 5h window any plan has; other plan-specific windows (`weekly_fable`) are left
    // out so they fall back to the weekly size.
    if (name === "short" || vals.length === plans.length) out[name] = Math.min(...vals);
  }
  return out;
}

/** Capacity table for `(provider, plan)`; unknown plan -> conservative table; no plan tiers for the provider -> null. */
export function planCapacityTable(provider, planKey) {
  const prov = String(provider || "").toLowerCase();
  if (!plansProviders.includes(prov)) return null;
  return (planKey && PLAN_CAPACITY_USD[prov][planKey]) || conservativePlanTable(prov);
}

export function clamp(val, low, high) {
  return Math.max(low, Math.min(high, val));
}

export function round4(val) {
  return pyRound(val, 4);
}

export function round3(val) {
  return Math.round(val * 1000) / 1000;
}

/**
 * Abramowitz & Stegun 7.1.26 error function `erf(x)` (max error < 1.5e-7).
 */
export function erf(x) {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1.0 / (1.0 + 0.3275911 * ax);
  const y =
    1.0 -
    (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t +
      0.254829592) *
      t) *
      Math.exp(-ax * ax);
  return sign * y;
}

/**
 * E[(D - remaining)+] in percent for demand D ~ Normal(mean, sd) (`allocator.py:expected_shortage`).
 */
export function expectedShortage(remaining, mean, sd) {
  const safeSd = Math.max(Number(sd) || 1e-6, 1e-6);
  const z = (mean - remaining) / safeSd;
  return (
    (safeSd * Math.exp((-z * z) / 2.0)) / SQRT_2PI +
    ((mean - remaining) * (1.0 + erf(z / SQRT2))) / 2.0
  );
}

/**
 * P(D > remaining) for demand D ~ Normal(mean, sd) (`allocator.py:shortage_probability`).
 */
export function shortageProbability(remaining, mean, sd) {
  const safeSd = Math.max(Number(sd) || 1e-6, 1e-6);
  return (1.0 + erf((mean - remaining) / (safeSd * SQRT2))) / 2.0;
}

/**
 * Infer provider family (`claude`, `codex`, `antigravity`, `gemini`, `grok`, `deepseek`, `zai`, `kimi`, `minimax`, `qwen`)
 * from a model ID or provider prefix.
 */
export function inferModelProviderFamily(rawModel, aliases = MODEL_ALIASES) {
  if (!rawModel || typeof rawModel !== "string") return "";
  const clean = rawModel.trim().toLowerCase();
  const slashIdx = clean.indexOf("/");
  if (slashIdx > 0) {
    const prefix = clean.slice(0, slashIdx);
    if (
      [
        "claude",
        "anthropic",
        "codex",
        "openai",
        "antigravity",
        "gemini",
        "google",
        "grok",
        "xai",
        "deepseek",
        "zai",
        "glm",
        "kimi",
        "moonshot",
        "minimax",
        "qwen",
      ].includes(prefix)
    ) {
      if (prefix === "anthropic") return "claude";
      if (prefix === "openai") return "codex";
      if (prefix === "google") return "gemini";
      if (prefix === "xai") return "grok";
      if (prefix === "glm") return "zai";
      if (prefix === "moonshot") return "kimi";
      return prefix;
    }
  }
  const canon = canonicalModel(rawModel, aliases).toLowerCase();
  if (canon.startsWith("claude")) return "claude";
  if (canon.startsWith("gpt") || canon.includes("codex") || /^o\d/.test(canon)) {
    return "codex";
  }
  if (canon.startsWith("gemini") || canon.includes("argon")) return "gemini";
  if (canon.startsWith("grok")) return "grok";
  if (canon.startsWith("deepseek")) return "deepseek";
  if (canon.startsWith("glm")) return "zai";
  if (canon.startsWith("kimi")) return "kimi";
  if (canon.startsWith("minimax")) return "minimax";
  if (canon.startsWith("qwen")) return "qwen";
  return "";
}

/**
 * Normalize a provider identifier (`claude`/`anthropic`, `codex`/`openai`, `antigravity`/`gemini`/`google`, etc.)
 * into its canonical provider family.
 */
export function normalizeProviderFamily(raw) {
  const s = String(raw || "").trim().toLowerCase();
  if (!s) return "";
  const slash = s.indexOf("/");
  const p = slash > 0 ? s.slice(0, slash) : s;
  if (p === "anthropic" || p === "claude" || p === "claude-code") return "claude";
  if (p === "openai" || p === "codex" || p === "chatgpt") return "codex";
  if (p === "antigravity" || p === "gemini" || p === "google") return "antigravity";
  if (p === "xai" || p === "grok") return "grok";
  if (p === "glm" || p === "zai") return "zai";
  if (p === "moonshot" || p === "kimi") return "kimi";
  return p;
}

/**
 * Check whether two provider identifiers belong to the same provider family.
 */
export function providersMatch(provA, provB) {
  const fa = normalizeProviderFamily(provA);
  const fb = normalizeProviderFamily(provB);
  if (!fa || !fb) return false;
  return fa === fb;
}

/**
 * Check whether an account's provider matches a candidate model's provider family.
 */
export function providerMatchesModel(accountProvider, rawModel, aliases = MODEL_ALIASES) {
  const acctProv = String(accountProvider || "").trim().toLowerCase();
  if (!acctProv) return false;
  const rawLower = String(rawModel || "").trim().toLowerCase();
  const canon = canonicalModel(rawModel, aliases).toLowerCase();
  const modelProv = inferModelProviderFamily(rawModel, aliases);

  if (acctProv === rawLower || acctProv === canon || acctProv === modelProv) {
    return true;
  }
  if (
    (acctProv === "anthropic" || acctProv === "claude" || acctProv === "claude-code") &&
    modelProv === "claude"
  ) {
    return true;
  }
  if (
    (acctProv === "openai" || acctProv === "codex" || acctProv === "chatgpt") &&
    modelProv === "codex"
  ) {
    return true;
  }
  if (
    (acctProv === "antigravity" || acctProv === "gemini" || acctProv === "google") &&
    modelProv === "gemini"
  ) {
    return true;
  }
  if (acctProv === "antigravity" && (modelProv === "claude" || modelProv === "codex")) {
    // Antigravity can also host 3P models when explicitly configured
    return rawLower.startsWith("antigravity/");
  }
  return false;
}

/**
 * Check whether a specific quota window applies to the given canonical model
 * (`allocator.py:legacy_model_binds_window` and Magpie `strategies.go:124`).
 */
export function windowBindsModel(windowName, canonicalModelName, explicitModels = null, rawModel = null) {
  const wid = String(windowName || "").trim().toLowerCase();
  const model = String(canonicalModelName || "").trim().toLowerCase();
  // The window may be written against the key or the name as routed (contract R1).
  const raw = String(rawModel || model).trim().toLowerCase();
  const names = (tag) => model.includes(tag) || raw.includes(tag);

  if (Array.isArray(explicitModels) && explicitModels.length > 0) {
    const matchesExplicit = explicitModels.some((tag) => {
      const t = String(tag || "").toLowerCase();
      return t && (names(t) || wid.includes(t) || canonicalModel(t).toLowerCase() === model);
    });
    if (!matchesExplicit) return false;
  }

  if (wid.includes("fable") && !names("fable")) return false;
  if (wid.includes("opus") && !names("opus")) return false;
  if (wid.includes("sonnet") && !names("sonnet")) return false;
  if (wid.includes("haiku") && !names("haiku")) return false;
  if (wid.startsWith("gemini-") && !model.startsWith("gemini")) return false;
  if (wid.startsWith("3p-") && model.startsWith("gemini")) return false;

  return true;
}

/**
 * Parse a timestamp (ISO string, unix seconds, or unix ms) into milliseconds since epoch.
 */
export function parseResetTimeMs(val) {
  if (val === null || val === undefined || val === "") return null;
  if (typeof val === "number" && Number.isFinite(val)) {
    if (val <= 0) return null;
    // Heuristic: < 1e11 is unix seconds, >= 1e11 is unix ms
    return val < 1e11 ? val * 1000 : val;
  }
  if (typeof val === "string") {
    const trimmed = val.trim();
    if (!trimmed) return null;
    const asNum = Number(trimmed);
    if (Number.isFinite(asNum) && asNum > 0) {
      return asNum < 1e11 ? asNum * 1000 : asNum;
    }
    const parsed = Date.parse(trimmed);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

/**
 * Infer a window's total duration in hours from its name/id or explicit `hours`.
 */
export function inferWindowHours(windowObj = {}, resetHours = null) {
  if (typeof windowObj.hours === "number" && windowObj.hours > 0) {
    return windowObj.hours;
  }
  if (typeof windowObj.spanHours === "number" && windowObj.spanHours > 0) {
    return windowObj.spanHours;
  }
  const name = String(windowObj.name || windowObj.id || "").trim().toLowerCase();
  if (
    name === "5h" ||
    name === "five_hour" ||
    name === "5-hour" ||
    name.endsWith("-5h") ||
    name.includes("5h") ||
    name === "session"
  ) {
    return 5;
  }
  if (name === "3h" || name === "three_hour" || name.includes("3h")) {
    return 3;
  }
  if (name === "1d" || name === "24h" || name === "daily" || name === "day") {
    return 24;
  }
  if (
    name === "30d" ||
    name === "month" ||
    name === "monthly" ||
    name.includes("month")
  ) {
    return 720;
  }
  if (
    name === "7d" ||
    name === "seven_day" ||
    name === "7-day" ||
    name === "week" ||
    name === "weekly" ||
    name === "usage" ||
    name.includes("week") ||
    name.includes("7d") ||
    name === "opus" ||
    name === "sonnet"
  ) {
    return 168;
  }
  if (typeof resetHours === "number" && resetHours > 0) {
    if (resetHours <= 5.5) return 5;
    if (resetHours <= 24.5) return 24;
    if (resetHours <= 170) return 168;
    return 720;
  }
  return 168;
}

/**
 * Resolve API-equivalent USD capacity for a subscription window (`allocator.py:window_capacity_usd`).
 * Order: explicit `capacity_usd` > custom table > the account's plan table (providers with plan tiers;
 * `planKey` null/unknown is sized conservatively) > the provider-wide priors.
 * `plan` is `undefined` for callers that carry no plan information, which keeps the provider priors.
 */
export function resolveWindowCapacityUsd(
  provider,
  windowName,
  windowHours,
  explicitCapUsd = null,
  customTable = null,
  plan = undefined
) {
  if (typeof explicitCapUsd === "number" && explicitCapUsd > 0) {
    return explicitCapUsd;
  }
  const prov = String(provider || "default").toLowerCase();
  const wid = String(windowName || "").toLowerCase();
  const planTable = plan === undefined || customTable ? null : planCapacityTable(prov, plan);
  const table =
    planTable ||
    (customTable && (customTable[prov] || customTable.default)) ||
    DEFAULT_WINDOW_CAPACITY_USD[prov] ||
    DEFAULT_WINDOW_CAPACITY_USD.default;

  if (table && typeof table[wid] === "number") {
    return table[wid];
  }
  if (windowHours < 24) {
    if (planTable) {
      return typeof planTable.short === "number" ? planTable.short : null;
    }
    return Number(table?.short ?? DEFAULT_WINDOW_CAPACITY_USD.default.short);
  }
  if (windowHours <= 170) {
    return Number(table?.weekly ?? DEFAULT_WINDOW_CAPACITY_USD.default.weekly);
  }
  return Number(table?.monthly ?? (planTable ? planTable.weekly * 4 : DEFAULT_WINDOW_CAPACITY_USD.default.monthly));
}

/**
 * Normalize a single window observation into AgentCompass's canonical `windowState`
 * (`allocator.py:window_state`).
 */
export function normalizeWindowState(
  rawWin,
  provider = "default",
  nowMs = Date.now(),
  policy = DEFAULT_SCORING_POLICY,
  accountStale = false,
  planKey = undefined,
  keepRates = true
) {
  if (!rawWin || typeof rawWin !== "object") return null;
  if (rawWin.unlimited === true) {
    return {
      id: String(rawWin.name || rawWin.id || "unlimited"),
      unlimited: true,
      known: true,
      short: false,
      remaining: 100,
      used: 0,
      hours: 168,
      resetHours: 168,
      endHours: 168,
      targetRemaining: 100,
      stale: false,
    };
  }

  const id = String(rawWin.name || rawWin.id || "usage");
  let remaining = null;
  let used = null;

  if (typeof rawWin.remaining === "number" && Number.isFinite(rawWin.remaining)) {
    remaining = clamp(rawWin.remaining, 0, 100);
    used = clamp(100 - remaining, 0, 100);
  } else if (
    typeof rawWin.remaining_pct_before_reset === "number" &&
    Number.isFinite(rawWin.remaining_pct_before_reset)
  ) {
    remaining = clamp(rawWin.remaining_pct_before_reset, 0, 100);
    used = clamp(100 - remaining, 0, 100);
  } else if (
    typeof rawWin.observed_remaining_pct === "number" &&
    Number.isFinite(rawWin.observed_remaining_pct)
  ) {
    remaining = clamp(rawWin.observed_remaining_pct, 0, 100);
    used = clamp(100 - remaining, 0, 100);
  } else if (typeof rawWin.used === "number" && Number.isFinite(rawWin.used)) {
    used = clamp(rawWin.used, 0, 100);
    remaining = clamp(100 - used, 0, 100);
  } else if (typeof rawWin.used_pct === "number" && Number.isFinite(rawWin.used_pct)) {
    used = clamp(rawWin.used_pct, 0, 100);
    remaining = clamp(100 - used, 0, 100);
  } else if (
    typeof rawWin.amount === "number" &&
    typeof rawWin.limit === "number" &&
    rawWin.limit > 0
  ) {
    used = clamp((rawWin.amount / rawWin.limit) * 100, 0, 100);
    remaining = clamp(100 - used, 0, 100);
  }

  let resetHours = null;
  if (typeof rawWin.resetHours === "number" && Number.isFinite(rawWin.resetHours)) {
    resetHours = rawWin.resetHours;
  } else if (typeof rawWin.resetsInHours === "number" && Number.isFinite(rawWin.resetsInHours)) {
    resetHours = rawWin.resetsInHours;
  } else if (
    typeof rawWin.seconds_until_reset === "number" &&
    Number.isFinite(rawWin.seconds_until_reset)
  ) {
    resetHours = rawWin.seconds_until_reset / 3600;
  } else {
    const resetMs = parseResetTimeMs(rawWin.resetsAt ?? rawWin.resets_at ?? rawWin.resetAt);
    if (resetMs !== null) {
      resetHours = (resetMs - nowMs) / 3600000;
    }
  }

  const hours = inferWindowHours(rawWin, resetHours);
  const shortHoursThreshold = Number(policy?.short_window_hours ?? 24);
  const short = hours < shortHoursThreshold;

  // If a rolling window hasn't started yet (remaining === 100, resetsAt === null), its full span is available
  if (remaining === 100 && (resetHours === null || resetHours === undefined)) {
    resetHours = hours;
  }
  // If reset time already passed in the past, treat remaining as unknown or renewed unless explicitly rate_limited
  if (resetHours !== null && resetHours <= 0 && rawWin.rate_limited !== true) {
    if (remaining !== null && remaining <= 0) {
      // A window whose reset timestamp is in the past has already renewed!
      remaining = 100;
      used = 0;
      resetHours = hours;
    } else {
      resetHours = hours;
    }
  }

  if (remaining === null) {
    return {
      id,
      hours,
      short,
      known: false,
      stale: Boolean(accountStale || rawWin.stale),
      models: rawWin.models || null,
    };
  }

  // Default resetHours if omitted in shorthand test fixtures (`{ remaining: 80 }`)
  if (resetHours === null || !Number.isFinite(resetHours) || resetHours <= 0) {
    resetHours = short ? hours : hours * Math.max(0.1, remaining / 100);
  }

  const endHours = Math.max(0, resetHours);
  const targetRemaining = 100 * Math.min(1.0, endHours / hours);
  const capacityUsd = resolveWindowCapacityUsd(
    provider,
    id,
    hours,
    rawWin.capacity_usd ?? rawWin.capacityUsd,
    null,
    planKey
  );

  return {
    id,
    hours,
    short,
    known: true,
    stale: Boolean(accountStale || rawWin.stale),
    remaining: round4(remaining),
    used: round4(used),
    resetHours: round4(resetHours),
    endHours: round4(endHours),
    targetRemaining: round4(targetRemaining),
    capacityUsd,
    rateLimited: Boolean(rawWin.rate_limited || rawWin.rateLimited),
    pctPerHour3h: keepRates && typeof rawWin.pct_per_hour_3h === "number" ? rawWin.pct_per_hour_3h : null,
    pctPerHour24h: keepRates && typeof rawWin.pct_per_hour_24h === "number" ? rawWin.pct_per_hour_24h : null,
    models: Array.isArray(rawWin.models) ? rawWin.models : null,
  };
}

function firstTimeMs(...vals) {
  for (const v of vals) {
    const ms = parseResetTimeMs(v);
    if (ms !== null) return ms;
  }
  return null;
}

function httpStatusOf(item) {
  const err = item.error ?? item.quota?.error;
  const cands = [
    item.http_status,
    item.httpStatus,
    item.lastErrorHttpStatus,
    item.quota?.http_status,
    typeof item.status === "number" ? item.status : null,
    err && typeof err === "object"
      ? (err.status ?? err.http_status ?? (typeof err.code === "number" || /^\d+$/.test(String(err.code || "")) ? err.code : null))
      : null,
  ];
  for (const c of cands) {
    const n = Number(c);
    if (Number.isFinite(n) && n > 0) return n;
  }
  return null;
}

function errorCodeOf(item) {
  const err = item.error ?? item.quota?.error;
  const cands = [
    err && typeof err === "object" ? (err.code ?? err.error_code ?? err.type) : null,
    item.error_code,
    item.errorCode,
    item.lastErrorClass,
    item.failure_class,
    item.quota?.error_code,
  ];
  for (const c of cands) {
    if (c === null || c === undefined || typeof c === "number") continue;
    const s = planSlug(c);
    if (s && !/^\d+$/.test(s)) return s;
  }
  return null;
}

function fmtHours(ms) {
  return `${(ms / 3600000).toFixed(1)}h`;
}

/**
 * One account's state (`ready` / `plan_inactive` / `usage_failing` / `auth_failed`, plus operator `disabled`),
 * why, when it last read successfully, and whether it is out of dispatch. Reads structured fields only
 * (`status`, `error`, HTTP codes, timestamps); never message text.
 */
function classifyAccount(item, entry, plan, nowMs, maxAgeMs, freshMs) {
  const warnings = [];
  const status = String(item.status ?? item.quota?.status ?? "").trim().toLowerCase().replace(/[\s-]+/g, "_");
  const errorValue = item.error ?? item.quota?.error ?? null;
  const http = httpStatusOf(item);
  const errorCode = errorCodeOf(item);
  const hasReading = entry.unlimited || entry.windows.some((w) => w.known || w.unlimited);
  const metered = ["balance", "metered", "api", "payg"].includes(entry.kind);

  const prevOk =
    item.previous && typeof item.previous === "object" && item.previous.status === ACCOUNT_STATES.READY
      ? firstTimeMs(item.previous.last_success_at, item.previous.updated_at, item.previous.fetched_at)
      : null;
  const explicitOk = firstTimeMs(
    item.last_success_at,
    item.lastSuccessAt,
    item.last_ok_at,
    item.lastOkAt,
    item.quota?.last_success_at,
    prevOk
  );
  const readAt = firstTimeMs(
    item.fetched_at,
    item.fetchedAt,
    item.updated_at,
    item.updatedAt,
    item.read_at,
    item.quota?.fetched_at
  );

  let state = ACCOUNT_STATES.READY;
  let reason = null;
  let failing = false;

  if (item.disabled === true || item.enabled === false || status === "disabled") {
    state = ACCOUNT_STATES.DISABLED;
    reason = "account is disabled";
  } else if (
    plan.inactive ||
    STATUS_PLAN_INACTIVE.has(status) ||
    http === 402 ||
    (errorCode && INACTIVE_ERROR_CODES.has(errorCode))
  ) {
    state = ACCOUNT_STATES.PLAN_INACTIVE;
    const why = plan.inactive
      ? plan.raw
      : errorCode && INACTIVE_ERROR_CODES.has(errorCode)
        ? errorCode
        : http === 402
          ? "HTTP 402"
          : status;
    reason = `subscription is not active (${why})`;
  } else if (
    http === 401 ||
    http === 403 ||
    STATUS_AUTH_FAILED.has(status) ||
    (errorCode && STATUS_AUTH_FAILED.has(errorCode))
  ) {
    state = ACCOUNT_STATES.AUTH_FAILED;
    reason = `sign-in rejected${http ? ` (HTTP ${http})` : ""}`;
  } else if (
    errorValue ||
    item.available === false ||
    entry.stale ||
    STATUS_USAGE_FAILING.has(status) ||
    (!hasReading && !metered)
  ) {
    failing = true;
    const detail =
      typeof errorValue === "string"
        ? errorValue
        : errorValue && typeof errorValue === "object" && typeof errorValue.message === "string"
          ? errorValue.message
          : item.available === false
            ? "provider unavailable"
            : entry.stale
              ? "reading is stale"
              : !hasReading
                ? "no usage reading"
                : status || "reading failed";
    reason = `usage reading failing: ${detail}`;
  }

  // Last success: an explicit field, else the time of a good reading when the poll did not fail.
  let lastSuccessMs = explicitOk;
  if (lastSuccessMs === null && state === ACCOUNT_STATES.READY && !failing && (hasReading || metered)) {
    lastSuccessMs = readAt;
  }

  // No error but no fresh reading either: failing once the reading is older than the freshness limit (2h).
  if (state === ACCOUNT_STATES.READY && !failing && lastSuccessMs !== null && nowMs - lastSuccessMs > freshMs) {
    failing = true;
    reason = `usage reading failing: no successful reading for ${fmtHours(nowMs - lastSuccessMs)}`;
  }

  const explicitFailingSinceMs = firstTimeMs(
    item.failing_since,
    item.failingSince,
    item.quota?.failing_since,
    item.previous && typeof item.previous === "object" && item.previous.status === ACCOUNT_STATES.USAGE_FAILING
      ? item.previous.failing_since
      : null
  );

  const poolLevel = item.pool_level === true;
  let excluded = false;
  if (failing) {
    state = ACCOUNT_STATES.USAGE_FAILING;
    const anchorMs = lastSuccessMs ?? explicitFailingSinceMs ?? (poolLevel ? nowMs : null);
    excluded = anchorMs === null || nowMs - anchorMs >= maxAgeMs;
  } else if (state !== ACCOUNT_STATES.READY) {
    excluded = true;
  }

  if (state !== ACCOUNT_STATES.READY) {
    const lastTxt =
      lastSuccessMs === null ? "never read successfully" : `last success ${new Date(lastSuccessMs).toISOString()}`;
    warnings.push(
      poolLevel
        ? excluded
          ? `${entry.key}: ${state}, excluded from dispatch: ${reason} (${lastTxt})`
          : `${entry.key}: ${state}: ${reason} (${lastTxt}); quota limits may be missing or out of date`
        : excluded
          ? `${entry.key}: ${state}, excluded from dispatch: ${reason} (${lastTxt})`
          : `${entry.key}: ${state}: ${reason} (${lastTxt}); excluded after ${fmtHours(maxAgeMs)} without a successful reading`
    );
  }
  // Where the failure began: an explicit `failing_since`, else the last success (readings stopped after it).
  const failingSinceMs =
    state === ACCOUNT_STATES.USAGE_FAILING
      ? explicitFailingSinceMs ?? lastSuccessMs
      : null;
  return { state, reason, lastSuccessMs, excluded, warnings, failingSinceMs };
}

function planInfoOf(item, provider, key) {
  const online =
    item.plan ?? item.quota?.plan ?? item.plan_type ?? item.planType ?? item.plan_label ?? item.quota?.plan_type ?? null;
  const claim = item.plan_claim ?? item.planClaim ?? null;
  // The online plan wins; the credential claim is only a fallback when there is no online plan.
  const rawPlan = normalizePlan(provider, online).raw !== null ? online : claim;
  const plan = normalizePlan(provider, rawPlan);
  const warnings = [];
  let conflict = null;
  if (claim && normalizePlan(provider, online).raw !== null) {
    const claimed = normalizePlan(provider, claim);
    if (claimed.key !== plan.key || claimed.inactive !== plan.inactive) {
      conflict = { online: String(online), credential: String(claim) };
      warnings.push(
        `${key}: plan sources disagree (online "${online}" vs credential claim "${claim}"); using the online value`
      );
    }
  }
  if (!plan.known && !plan.inactive && plansProviders.includes(provider)) {
    warnings.push(
      `${key}: plan ${plan.raw ? `"${plan.raw}" is not recognized` : "was not reported"}; capacity is sized as the smallest known ${provider} plan`
    );
  }
  return { plan, warnings, conflict };
}

/** Build one canonical account record from a raw quota item (shared by every snapshot format). */
function buildAccountEntry(
  item,
  { key, provider, account, rawWindows, kind, unlimited, balance, stale, nowMs, policy, maxAgeMs, freshMs }
) {
  const { plan, warnings: planWarnings, conflict } = planInfoOf(item, provider, key);
  const planKey = plansProviders.includes(provider) ? plan.key : undefined;
  // Rates measured under another plan must not be carried over (a tier change reads as a demand jump).
  const ratePlan = item.rate_plan ?? item.ratePlan;
  const keepRates = !ratePlan || normalizePlan(provider, ratePlan).key === plan.key;
  const windows = rawWindows
    .map((w) => normalizeWindowState(w, provider, nowMs, policy, stale, planKey, keepRates))
    .filter(Boolean);
  const entry = {
    key,
    provider,
    account,
    kind,
    poolLevel: Boolean(item.pool_level),
    unlimited,
    stale,
    balance,
    error: item.error || item.quota?.error || null,
    plan: plan.raw,
    planKey: plan.key,
    planKnown: plan.known,
    planConflict: conflict,
    maxAgeHours: maxAgeMs / 3600000,
    windows,
  };
  const cls = classifyAccount(item, entry, plan, nowMs, maxAgeMs, freshMs);
  entry.state = cls.state;
  entry.reason = cls.reason;
  entry.lastSuccessAt = cls.lastSuccessMs === null ? null : new Date(cls.lastSuccessMs).toISOString();
  entry.excluded = cls.excluded;
  entry.failingSince = cls.failingSinceMs === null ? null : new Date(cls.failingSinceMs).toISOString();
  entry.warnings = [...(cls.state === ACCOUNT_STATES.PLAN_INACTIVE ? [] : planWarnings), ...cls.warnings];
  return entry;
}

/**
 * Normalize Magpie's `quotas.json`, `/v1/magpie/quotas` array, AgentCompass `usage.json`,
 * or shorthand `{ provider: { remaining, resetsInHours, windows } }` into a uniform array
 * of account quota records:
 *   `[{ key, provider, account, kind, unlimited, stale, balance, plan, windows: [windowState, ...],
 *       state, reason, lastSuccessAt, excluded, warnings }]`
 * `state` is the account status shared by dispatch and the account list (see `ACCOUNT_STATES`).
 * `opts.usageMaxAgeHours` (else `policy.usage_max_age_hours`, else a top-level `usage_max_age_hours`
 * in the snapshot, else 6) is how long an account may go without a successful reading before it is excluded;
 * `usageFreshHours` / `usage_fresh_hours` (default 2) is when a reading with no error starts counting as failing.
 */
export function normalizeQuotaSnapshot(rawQuotas, nowMs = Date.now(), policy = DEFAULT_SCORING_POLICY, opts = {}) {
  if (!rawQuotas || typeof rawQuotas !== "object") return [];

  const maxAgeHours =
    [opts?.usageMaxAgeHours, policy?.usage_max_age_hours, rawQuotas.usage_max_age_hours, rawQuotas.usageMaxAgeHours]
      .map(Number)
      .find((n) => Number.isFinite(n) && n > 0) ?? DEFAULT_USAGE_MAX_AGE_HOURS;
  const maxAgeMs = maxAgeHours * 3600000;
  const freshHours =
    [opts?.usageFreshHours, policy?.usage_fresh_hours, rawQuotas.usage_fresh_hours, rawQuotas.usageFreshHours]
      .map(Number)
      .find((n) => Number.isFinite(n) && n > 0) ?? DEFAULT_USAGE_FRESH_HOURS;
  const freshMs = Math.min(freshHours, maxAgeHours) * 3600000;
  const entries = [];

  // Format A: Array from `/v1/magpie/quotas`
  if (Array.isArray(rawQuotas)) {
    for (const item of rawQuotas) {
      if (!item || typeof item !== "object") continue;
      const provider = String(item.provider || item.type || "").trim().toLowerCase();
      const account = String(item.name || item.account || item.user || provider || "default").trim();
      const key = item.key || (provider ? `${provider}/${account}` : account);
      const rawWindows = Array.isArray(item.windows)
        ? item.windows
        : Array.isArray(item.quota?.windows)
          ? item.quota.windows
          : [];
      const stale = Boolean(item.stale || item.status === "stale" || item.error);
      const kind = String(
        item.kind || item.billing || (rawWindows.length > 0 ? "subscription" : item.balance ? "balance" : "subscription")
      ).toLowerCase();
      entries.push(
        buildAccountEntry(item, {
          key,
          provider: provider || inferModelProviderFamily(key),
          account,
          rawWindows,
          kind,
          unlimited: Boolean(item.unlimited),
          balance: item.balance || null,
          stale,
          nowMs,
          policy,
          maxAgeMs,
          freshMs,
        })
      );
    }
    return entries;
  }

  // Format B: AgentCompass `usage.json` (`{ accounts: { claude: { status, windows: [...] } } }`)
  if (rawQuotas.accounts && typeof rawQuotas.accounts === "object" && !Array.isArray(rawQuotas.accounts)) {
    return normalizeQuotaSnapshot(rawQuotas.accounts, nowMs, policy, { usageMaxAgeHours: maxAgeHours, usageFreshHours: freshHours, ...opts });
  }

  // Format C: Magpie `quotas.json` object (`{ "claude/user@example.com": { quota: { windows: [...] } } }`)
  // or shorthand (`{ claude: { remaining: 75, resetsInHours: 12 } }`)
  for (const [rawKey, item] of Object.entries(rawQuotas)) {
    if (!item || typeof item !== "object") continue;
    const slashIdx = rawKey.indexOf("/");
    const keyProvider = slashIdx > 0 ? rawKey.slice(0, slashIdx) : rawKey;
    const keyAccount = slashIdx > 0 ? rawKey.slice(slashIdx + 1) : rawKey;
    const provider = String(item.provider || keyProvider || "").trim().toLowerCase();
    const account = String(item.name || item.account || keyAccount || provider).trim();
    const stale = Boolean(item.stale || item.status === "stale");

    let rawWindows = [];
    if (Array.isArray(item.windows)) {
      rawWindows = item.windows;
    } else if (Array.isArray(item.quota?.windows)) {
      rawWindows = item.quota.windows;
    } else if (
      typeof item.remaining === "number" ||
      typeof item.used === "number" ||
      typeof item.remaining_pct_before_reset === "number" ||
      typeof item.used_pct === "number"
    ) {
      rawWindows = [
        {
          name: item.window || item.name || "7d",
          remaining: item.remaining ?? item.remaining_pct_before_reset,
          used: item.used ?? item.used_pct,
          resetsAt: item.resetsAt ?? item.resets_at,
          resetHours: item.resetHours ?? item.resetsInHours,
          hours: item.hours,
          capacity_usd: item.capacity_usd ?? item.capacityUsd,
          pct_per_hour_24h: item.pct_per_hour_24h ?? item.pctPerHour24h,
          pct_per_hour_3h: item.pct_per_hour_3h ?? item.pctPerHour3h,
          unlimited: item.unlimited,
        },
      ];
    }

    const kind = String(
      item.kind ||
        item.billing ||
        item.quota?.kind ||
        (rawWindows.length > 0 ? "subscription" : item.balance || item.quota?.balance ? "balance" : "subscription")
    ).toLowerCase();

    entries.push(
      buildAccountEntry(item, {
        key: rawKey,
        provider: provider || inferModelProviderFamily(rawKey),
        account,
        rawWindows,
        kind,
        unlimited: Boolean(item.unlimited || item.quota?.unlimited),
        balance: item.balance || item.quota?.balance || null,
        stale,
        nowMs,
        policy,
        maxAgeMs,
        freshMs,
      })
    );
  }

  return entries;
}

/**
 * The account list: one row per account with its state, why, plan and last successful reading.
 * This is what the plugin shows users (`accounts` in decisions, `formatQuotaAccountList`).
 */
export function listQuotaAccounts(rawQuotas, nowMs = Date.now(), policy = DEFAULT_SCORING_POLICY, opts = {}) {
  if (!rawQuotas || typeof rawQuotas !== "object") return [];
  const entries = Array.isArray(rawQuotas._normalized)
    ? rawQuotas._normalized
    : normalizeQuotaSnapshot(rawQuotas, nowMs, policy, opts);
  return entries.map((e) => ({
    key: e.key,
    provider: e.provider,
    account: e.account,
    plan: e.plan,
    plan_known: e.planKnown,
    // `status` is AgentCompass's field name; `state` is the same value.
    status: e.state,
    state: e.state,
    reason: e.reason,
    last_success_at: e.lastSuccessAt,
    ...(e.state === ACCOUNT_STATES.USAGE_FAILING
      ? { failing_since: e.failingSince, exclude_after_hours: e.maxAgeHours, usage_excluded: e.excluded }
      : {}),
    ...(e.planConflict ? { plan_conflict: e.planConflict } : {}),
    dispatchable: !e.excluded,
    warnings: e.warnings,
  }));
}

/** Plain-text account list for CLI / status output. */
export function formatQuotaAccountList(accounts) {
  if (!Array.isArray(accounts) || accounts.length === 0) return "No quota accounts.";
  return accounts
    .map((a) => {
      const flag = a.dispatchable ? "" : " [excluded from dispatch]";
      const why = a.reason ? ` - ${a.reason}` : "";
      return `${a.key}  plan=${a.plan ?? "unknown"}  state=${a.state}${flag}  last_success=${a.last_success_at ?? "never"}${why}`;
    })
    .join("\n");
}

/** Every account's warning (plan unknown, reading failing, excluded, ...), deduplicated. */
export function quotaAccountWarnings(rawQuotas, nowMs = Date.now(), policy = DEFAULT_SCORING_POLICY) {
  return [...new Set(listQuotaAccounts(rawQuotas, nowMs, policy).flatMap((a) => a.warnings))];
}

/**
 * Quotas for the "everything was filtered" fallback: only account health still applies (expired, failing,
 * signed-out and disabled accounts stay excluded) while exhausted windows are ignored.
 */
export function healthOnlyQuotas(rawQuotas, nowMs = Date.now(), policy = DEFAULT_SCORING_POLICY) {
  if (!rawQuotas || typeof rawQuotas !== "object") return rawQuotas;
  const normalized = Array.isArray(rawQuotas._normalized)
    ? rawQuotas._normalized
    : normalizeQuotaSnapshot(rawQuotas, nowMs, policy);
  return { _normalized: normalized, _healthOnly: true };
}

/**
 * Forecast percent of the window that other work will draw before the window resets
 * (`allocator.py:demand_forecast`).
 */
export function demandForecast(windowState, policy = DEFAULT_SCORING_POLICY, forceShortForecast = false) {
  if (!windowState || !windowState.known || windowState.unlimited) return null;

  let rate = null;
  let source = "observed";

  if (windowState.short && !forceShortForecast) {
    rate = windowState.pctPerHour3h ?? windowState.pctPerHour24h;
    if (typeof rate !== "number" || !Number.isFinite(rate) || rate < 0) {
      return null;
    }
  } else {
    rate = windowState.pctPerHour24h;
    const elapsed = Math.max(0, windowState.hours - windowState.resetHours);
    const minElapsed = Number(policy?.demand_min_elapsed_hours ?? 6);
    if (
      (typeof rate !== "number" || !Number.isFinite(rate) || rate < 0) &&
      elapsed >= minElapsed &&
      typeof windowState.remaining === "number"
    ) {
      rate = Math.max(0.0, 100.0 - windowState.remaining) / elapsed;
      source = "period average";
    }
    if (typeof rate !== "number" || !Number.isFinite(rate) || rate < 0) {
      rate = 100.0 / Math.max(1.0, windowState.hours);
      source = "straight-line schedule";
    }
    if (
      source === "observed" &&
      windowState.remaining < (windowState.targetRemaining ?? 0)
    ) {
      const paceFloorRatio = Number(policy?.demand_pace_floor_ratio ?? 0.5);
      rate = Math.max(rate, (paceFloorRatio * 100.0) / Math.max(1.0, windowState.hours));
    }
  }

  const endHours = Math.max(0.0, windowState.endHours);
  const mean = rate * endHours;
  const sdFloor =
    Number(policy?.demand_sd_floor_pct ?? 5) * Math.min(1.0, Math.max(0.0, endHours) / 24.0);
  const cv = Number(policy?.demand_cv ?? 0.35);
  const sd = Math.max(cv * mean, sdFloor, 1e-6);

  return { mean, sd, rate, source };
}

/**
 * Compute one window's opportunity cost, scarcity cost, and pre-reset surplus bonus
 * (`allocator.py:window_cost`).
 */
export function computeWindowCost(windowState, stepPct, forecast, policy = DEFAULT_SCORING_POLICY, stepUsd = null) {
  const { mean, sd, rate, source } = forecast;
  const remaining = windowState.remaining;
  const endHours = windowState.endHours;
  const safetyMarginPct = Number(policy?.safety_margin_pct ?? 5);
  const usable = remaining - safetyMarginPct;

  const shortage = Math.max(
    0.0,
    expectedShortage(usable - stepPct, mean, sd) - expectedShortage(usable, mean, sd)
  );
  const shortageProb = shortageProbability(usable - stepPct, mean, sd);
  const capacityUsd = windowState.capacityUsd || 180.0;
  const realStepUsd =
    stepUsd !== null && stepUsd !== undefined
      ? stepUsd
      : (stepPct * capacityUsd) / 100.0;
  const displacedUsd = (shortage * capacityUsd) / 100.0;
  const scorePerUsd = Number(
    policy?.opportunity_exchange?.score_per_displaced_usd ??
      policy?.shortage_loss_per_usd ??
      0.012
  );
  let cost = scorePerUsd * displacedUsd;

  const scarcityRef = Number(policy?.scarcity_usable_ref_pct ?? 45.0);
  const scarcity =
    !windowState.short && scarcityRef > 0
      ? shortageProb *
        Math.pow(clamp(1.0 - Math.max(0.0, usable - stepPct) / scarcityRef, 0.0, 1.0), 1.5)
      : 0.0;
  let scarcityCost = Number(policy?.scarcity_penalty_max ?? 0.06) * scarcity;
  cost += scarcityCost;

  const horizon = Number(policy?.reset_horizon_hours ?? 36);
  let sunkWeight = 0.0;
  const excessPct = Math.max(0.0, usable - stepPct - mean);

  if (
    horizon > 0 &&
    !windowState.stale &&
    windowState.resetHours > 0 &&
    windowState.resetHours <= horizon &&
    usable - stepPct > 0
  ) {
    const base = clamp(1.0 - windowState.resetHours / horizon, 0.0, 1.0);
    const urgencyPower = Number(policy?.reset_urgency_power ?? 1.0);
    const urgency = Math.pow(
      0.4 * base + 0.6 * Math.exp(-windowState.resetHours / 12.0),
      urgencyPower
    );
    const surplusRef = Math.max(1e-6, Number(policy?.reset_surplus_ref_pct ?? 15));
    const surplus = (1.0 - shortageProb) * clamp(excessPct / surplusRef, 0.0, 1.0);
    sunkWeight = round4(urgency * surplus);
  }

  const resetBonus =
    sunkWeight > 0
      ? round4(Number(policy?.reset_bonus_max ?? 1.2) * sunkWeight)
      : 0.0;

  cost *= 1.0 - sunkWeight;
  scarcityCost *= 1.0 - sunkWeight;

  return {
    window: windowState.id,
    remaining: round3(remaining),
    after: round3(remaining - stepPct),
    step_pct: round3(stepPct),
    capacity_usd: round3(capacityUsd),
    expected_usd: round3(realStepUsd),
    reset_hours: round3(windowState.resetHours),
    end_hours: round3(endHours),
    demand_pct: round3(mean),
    demand_source: source,
    rate_pct_per_hour: round3(rate),
    shortage_probability: round3(shortageProb),
    scarcity_cost: round4(scarcityCost),
    usd_displaced: round3(displacedUsd),
    reset_sunk_weight: sunkWeight,
    reset_bonus: resetBonus,
    surplus_pct: round3(excessPct),
    opportunity_cost: round4(cost),
  };
}

/**
 * Evaluate a single account's quota eligibility and opportunity cost for `(canonicalModel, effort)`.
 */
export function evaluateAccountQuotaForModel(
  accountEntry,
  canonicalModelName,
  modelSpec,
  effort,
  effortScale,
  taskHours,
  policy = DEFAULT_SCORING_POLICY,
  rawModel = null
) {
  const excluded = [];
  const kind = String(accountEntry.kind || "subscription").toLowerCase();
  const isMetered = kind === "balance" || kind === "metered" || kind === "api" || kind === "payg";

  // Account health comes first: an expired, signed-out, disabled or long-failing account is never
  // selectable, whatever its (old) readings say, and `unlimited` does not rescue it.
  if (accountEntry.excluded) {
    return {
      account: accountEntry.key,
      provider: accountEntry.provider,
      kind,
      state: accountEntry.state,
      eligible: false,
      accountUnavailable: true,
      excluded: [`${accountEntry.key}: ${accountEntry.state}: ${accountEntry.reason}`],
      opportunity_cost: 0.0,
      reset_bonus: 0.0,
      reset_sunk_weight: 0.0,
      binding_window: null,
    };
  }

  if (accountEntry.unlimited) {
    return {
      account: accountEntry.key,
      provider: accountEntry.provider,
      kind,
      eligible: true,
      excluded: [],
      opportunity_cost: 0.0,
      free_usage: Number(policy?.unlimited_bonus ?? 0.18),
      reset_bonus: 0.0,
      reset_sunk_weight: 0.0,
      binding_window: null,
    };
  }

  const applicable = (accountEntry.windows || []).filter((w) =>
    windowBindsModel(w.id, canonicalModelName, w.models, rawModel)
  );

  // If this is a metered / balance-only account with no subscription windows
  if (applicable.length === 0) {
    if (isMetered) {
      const bal = accountEntry.balance;
      if (bal && typeof bal.amount === "number" && bal.amount <= 0) {
        excluded.push(`${accountEntry.key}: prepaid balance is exhausted (${bal.amount})`);
      }
      const usdPerHour = Number(modelSpec?.usd_per_hour ?? 4.0);
      const stepUsd = taskHours * usdPerHour * effortScale;
      const scorePerUsd = Number(
        policy?.opportunity_exchange?.score_per_displaced_usd ??
          policy?.shortage_loss_per_usd ??
          0.012
      );
      // Metered API keys pay direct list-price opportunity cost plus a small reserve preference penalty
      const meteredCost = round4(scorePerUsd * stepUsd + 0.035);
      return {
        account: accountEntry.key,
        provider: accountEntry.provider,
        kind: "metered",
        eligible: excluded.length === 0,
        excluded,
        opportunity_cost: meteredCost,
        reset_bonus: 0.0,
        reset_sunk_weight: 0.0,
        binding_window: null,
      };
    }
    return {
      account: accountEntry.key,
      provider: accountEntry.provider,
      kind,
      eligible: true,
      excluded: [],
      opportunity_cost: Number(policy?.unknown_capacity_penalty ?? 0.03),
      reset_bonus: 0.0,
      reset_sunk_weight: 0.0,
      binding_window: null,
    };
  }

  const usdPerHour = Number(modelSpec?.usd_per_hour ?? 4.0);
  const durMult = Number(modelSpec?.duration_multiplier ?? 1.0);
  const effectiveHours = taskHours * durMult;
  const stepUsd = effectiveHours * usdPerHour * effortScale;

  const hasLongWindow = applicable.some((w) => !w.short && w.known && w.capacityUsd !== null);
  let totalOpportunityCost = 0.0;
  let binding = null;

  for (const w of applicable) {
    if (!w.known) continue;
    if (w.unlimited) continue;

    if (w.capacityUsd === null) {
      if (w.rateLimited || w.remaining <= 0) {
        excluded.push(
          `${accountEntry.key} ${w.id}: ${w.remaining.toFixed(1)}% remaining is exhausted`
        );
      }
      continue;
    }

    const unitRatePct = w.capacityUsd > 0 ? (100.0 * usdPerHour * effortScale) / w.capacityUsd : 2.0 * effortScale;
    const stepPct = effectiveHours * unitRatePct;

    if (w.short) {
      const smallShort = w.capacityUsd < Number(policy?.small_short_capacity_usd ?? 50);
      const floorPct = smallShort
        ? Number(policy?.small_short_window_floor_pct ?? 25)
        : Number(policy?.short_window_floor_pct ?? 10);
      const marginPct = smallShort
        ? Number(policy?.small_short_safety_margin_pct ?? 12)
        : Number(policy?.safety_margin_pct ?? 5);

      if (w.rateLimited || w.remaining <= floorPct || w.remaining - stepPct < marginPct) {
        excluded.push(
          `${accountEntry.key} ${w.id}: ${w.remaining.toFixed(1)}% remaining is below short-window floor (${floorPct}% / ${marginPct}% margin)`
        );
        continue;
      }
      // If the account has a weekly/long window, the short window is a rate-cap gate and the weekly window prices the loss.
      // If the account ONLY has a short window (e.g. Claude Enterprise 5h-only), price it directly.
      if (hasLongWindow) {
        continue;
      }
    }

    const safetyMarginPct = Number(policy?.safety_margin_pct ?? 5);
    if (w.rateLimited || w.remaining <= 0 || w.remaining - stepPct < safetyMarginPct) {
      excluded.push(
        `${accountEntry.key} ${w.id}: ${w.remaining.toFixed(1)}% remaining is below ${safetyMarginPct}% safety margin (needs ~${stepPct.toFixed(1)}%)`
      );
      continue;
    }

    const forecast = demandForecast(w, policy, !hasLongWindow);
    if (!forecast) continue;

    const row = computeWindowCost(w, stepPct, forecast, policy, stepUsd);
    totalOpportunityCost += row.opportunity_cost;

    if (
      binding === null ||
      row.opportunity_cost > binding.opportunity_cost ||
      (row.opportunity_cost === binding.opportunity_cost && row.after < binding.after)
    ) {
      binding = row;
    }
  }

  return {
    account: accountEntry.key,
    provider: accountEntry.provider,
    kind,
    eligible: excluded.length === 0,
    excluded,
    opportunity_cost: round4(totalOpportunityCost),
    reset_bonus: binding ? binding.reset_bonus : 0.0,
    reset_sunk_weight: binding ? binding.reset_sunk_weight : 0.0,
    shortage_probability: binding ? binding.shortage_probability : 0.0,
    binding_window: binding,
  };
}

/**
 * Evaluate subscription quota economics across all matching accounts for a candidate `(rawModel, effort)`.
 * If a provider has multiple accounts in `quotas.json`, selects the best eligible account
 * (highest `reset_bonus - opportunity_cost`).
 */
export function evaluateModelQuota(
  rawModel,
  effort,
  taskCtx = {},
  rawQuotas = null,
  catalog = {},
  nowMs = Date.now()
) {
  if (!rawQuotas) return null;

  const models = catalog.models || DEFAULT_MODELS;
  const aliases = catalog.aliases || MODEL_ALIASES;
  const policy = catalog.policy || DEFAULT_SCORING_POLICY;
  const curvesTable = catalog.effortCurves || EFFORT_CURVES;

  const { canonical, spec: info } = resolveModelSpec(rawModel, models, aliases);
  const curves = curvesTable[canonical] || {};
  const refBurn = curves?.burn?.high ?? policy?.effort_burn?.high ?? 0.7;
  const effBurn = curves?.burn?.[effort] ?? policy?.effort_burn?.[effort] ?? refBurn;
  const effortScale = refBurn > 0 ? effBurn / refBurn : 1.0;

  const taskHours = Number(
    taskCtx.hours ??
      taskCtx.profile?.expected_hours ??
      policy?.default_hours?.[taskCtx.difficulty] ??
      0.5
  );

  const normalized = Array.isArray(rawQuotas._normalized)
    ? rawQuotas._normalized
    : normalizeQuotaSnapshot(rawQuotas, nowMs, policy);

  if (normalized.length === 0) return null;

  const expiredPool = normalized.filter((entry) => entry.poolLevel && entry.excluded);
  if (expiredPool.length > 0) {
    const allWarnings = [...new Set(normalized.flatMap((e) => e.warnings || []))];
    return {
      eligible: false,
      accountUnavailable: true,
      warnings: allWarnings,
      excluded: expiredPool.map((e) => `${e.key}: ${e.state}: ${e.reason}`),
      account: expiredPool[0].key,
      provider: expiredPool[0].provider,
      kind: "subscription",
      opportunity_cost: 0.0,
      reset_bonus: 0.0,
      reset_sunk_weight: 0.0,
      binding_window: null,
    };
  }

  const matchingAccounts = normalized.filter((entry) =>
    providerMatchesModel(entry.provider, rawModel, aliases) ||
    providerMatchesModel(entry.key, rawModel, aliases)
  );

  if (matchingAccounts.length === 0) {
    return null;
  }

  const evaluations = matchingAccounts.map((entry) =>
    evaluateAccountQuotaForModel(
      rawQuotas._healthOnly === true ? { ...entry, windows: [] } : entry,
      canonical,
      info,
      effort,
      effortScale,
      taskHours,
      policy,
      rawModel
    )
  );
  // Every matching account that is not `ready` is reported, selected or not (no silent failures).
  const warnings = [...new Set(matchingAccounts.flatMap((e) => e.warnings || []))];

  evaluations.forEach((e, i) => {
    e.state = matchingAccounts[i].state;
  });
  const eligibleAccounts = evaluations.filter((e) => e.eligible);
  if (eligibleAccounts.length === 0) {
    return {
      eligible: false,
      accountUnavailable: evaluations.length > 0 && evaluations.every((e) => e.accountUnavailable === true),
      warnings,
      excluded: evaluations.flatMap((e) => e.excluded),
      account: evaluations[0]?.account || null,
      provider: evaluations[0]?.provider || null,
      kind: evaluations[0]?.kind || "subscription",
      opportunity_cost: 0.0,
      reset_bonus: 0.0,
      reset_sunk_weight: 0.0,
      binding_window: null,
    };
  }

  // Prefer subscription accounts over metered accounts unless metered is the only eligible option;
  // among subscription accounts, pick the one with the best net quota score (reset_bonus - opportunity_cost).
  eligibleAccounts.sort((a, b) => {
    const aSub = a.kind !== "metered" ? 1 : 0;
    const bSub = b.kind !== "metered" ? 1 : 0;
    if (aSub !== bSub) return bSub - aSub;
    // An account whose reading is failing (still inside the grace period) is the last resort among peers.
    const aReady = a.state === ACCOUNT_STATES.USAGE_FAILING ? 0 : 1;
    const bReady = b.state === ACCOUNT_STATES.USAGE_FAILING ? 0 : 1;
    if (aReady !== bReady) return bReady - aReady;
    const aNet = (a.reset_bonus || 0) - (a.opportunity_cost || 0);
    const bNet = (b.reset_bonus || 0) - (b.opportunity_cost || 0);
    return bNet - aNet;
  });

  const best = eligibleAccounts[0];
  if (best.kind === "metered" && taskCtx.metered_ok === false) {
    return {
      ...best,
      warnings,
      eligible: false,
      excluded: [`${best.account}: metered billing is disabled (metered_ok: false)`],
    };
  }

  return { ...best, warnings };
}

/**
 * Quick check for `evaluateProviderContinuation`: returns `true` when the sticky session's
 * current model/provider has exhausted its subscription quota (`remaining <= safety_margin_pct`),
 * so the session hold should break with `why: "quota_exhausted"` (matching Magpie's `spent`).
 */
export function isModelQuotaExhausted(
  rawModel,
  rawQuotas,
  policy = DEFAULT_SCORING_POLICY,
  nowMs = Date.now(),
  preferredProvider = ""
) {
  if (!rawQuotas || !rawModel) return false;
  const evalResult = evaluateModelQuota(
    preferredProvider && !rawModel.includes("/") ? `${preferredProvider}/${rawModel}` : rawModel,
    "medium",
    { hours: 0.1, difficulty: "routine" },
    rawQuotas,
    { policy },
    nowMs
  );
  if (!evalResult) return false;
  return evalResult.eligible === false;
}

/**
 * True when `rawQuotas` is present and every model in `candidateModels` (or the default catalog
 * when `candidateModels` is omitted) is blocked by account non-dispatchability (`excluded`,
 * `pool/quotas` expired >= 6h, exhausted prepaid balance <= 0, or `meteredOk: false`).
 */
export function areCandidatesAccountBlocked(
  candidateModels,
  rawQuotas,
  options = {},
  nowMs = Date.now()
) {
  if (!rawQuotas || typeof rawQuotas !== "object") return false;
  const catalog = options?.catalog || {};
  const policy = catalog.policy || DEFAULT_SCORING_POLICY;
  const models = catalog.models || DEFAULT_MODELS;
  const hq = healthOnlyQuotas(rawQuotas, nowMs, policy);
  const list = Array.isArray(hq?._normalized) ? hq._normalized : [];
  if (list.length === 0) return false;
  if (list.some((e) => e.poolLevel && e.excluded)) return true;

  const rawCandidates =
    Array.isArray(candidateModels) && candidateModels.length > 0
      ? candidateModels
      : Object.keys(models).filter((m) => m !== "deepseek-v4.1-flash-free" && !models[m]?.disabled_by_default);
  const modelNames = rawCandidates
    .map((item) => {
      if (typeof item === "string") return item.trim();
      if (item && typeof item === "object") return String(item.model || item.id || item.name || "").trim();
      return "";
    })
    .filter(Boolean);
  if (modelNames.length === 0) return false;

  const taskCtx = {
    hours: 0.5,
    metered_ok: options?.meteredOk !== false,
  };
  return modelNames.every((m) => {
    const qEval = evaluateModelQuota(m, "high", taskCtx, hq, catalog, nowMs);
    return Boolean(qEval && !qEval.eligible);
  });
}

/**
 * Magpie Pool Seat Ordering Strategy (`strategies.agentcompass` & `strategies["agentcompass-pure"]`).
 *
 * Combines Magpie's `Seat` telemetry (`internal/middleware/points.go:201-212`:
 * `{ key, account, known, learns, used, soon, pace, weight, resting, tokens }`)
 * with AgentCompass's subscription opportunity-cost and pre-reset surplus sprint economics:
 *
 * 1. Partitions seats into 4 availability tiers:
 *    - Tier 0 (`fine`): `!excluded && !resting && used < keepPercent (default 90%)`
 *    - Tier 1 (`low` reserve): `!excluded && !resting && used >= keepPercent && used < 100%`
 *    - Tier 2 (`spent` / `resting`): `!excluded && (resting || used >= 100%)`
 *    - Tier 3 (`excluded`): `plan_inactive`, `auth_failed`, `disabled`, or `usage_failing` past 6h
 * 2. Within each tier, non-expired `usage_failing` seats rank after `ready` seats.
 * 3. Within Tier 0 (`fine`):
 *    - `seat.learns === true` goes first once (so unknown accounts that learn their quota from
 *      response headers get probed, matching Magpie's `Smart` and `Pace`).
 *    - Subscription seats (`seat.account !== false` and known window/use) rank ahead of
 *      pay-as-you-go API keys (`seat.account === false` with no window), preventing paid wallet
 *      balance from burning while subscription quota sits idle.
 *    - Computes each subscription seat's net AgentCompass quota score (`resetBonus - opportunityCost`)
 *      from `remaining = 100 - used`, `soon[0]` (window renewal unix ms), and `pace` (% left/hour):
 *      * Seats inside the 36h pre-reset horizon with surplus quota (`resetBonus > 0`) go FIRST
 *        so expiring allowance is harvested before reset!
 *      * Otherwise, seats with the lowest opportunity cost / highest sustainable pace go first.
 * 4. Returns a permutation of seat indices `[0 .. seats.length - 1]`.
 */
const EXCLUDED_SEAT_STATES = new Set([
  ACCOUNT_STATES.PLAN_INACTIVE,
  ACCOUNT_STATES.AUTH_FAILED,
  ACCOUNT_STATES.DISABLED,
]);

/** A seat's weekly capacity: its own `capacityUsd`, else its plan's, else the smallest known plan of its provider. */
function seatCapacityUsd(seat, matchedEntry = null, poolProvider = "") {
  if (typeof seat.capacityUsd === "number" && seat.capacityUsd > 0) return seat.capacityUsd;
  const provider =
    String(seat.provider || matchedEntry?.provider || poolProvider || "").toLowerCase() ||
    inferModelProviderFamily(String(seat.key || ""));
  const planKey = normalizePlan(provider, seat.plan ?? matchedEntry?.plan).key;
  return resolveWindowCapacityUsd(provider || "default", "weekly", 168, null, null, planKey);
}

function matchSeatQuotaEntry(seat, poolProvider, normalizedQuotas) {
  if (!Array.isArray(normalizedQuotas) || normalizedQuotas.length === 0 || !seat) return null;
  const prov =
    String(seat.provider || poolProvider || "").trim().toLowerCase() ||
    inferModelProviderFamily(String(seat.key || ""));
  const key = typeof seat.key === "string" ? seat.key.trim() : "";
  const acct = typeof seat.account === "string" ? seat.account.trim() : "";
  const name = typeof seat.name === "string" ? seat.name.trim() : "";

  for (const e of normalizedQuotas) {
    if (e.poolLevel) continue;
    const provOk = !prov || !e.provider || providersMatch(prov, e.provider);
    if (
      key &&
      (e.key === key ||
        (prov && e.key === `${prov}/${key}`) ||
        (!key.includes("/") && provOk && e.account === key))
    ) {
      return e;
    }
    if (
      acct &&
      (e.key === acct ||
        (prov && e.key === `${prov}/${acct}`) ||
        (!acct.includes("/") && provOk && e.account === acct))
    ) {
      return e;
    }
    if (
      name &&
      (e.key === name ||
        (prov && e.key === `${prov}/${name}`) ||
        (!name.includes("/") && provOk && e.account === name))
    ) {
      return e;
    }
  }
  return null;
}

function resolveSeatHealth(seat, poolProvider, matchedEntry, nowMs, maxAgeMs) {
  if (!seat || typeof seat !== "object") {
    return { state: ACCOUNT_STATES.READY, excluded: false };
  }
  const prov =
    String(seat.provider || matchedEntry?.provider || poolProvider || "").trim().toLowerCase() ||
    inferModelProviderFamily(String(seat.key || ""));
  const rawState = String(seat.state || "").trim().toLowerCase().replace(/[\s-]+/g, "_");
  const rawStatus = String(seat.status || "").trim().toLowerCase().replace(/[\s-]+/g, "_");
  const planInfo = normalizePlan(prov, seat.plan ?? matchedEntry?.plan);
  const http = httpStatusOf(seat);
  const errCode = errorCodeOf(seat);

  let state = ACCOUNT_STATES.READY;
  let excluded = Boolean(seat.excluded === true || seat.usage_excluded === true || seat.dispatchable === false);

  if (seat.disabled === true || seat.enabled === false || rawState === ACCOUNT_STATES.DISABLED || rawStatus === "disabled") {
    state = ACCOUNT_STATES.DISABLED;
    excluded = true;
  } else if (
    rawState === ACCOUNT_STATES.PLAN_INACTIVE ||
    planInfo.inactive ||
    STATUS_PLAN_INACTIVE.has(rawStatus) ||
    http === 402 ||
    (errCode && INACTIVE_ERROR_CODES.has(errCode))
  ) {
    state = ACCOUNT_STATES.PLAN_INACTIVE;
    excluded = true;
  } else if (
    rawState === ACCOUNT_STATES.AUTH_FAILED ||
    http === 401 ||
    http === 403 ||
    STATUS_AUTH_FAILED.has(rawStatus) ||
    (errCode && STATUS_AUTH_FAILED.has(errCode))
  ) {
    state = ACCOUNT_STATES.AUTH_FAILED;
    excluded = true;
  } else if (
    rawState === ACCOUNT_STATES.USAGE_FAILING ||
    STATUS_USAGE_FAILING.has(rawStatus) ||
    seat.error ||
    seat.stale === true
  ) {
    state = ACCOUNT_STATES.USAGE_FAILING;
    const anchorMs = firstTimeMs(
      seat.last_success_at,
      seat.lastSuccessAt,
      seat.failing_since,
      seat.failingSince
    );
    if (anchorMs !== null && nowMs - anchorMs >= maxAgeMs) {
      excluded = true;
    }
  } else if (EXCLUDED_SEAT_STATES.has(rawState)) {
    state = rawState;
    excluded = true;
  }

  if (matchedEntry) {
    if (matchedEntry.excluded) {
      excluded = true;
      state = matchedEntry.state;
    } else if (state === ACCOUNT_STATES.READY && matchedEntry.state !== ACCOUNT_STATES.READY) {
      state = matchedEntry.state;
    }
  }

  return { state, excluded };
}

export function orderSeatsByQuotaEconomics(_pool, seats, options = {}) {
  if (!Array.isArray(seats)) return [];
  if (seats.length <= 1) return seats.map((_, i) => i);

  const nowMs = Number.isFinite(options?.now) ? Number(options.now) : Date.now();
  const keepPercent = Number(options?.keepPercent ?? 90);
  const policy = options?.catalog?.policy || DEFAULT_SCORING_POLICY;
  const maxAgeHours = Number(options?.usageMaxAgeHours ?? policy?.usage_max_age_hours ?? DEFAULT_USAGE_MAX_AGE_HOURS);
  const maxAgeMs = maxAgeHours * 3600000;
  const poolProvider = typeof _pool === "string" ? _pool.trim().toLowerCase() : "";

  const rawQuotas =
    options?.quotas && typeof options.quotas === "object"
      ? options.quotas
      : loadLiveQuotas(options, nowMs);
  const normalizedQuotas =
    rawQuotas && typeof rawQuotas === "object"
      ? Array.isArray(rawQuotas._normalized)
        ? rawQuotas._normalized
        : normalizeQuotaSnapshot(rawQuotas, nowMs, policy)
      : [];

  const matchedEntries = seats.map((s) => matchSeatQuotaEntry(s, poolProvider, normalizedQuotas));
  const seatHealth = seats.map((s, i) => resolveSeatHealth(s, poolProvider, matchedEntries[i], nowMs, maxAgeMs));

  const isSharedSubscription = (s) => {
    if (s.account === false) {
      return Boolean(s.known && Array.isArray(s.soon) && s.soon.some((t) => Number(t) > 0));
    }
    if (!s.known && s.account !== true) return false;
    return true;
  };

  const seatTier = (idx) => {
    const s = seats[idx] || {};
    if (seatHealth[idx]?.excluded) return 3;
    if (s.resting) return 2;
    const used = typeof s.used === "number" ? s.used : 0;
    if (used >= 100) return 2;
    if (used >= keepPercent) return 1;
    return 0;
  };

  // Pre-compute AgentCompass net quota score (`resetBonus - opportunityCost`) for each seat
  const netScores = seats.map((s, idx) => {
    const used = typeof s.used === "number" ? clamp(s.used, 0, 100) : 0;
    const remaining = 100 - used;
    const soonMs =
      Array.isArray(s.soon) && s.soon.length > 0 && Number(s.soon[0]) > 0
        ? Number(s.soon[0])
        : null;
    let resetHours =
      soonMs !== null && soonMs > nowMs ? (soonMs - nowMs) / 3600000 : null;
    if (
      (resetHours === null || resetHours <= 0) &&
      typeof s.pace === "number" &&
      s.pace > 0 &&
      remaining > 0
    ) {
      resetHours = remaining / s.pace;
    }
    if (resetHours === null || resetHours <= 0) {
      resetHours = 168;
    }

    const winState = {
      id: "week",
      hours: 168,
      short: false,
      known: Boolean(s.known || soonMs !== null || s.used > 0),
      stale: false,
      remaining,
      used,
      resetHours,
      endHours: resetHours,
      targetRemaining: 100 * Math.min(1.0, resetHours / 168),
      capacityUsd: seatCapacityUsd(s, matchedEntries[idx], poolProvider),
    };

    if (!winState.known) {
      return 0.0;
    }
    const forecast = demandForecast(winState, policy);
    if (!forecast) return 0.0;
    const costRow = computeWindowCost(winState, 1.5, forecast, policy);
    return round4((costRow.reset_bonus || 0) - (costRow.opportunity_cost || 0));
  });

  return seats
    .map((_, i) => i)
    .sort((ia, ib) => {
      const a = seats[ia] || {};
      const b = seats[ib] || {};

      const ta = seatTier(ia);
      const tb = seatTier(ib);
      if (ta !== tb) return ta - tb;

      // Within any tier, non-expired `usage_failing` seats rank after `ready` seats
      const aFailing = seatHealth[ia]?.state === ACCOUNT_STATES.USAGE_FAILING ? 1 : 0;
      const bFailing = seatHealth[ib]?.state === ACCOUNT_STATES.USAGE_FAILING ? 1 : 0;
      if (aFailing !== bFailing) return aFailing - bFailing;

      // Within Tier 0 (healthy seats)
      if (ta === 0) {
        if (Boolean(a.learns) !== Boolean(b.learns)) {
          return a.learns ? -1 : 1;
        }
        const aSub = isSharedSubscription(a);
        const bSub = isSharedSubscription(b);
        if (aSub !== bSub) {
          return aSub ? -1 : 1;
        }
        const diffScore = netScores[ib] - netScores[ia];
        if (Math.abs(diffScore) > 0.002) {
          return diffScore < 0 ? -1 : 1;
        }
        if (typeof a.pace === "number" && typeof b.pace === "number" && Math.abs(a.pace - b.pace) > 0.05) {
          return b.pace - a.pace;
        }
      }

      const aUsed = typeof a.used === "number" ? a.used : 50;
      const bUsed = typeof b.used === "number" ? b.used : 50;
      if (aUsed !== bUsed) return aUsed - bUsed;

      const aTokens = typeof a.tokens === "number" ? a.tokens : 0;
      const bTokens = typeof b.tokens === "number" ? b.tokens : 0;
      if (aTokens !== bTokens) return aTokens - bTokens;

      return ia - ib;
    });
}

export function shouldSkipQuotaAccount(acct, ctx, quotaAccounts) {
  if (!Array.isArray(quotaAccounts) || quotaAccounts.length === 0) return false;
  if (quotaAccounts.some((a) => a.key === "pool/quotas" && a.dispatchable === false)) {
    return true;
  }
  const rawProv =
    acct.provider ||
    ctx?.provider?.id ||
    ctx?.provider?.name ||
    (typeof acct.key === "string" && acct.key.includes("/") ? acct.key.split("/")[0] : "") ||
    (typeof acct.ref === "string" && acct.ref.includes("/") ? acct.ref.split("/")[0] : "");
  const acctProv = normalizeProviderFamily(rawProv);
  const fullKeys = new Set(
    [
      typeof acct.key === "string" && acct.key.includes("/") ? acct.key : "",
      typeof acct.ref === "string" && acct.ref.includes("/") ? acct.ref : "",
      acct.provider && acct.name ? `${acct.provider}/${acct.name}` : "",
      acct.provider && acct.account ? `${acct.provider}/${acct.account}` : "",
      acctProv && acct.name ? `${acctProv}/${acct.name}` : "",
      acctProv && acct.account ? `${acctProv}/${acct.account}` : "",
    ]
      .filter((v) => typeof v === "string" && v.trim())
      .map((v) => v.trim())
  );
  const bareNames = new Set(
    [
      acct.name,
      acct.account,
      typeof acct.key === "string" && !acct.key.includes("/") ? acct.key : "",
      typeof acct.ref === "string" && !acct.ref.includes("/") ? acct.ref : "",
    ]
      .filter((v) => typeof v === "string" && v.trim())
      .map((v) => v.trim())
  );
  const exactMatches = quotaAccounts.filter((a) => {
    if (a.key === "pool/quotas") return false;
    if (fullKeys.has(a.key)) return true;
    const provOk = !acctProv || !a.provider || providersMatch(acctProv, a.provider);
    if (!provOk) return false;
    return bareNames.has(a.account) || bareNames.has(a.key);
  });
  if (exactMatches.length > 0) {
    if (exactMatches.some((a) => a.dispatchable === false)) {
      return true;
    }
  } else if (acctProv) {
    const provMatches = quotaAccounts.filter(
      (a) => a.key !== "pool/quotas" && providersMatch(acctProv, a.provider)
    );
    if (provMatches.length > 0 && provMatches.every((a) => a.dispatchable === false)) {
      return true;
    }
  }
  return false;
}
