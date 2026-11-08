/**
 * Multi-Protocol Request Extractor & Task Profile Schema
 *
 * A task profile comes from the classifier (`src/scorer-live.js`) or from the caller, never from the task's text:
 * without one the router does not route (`unroutedDecision`).
 * Pure ES2022 module (zero Node-only imports) so it runs identically in both Bun and `moejs`.
 */

import { INTERACTION_STYLES, KIND_TRAITS } from "./catalog.js";
import { analyzeTurnState } from "./continuity.js";

export const PROFILE_INTEGERS = {
  reasoning_depth: [1, 4],
  code_complexity: [0, 3],
  context_scale: [1, 3],
  divergence: [1, 3],
  hallucination_risk: [1, 3],
  error_cost: [1, 3],
};

export const PROFILE_FLAGS = ["needs_visual", "needs_gui", "local_only", "realtime_loop"];

export const REALTIME_LOOP_QUESTION =
  "True ONLY when the worker agent itself must stay inside a running external system's continuous observe-act loop across turns where slow per-turn model latency directly degrades the live outcome (e.g. personally playing a live game / dailies / dungeons / AFK loop, sniping tickets / courses / appointments / flash sales live, watching a live order book or API stream and placing orders in real time, polling a live page / app / API to act immediately when a state appears, interacting with live-stream chat). False for ALL software engineering, design, debugging, or analysis tasks—even when the subject matter is real-time, polling, games, dashboards, or flash sales (e.g. writing or fixing a ticket/game/automation script or bot, implementing real-time page refresh or message push, fixing page polling logic, designing a live trading dashboard frontend, diagnosing client loop waits or refresh latency, adding a countdown modal or live chat component, fixing flash-sale page CSS, analyzing game/AFK logs, or a one-off GUI click/setting toggle).";

export const MULTI_TASK_QUESTION =
  "True when the brief bundles more than one separate task: deliverables that different workers could produce at the same time without waiting for each other's result (e.g. collect reference images, compare game engines, design the gameplay loop). False for one task, however many steps: investigate-fix-test-release, implement plus its tests, a review plus its report, the remaining steps of a handoff, a brief that already assigns one share of a larger task, or a sequence where each part needs the previous part's output.";

export const TASK_COUNT_QUESTION =
  "How many separate tasks, in the sense of multi_task, the brief holds: 1 when multi_task is false.";

export const TASK_PROFILE_SCHEMA = {
  type: "object",
  properties: {
    work_kind: {
      type: "string",
      enum: Object.keys(KIND_TRAITS),
      description: "Primary work category.",
    },
    reasoning_depth: {
      type: "integer",
      minimum: 1,
      maximum: 4,
      description:
        "How hard the thinking is, not how long the brief is. 1 = routine or direct execution with obvious steps; 2 = standard engineering or analysis with a clear direction; 3 = cross-module architectural design/refactor, subtle concurrency or distributed-invariant debugging; 4 = frontier mathematical, strategic or architectural critique and formulation.",
    },
    code_complexity: {
      type: "integer",
      minimum: 0,
      maximum: 3,
      description:
        "Production code the worker itself writes or modifies (NOT the complexity of code being read or reviewed). 0 = none (read-only review/audit, CLI commands, log/data exploration, text drafting); 1 = small script, config/constant tweak, localized helper; 2 = production module implemented, fixed or refactored with unit tests; 3 = complex multi-module, concurrency, protocol or core-library changes.",
    },
    context_scale: {
      type: "integer",
      minimum: 1,
      maximum: 3,
      description:
        "Functional scope of the code or system that must be understood, NOT the number of paths the brief mentions. 1 = one module/component or one subsystem's runbook; 2 = several coupled functional modules or subsystems; 3 = whole-repository or cross-system architecture.",
    },
    divergence: {
      type: "integer",
      minimum: 1,
      maximum: 3,
      description:
        "1 = convergent (known target, spec, checklist, review scope or identified bug cause); 2 = multi-hypothesis (unknown root cause or comparing design options); 3 = open-ended/creative (formulate a new scheme, creative ideation, or first-principles architecture redesign).",
    },
    hallucination_risk: {
      type: "integer",
      minimum: 1,
      maximum: 3,
      description:
        "1 = a compiler, unit test or script deterministically verifies the result; 2 = mixed engineering plus judgment; 3 = a fabricated claim or missed defect would not be caught by a test runner.",
    },
    error_cost: {
      type: "integer",
      minimum: 1,
      maximum: 3,
      description:
        "1 = low (read-only, sandboxed, can be rerun); 2 = medium; 3 = high (production release or a critical irreversible operation).",
    },
    needs_visual: {
      type: "boolean",
      description: "True when the work must look at images, screenshots or other visual input.",
    },
    needs_gui: {
      type: "boolean",
      description: "True when the work must operate a desktop or browser GUI through screenshots, mouse and keyboard.",
    },
    local_only: {
      type: "boolean",
      description:
        "True when the task explicitly depends on existing files, directories, or repositories on this machine or on local processes/GUI/environment.",
    },
    realtime_loop: {
      type: "boolean",
      description: REALTIME_LOOP_QUESTION,
    },
    interaction_style: {
      type: "string",
      enum: [...INTERACTION_STYLES],
      description:
        "engineering = write or change software; exploration = explore a codebase, run experiments or backtests, analyze logs or data; text_only = pure text in and out or public web lookup; advisory = direction reasoning or complex build planning.",
    },
    expected_hours: {
      type: "number",
      minimum: 0.05,
      maximum: 24.0,
      description: "Active worker-hours this task needs at an ordinary agent pace.",
    },
    hours_confidence: {
      type: "string",
      enum: ["low", "medium", "high"],
      description: "How well the text supports expected_hours.",
    },
    urgency: {
      type: "string",
      enum: ["unknown", "low", "normal", "high"],
      description: "unknown unless the text itself states a deadline, an incident or 'asap'.",
    },
    multi_task: {
      type: "boolean",
      description: MULTI_TASK_QUESTION,
    },
    task_count: {
      type: "string",
      enum: ["1", "2", "3", ">3"],
      description: TASK_COUNT_QUESTION,
    },
  },
  required: [
    ...Object.keys(PROFILE_INTEGERS),
    ...PROFILE_FLAGS,
    "expected_hours",
    "hours_confidence",
    "urgency",
    "interaction_style",
    "multi_task",
    "task_count",
  ],
};

export const UNROUTED_NOTICE = "未路由：分类不可用";

/**
 * What a request is sent on with when it has no task profile: no routing decision, the client's model as it asked.
 */
export function unroutedDecision(reason) {
  return { ok: false, routed: false, notice: UNROUTED_NOTICE, reason, targetModel: null, targetEffort: null };
}

/**
 * A profile the caller passed, checked like the classifier's: `{ profile }`, `{ error }`, or `null` when there is none.
 * Image parts in the request are a structured signal, so they hold whatever the profile says.
 */
export function resolveCallerProfile(raw, hasVisualInput = false) {
  if (raw === undefined || raw === null) return null;
  try {
    const profile = validateProfile(raw);
    return { profile: hasVisualInput ? { ...profile, needs_visual: true } : profile };
  } catch (err) {
    return { error: `invalid_caller_profile: ${String(err?.message || err).slice(0, 120)}` };
  }
}

function extractSystemSummary(rawSystem, maxPreviewChars = 2000) {
  if (!rawSystem) return { preview: "", totalChars: 0 };
  if (typeof rawSystem === "string") {
    return {
      preview: rawSystem.slice(0, maxPreviewChars),
      totalChars: rawSystem.length,
    };
  }
  if (!Array.isArray(rawSystem)) return { preview: "", totalChars: 0 };
  let preview = "";
  let totalChars = 0;
  for (const part of rawSystem) {
    if (!part) continue;
    const t =
      typeof part === "string"
        ? part
        : typeof part.text === "string"
          ? part.text
          : typeof part.input_text === "string"
            ? part.input_text
            : typeof part.content === "string"
              ? part.content
              : "";
    if (t) {
      totalChars += t.length;
      if (preview.length < maxPreviewChars) {
        preview += (preview ? "\n" : "") + t.slice(0, maxPreviewChars - preview.length);
      }
    }
  }
  return { preview, totalChars };
}

/**
 * Parse a Magpie request body (`chat`, `responses`, `anthropic`, or `gemini`)
 * into a normalized routing evidence envelope in a single bounded O(N) pass.
 */
export function extractRequestFacts(body, ctx = {}) {
  if (!body || typeof body !== "object") {
    return {
      model: "",
      objective: "",
      turnText: "",
      lastUserText: "",
      previousUser: "",
      previousUserText: "",
      previousAssistantText: "",
      systemText: "",
      hasVisualInput: false,
      hasVisual: false,
      hasCurrentTurnVisual: false,
      hasComputerUse: false,
      demandsFrontierEscalation: false,
      isToolFollowup: false,
      within: false,
      turn: 0,
      toolCount: 0,
      scannedChars: 0,
      extractedTurns: 0,
      estimatedTokens: 0,
      estimatedInputTokens: 0,
      firstWords: "",
      sessionKey: "",
      agent: ctx?.agent || "",
      protocol: ctx?.protocol || "",
    };
  }

  const model = typeof body.model === "string" ? body.model : "";
  const sysSource =
    body.system !== undefined
      ? body.system
      : body.instructions !== undefined
        ? body.instructions
        : body.systemInstruction?.parts;
  const { preview: systemText, totalChars: sysChars } = extractSystemSummary(sysSource, 2000);

  const turnState = analyzeTurnState(body, ctx);
  const effectiveObjective = turnState.lastUserText || "";
  const effectivePrevUser = turnState.previousUserText || "";
  const hasVisual = Boolean(turnState.hasAnyVisual);
  const totalChars = sysChars + (turnState.totalChars || 0);
  const hasExplicitEstimatedTokens =
    Number.isFinite(body?.estimated_input_tokens) ||
    Number.isFinite(ctx?.estimated_input_tokens);
  const estimatedTokens = Number(
    body?.estimated_input_tokens ||
      ctx?.estimated_input_tokens ||
      Math.ceil(totalChars / 4)
  );

  const tools = Array.isArray(body.tools) ? body.tools : [];
  const needsObj = body?.needs && typeof body.needs === "object" ? body.needs : null;
  const explicitRealtimeLoop =
    ctx && typeof ctx.realtime_loop === "boolean"
      ? ctx.realtime_loop
      : needsObj && typeof needsObj.realtime_loop === "boolean"
        ? needsObj.realtime_loop
        : body && typeof body.realtime_loop === "boolean"
          ? body.realtime_loop
          : undefined;
  const callerProfile = [ctx?.state?.profile, ctx?.profile, body.profile].find((p) => p && typeof p === "object");

  return {
    model,
    objective: effectiveObjective,
    turnText: effectiveObjective,
    lastUserText: effectiveObjective,
    previousUser: effectivePrevUser,
    previousUserText: effectivePrevUser,
    previousAssistantText: turnState.previousAssistantText || "",
    systemText,
    hasVisualInput: hasVisual,
    hasVisual,
    hasCurrentTurnVisual: Boolean(turnState.hasCurrentTurnVisual),
    hasComputerUse: Boolean(needsObj?.gui === true || callerProfile?.needs_gui === true),
    demandsFrontierEscalation: Number(callerProfile?.reasoning_depth) >= 3,
    ...(callerProfile ? { callerProfile } : {}),
    ...(explicitRealtimeLoop !== undefined ? { realtime_loop: explicitRealtimeLoop } : {}),
    isToolFollowup: Boolean(turnState.isToolFollowup || turnState.within),
    within: Boolean(turnState.within || turnState.isToolFollowup),
    turn: turnState.turn,
    toolCount: tools.length,
    scannedChars: turnState.scannedChars || 0,
    extractedTurns: turnState.extractedTurns || 0,
    estimatedTokens,
    estimatedInputTokens: estimatedTokens,
    hasExplicitEstimatedTokens,
    firstWords: turnState.firstWords,
    sessionKey: turnState.sessionKey,
    agent: ctx?.agent || "",
    protocol: ctx?.protocol || "",
  };
}

/**
 * A profile from a request's structured signals alone (image parts, explicit `needs.gui`, `realtime_loop`), used only
 * when the turn has no text (`!facts.objective`) and the user set a candidate pool or a pin (contract R2, R3).
 */
export function noTextProfile(facts = {}) {
  const gui = Boolean(facts.hasComputerUse);
  return {
    work_kind: gui ? "computer_use" : "other_or_mixed",
    reasoning_depth: 1,
    code_complexity: gui ? 0 : 1,
    context_scale: 1,
    divergence: 1,
    hallucination_risk: gui ? 2 : 1,
    error_cost: gui ? 2 : 1,
    needs_visual: Boolean(facts.hasVisualInput || gui),
    needs_gui: gui,
    local_only: gui,
    realtime_loop: Boolean(facts.realtime_loop),
    interaction_style: "engineering",
    expected_hours: 0.4,
    hours_confidence: "medium",
    urgency: "unknown",
    multi_task: false,
    task_count: "1",
  };
}

const TASK_COUNTS = ["1", "2", "3", ">3"];

function taskCountBucket(value) {
  if (TASK_COUNTS.includes(value)) return value;
  if (typeof value === "number" && Number.isInteger(value) && value >= 1) {
    return TASK_COUNTS[Math.min(value, 4) - 1];
  }
  throw new Error(`task.profile.task_count must be one of ${TASK_COUNTS.join(", ")} or a positive whole number`);
}

export function validateProfile(raw) {
  if (!raw || typeof raw !== "object") {
    throw new Error("task.profile must be an object");
  }
  const profile = {};
  if (raw.work_kind !== undefined && raw.work_kind !== null) {
    if (typeof raw.work_kind !== "string" || !(raw.work_kind in KIND_TRAITS)) {
      throw new Error(`task.profile.work_kind must be one of ${Object.keys(KIND_TRAITS).join(", ")}`);
    }
    profile.work_kind = raw.work_kind;
  }
  for (const [name, [low, high]] of Object.entries(PROFILE_INTEGERS)) {
    if (name === "divergence" && !(name in raw)) {
      profile.divergence = 1;
      continue;
    }
    const val = Number(raw[name]);
    if (!Number.isInteger(val) || val < low || val > high) {
      throw new Error(`task.profile.${name} must be an integer from ${low} to ${high}`);
    }
    profile[name] = val;
  }
  for (const flag of PROFILE_FLAGS) {
    if (flag === "realtime_loop") {
      if ("realtime_loop" in raw && raw.realtime_loop !== undefined) {
        if (typeof raw.realtime_loop !== "boolean") {
          throw new Error("task.profile.realtime_loop must be true or false");
        }
        profile.realtime_loop = raw.realtime_loop;
      } else {
        profile.realtime_loop = false;
      }
    } else {
      profile[flag] = Boolean(raw[flag]);
    }
  }
  const style = raw.interaction_style || "engineering";
  if (!INTERACTION_STYLES.includes(style)) {
    throw new Error(`task.profile.interaction_style must be one of ${INTERACTION_STYLES.join(", ")}`);
  }
  profile.interaction_style = style;
  const hours = Number(raw.expected_hours ?? 0.5);
  profile.expected_hours = Number.isFinite(hours)
    ? Math.round(Math.min(Math.max(hours, 0.05), 24.0) * 100) / 100
    : 0.5;
  profile.hours_confidence = ["low", "medium", "high"].includes(raw.hours_confidence)
    ? raw.hours_confidence
    : "medium";
  profile.urgency = ["unknown", "low", "normal", "high"].includes(raw.urgency) ? raw.urgency : "unknown";
  if (typeof raw.reason === "string" && raw.reason.trim()) {
    profile.reason = raw.reason.trim().slice(0, 300);
  }
  if (typeof raw.multi_task !== "boolean") {
    throw new Error("task.profile.multi_task must be true or false");
  }
  profile.multi_task = raw.multi_task;
  profile.task_count = taskCountBucket(raw.task_count);
  if (profile.multi_task !== (profile.task_count !== "1")) {
    throw new Error(
      `task.profile.multi_task ${profile.multi_task} contradicts task.profile.task_count ${profile.task_count}`
    );
  }
  return profile;
}
