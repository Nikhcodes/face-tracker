/**
 * Temporal smoothing for frame verdicts.
 *
 *  - A pose must win for `enterMs` and `enterFrames` before it is shown.
 *  - Once shown, it survives brief drop-outs for `releaseMs` (hysteresis).
 *  - "No hands" becomes `idle` after `idleMs`.
 *  - "Hands but no pose" is reported separately from "no hands".
 */
import type { FrameResult, UnknownReason } from './classify';
import { EXPERIMENTAL_POSES, type DetectablePoseId } from './poseIds';

export type DisplayState =
  | { kind: 'none' }
  | { kind: 'idle'; since: number }
  | { kind: 'unknown'; reason: UnknownReason }
  | { kind: 'pose'; pose: DetectablePoseId; confidence: number; since: number };

export type StabilizerEvent =
  | { type: 'enter'; pose: DetectablePoseId | 'idle' }
  | { type: 'exit'; pose: DetectablePoseId | 'idle' }
  | { type: 'held'; pose: DetectablePoseId };

export interface StabilizerOptions {
  enterMs: number;
  /** Extra time experimental poses must be held before they show. */
  experimentalEnterMs: number;
  enterFrames: number;
  releaseMs: number;
  idleMs: number;
  holdMs: number;
  minConfidence: number;
  /** A candidate survives a single missed frame if seen this recently. */
  missToleranceMs: number;
}

export const DEFAULT_STABILIZER: StabilizerOptions = {
  enterMs: 160,
  experimentalEnterMs: 320,
  enterFrames: 3,
  releaseMs: 300,
  idleMs: 1500,
  holdMs: 1200,
  minConfidence: 0.6,
  missToleranceMs: 90,
};

interface Candidate {
  pose: DetectablePoseId;
  start: number;
  lastSeen: number;
  frames: number;
  confidence: number;
}

interface Active {
  pose: DetectablePoseId;
  since: number;
  lastSeen: number;
  confidence: number;
  heldFired: boolean;
}

export class PoseStabilizer {
  private readonly opts: StabilizerOptions;
  private candidate: Candidate | null = null;
  private active: Active | null = null;
  private lastHandsAt: number | null = null;
  private idleSince: number | null = null;
  private lastUnknown: UnknownReason = 'unrecognized';
  private lastKind: FrameResult['kind'] = 'none';

  constructor(options: Partial<StabilizerOptions> = {}) {
    this.opts = { ...DEFAULT_STABILIZER, ...options };
  }

  reset(now: number | null = null): void {
    this.candidate = null;
    this.active = null;
    this.idleSince = null;
    this.lastHandsAt = now;
    this.lastKind = 'none';
  }

  update(result: FrameResult, now: number): { state: DisplayState; events: StabilizerEvent[] } {
    const o = this.opts;
    const events: StabilizerEvent[] = [];
    this.lastKind = result.kind;
    if (this.lastHandsAt === null) this.lastHandsAt = now;

    if (result.kind !== 'none') {
      this.lastHandsAt = now;
      if (this.idleSince !== null) {
        this.idleSince = null;
        events.push({ type: 'exit', pose: 'idle' });
      }
    }
    if (result.kind === 'unknown') this.lastUnknown = result.reason;

    // Track the candidate pose.
    if (result.kind === 'pose' && result.confidence >= o.minConfidence) {
      if (this.candidate && this.candidate.pose === result.pose) {
        this.candidate.frames += 1;
        this.candidate.lastSeen = now;
        this.candidate.confidence = result.confidence;
      } else {
        this.candidate = { pose: result.pose, start: now, lastSeen: now, frames: 1, confidence: result.confidence };
      }
      if (this.active && this.active.pose === result.pose) {
        this.active.lastSeen = now;
        this.active.confidence = result.confidence;
      }
    } else if (this.candidate && now - this.candidate.lastSeen > o.missToleranceMs) {
      this.candidate = null;
    }

    const qualifies = (c: Candidate) =>
      now - c.start >= (EXPERIMENTAL_POSES.has(c.pose) ? o.experimentalEnterMs : o.enterMs) && c.frames >= o.enterFrames;

    if (this.candidate && qualifies(this.candidate) && this.active?.pose !== this.candidate.pose) {
      if (this.active) events.push({ type: 'exit', pose: this.active.pose });
      this.active = {
        pose: this.candidate.pose,
        since: now,
        lastSeen: now,
        confidence: this.candidate.confidence,
        heldFired: false,
      };
      events.push({ type: 'enter', pose: this.active.pose });
    } else if (this.active && now - this.active.lastSeen > o.releaseMs) {
      events.push({ type: 'exit', pose: this.active.pose });
      this.active = null;
    }

    if (this.active && !this.active.heldFired && now - this.active.since >= o.holdMs) {
      this.active.heldFired = true;
      events.push({ type: 'held', pose: this.active.pose });
    }

    if (!this.active && result.kind === 'none' && this.idleSince === null && now - this.lastHandsAt >= o.idleMs) {
      this.idleSince = now;
      events.push({ type: 'enter', pose: 'idle' });
    }

    return { state: this.state(), events };
  }

  state(): DisplayState {
    if (this.active) {
      return { kind: 'pose', pose: this.active.pose, confidence: this.active.confidence, since: this.active.since };
    }
    if (this.idleSince !== null) return { kind: 'idle', since: this.idleSince };
    if (this.lastKind === 'none') return { kind: 'none' };
    return { kind: 'unknown', reason: this.lastUnknown };
  }
}

/** Per-key cooldown so big reactions and sounds do not repeat when a pose flickers. */
export class Cooldown {
  private readonly last = new Map<string, number>();
  constructor(private readonly ms: number) {}

  /** Returns true (and starts the cooldown) if `key` is not cooling down. */
  take(key: string, now: number): boolean {
    const prev = this.last.get(key);
    if (prev !== undefined && now - prev < this.ms) return false;
    this.last.set(key, now);
    return true;
  }

  clear(): void {
    this.last.clear();
  }
}
