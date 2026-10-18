import { livePoseIds } from '../config/registry';
import { GESTURE_POSES, isCustomPoseId, type PoseId } from '../gestures/poseIds';
import { readJson, safeStorage } from '../lib/storage';
import { createStore } from '../lib/store';

/** Poses this viewer has performed (kept in this browser only). */
const KEY = 'posecam.unlocked.v1';
const BUILTIN = new Set<string>(GESTURE_POSES);
const valid = (x: unknown): x is PoseId => typeof x === 'string' && (BUILTIN.has(x) || isCustomPoseId(x));

function load(): ReadonlySet<PoseId> {
  const raw = readJson(KEY);
  return new Set(Array.isArray(raw) ? raw.filter(valid) : []);
}

export const unlockedStore = createStore<ReadonlySet<PoseId>>(load());
unlockedStore.subscribe(() => safeStorage.set(KEY, JSON.stringify([...unlockedStore.get()])));

/** Returns true if this call unlocked the pose for the first time. */
export function unlockPose(id: PoseId): boolean {
  if (!valid(id) || unlockedStore.get().has(id)) return false;
  unlockedStore.set((prev) => new Set([...prev, id]));
  return true;
}

export function resetProgress(): void {
  unlockedStore.set(new Set());
}

/** True once every enabled live built-in pose has been performed. */
export function hasAllLivePoses(set: ReadonlySet<PoseId> = unlockedStore.get()): boolean {
  const live = livePoseIds();
  return live.length > 0 && live.every((id) => set.has(id));
}
