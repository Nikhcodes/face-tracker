import { readJson, safeStorage } from '../lib/storage';
import { createStore } from '../lib/store';

export const FILTERS = ['original', 'cinematic', 'mono', 'warm', 'cool'] as const;
export type FilterId = (typeof FILTERS)[number];
export const QUALITIES = ['performance', 'balanced', 'high'] as const;
export type Quality = (typeof QUALITIES)[number];
export const MOTION_PREFS = ['system', 'reduced', 'full'] as const;
export type MotionPref = (typeof MOTION_PREFS)[number];
export const PHOTO_FORMATS = ['png', 'jpeg'] as const;
export type PhotoFormat = (typeof PHOTO_FORMATS)[number];

export interface Settings {
  filter: FilterId;
  vignette: boolean;
  grain: boolean;
  /** Meme stickers for detected poses. */
  stickers: boolean;
  /** Captions, particles and other reactions. */
  reactions: boolean;
  /** Draw hand landmarks on the preview (and in photos). */
  landmarks: boolean;
  /** Debug panel with classifier internals. */
  debug: boolean;
  mirror: boolean;
  quality: Quality;
  sound: boolean;
  music: boolean;
  volume: number;
  motion: MotionPref;
  /** Three fingers starts the photo timer. */
  photoTimer: boolean;
  format: PhotoFormat;
  /** Easter egg. */
  chaos: boolean;
  /** Gold frame — only drawn once every live pose has been collected. */
  goldFrame: boolean;
}

export const DEFAULT_SETTINGS: Settings = {
  filter: 'original',
  vignette: true,
  grain: false,
  stickers: true,
  reactions: true,
  landmarks: false,
  debug: false,
  mirror: true,
  quality: 'balanced',
  sound: false,
  music: false,
  volume: 0.6,
  motion: 'system',
  photoTimer: true,
  format: 'png',
  chaos: false,
  goldFrame: true,
};

/** Settings that "Reset effects" restores (camera/audio choices are left alone). */
export const EFFECT_KEYS = ['filter', 'vignette', 'grain', 'stickers', 'reactions', 'landmarks', 'debug', 'chaos'] as const;

const oneOf = <T extends string>(list: readonly T[], v: unknown, fallback: T): T =>
  typeof v === 'string' && (list as readonly string[]).includes(v) ? (v as T) : fallback;
const bool = (v: unknown, fallback: boolean) => (typeof v === 'boolean' ? v : fallback);

export function sanitizeSettings(raw: unknown): Settings {
  const r = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>;
  const d = DEFAULT_SETTINGS;
  const volume = typeof r.volume === 'number' && Number.isFinite(r.volume) ? Math.min(1, Math.max(0, r.volume)) : d.volume;
  return {
    filter: oneOf(FILTERS, r.filter, d.filter),
    vignette: bool(r.vignette, d.vignette),
    grain: bool(r.grain, d.grain),
    stickers: bool(r.stickers, d.stickers),
    reactions: bool(r.reactions, d.reactions),
    landmarks: bool(r.landmarks, d.landmarks),
    debug: bool(r.debug, d.debug),
    mirror: bool(r.mirror, d.mirror),
    quality: oneOf(QUALITIES, r.quality, d.quality),
    // audio always starts off: browsers need a gesture first, and surprise sound is rude
    sound: false,
    music: false,
    volume,
    motion: oneOf(MOTION_PREFS, r.motion, d.motion),
    photoTimer: bool(r.photoTimer, d.photoTimer),
    format: oneOf(PHOTO_FORMATS, r.format, d.format),
    chaos: bool(r.chaos, d.chaos),
    goldFrame: bool(r.goldFrame, d.goldFrame),
  };
}

export function resetEffects(s: Settings): Settings {
  const next = { ...s };
  for (const k of EFFECT_KEYS) (next as Record<string, unknown>)[k] = DEFAULT_SETTINGS[k];
  return next;
}

const KEY = 'posecam.settings.v1';
export const settingsStore = createStore<Settings>(sanitizeSettings(readJson(KEY)));
settingsStore.subscribe(() => safeStorage.set(KEY, JSON.stringify(settingsStore.get())));

export function updateSettings(patch: Partial<Settings>): void {
  settingsStore.set((prev) => ({ ...prev, ...patch }));
}

/** Whether animations should be reduced, combining the user's choice with the OS preference. */
export function prefersReducedMotion(s: Settings): boolean {
  if (s.motion === 'reduced') return true;
  if (s.motion === 'full') return false;
  try {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  } catch {
    return false;
  }
}
