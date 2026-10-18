/**
 * The live pose registry: built-in poses + editor overrides + custom poses.
 *
 * Source of truth, in priority order:
 *   1. the local pack edited in this browser (IndexedDB)
 *   2. the site pack shipped with the deployment (/packs/default.json)
 *   3. the built-in catalog (config/poses.ts)
 */
import type { CustomRule } from '../gestures/customRules';
import { computeFeatures } from '../gestures/features';
import { EXPERIMENTAL_POSES, type CustomPoseId, type PoseId } from '../gestures/poseIds';
import { F_EXT, F_FOLD, F_HALF, T_EXT, T_REST, T_TUCK, rollToPointThumb, synthHand, type GlyphHand, type HandSpec } from '../gestures/synthetic';
import { matchRule } from '../gestures/customRules';
import { idb } from '../lib/idb';
import { createStore } from '../lib/store';
import { emptyPack, sanitizePack, type CustomPoseData, type PoseOverride, type PosePack, type StickerImage } from './pack';
import { BUILTIN_BY_ID, BUILTIN_POSES, type ParticleKind, type PoseDefinition, type ReactionSpec, type StickerRef } from './poses';

// ── Building definitions ───────────────────────────────────────────────────

const PARTICLE_COUNT: Record<ParticleKind, number> = { sparkles: 18, stars: 16, confetti: 90, hearts: 22, notes: 12, rain: 40 };
const PARTICLE_ANCHOR: Record<ParticleKind, 'hand' | 'top'> = {
  sparkles: 'hand',
  stars: 'hand',
  confetti: 'top',
  hearts: 'hand',
  notes: 'hand',
  rain: 'top',
};

const toSticker = (s: StickerImage): StickerRef => ({ kind: 'image', src: s.src, width: s.width, height: s.height });

function reactionWith(base: ReactionSpec, o: Pick<PoseOverride, 'particles' | 'shake' | 'spotlight'>): ReactionSpec {
  const r: ReactionSpec = { ...base };
  if (o.particles === 'none') delete r.particles;
  else if (o.particles) r.particles = { kind: o.particles, count: PARTICLE_COUNT[o.particles], anchor: PARTICLE_ANCHOR[o.particles] };
  if (o.shake === true) r.shake = r.shake ?? 6;
  if (o.shake === false) delete r.shake;
  if (o.spotlight !== undefined) r.spotlight = o.spotlight;
  return r;
}

export function applyOverride(base: PoseDefinition, o: PoseOverride | undefined): PoseDefinition {
  if (!o) return base;
  return {
    ...base,
    name: o.name ?? base.name,
    label: o.name ?? base.label,
    blurb: o.blurb ?? base.blurb,
    howTo: o.howTo ?? base.howTo,
    captions: o.captions ?? base.captions,
    stickers: [...(o.keepOriginal === false ? [] : base.stickers), ...(o.stickers ?? []).map(toSticker)],
    sound: o.sound ?? base.sound,
    reaction: reactionWith(base.reaction, o),
  };
}

const DIGIT_POSE = { up: F_EXT, down: F_FOLD, any: F_HALF } as const;
const THUMB_POSE = { up: T_EXT, down: T_TUCK, any: T_REST } as const;

/** Illustration for a custom rule, drawn with the same hand model as the built-ins. */
export function ruleToGlyph(rule: CustomRule): GlyphHand[] {
  const spec: HandSpec = {
    index: { ...DIGIT_POSE[rule.index], spread: rule.index === 'up' ? 8 : 0 },
    middle: DIGIT_POSE[rule.middle],
    ring: { ...DIGIT_POSE[rule.ring], spread: rule.ring === 'up' ? -6 : 0 },
    pinky: { ...DIGIT_POSE[rule.pinky], spread: rule.pinky === 'up' ? -14 : 0 },
    thumb: THUMB_POSE[rule.thumb],
  };
  const roll = rule.thumbDirection === 'any' ? 0 : rollToPointThumb(spec, rule.thumbDirection);
  if (rule.hands === 2) {
    return [
      { spec, roll, dx: -0.11 },
      { spec, roll: -roll, mirror: true, dx: 0.11 },
    ];
  }
  return [{ spec, roll }];
}

export function customToDefinition(c: CustomPoseData): PoseDefinition {
  const stamp = c.name.split(/\s+/)[0].slice(0, 7).toUpperCase() || 'POSE';
  return {
    id: c.id,
    name: c.name,
    label: c.name,
    blurb: c.blurb || 'A custom pose made in the pose editor.',
    howTo: c.howTo || describeRule(c.rule),
    status: 'custom',
    source: 'custom',
    hands: c.rule.hands,
    stickers: c.stickers.length ? c.stickers.map(toSticker) : [{ kind: 'stamp', label: stamp }],
    captions: c.captions,
    reaction: reactionWith({}, c),
    sound: c.sound,
    glyph: ruleToGlyph(c.rule),
    rule: c.rule,
  };
}

const WORD = { up: 'up', down: 'folded', any: 'any' } as const;
export function describeRule(rule: CustomRule): string {
  const parts = (['thumb', 'index', 'middle', 'ring', 'pinky'] as const)
    .filter((d) => rule[d] !== 'any')
    .map((d) => `${d} ${WORD[rule[d]]}`);
  let s = parts.join(', ');
  s = s.charAt(0).toUpperCase() + s.slice(1);
  if (rule.thumbDirection !== 'any') s += `, thumb pointing ${rule.thumbDirection}`;
  if (rule.zone !== 'any') s += `, hand in the ${rule.zone} half`;
  if (rule.hands === 2) s = `Both hands: ${s.charAt(0).toLowerCase()}${s.slice(1)}`;
  return `${s}.`;
}

export interface Registry {
  /** Enabled poses in display order (built-ins, then custom, idle last). */
  poses: PoseDefinition[];
  /** Every pose, including switched-off ones (for the editor). */
  all: PoseDefinition[];
  byId: Map<string, PoseDefinition>;
  disabled: ReadonlySet<string>;
  custom: ReadonlyArray<{ id: CustomPoseId; rule: CustomRule }>;
}

export function buildRegistry(pack: PosePack): Registry {
  const disabled = new Set<string>();
  const builtins = BUILTIN_POSES.map((p) => {
    const o = pack.overrides[p.id as keyof PosePack['overrides']];
    if (o?.disabled && p.id !== 'idle') disabled.add(p.id);
    return applyOverride(p, o);
  });
  const customs = pack.custom.map((c) => {
    if (c.disabled) disabled.add(c.id);
    return customToDefinition(c);
  });
  const idle = builtins.filter((p) => p.id === 'idle');
  const all = [...builtins.filter((p) => p.id !== 'idle'), ...customs, ...idle];
  return {
    all,
    poses: all.filter((p) => !disabled.has(p.id)),
    byId: new Map(all.map((p) => [p.id, p])),
    disabled,
    custom: pack.custom.filter((c) => !c.disabled).map((c) => ({ id: c.id, rule: c.rule })),
  };
}

/** Which other poses a rule would also catch (shown as a warning in the editor). */
export function findConflicts(rule: CustomRule, registry: Registry, selfId?: string): PoseDefinition[] {
  const out: PoseDefinition[] = [];
  for (const p of registry.all) {
    if (p.id === selfId || p.id === 'idle' || registry.disabled.has(p.id) || p.hands !== rule.hands || !p.glyph.length) continue;
    // test the other pose's canonical hand shape against this rule
    const hands = p.glyph.map((g) =>
      computeFeatures(
        synthHand(g.spec, { roll: g.roll ?? 0, handedness: g.mirror ? 'Left' : 'Right', cy: rule.zone === 'lower' ? 0.7 : 0.35 }),
        16 / 9,
      ),
    );
    if (matchRule(rule, hands) !== null) out.push(p);
  }
  return out;
}

// ── State and persistence ──────────────────────────────────────────────────

interface PackState {
  site: PosePack | null;
  local: PosePack | null;
  ready: boolean;
  storage: 'idb' | 'memory';
  save: 'idle' | 'saving' | 'saved' | 'error';
}

export const packState = createStore<PackState>({ site: null, local: null, ready: false, storage: 'idb', save: 'idle' });

export function effectivePack(s: PackState = packState.get()): PosePack {
  return s.local ?? s.site ?? emptyPack();
}

export const registryStore = createStore<Registry>(buildRegistry(emptyPack()));
let lastPack: PosePack | null = null;
packState.subscribe(() => {
  const pack = effectivePack();
  if (pack !== lastPack) {
    lastPack = pack;
    registryStore.set(buildRegistry(pack));
  }
});

export function getPose(id: PoseId | string): PoseDefinition {
  return registryStore.get().byId.get(id) ?? BUILTIN_BY_ID[id] ?? BUILTIN_BY_ID.idle;
}

/** Live, enabled built-in poses (collecting them all unlocks the gold frame). */
export function livePoseIds(reg: Registry = registryStore.get()): PoseId[] {
  return reg.poses.filter((p) => p.status === 'live').map((p) => p.id);
}

export const isExperimental = (id: string) => EXPERIMENTAL_POSES.has(id);

const LOCAL_KEY = 'pack';
let initPromise: Promise<void> | null = null;

/** Loads the site pack and this browser's edits. Safe to call more than once. */
export function initPacks(fetchImpl: typeof fetch = fetch): Promise<void> {
  if (initPromise) return initPromise;
  initPromise = (async () => {
    let site: PosePack | null = null;
    try {
      const res = await fetchImpl('/packs/default.json', { cache: 'no-cache' });
      if (res.ok && (res.headers.get('content-type') ?? '').includes('json')) site = sanitizePack(await res.json()).pack;
    } catch {
      /* no site pack — fine */
    }
    let local: PosePack | null = null;
    let storage: PackState['storage'] = 'idb';
    try {
      const raw = await idb.get(LOCAL_KEY);
      if (raw) local = sanitizePack(raw).pack;
    } catch {
      storage = 'memory';
    }
    packState.set({ site, local, ready: true, storage, save: 'idle' });
  })();
  return initPromise;
}

let saveTimer: ReturnType<typeof setTimeout> | null = null;
let saveWaiters: Array<(ok: boolean) => void> = [];

/**
 * Applies an edited pack immediately (the app updates live) and writes it to
 * IndexedDB shortly after, so rapid typing doesn't rewrite large images each keystroke.
 * Resolves false if the browser blocks storage (edits then last until reload).
 */
export function saveLocalPack(pack: PosePack, delayMs = 350): Promise<boolean> {
  const stamped = { ...pack, updatedAt: new Date().toISOString() };
  packState.set((s) => ({ ...s, local: stamped, save: 'saving' }));
  if (saveTimer) clearTimeout(saveTimer);
  return new Promise((resolve) => {
    saveWaiters.push(resolve);
    saveTimer = setTimeout(async () => {
      saveTimer = null;
      const waiters = saveWaiters;
      saveWaiters = [];
      const latest = packState.get().local;
      let ok = true;
      try {
        if (latest) await idb.set(LOCAL_KEY, latest);
      } catch {
        ok = false;
      }
      packState.set((s) => ({ ...s, storage: ok ? 'idb' : 'memory', save: ok ? 'saved' : 'error' }));
      waiters.forEach((w) => w(ok));
    }, delayMs);
  });
}

/** Drops this browser's edits and goes back to the site pack (or built-ins). */
export async function resetLocalPack(): Promise<void> {
  if (saveTimer) clearTimeout(saveTimer);
  saveTimer = null;
  packState.set((s) => ({ ...s, local: null, save: 'idle' }));
  try {
    await idb.del(LOCAL_KEY);
  } catch {
    /* nothing stored */
  }
}

export function packToJson(pack: PosePack): string {
  return JSON.stringify({ ...pack, updatedAt: new Date().toISOString() }, null, 2);
}
