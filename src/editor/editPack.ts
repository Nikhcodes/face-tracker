/** Pure helpers the pose editor uses to change a pack (easy to test, no side effects). */
import { DEFAULT_RULE, type CustomRule } from '../gestures/customRules';
import { isCustomPoseId, type CustomPoseId } from '../gestures/poseIds';
import {
  newCustomId,
  type CustomPoseData,
  type OverridableId,
  type ParticleChoice,
  type PoseOverride,
  type PosePack,
  type SoundCue,
  type StickerImage,
} from '../config/pack';
import { BUILTIN_BY_ID } from '../config/poses';

/** Everything the editor form shows for one pose, merged with defaults. */
export interface PoseDraft {
  id: string;
  kind: 'builtin' | 'custom';
  name: string;
  blurb: string;
  howTo: string;
  captions: string[];
  /** Pictures added in the editor. */
  stickers: StickerImage[];
  keepOriginal: boolean;
  sound: SoundCue;
  particles: ParticleChoice;
  shake: boolean;
  spotlight: boolean;
  enabled: boolean;
  rule?: CustomRule;
  /** True when a built-in pose differs from its defaults. */
  edited: boolean;
}

export function draftFor(pack: PosePack, id: string): PoseDraft | null {
  if (isCustomPoseId(id)) {
    const c = pack.custom.find((x) => x.id === id);
    if (!c) return null;
    const { disabled, ...rest } = c;
    return { ...rest, kind: 'custom', keepOriginal: false, enabled: !disabled, edited: true };
  }
  const base = BUILTIN_BY_ID[id];
  if (!base) return null;
  const o = pack.overrides[id as OverridableId] ?? {};
  return {
    id,
    kind: 'builtin',
    name: o.name ?? base.name,
    blurb: o.blurb ?? base.blurb,
    howTo: o.howTo ?? base.howTo,
    captions: o.captions ?? base.captions,
    stickers: o.stickers ?? [],
    keepOriginal: o.keepOriginal !== false,
    sound: o.sound ?? base.sound,
    particles: o.particles ?? base.reaction.particles?.kind ?? 'none',
    shake: o.shake ?? !!base.reaction.shake,
    spotlight: o.spotlight ?? !!base.reaction.spotlight,
    enabled: !o.disabled,
    edited: Object.keys(o).length > 0,
  };
}

export type DraftPatch = Partial<Omit<PoseDraft, 'id' | 'kind' | 'edited'>>;

/** Applies a form change to a pose, returning a new pack. */
export function applyPatch(pack: PosePack, id: string, patch: DraftPatch): PosePack {
  if (isCustomPoseId(id)) {
    const { enabled, keepOriginal: _keep, ...rest } = patch;
    return {
      ...pack,
      custom: pack.custom.map((c) => (c.id === id ? { ...c, ...rest, ...(enabled !== undefined ? { disabled: !enabled } : {}) } : c)),
    };
  }
  const base = BUILTIN_BY_ID[id];
  if (!base) return pack;
  const key = id as OverridableId;
  const next: PoseOverride = { ...(pack.overrides[key] ?? {}) };
  const { enabled, rule: _rule, ...rest } = patch;
  Object.assign(next, rest);
  if (enabled !== undefined) next.disabled = !enabled;
  // drop values that equal the built-in defaults so "edited" stays meaningful
  if (next.name === base.name) delete next.name;
  if (next.blurb === base.blurb) delete next.blurb;
  if (next.howTo === base.howTo) delete next.howTo;
  if (next.captions && next.captions.join('\n') === base.captions.join('\n')) delete next.captions;
  if (next.stickers && next.stickers.length === 0) delete next.stickers;
  if (next.keepOriginal === true) delete next.keepOriginal;
  if (next.sound === base.sound) delete next.sound;
  if (next.particles === (base.reaction.particles?.kind ?? 'none')) delete next.particles;
  if (next.shake === !!base.reaction.shake) delete next.shake;
  if (next.spotlight === !!base.reaction.spotlight) delete next.spotlight;
  if (next.disabled === false) delete next.disabled;
  const overrides = { ...pack.overrides };
  if (Object.keys(next).length) overrides[key] = next;
  else delete overrides[key];
  return { ...pack, overrides };
}

export function resetBuiltin(pack: PosePack, id: string): PosePack {
  const overrides = { ...pack.overrides };
  delete overrides[id as OverridableId];
  return { ...pack, overrides };
}

export function addCustomPose(pack: PosePack, rule: CustomRule = DEFAULT_RULE): { pack: PosePack; id: CustomPoseId } {
  const id = newCustomId();
  const n = pack.custom.length + 1;
  const pose: CustomPoseData = {
    id,
    name: n === 1 ? 'My pose' : `My pose ${n}`,
    blurb: '',
    howTo: '',
    rule: { ...rule },
    captions: ['NEW POSE UNLOCKED.'],
    stickers: [],
    sound: 'pop',
    particles: 'sparkles',
    shake: false,
    spotlight: false,
    disabled: false,
  };
  return { pack: { ...pack, custom: [...pack.custom, pose] }, id };
}

export function deleteCustomPose(pack: PosePack, id: string): PosePack {
  return { ...pack, custom: pack.custom.filter((c) => c.id !== id) };
}

export function duplicateCustomPose(pack: PosePack, id: string): { pack: PosePack; id: CustomPoseId } | null {
  const src = pack.custom.find((c) => c.id === id);
  if (!src) return null;
  const copy: CustomPoseData = { ...src, id: newCustomId(), name: `${src.name} copy`.slice(0, 40), rule: { ...src.rule } };
  return { pack: { ...pack, custom: [...pack.custom, copy] }, id: copy.id };
}

/** Bytes of embedded image data (helps warn before a pack gets huge). */
export function packImageBytes(pack: PosePack): number {
  let chars = 0;
  const add = (list?: StickerImage[]) => list?.forEach((s) => s.src.startsWith('data:') && (chars += s.src.length));
  Object.values(pack.overrides).forEach((o) => add(o?.stickers));
  pack.custom.forEach((c) => add(c.stickers));
  return Math.round(chars * 0.75);
}
