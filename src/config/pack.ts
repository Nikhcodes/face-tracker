/**
 * Pose packs: the portable format the pose editor saves, exports and imports.
 *
 * A pack holds overrides for built-in poses (name, pictures, captions,
 * reactions, on/off) and fully custom poses. Everything coming from a file is
 * validated here before it can reach the app.
 */
import { isRule, ruleIssues, type CustomRule } from '../gestures/customRules';
import { GESTURE_POSES, isCustomPoseId, type CustomPoseId, type GesturePoseId } from '../gestures/poseIds';
import type { ParticleKind, PoseDefinition } from './poses';

export const PACK_FORMAT = 'posecam-pack';
export const PACK_VERSION = 1;

export const SOUNDS = ['pop', 'chime', 'boom', 'pew', 'sad', 'sparkle', 'none'] as const;
export type SoundCue = PoseDefinition['sound'];
export const PARTICLES = ['none', 'sparkles', 'stars', 'confetti', 'hearts', 'notes', 'rain'] as const;
export type ParticleChoice = ParticleKind | 'none';

export const LIMITS = {
  name: 40,
  blurb: 140,
  howTo: 200,
  caption: 48,
  captions: 8,
  stickers: 6,
  custom: 40,
  /** Characters in one image data URL (~2.2 MB of image). */
  imageChars: 3_000_000,
} as const;

export interface StickerImage {
  /** data:image/... URL (from the editor) or a site path like /overlays/my.webp */
  src: string;
  width: number;
  height: number;
}

export interface PoseOverride {
  name?: string;
  blurb?: string;
  howTo?: string;
  captions?: string[];
  /** Pictures added in the editor. */
  stickers?: StickerImage[];
  /** Keep the built-in pictures alongside added ones (default true). */
  keepOriginal?: boolean;
  sound?: SoundCue;
  particles?: ParticleChoice;
  shake?: boolean;
  spotlight?: boolean;
  disabled?: boolean;
}

export interface CustomPoseData {
  id: CustomPoseId;
  name: string;
  blurb: string;
  howTo: string;
  rule: CustomRule;
  captions: string[];
  stickers: StickerImage[];
  sound: SoundCue;
  particles: ParticleChoice;
  shake: boolean;
  spotlight: boolean;
  disabled: boolean;
}

export type OverridableId = GesturePoseId | 'idle';

export interface PosePack {
  format: typeof PACK_FORMAT;
  version: typeof PACK_VERSION;
  name: string;
  updatedAt: string;
  overrides: Partial<Record<OverridableId, PoseOverride>>;
  custom: CustomPoseData[];
}

export function emptyPack(name = 'My poses'): PosePack {
  return { format: PACK_FORMAT, version: PACK_VERSION, name, updatedAt: new Date(0).toISOString(), overrides: {}, custom: [] };
}

const DATA_URL = /^data:image\/(png|jpeg|webp|gif);base64,[A-Za-z0-9+/=]+$/;
const SITE_PATH = /^\/[A-Za-z0-9/_.-]+\.(webp|png|jpe?g|gif)$/;
const OVERRIDABLE = new Set<string>([...GESTURE_POSES, 'idle']);

const stripControl = (s: string) => Array.from(s, (ch) => (ch.charCodeAt(0) < 32 || ch.charCodeAt(0) === 127 ? ' ' : ch)).join('');
const str = (v: unknown, max: number): string | undefined =>
  typeof v === 'string' ? stripControl(v).trim().slice(0, max) : undefined;
const oneOf = <T extends string>(list: readonly T[], v: unknown): T | undefined =>
  typeof v === 'string' && (list as readonly string[]).includes(v) ? (v as T) : undefined;
const bool = (v: unknown): boolean | undefined => (typeof v === 'boolean' ? v : undefined);

export function sanitizeImage(v: unknown): StickerImage | null {
  if (!v || typeof v !== 'object') return null;
  const r = v as Record<string, unknown>;
  if (typeof r.src !== 'string') return null;
  const okSrc = (r.src.length <= LIMITS.imageChars && DATA_URL.test(r.src)) || (SITE_PATH.test(r.src) && !r.src.includes('..'));
  const dim = (n: unknown) => (typeof n === 'number' && Number.isFinite(n) && n >= 1 && n <= 8192 ? Math.round(n) : null);
  const width = dim(r.width);
  const height = dim(r.height);
  if (!okSrc || !width || !height) return null;
  return { src: r.src, width, height };
}

function sanitizeCaptions(v: unknown): string[] | undefined {
  if (!Array.isArray(v)) return undefined;
  return v
    .map((c) => str(c, LIMITS.caption))
    .filter((c): c is string => !!c)
    .slice(0, LIMITS.captions);
}

function sanitizeStickers(v: unknown): StickerImage[] | undefined {
  if (!Array.isArray(v)) return undefined;
  return v
    .map(sanitizeImage)
    .filter((x): x is StickerImage => !!x)
    .slice(0, LIMITS.stickers);
}

function sanitizeOverride(v: unknown): PoseOverride | null {
  if (!v || typeof v !== 'object') return null;
  const r = v as Record<string, unknown>;
  const o: PoseOverride = {};
  const name = str(r.name, LIMITS.name);
  if (name) o.name = name;
  const blurb = str(r.blurb, LIMITS.blurb);
  if (blurb !== undefined) o.blurb = blurb;
  const howTo = str(r.howTo, LIMITS.howTo);
  if (howTo !== undefined) o.howTo = howTo;
  const captions = sanitizeCaptions(r.captions);
  if (captions) o.captions = captions;
  const stickers = sanitizeStickers(r.stickers);
  if (stickers) o.stickers = stickers;
  const sound = oneOf(SOUNDS, r.sound);
  if (sound) o.sound = sound;
  const particles = oneOf(PARTICLES, r.particles);
  if (particles) o.particles = particles;
  for (const k of ['shake', 'spotlight', 'disabled', 'keepOriginal'] as const) {
    const b = bool(r[k]);
    if (b !== undefined) o[k] = b;
  }
  return Object.keys(o).length ? o : null;
}

function sanitizeCustom(v: unknown): CustomPoseData | null {
  if (!v || typeof v !== 'object') return null;
  const r = v as Record<string, unknown>;
  if (typeof r.id !== 'string' || !isCustomPoseId(r.id)) return null;
  if (!isRule(r.rule) || ruleIssues(r.rule).length) return null;
  const name = str(r.name, LIMITS.name);
  if (!name) return null;
  return {
    id: r.id,
    name,
    blurb: str(r.blurb, LIMITS.blurb) ?? '',
    howTo: str(r.howTo, LIMITS.howTo) ?? '',
    rule: { ...r.rule },
    captions: sanitizeCaptions(r.captions) ?? [],
    stickers: sanitizeStickers(r.stickers) ?? [],
    sound: oneOf(SOUNDS, r.sound) ?? 'pop',
    particles: oneOf(PARTICLES, r.particles) ?? 'sparkles',
    shake: bool(r.shake) ?? false,
    spotlight: bool(r.spotlight) ?? false,
    disabled: bool(r.disabled) ?? false,
  };
}

export interface SanitizeResult {
  pack: PosePack | null;
  /** Human-readable notes about anything that was dropped. */
  warnings: string[];
}

export function sanitizePack(raw: unknown): SanitizeResult {
  const warnings: string[] = [];
  if (!raw || typeof raw !== 'object') return { pack: null, warnings: ['This file isn’t a POSE CAM pack.'] };
  const r = raw as Record<string, unknown>;
  if (r.format !== PACK_FORMAT) return { pack: null, warnings: ['This file isn’t a POSE CAM pack.'] };
  if (r.version !== PACK_VERSION) return { pack: null, warnings: [`Unsupported pack version (${String(r.version)}).`] };

  const overrides: PosePack['overrides'] = {};
  if (r.overrides && typeof r.overrides === 'object') {
    for (const [id, value] of Object.entries(r.overrides as Record<string, unknown>)) {
      if (!OVERRIDABLE.has(id)) {
        warnings.push(`Skipped unknown pose “${id.slice(0, 30)}”.`);
        continue;
      }
      const o = sanitizeOverride(value);
      if (o) overrides[id as OverridableId] = o;
    }
  }

  const custom: CustomPoseData[] = [];
  const seen = new Set<string>();
  if (Array.isArray(r.custom)) {
    for (const item of r.custom) {
      const c = sanitizeCustom(item);
      if (!c) {
        warnings.push('Skipped a custom pose with missing or invalid data.');
        continue;
      }
      if (seen.has(c.id)) continue;
      seen.add(c.id);
      if (custom.length >= LIMITS.custom) {
        warnings.push(`Only the first ${LIMITS.custom} custom poses were kept.`);
        break;
      }
      custom.push(c);
    }
  }

  return {
    pack: {
      format: PACK_FORMAT,
      version: PACK_VERSION,
      name: str(r.name, 60) || 'Pose pack',
      updatedAt: typeof r.updatedAt === 'string' && !Number.isNaN(Date.parse(r.updatedAt)) ? r.updatedAt : new Date().toISOString(),
      overrides,
      custom,
    },
    warnings,
  };
}

export function newCustomId(): CustomPoseId {
  const rand = Math.random().toString(36).slice(2, 8);
  return `custom-${Date.now().toString(36)}-${rand}`;
}
