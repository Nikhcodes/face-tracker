/**
 * Central pose catalog: names, instructions, stickers, captions and reactions.
 * The classifier decides *what* was detected; this file decides how it looks.
 */
import { EXPERIMENTAL_POSES, type PoseId } from '../gestures/poseIds';
import type { CustomRule } from '../gestures/customRules';
import { SPECS, rollToPointThumb, type GlyphHand } from '../gestures/synthetic';

export type { GlyphHand };

export type StickerRef =
  | { kind: 'image'; src: string; width: number; height: number }
  | { kind: 'sprite'; src: string; frameWidth: number; frameHeight: number; columns: number; frames: number; frameMs: number }
  /** A designed badge: the pose's hand glyph plus a short label, drawn as vector graphics. */
  | { kind: 'stamp'; label: string };

export type ParticleKind = 'confetti' | 'sparkles' | 'stars' | 'hearts' | 'rain' | 'notes';

export interface ReactionSpec {
  particles?: { kind: ParticleKind; count: number; anchor: 'hand' | 'screen' | 'top' };
  /** Small camera shake on entry (disabled with reduced motion). */
  shake?: number;
  spotlight?: boolean;
  letterbox?: boolean;
  /** Target reticle on the index fingertip. */
  reticle?: boolean;
  /** Muzzle flash on the index fingertip. */
  muzzle?: boolean;
  /** Thought bubble above the hand. */
  thought?: boolean;
  /** Comic impact burst behind the caption. */
  impact?: boolean;
  /** Briefly desaturate the frame. */
  desaturate?: boolean;
}

export interface PoseDefinition {
  id: PoseId;
  name: string;
  /** One-line description. */
  blurb: string;
  /** How to perform it. */
  howTo: string;
  status: 'live' | 'experimental' | 'state' | 'custom';
  source: 'builtin' | 'custom';
  /** Finger rule for poses made in the editor. */
  rule?: CustomRule;
  hands: 0 | 1 | 2;
  stickers: StickerRef[];
  captions: string[];
  reaction: ReactionSpec;
  /** Sound cue played on entry when sound is on. */
  sound: 'pop' | 'chime' | 'boom' | 'pew' | 'sad' | 'sparkle' | 'none';
  glyph: GlyphHand[];
  /** Short label for the live status chip. */
  label: string;
}

const img = (name: string, width: number, height: number): StickerRef => ({
  kind: 'image',
  src: `/overlays/${name}.webp`,
  width,
  height,
});

const thumbUpRoll = rollToPointThumb(SPECS.thumb, 'up');
const thumbDownRoll = rollToPointThumb(SPECS.thumb, 'down');
const status = (id: string): 'experimental' | 'live' => (EXPERIMENTAL_POSES.has(id) ? 'experimental' : 'live');

export const BUILTIN_POSES: PoseDefinition[] = (
  [
  {
    id: 'hello',
    name: 'Hello',
    label: 'Hello',
    blurb: 'A friendly wave back from the internet.',
    howTo: 'Show an open hand to the camera, all five fingers out.',
    status: status('hello'),
    hands: 1,
    stickers: [{ kind: 'sprite', src: '/overlays/hello-sprite.webp', frameWidth: 200, frameHeight: 152, columns: 8, frames: 38, frameMs: 70 }],
    captions: ['HELLO THERE.', 'WAVE RECEIVED.', 'HI. YOU LOOK GREAT.'],
    reaction: { particles: { kind: 'sparkles', count: 14, anchor: 'hand' } },
    sound: 'pop',
    glyph: [{ spec: SPECS.open }],
  },
  {
    id: 'peace',
    name: 'Peace',
    label: 'Peace',
    blurb: 'The universal photo-ready pose.',
    howTo: 'Index and middle finger up in a V. Ring and pinky folded down.',
    status: status('peace'),
    hands: 1,
    stickers: [img('peace', 720, 717)],
    captions: ['PLEASE REMAIN ICONIC.', 'PEACE AND LOVE, ONLY.'],
    reaction: { particles: { kind: 'sparkles', count: 22, anchor: 'hand' } },
    sound: 'sparkle',
    glyph: [{ spec: SPECS.peace }],
  },
  {
    id: 'point',
    name: 'Point',
    label: 'Point',
    blurb: 'Locks a target on your fingertip. Occasionally, a correction.',
    howTo: 'Index finger out, other fingers curled, thumb tucked in. Works in any direction.',
    status: status('point'),
    hands: 1,
    stickers: [img('point', 720, 717), img('point-alt', 260, 194)],
    captions: ['MAIN CHARACTER DETECTED.', 'ERM, ACTUALLY—', 'NOTED.'],
    reaction: { reticle: true },
    sound: 'pop',
    glyph: [{ spec: SPECS.point, roll: -20 }],
  },
  {
    id: 'thumbsup',
    name: 'Thumbs Up',
    label: 'Thumbs up',
    blurb: 'Official approval from a cat.',
    howTo: 'Make a fist and point your thumb straight up.',
    status: status('thumbsup'),
    hands: 1,
    stickers: [img('thumbsup', 220, 203)],
    captions: ['THE COUNCIL APPROVES.', 'CERTIFIED GOOD.'],
    reaction: { particles: { kind: 'stars', count: 12, anchor: 'hand' } },
    sound: 'chime',
    glyph: [{ spec: SPECS.thumb, roll: thumbUpRoll }],
  },
  {
    id: 'thumbsdown',
    name: 'Thumbs Down',
    label: 'Thumbs down',
    blurb: 'The council has reviewed your request.',
    howTo: 'Make a fist and point your thumb straight down.',
    status: status('thumbsdown'),
    hands: 1,
    stickers: [{ kind: 'stamp', label: 'DENIED' }],
    captions: ['THE COUNCIL DISAPPROVES.', 'HARD PASS.'],
    reaction: { particles: { kind: 'rain', count: 40, anchor: 'top' }, desaturate: true },
    sound: 'sad',
    glyph: [{ spec: SPECS.thumb, roll: thumbDownRoll }],
  },
  {
    id: 'rockon',
    name: 'Rock On',
    label: 'Rock on',
    blurb: 'Raccoon-certified rock-star energy.',
    howTo: 'Index finger and pinky up, middle and ring folded. Thumb can do whatever it likes.',
    status: status('rockon'),
    hands: 1,
    stickers: [img('rockon', 612, 406)],
    captions: ['TOO MUCH AURA.', 'ROCK AND ROLL, BABY.'],
    reaction: { particles: { kind: 'notes', count: 12, anchor: 'hand' }, shake: 6, impact: true },
    sound: 'boom',
    glyph: [{ spec: SPECS.rockon }],
  },
  {
    id: 'three',
    name: 'Three — Photo Timer',
    label: 'Photo timer',
    blurb: 'Hold up three fingers to start a 3-second photo countdown.',
    howTo: 'Index, middle and ring finger up; pinky folded. Hold it for a moment.',
    status: status('three'),
    hands: 1,
    stickers: [{ kind: 'stamp', label: '3·2·1' }],
    captions: ['SAY CHEESE.'],
    reaction: {},
    sound: 'none',
    glyph: [{ spec: SPECS.three }],
  },
  {
    id: 'shaka',
    name: 'Hang Loose',
    label: 'Hang loose',
    blurb: 'Thumb and pinky out. Instantly on holiday.',
    howTo: 'Fold your three middle fingers, stretch out your thumb and pinky.',
    status: status('shaka'),
    hands: 1,
    stickers: [{ kind: 'stamp', label: 'CHILL' }],
    captions: ['VIBES: IMMACULATE.', 'HANG LOOSE.'],
    reaction: { particles: { kind: 'sparkles', count: 16, anchor: 'hand' } },
    sound: 'sparkle',
    glyph: [{ spec: SPECS.shaka }],
  },
  {
    id: 'fingergun',
    name: 'Finger Gun',
    label: 'Finger gun',
    blurb: 'Pew. Comes with widescreen bars.',
    howTo: 'Index finger forward, thumb up like a hammer, other fingers curled.',
    status: status('fingergun'),
    hands: 1,
    stickers: [{ kind: 'stamp', label: 'PEW!' }],
    captions: ['ABSOLUTE CINEMA.', 'PEW PEW.'],
    reaction: { muzzle: true, letterbox: true },
    sound: 'pew',
    glyph: [{ spec: SPECS.gun, roll: 70 }],
  },
  {
    id: 'think',
    name: 'Thinking',
    label: 'Thinking',
    blurb: 'Deep in thought. Possibly about snacks.',
    howTo: 'Rest a fist under your chin with one hand, roughly in the middle of the frame.',
    status: status('think'),
    hands: 1,
    stickers: [img('think', 226, 223)],
    captions: ['BRO IS LOCKED IN.', 'HMM. YES. INDEED.'],
    reaction: { thought: true },
    sound: 'chime',
    glyph: [{ spec: SPECS.fist }],
  },
  {
    id: 'doublethumbs',
    name: 'Double Thumbs',
    label: 'Max approval',
    blurb: 'Two thumbs up. Maximum approval unlocked.',
    howTo: 'Both hands in a thumbs up, side by side.',
    status: status('doublethumbs'),
    hands: 2,
    stickers: [img('thumbsup', 220, 203)],
    captions: ['CERTIFIED LEGEND.', 'MAXIMUM APPROVAL.'],
    reaction: { particles: { kind: 'stars', count: 30, anchor: 'screen' }, spotlight: true },
    sound: 'chime',
    glyph: [
      { spec: SPECS.thumb, roll: thumbUpRoll, dx: -0.11 },
      { spec: SPECS.thumb, roll: -thumbUpRoll, mirror: true, dx: 0.11 },
    ],
  },
  {
    id: 'celebrate',
    name: 'Celebrate',
    label: 'Celebrate',
    blurb: 'Both hands open. Confetti happens.',
    howTo: 'Raise both hands with all fingers open.',
    status: status('celebrate'),
    hands: 2,
    stickers: [{ kind: 'stamp', label: 'YAY' }],
    captions: ['THE CAMERA HAS SPOKEN.', 'WE ARE SO BACK.'],
    reaction: { particles: { kind: 'confetti', count: 90, anchor: 'top' }, shake: 4 },
    sound: 'boom',
    glyph: [
      { spec: SPECS.open, roll: -14, dx: -0.12 },
      { spec: SPECS.open, roll: 14, mirror: true, dx: 0.12 },
    ],
  },
  {
    id: 'rodrick',
    name: 'Rodrick',
    label: 'RODRICK!!',
    blurb: 'Two fists raised. You know exactly what this is.',
    howTo: 'Raise both fists up to shoulder height or higher.',
    status: status('rodrick'),
    hands: 2,
    stickers: [img('rodrick', 360, 360)],
    captions: ['RODRICK!!'],
    reaction: { impact: true, shake: 10, spotlight: true, particles: { kind: 'stars', count: 18, anchor: 'screen' } },
    sound: 'boom',
    glyph: [
      { spec: SPECS.fist, dx: -0.1 },
      { spec: SPECS.fist, mirror: true, dx: 0.1 },
    ],
  },
  {
    id: 'heart',
    name: 'Heart Hands',
    label: 'Heart',
    blurb: 'Two hands, one heart. Wholesome mode.',
    howTo: 'Curve both hands so your index fingertips touch at the top and your thumb tips touch below.',
    status: status('heart'),
    hands: 2,
    stickers: [{ kind: 'stamp', label: 'LOVE' }],
    captions: ['WHOLESOME CONTENT DETECTED.', 'LOVE THAT FOR YOU.'],
    reaction: { particles: { kind: 'hearts', count: 24, anchor: 'hand' } },
    sound: 'sparkle',
    glyph: [
      { spec: SPECS.heartHalf, roll: -40, dx: -0.05 },
      { spec: SPECS.heartHalf, roll: 40, mirror: true, dx: 0.05 },
    ],
  },
  {
    id: 'idle',
    name: 'Idle',
    label: 'Idle',
    blurb: 'Hands out of frame for a moment? The cat waits with you.',
    howTo: 'Lower your hands for about 1.5 seconds.',
    status: 'state',
    hands: 0,
    stickers: [img('idle', 720, 720)],
    captions: [],
    reaction: {},
    sound: 'none',
    glyph: [],
  },
  ] satisfies Omit<PoseDefinition, 'source'>[]
).map((p) => ({ ...p, source: 'builtin' as const }));

/** Built-in poses by id (before any editor overrides). */
export const BUILTIN_BY_ID: Record<string, PoseDefinition> = Object.fromEntries(BUILTIN_POSES.map((p) => [p.id, p]));

export const HELD_MESSAGES = ['POSE ACQUIRED.', 'LOCKED IN.', 'HOLD THAT.'];
export const CHAOS_CAPTIONS = [
  'THE CAMERA HAS SPOKEN.',
  'ABSOLUTE CINEMA.',
  'TOO MUCH AURA.',
  'PLEASE REMAIN ICONIC.',
  'MAIN CHARACTER DETECTED.',
  'THE COUNCIL APPROVES.',
  'RODRICK!!',
];
