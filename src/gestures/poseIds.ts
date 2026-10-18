/** Every pose the classifier can output. Display metadata lives in src/config/poses.ts. */
export const GESTURE_POSES = [
  'hello',
  'peace',
  'point',
  'thumbsup',
  'thumbsdown',
  'rockon',
  'three',
  'shaka',
  'fingergun',
  'think',
  'doublethumbs',
  'celebrate',
  'rodrick',
  'heart',
] as const;

export type GesturePoseId = (typeof GESTURE_POSES)[number];
/** Poses created in the pose editor. */
export type CustomPoseId = `custom-${string}`;
/** Anything the classifier can report. */
export type DetectablePoseId = GesturePoseId | CustomPoseId;
/** `idle` is a state (no hands for a while), not a gesture, but it has a reaction too. */
export type PoseId = DetectablePoseId | 'idle';

export const isCustomPoseId = (id: string): id is CustomPoseId => /^custom-[a-z0-9-]{1,40}$/.test(id);

/** Poses whose recognition has not been validated broadly enough to be called reliable. */
export const EXPERIMENTAL_POSES: ReadonlySet<string> = new Set<GesturePoseId>(['think', 'rodrick', 'heart', 'fingergun']);
