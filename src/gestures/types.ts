/** A single landmark. Image landmarks are normalised to [0,1]; world landmarks are metres. */
export interface Point3 {
  x: number;
  y: number;
  z: number;
}

export type Handedness = 'Left' | 'Right' | 'Unknown';

/** One detected hand, in the raw (un-mirrored) camera frame. */
export interface HandObservation {
  /** 21 normalised image landmarks (x by frame width, y by frame height). */
  image: Point3[];
  /** 21 metric landmarks (camera-aligned axes, y points down). */
  world: Point3[];
  handedness: Handedness;
  /** Detector handedness confidence, 0..1. */
  score: number;
}

export interface FrameObservation {
  hands: HandObservation[];
  /** Frame width / height. Needed because normalised x and y have different units. */
  aspect: number;
}

export const LM = {
  WRIST: 0,
  THUMB_CMC: 1,
  THUMB_MCP: 2,
  THUMB_IP: 3,
  THUMB_TIP: 4,
  INDEX_MCP: 5,
  INDEX_PIP: 6,
  INDEX_DIP: 7,
  INDEX_TIP: 8,
  MIDDLE_MCP: 9,
  MIDDLE_PIP: 10,
  MIDDLE_DIP: 11,
  MIDDLE_TIP: 12,
  RING_MCP: 13,
  RING_PIP: 14,
  RING_DIP: 15,
  RING_TIP: 16,
  PINKY_MCP: 17,
  PINKY_PIP: 18,
  PINKY_DIP: 19,
  PINKY_TIP: 20,
} as const;

export const HAND_CONNECTIONS: ReadonlyArray<readonly [number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

export const FINGERS = ['index', 'middle', 'ring', 'pinky'] as const;
export type FingerName = (typeof FINGERS)[number];

/** Joint chains [MCP, PIP, DIP, TIP] for the four fingers. */
export const FINGER_JOINTS: Record<FingerName, readonly [number, number, number, number]> = {
  index: [5, 6, 7, 8],
  middle: [9, 10, 11, 12],
  ring: [13, 14, 15, 16],
  pinky: [17, 18, 19, 20],
};
