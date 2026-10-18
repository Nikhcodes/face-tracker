/**
 * MediaPipe assets are served from this site's own origin:
 *  - WASM runtime: copied from the exact npm version pinned in package-lock.json
 *  - Model: Google's hand_landmarker.task (float16), committed in /public/models
 * No camera data, and no request for these files, goes to a third party.
 */
export const MEDIAPIPE_VERSION = __MEDIAPIPE_VERSION__;

export function assetUrls(origin: string = self.location.origin) {
  return {
    wasmBase: `${origin}/mediapipe/${MEDIAPIPE_VERSION}`,
    modelUrl: `${origin}/models/hand_landmarker.task`,
  };
}
