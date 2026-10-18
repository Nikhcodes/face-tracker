// Copies the MediaPipe Tasks Vision WASM runtime from the pinned npm package
// into /public so the app serves it from its own origin (no third-party CDN).
// The output folder is versioned, so it can be cached as immutable.
import { cpSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const pkgDir = join(root, 'node_modules', '@mediapipe', 'tasks-vision');
const { version } = JSON.parse(readFileSync(join(pkgDir, 'package.json'), 'utf8'));
const outRoot = join(root, 'public', 'mediapipe');
const outDir = join(outRoot, version);

// Only the files the app can actually request:
// - module build (used inside the module Web Worker)
// - classic SIMD + no-SIMD builds (used by the main-thread fallback)
const wanted = [
  'vision_wasm_module_internal.js',
  'vision_wasm_module_internal.wasm',
  'vision_wasm_internal.js',
  'vision_wasm_internal.wasm',
  'vision_wasm_nosimd_internal.js',
  'vision_wasm_nosimd_internal.wasm',
];

if (existsSync(outRoot)) {
  for (const entry of readdirSync(outRoot)) {
    if (entry !== version) rmSync(join(outRoot, entry), { recursive: true, force: true });
  }
}
mkdirSync(outDir, { recursive: true });
for (const file of wanted) {
  const src = join(pkgDir, 'wasm', file);
  if (!existsSync(src)) throw new Error(`MediaPipe file missing from package: ${file}`);
  cpSync(src, join(outDir, file));
}
console.log(`[pose-cam] MediaPipe ${version} WASM copied to public/mediapipe/${version}`);
