import { useEffect } from 'react';
import { SiteLayout } from '../components/SiteLayout';
import { HOSTING_NOTE, PRIVACY_STATEMENT } from '../config/copy';
import { registryStore } from '../config/registry';
import { useStore } from '../lib/useStore';
import { Link } from '../lib/router';

const SECTIONS = [
  ['camera', 'Camera permissions'],
  ['poses', 'Poses'],
  ['performance', 'Performance'],
  ['editor', 'Pose editor'],
  ['effects', 'Effects'],
  ['sound', 'Sound'],
  ['privacy', 'Privacy'],
  ['troubleshooting', 'Troubleshooting'],
  ['keys', 'Keyboard shortcuts'],
] as const;

export function Help() {
  useEffect(() => {
    document.title = 'Help — POSE CAM';
    const id = window.location.hash.slice(1);
    if (id) requestAnimationFrame(() => document.getElementById(id)?.scrollIntoView());
  }, []);
  const { poses } = useStore(registryStore);
  const live = poses.filter((p) => p.status === 'live').map((p) => p.name);
  const experimental = poses.filter((p) => p.status === 'experimental').map((p) => p.name);

  return (
    <SiteLayout>
      <header className="page-head">
        <h1>Help</h1>
        <p>Short answers for getting the camera working and the poses landing.</p>
      </header>
      <div className="help">
        <nav className="help__toc" aria-label="Help sections">
          {SECTIONS.map(([id, label]) => (
            <a key={id} href={`#${id}`}>
              {label}
            </a>
          ))}
        </nav>
        <div>
          <section id="camera" aria-labelledby="h-camera">
            <h2 id="h-camera">Camera permissions</h2>
            <div className="prose">
              <p>
                POSE CAM only asks for your camera when you press <strong>Turn on camera</strong> or <strong>Enter the camera</strong>.
                Your browser then shows its own permission prompt.
              </p>
              <ul>
                <li>
                  <strong>Blocked by mistake?</strong> Click the camera or lock icon in the address bar, allow the camera for this site,
                  then press Try again.
                </li>
                <li>
                  <strong>On iPhone or iPad:</strong> Settings → Safari → Camera → Allow (or Ask). Use Safari or another browser
                  updated to a recent iOS.
                </li>
                <li>
                  <strong>Camera busy?</strong> Close video calls and other apps or tabs using it.
                </li>
                <li>
                  Cameras only work on secure pages (<strong>https://</strong>).
                </li>
              </ul>
            </div>
          </section>

          <section id="poses" aria-labelledby="h-poses">
            <h2 id="h-poses">Poses</h2>
            <div className="prose">
              <p>
                <strong>Live:</strong> {live.join(', ')}.
              </p>
              <p>
                <strong>Experimental:</strong> {experimental.join(', ')}. These depend more on angle and framing, so they need to be
                held a little longer before they show.
              </p>
              <p>
                Finger poses are read from the shape of your hand in 3D, so they work upright, sideways or tilted. Thumbs up and down
                need the thumb pointing roughly vertical. See <Link to="/poses">all poses</Link> for how to do each one.
              </p>
              <ul>
                <li>Keep your whole hand in the frame, fingers clearly apart.</li>
                <li>Hold the pose for a moment; the camera waits briefly so reactions don’t flicker.</li>
                <li>Good, even light helps more than anything else.</li>
              </ul>
            </div>
          </section>

          <section id="performance" aria-labelledby="h-performance">
            <h2 id="h-performance">Performance</h2>
            <div className="prose">
              <p>
                Hand tracking runs on your device, using your graphics chip when the browser allows it. If the preview stutters, open{' '}
                <strong>Effects → Performance</strong> and choose <strong>Smooth</strong>. Turning off film grain also helps on older
                phones. Detection pauses automatically when the tab is hidden.
              </p>
            </div>
          </section>

          <section id="editor" aria-labelledby="h-editor">
            <h2 id="h-editor">Pose editor</h2>
            <div className="prose">
              <p>
                The <Link to="/admin">pose editor</Link> lets you create poses from a finger pattern (or copy your hand from the camera),
                add your own pictures to any pose, rewrite captions, pick effects and sounds, and switch poses off.
              </p>
              <ul>
                <li>Changes are saved in this browser and show up in the camera straight away.</li>
                <li>
                  <strong>Export pack</strong> downloads everything as one file. Import it in another browser, or publish it for every
                  visitor by adding it to the site as <code>public/packs/default.json</code>.
                </li>
                <li>Custom poses are checked before built-in ones, so the editor warns you when a shape overlaps an existing pose.</li>
              </ul>
            </div>
          </section>

          <section id="effects" aria-labelledby="h-effects">
            <h2 id="h-effects">Effects</h2>
            <div className="prose">
              <p>
                Pick a look (Original, Cinematic, Mono, Warm, Cool), add vignette or grain, and switch stickers or reactions on and off.
                Photos contain exactly what the preview shows, including hand landmarks if you turn them on. <strong>Reset effects</strong>{' '}
                brings everything back to the defaults.
              </p>
            </div>
          </section>

          <section id="sound" aria-labelledby="h-sound">
            <h2 id="h-sound">Sound</h2>
            <div className="prose">
              <p>
                Sound is off every time the page loads. Turn it on with the speaker button. Effects and the optional music loop are
                generated in your browser, so there are no audio files and nothing plays on its own. The app works fully without sound.
              </p>
            </div>
          </section>

          <section id="privacy" aria-labelledby="h-privacy">
            <h2 id="h-privacy">Privacy</h2>
            <div className="prose">
              <p>{PRIVACY_STATEMENT}</p>
              <ul>
                <li>The hand-tracking model and its runtime are downloaded from this site, then run inside your browser.</li>
                <li>Video is never recorded. Frames are analyzed in memory and discarded.</li>
                <li>No accounts, no analytics, no cookies. Settings, found poses and pose-editor changes are remembered only in this browser.</li>
                <li>Turning the camera off or leaving the camera page stops the camera completely.</li>
              </ul>
              <p className="callout">{HOSTING_NOTE}</p>
            </div>
          </section>

          <section id="troubleshooting" aria-labelledby="h-trouble">
            <h2 id="h-trouble">Troubleshooting</h2>
            <div className="prose">
              <ul>
                <li>
                  <strong>Black preview:</strong> another app may hold the camera, or the wrong camera is selected. Try Effects → Camera.
                </li>
                <li>
                  <strong>“Hand tracking couldn’t load”:</strong> check your connection and press Retry. Very old browsers without
                  WebAssembly can’t run it.
                </li>
                <li>
                  <strong>Pose not recognized:</strong> turn on <strong>Show hand landmarks</strong> to see what the tracker sees, and{' '}
                  <strong>Debug info</strong> for finger states (E = extended, f = folded, ? = unsure).
                </li>
                <li>
                  <strong>Everything is reversed:</strong> toggle Mirror preview (or press M).
                </li>
              </ul>
            </div>
          </section>

          <section id="keys" aria-labelledby="h-keys">
            <h2 id="h-keys">Keyboard shortcuts</h2>
            <div className="kbd-list">
              <kbd>Space</kbd>
              <span>Take a photo</span>
              <kbd>T</kbd>
              <span>Start the 3-second timer</span>
              <kbd>M</kbd>
              <span>Mirror preview on/off</span>
              <kbd>D</kbd>
              <span>Debug info on/off</span>
              <kbd>Esc</kbd>
              <span>Close the photo or panel</span>
            </div>
          </section>
        </div>
      </div>
    </SiteLayout>
  );
}
