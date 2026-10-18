import { Camera, LockKeyhole } from 'lucide-react';
import { HandGlyph } from '../components/HandGlyph';
import { HeroPreview } from '../components/HeroPreview';
import { SiteLayout } from '../components/SiteLayout';
import { BUILTIN_POSES } from '../config/poses';
import { registryStore } from '../config/registry';
import { useStore } from '../lib/useStore';
import { Link } from '../lib/router';
import { PRIVACY_STATEMENT } from '../config/copy';

const FEATURED = ['hello', 'peace', 'thumbsup', 'rockon', 'point', 'three', 'celebrate'] as const;

export function Landing() {
  const registry = useStore(registryStore);
  return (
    <SiteLayout>
      <section className="hero" aria-labelledby="hero-title">
        <div>
          <h1 id="hero-title" className="hero__title">
            <span>Your camera.</span>
            <span>Your poses.</span>
            <span>Your moment.</span>
          </h1>
          <p className="hero__lede">
            Make a gesture and POSE CAM answers with stickers, captions and confetti. Then keep the shot.
          </p>
          <div className="hero__actions">
            <Link to="/studio" state={{ autostart: true }} className="btn btn--primary btn--lg">
              <Camera aria-hidden="true" />
              Enter the camera
            </Link>
            <Link to="/poses" className="btn btn--ghost btn--lg">
              Explore poses
            </Link>
          </div>
          <p className="hero__privacy">
            <LockKeyhole aria-hidden="true" />
            <span>{PRIVACY_STATEMENT}</span>
          </p>
        </div>
        <HeroPreview />
      </section>

      <ol className="how" aria-label="How it works">
        <li>
          <span className="how__n" aria-hidden="true">
            1
          </span>
          <h3>Turn on your camera</h3>
          <p className="muted">Your browser asks first. Nothing starts until you say so, and one tap turns it off.</p>
        </li>
        <li>
          <span className="how__n" aria-hidden="true">
            2
          </span>
          <h3>Strike a pose</h3>
          <p className="muted">Hand tracking runs on your device and spots peace signs, thumbs, rock-on hands and more.</p>
        </li>
        <li>
          <span className="how__n" aria-hidden="true">
            3
          </span>
          <h3>Keep the shot</h3>
          <p className="muted">Snap a photo with every sticker and effect in it, or hold up three fingers for a timer.</p>
        </li>
      </ol>

      <section className="strip" aria-labelledby="strip-title">
        <div className="strip__head">
          <h2 id="strip-title">{registry.poses.filter((p) => p.status !== 'state').length} poses to find</h2>
          <Link to="/poses" className="text-link">
            See every pose
          </Link>
        </div>
        <ul className="strip__list">
          {FEATURED.map((id) => {
            const p = BUILTIN_POSES.find((x) => x.id === id)!;
            return (
              <li key={id} className="strip__item">
                <HandGlyph hands={p.glyph} fill="var(--text)" outline="#1d1d1a" />
                {p.name.replace(' — Photo Timer', '')}
              </li>
            );
          })}
        </ul>
      </section>
    </SiteLayout>
  );
}
