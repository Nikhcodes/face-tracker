import type { ReactNode } from 'react';
import { Link } from '../lib/router';
import { Wordmark } from './Wordmark';

export function SiteLayout({ children, wide = false }: { children: ReactNode; wide?: boolean }) {
  return (
    <div className="site">
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <header className={`site-header${wide ? ' site-header--wide' : ''}`}>
        <Wordmark />
        <nav className="site-nav" aria-label="Main">
          <Link to="/poses">Poses</Link>
          <Link to="/admin">Editor</Link>
          <Link to="/help">Help</Link>
          <Link to="/studio" state={{ autostart: false }}>
            Camera
          </Link>
        </nav>
      </header>
      <main id="main" className={`site-main${wide ? ' site-main--wide' : ''}`}>
        {children}
      </main>
      <footer className="site-footer">
        <span>POSE CAM runs in your browser. No accounts, no uploads.</span>
        <nav aria-label="Footer">
          <Link to="/poses">Poses</Link>
          <Link to="/admin">Pose editor</Link>
          <Link to="/help">Help</Link>
          <Link to="/help#privacy">Privacy</Link>
        </nav>
      </footer>
    </div>
  );
}
