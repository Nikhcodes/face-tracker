import { useEffect } from 'react';
import { SiteLayout } from '../components/SiteLayout';
import { Link } from '../lib/router';

export function NotFound() {
  useEffect(() => {
    document.title = 'Not found — POSE CAM';
  }, []);
  return (
    <SiteLayout>
      <div className="notfound">
        <span className="tape" style={{ fontSize: '1.5rem' }}>
          THE CAMERA FOUND NOTHING.
        </span>
        <h1 style={{ fontSize: 'var(--fs-2xl)' }}>This page doesn’t exist</h1>
        <Link to="/" className="btn btn--primary">
          Go to the home page
        </Link>
      </div>
    </SiteLayout>
  );
}
