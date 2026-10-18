import { Link } from '../lib/router';

export function Wordmark({ to = '/' }: { to?: string }) {
  return (
    <Link to={to} className="wordmark" aria-label="POSE CAM home">
      <span className="wordmark__dot" aria-hidden="true" />
      <span>POSE CAM</span>
    </Link>
  );
}
