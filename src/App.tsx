import { Suspense, lazy, useCallback, useEffect } from 'react';
import { Toasts } from './components/Toasts';
import { useKonami } from './hooks/useKonami';
import { usePath } from './lib/router';
import { useStore } from './lib/useStore';
import { Help } from './pages/Help';
import { Landing } from './pages/Landing';
import { NotFound } from './pages/NotFound';
import { Poses } from './pages/Poses';
import { Studio } from './pages/Studio';

const Admin = lazy(() => import('./pages/Admin').then((m) => ({ default: m.Admin })));
import { prefersReducedMotion, settingsStore, updateSettings } from './state/settings';
import { pushToast } from './state/toasts';

export function App() {
  const path = usePath().replace(/\/+$/, '') || '/';
  const settings = useStore(settingsStore);

  useEffect(() => {
    document.documentElement.dataset.motion = prefersReducedMotion(settings) ? 'reduced' : 'full';
  }, [settings]);

  useEffect(() => {
    if (path === '/') document.title = 'POSE CAM — Your camera. Your poses. Your moment.';
  }, [path]);

  const onKonami = useCallback(() => {
    const on = !settingsStore.get().chaos;
    updateSettings({ chaos: on });
    pushToast(
      on
        ? { title: 'Chaos mode unlocked', body: 'Captions get louder. Turn it off in Effects.', tone: 'gold' }
        : { title: 'Chaos mode off', body: 'Order has been restored.' },
    );
  }, []);
  useKonami(onKonami);

  let page;
  if (path === '/') page = <Landing />;
  else if (path === '/studio') page = <Studio />;
  else if (path === '/poses') page = <Poses />;
  else if (path === '/help') page = <Help />;
  else if (path === '/admin')
    page = (
      <Suspense fallback={<div className="page-loading" role="status">Loading the pose editor…</div>}>
        <Admin />
      </Suspense>
    );
  else page = <NotFound />;

  return (
    <>
      {page}
      <Toasts />
    </>
  );
}
