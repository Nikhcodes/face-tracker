import '@fontsource-variable/bricolage-grotesque/wght.css';
import '@fontsource-variable/inter/wght.css';
import './styles/global.css';
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './App';
import { initPacks } from './config/registry';

// Load the site's pose pack and this browser's editor changes (non-blocking).
void initPacks();

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
