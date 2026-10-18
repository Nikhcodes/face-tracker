import { ArrowLeft, Camera, Copy, Download, FileUp, Plus, RotateCcw, Trash2, Upload } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { ConfirmDialog } from '../components/ConfirmDialog';
import { HandGlyph } from '../components/HandGlyph';
import { SiteLayout } from '../components/SiteLayout';
import { CameraTester } from '../components/editor/CameraTester';
import { CaptionEditor } from '../components/editor/CaptionEditor';
import { PictureEditor } from '../components/editor/PictureEditor';
import { PosePreview } from '../components/editor/PosePreview';
import { ReactionEditor } from '../components/editor/ReactionEditor';
import { ShapeBuilder } from '../components/editor/ShapeBuilder';
import { ToggleRow } from '../components/studio/Controls';
import { LIMITS, sanitizePack, type PosePack } from '../config/pack';
import { BUILTIN_BY_ID, type PoseDefinition } from '../config/poses';
import {
  effectivePack,
  findConflicts,
  packState,
  packToJson,
  registryStore,
  resetLocalPack,
  saveLocalPack,
} from '../config/registry';
import {
  addCustomPose,
  applyPatch,
  deleteCustomPose,
  draftFor,
  duplicateCustomPose,
  packImageBytes,
  resetBuiltin,
  type DraftPatch,
} from '../editor/editPack';
import { ruleIssues } from '../gestures/customRules';
import { formatBytes } from '../lib/image';
import { Link } from '../lib/router';
import { useStore } from '../lib/useStore';
import { downloadBlob } from '../render/capture';
import { pushToast } from '../state/toasts';

type Confirm =
  | { kind: 'reset-all' }
  | { kind: 'delete'; id: string; name: string }
  | { kind: 'import'; pack: PosePack; warnings: string[] }
  | null;

function readSelection(): string {
  const id = new URLSearchParams(window.location.search).get('pose');
  return id || 'hello';
}

function ListItem({
  pose,
  status,
  current,
  onSelect,
}: {
  pose: PoseDefinition;
  status: { label: string; cls: string } | null;
  current: boolean;
  onSelect: (id: string) => void;
}) {
  return (
    <li>
      <button type="button" className="ed-item" aria-current={current ? 'true' : undefined} onClick={() => onSelect(pose.id)}>
        <span className="ed-item__glyph" aria-hidden="true">
          {pose.glyph.length ? <HandGlyph hands={pose.glyph} fill="var(--text)" outline="#1d1d1a" /> : <span className="ed-item__idle">zz</span>}
        </span>
        <span className="ed-item__name">{pose.name}</span>
        {status ? <span className={`ed-pill ed-pill--${status.cls}`}>{status.label}</span> : null}
      </button>
    </li>
  );
}

function Section({ title, hint, children, id }: { title: string; hint?: string; children: ReactNode; id: string }) {
  return (
    <section className="ed-section" aria-labelledby={id}>
      <div className="ed-section__intro">
        <h2 id={id}>{title}</h2>
        {hint ? <p>{hint}</p> : null}
      </div>
      <div className="ed-section__body">{children}</div>
    </section>
  );
}

export function Admin() {
  const ps = useStore(packState);
  const registry = useStore(registryStore);
  const pack = effectivePack(ps);
  const [selected, setSelected] = useState(readSelection);
  const [mobileView, setMobileView] = useState<'list' | 'editor'>(() => (new URLSearchParams(window.location.search).get('pose') ? 'editor' : 'list'));
  const [confirm, setConfirm] = useState<Confirm>(null);
  const importRef = useRef<HTMLInputElement>(null);
  const editorRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    document.title = 'Pose editor — POSE CAM';
  }, []);

  const select = useCallback((id: string) => {
    setSelected(id);
    setMobileView('editor');
    window.history.replaceState(window.history.state, '', `/admin?pose=${encodeURIComponent(id)}`);
    requestAnimationFrame(() => editorRef.current?.focus({ preventScroll: true }));
  }, []);

  const draft = draftFor(pack, selected) ?? draftFor(pack, 'hello')!;
  const pose = registry.byId.get(draft.id) ?? BUILTIN_BY_ID.hello;
  const original = draft.kind === 'builtin' ? BUILTIN_BY_ID[draft.id] : null;

  const mutate = (fn: (p: PosePack) => PosePack) => void saveLocalPack(fn(effectivePack()));
  const patch = (p: DraftPatch) => mutate((pk) => applyPatch(pk, draft.id, p));

  const conflicts = useMemo(
    () => (draft.rule && !ruleIssues(draft.rule).length ? findConflicts(draft.rule, registry, draft.id) : []),
    [draft.rule, registry, draft.id],
  );

  const customs = registry.all.filter((p) => p.source === 'custom');
  const builtins = registry.all.filter((p) => p.source === 'builtin');
  const imageBytes = packImageBytes(pack);

  const saveLabel = !ps.ready
    ? 'Loading…'
    : ps.storage === 'memory' || ps.save === 'error'
      ? 'Not saved: this browser blocks storage. Export a pack to keep your changes.'
      : ps.save === 'saving'
        ? 'Saving…'
        : ps.local
          ? 'Saved in this browser'
          : ps.site
            ? 'Showing this site’s poses'
            : 'No changes yet';

  const exportPack = () => {
    const name = `posecam-pack-${new Date().toISOString().slice(0, 10)}.json`;
    downloadBlob(new Blob([packToJson(pack)], { type: 'application/json' }), name);
    pushToast({ title: 'Pack exported', body: name });
  };

  const onImportFile = async (file: File) => {
    try {
      const { pack: imported, warnings } = sanitizePack(JSON.parse(await file.text()));
      if (!imported) {
        pushToast({ title: 'Couldn’t import that file', body: warnings[0] });
        return;
      }
      setConfirm({ kind: 'import', pack: imported, warnings });
    } catch {
      pushToast({ title: 'Couldn’t import that file', body: 'It isn’t valid JSON.' });
    }
  };

  const newPose = () => {
    if (pack.custom.length >= LIMITS.custom) {
      pushToast({ title: `Up to ${LIMITS.custom} custom poses` });
      return;
    }
    const { pack: next, id } = addCustomPose(effectivePack());
    void saveLocalPack(next);
    select(id);
  };

  const statusFor = (id: string, source: 'builtin' | 'custom') => {
    if (registry.disabled.has(id)) return { label: 'Off', cls: 'off' };
    if (source === 'custom') return { label: 'Custom', cls: 'custom' };
    return draftFor(pack, id)?.edited ? { label: 'Edited', cls: 'edited' } : null;
  };

  return (
    <SiteLayout wide>
      <header className="ed-head">
        <div>
          <h1>Pose editor</h1>
          <p>Make your own poses, swap pictures and rewrite captions. Changes show up in the camera straight away.</p>
        </div>
        <div className="ed-head__actions">
          <span className="ed-save" data-state={ps.storage === 'memory' || ps.save === 'error' ? 'error' : ps.save} role="status">
            {saveLabel}
          </span>
          <div className="ed-head__buttons">
            <button type="button" className="btn btn--sm btn--ghost" onClick={() => importRef.current?.click()}>
              <FileUp aria-hidden="true" />
              Import
            </button>
            <button type="button" className="btn btn--sm" onClick={exportPack}>
              <Download aria-hidden="true" />
              Export pack
            </button>
            {ps.local ? (
              <button type="button" className="btn btn--sm btn--ghost" onClick={() => setConfirm({ kind: 'reset-all' })}>
                <RotateCcw aria-hidden="true" />
                Reset all
              </button>
            ) : null}
          </div>
          <input
            ref={importRef}
            type="file"
            accept="application/json,.json"
            className="sr-only"
            tabIndex={-1}
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void onImportFile(f);
              e.target.value = '';
            }}
          />
        </div>
      </header>

      <details className="ed-publish">
        <summary>
          <Upload aria-hidden="true" />
          <span>
            Edits are saved in <strong>this browser</strong> only. <span className="ed-publish__more">How do I publish them for everyone?</span>
          </span>
        </summary>
        <ol>
          <li>
            Click <strong>Export pack</strong> to download your poses as one <code>.json</code> file (pictures included).
          </li>
          <li>
            In the project, save it as <code>public/packs/default.json</code>.
          </li>
          <li>Commit and push. Vercel redeploys and every visitor gets your poses.</li>
        </ol>
        <p className="ed-help">
          Anyone can open this editor, but their changes only affect their own browser, never your site.
          {imageBytes > 0 ? ` Your pictures add ${formatBytes(imageBytes)} to the pack${imageBytes > 4_000_000 ? '; consider fewer or smaller images so the site loads quickly.' : '.'}` : ''}
        </p>
      </details>

      <div className="ed-layout" data-view={mobileView}>
        <button type="button" className="ed-back btn btn--sm btn--ghost" onClick={() => setMobileView('list')}>
          <ArrowLeft aria-hidden="true" />
          All poses
        </button>
        <nav className="ed-list" aria-label="Poses">
          <div className="ed-list__section">
            <div className="ed-list__head">
              <h2>Your poses</h2>
              <button type="button" className="btn btn--sm btn--primary" onClick={newPose}>
                <Plus aria-hidden="true" />
                New pose
              </button>
            </div>
            {customs.length ? (
              <ul>
                {customs.map((p) => (
                  <ListItem key={p.id} pose={p} status={statusFor(p.id, 'custom')} current={p.id === draft.id} onSelect={select} />
                ))}
              </ul>
            ) : (
              <p className="ed-empty">Pick fingers up or down, or copy your hand from the camera, and it becomes a new pose with its own pictures and captions.</p>
            )}
          </div>
          <div className="ed-list__section">
            <div className="ed-list__head">
              <h2>Built-in poses</h2>
            </div>
            <ul>
              {builtins.map((p) => (
                <ListItem key={p.id} pose={p} status={statusFor(p.id, 'builtin')} current={p.id === draft.id} onSelect={select} />
              ))}
            </ul>
          </div>
        </nav>

        <div className="ed-editor" ref={editorRef} tabIndex={-1} aria-label={`Editing ${draft.name}`}>
          <div className="ed-title">
            <div>
              <span className={`badge ${draft.kind === 'custom' ? 'badge--custom' : original?.status === 'experimental' ? 'badge--exp' : 'badge--live'}`}>
                {draft.kind === 'custom' ? 'Custom pose' : original?.status === 'experimental' ? 'Built-in, experimental' : original?.status === 'state' ? 'Built-in, automatic' : 'Built-in'}
              </span>
              <h2>{draft.name || 'Untitled pose'}</h2>
            </div>
            {draft.id !== 'idle' ? (
              <div className="ed-title__toggle">
                <ToggleRow label="Pose on" checked={draft.enabled} onChange={(v) => patch({ enabled: v })} />
              </div>
            ) : null}
          </div>

          <Section id="sec-details" title="Details" hint="Shown in the pose guide and on the poses page.">
            <label className="ed-field">
              <span>Name</span>
              <input className="ed-input" value={draft.name} maxLength={LIMITS.name} onChange={(e) => patch({ name: e.target.value })} />
            </label>
            <label className="ed-field">
              <span>Short description</span>
              <input className="ed-input" value={draft.blurb} maxLength={LIMITS.blurb} placeholder="What makes this pose fun" onChange={(e) => patch({ blurb: e.target.value })} />
            </label>
            <label className="ed-field">
              <span>How to do it</span>
              <textarea
                className="ed-input ed-textarea"
                value={draft.howTo}
                maxLength={LIMITS.howTo}
                rows={2}
                placeholder={draft.kind === 'custom' ? 'Leave empty to describe the hand shape automatically' : ''}
                onChange={(e) => patch({ howTo: e.target.value })}
              />
            </label>
          </Section>

          <Section
            id="sec-shape"
            title="Hand shape"
            hint={draft.kind === 'custom' ? 'Which fingers must be up or folded. Works with the hand tilted or sideways.' : undefined}
          >
            {draft.kind === 'custom' && draft.rule ? (
              <ShapeBuilder rule={draft.rule} conflicts={conflicts} onChange={(rule) => patch({ rule })} />
            ) : (
              <div className="ed-builtin-shape">
                <span className="ed-builtin-shape__glyph" aria-hidden="true">
                  {pose.glyph.length ? <HandGlyph hands={pose.glyph} fill="var(--text)" outline="#1c1c19" /> : null}
                </span>
                <p className="ed-help">
                  Built-in poses use tuned detection that can’t be edited. Want a different shape?{' '}
                  <button type="button" className="ed-linkbtn" onClick={newPose}>
                    Create a custom pose
                  </button>
                  .
                </p>
              </div>
            )}
          </Section>

          <Section id="sec-pictures" title="Pictures" hint="The sticker that pops up when the pose is detected.">
            <PictureEditor
              kind={draft.kind}
              originals={original?.stickers ?? []}
              keepOriginal={draft.keepOriginal}
              stickers={draft.stickers}
              onChange={(p) => patch(p)}
            />
          </Section>

          {draft.id !== 'idle' ? (
            <Section id="sec-captions" title="Captions" hint="The line on the lime tape.">
              <CaptionEditor captions={draft.captions} onChange={(captions) => patch({ captions })} />
            </Section>
          ) : null}

          {draft.id !== 'idle' ? (
            <Section id="sec-reaction" title="Reaction" hint="What happens on screen, and the sound it makes.">
              <ReactionEditor particles={draft.particles} shake={draft.shake} spotlight={draft.spotlight} cue={draft.sound} onChange={(p) => patch(p)} />
            </Section>
          ) : null}

          <div className="ed-footer-actions">
            {draft.kind === 'builtin' && draft.edited ? (
              <button type="button" className="btn btn--sm btn--ghost" onClick={() => mutate((pk) => resetBuiltin(pk, draft.id))}>
                <RotateCcw aria-hidden="true" />
                Restore {original?.name ?? 'pose'} defaults
              </button>
            ) : null}
            {draft.kind === 'custom' ? (
              <>
                <button
                  type="button"
                  className="btn btn--sm btn--ghost"
                  onClick={() => {
                    const r = duplicateCustomPose(effectivePack(), draft.id);
                    if (r) {
                      void saveLocalPack(r.pack);
                      select(r.id);
                    }
                  }}
                >
                  <Copy aria-hidden="true" />
                  Duplicate
                </button>
                <button type="button" className="btn btn--sm btn--danger-ghost" onClick={() => setConfirm({ kind: 'delete', id: draft.id, name: draft.name })}>
                  <Trash2 aria-hidden="true" />
                  Delete pose
                </button>
              </>
            ) : null}
          </div>
        </div>

        <aside className="ed-side" aria-label="Preview">
          <div className="ed-side__sticky">
            <PosePreview pose={pose} />
            <p className="ed-help ed-side__caption">Preview of the reaction. Pictures and captions rotate when there are several.</p>
            <CameraTester key={draft.id} pose={pose} onCopyShape={draft.kind === 'custom' ? (rule) => patch({ rule }) : undefined} />
            <Link to="/studio" state={{ autostart: true }} className="btn btn--sm btn--ghost ed-side__open">
              <Camera aria-hidden="true" />
              Open the camera
            </Link>
          </div>
        </aside>
      </div>

      {confirm?.kind === 'reset-all' ? (
        <ConfirmDialog
          title="Reset all changes?"
          body={ps.site ? 'Your edits in this browser are removed and the site’s own poses come back.' : 'Every custom pose and edit in this browser is removed. Export a pack first if you want to keep them.'}
          confirmLabel="Reset everything"
          tone="danger"
          onCancel={() => setConfirm(null)}
          onConfirm={() => {
            void resetLocalPack();
            setConfirm(null);
            setSelected('hello');
            pushToast({ title: 'Changes reset' });
          }}
        />
      ) : null}
      {confirm?.kind === 'delete' ? (
        <ConfirmDialog
          title={`Delete “${confirm.name}”?`}
          body="The pose, its pictures and captions are removed from this browser."
          confirmLabel="Delete pose"
          tone="danger"
          onCancel={() => setConfirm(null)}
          onConfirm={() => {
            mutate((pk) => deleteCustomPose(pk, confirm.id));
            setConfirm(null);
            select('hello');
            pushToast({ title: 'Pose deleted' });
          }}
        />
      ) : null}
      {confirm?.kind === 'import' ? (
        <ConfirmDialog
          title={`Import “${confirm.pack.name}”?`}
          body={`It has ${confirm.pack.custom.length} custom pose${confirm.pack.custom.length === 1 ? '' : 's'} and ${Object.keys(confirm.pack.overrides).length} edited built-in pose${Object.keys(confirm.pack.overrides).length === 1 ? '' : 's'}. It replaces your current edits in this browser.${confirm.warnings.length ? ` ${confirm.warnings.length} item(s) were skipped as invalid.` : ''}`}
          confirmLabel="Import pack"
          onCancel={() => setConfirm(null)}
          onConfirm={() => {
            void saveLocalPack(confirm.pack, 0);
            setConfirm(null);
            setSelected(confirm.pack.custom[0]?.id ?? 'hello');
            pushToast({ title: 'Pack imported', tone: 'accent' });
          }}
        />
      ) : null}
    </SiteLayout>
  );
}
