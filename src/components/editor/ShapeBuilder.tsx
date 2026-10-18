import { AlertTriangle, Info } from 'lucide-react';
import type { PoseDefinition } from '../../config/poses';
import { ruleToGlyph } from '../../config/registry';
import { DIGITS, ruleIssues, type CustomRule, type DigitReq } from '../../gestures/customRules';
import { HandGlyph } from '../HandGlyph';

const DIGIT_LABEL: Record<(typeof DIGITS)[number], string> = {
  thumb: 'Thumb',
  index: 'Index',
  middle: 'Middle',
  ring: 'Ring',
  pinky: 'Pinky',
};

function Choice<T extends string | number>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
}) {
  return (
    <div className="ed-choice" role="radiogroup" aria-label={label}>
      {options.map((o) => (
        <button key={String(o.value)} type="button" role="radio" aria-checked={value === o.value} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

const REQ_OPTIONS: { value: DigitReq; label: string }[] = [
  { value: 'up', label: 'Up' },
  { value: 'down', label: 'Folded' },
  { value: 'any', label: 'Any' },
];

export function ShapeBuilder({
  rule,
  conflicts,
  onChange,
}: {
  rule: CustomRule;
  conflicts: PoseDefinition[];
  onChange: (r: CustomRule) => void;
}) {
  const issues = ruleIssues(rule);
  const set = (p: Partial<CustomRule>) => onChange({ ...rule, ...p });
  return (
    <div className="ed-shape">
      <div className="ed-shape__glyph" aria-hidden="true">
        <HandGlyph hands={ruleToGlyph(rule)} fill="var(--text)" outline="#1c1c19" />
      </div>
      <div className="ed-shape__controls">
        <div className="ed-row">
          <span className="ed-row__label">Hands</span>
          <Choice
            label="Hands"
            value={rule.hands}
            onChange={(hands) => set({ hands })}
            options={[
              { value: 1, label: 'One' },
              { value: 2, label: 'Both' },
            ]}
          />
        </div>
        {DIGITS.map((d) => (
          <div className="ed-row" key={d}>
            <span className="ed-row__label">{DIGIT_LABEL[d]}</span>
            <Choice label={DIGIT_LABEL[d]} value={rule[d]} options={REQ_OPTIONS} onChange={(v) => set({ [d]: v } as Partial<CustomRule>)} />
          </div>
        ))}
        <div className="ed-row">
          <span className="ed-row__label">Thumb points</span>
          <Choice
            label="Thumb direction"
            value={rule.thumbDirection}
            onChange={(thumbDirection) => set({ thumbDirection, ...(thumbDirection !== 'any' ? { thumb: 'up' as const } : {}) })}
            options={[
              { value: 'any', label: 'Any way' },
              { value: 'up', label: 'Up' },
              { value: 'down', label: 'Down' },
            ]}
          />
        </div>
        <div className="ed-row">
          <span className="ed-row__label">Hand position</span>
          <Choice
            label="Hand position"
            value={rule.zone}
            onChange={(zone) => set({ zone })}
            options={[
              { value: 'any', label: 'Anywhere' },
              { value: 'upper', label: 'Top half' },
              { value: 'lower', label: 'Bottom half' },
            ]}
          />
        </div>
        {issues.map((i) => (
          <p key={i} className="ed-note ed-note--error" role="alert">
            <AlertTriangle aria-hidden="true" />
            {i}
          </p>
        ))}
        {!issues.length && conflicts.length ? (
          <p className="ed-note ed-note--warn">
            <Info aria-hidden="true" />
            This shape also matches {conflicts.map((c) => c.name).join(', ')}. Custom poses are checked first, so yours wins while it’s on.
          </p>
        ) : null}
      </div>
    </div>
  );
}
