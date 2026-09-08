import React, { useMemo, useState, useRef } from 'react';

type Stage = { title: string; detail: string; codeHtml: string };

type Props = { stages: Stage[] };

export default function ToolJourney({ stages }: Props) {
  const [active, setActive] = useState(0);
  const stage = useMemo(() => stages[active], [active, stages]);
  const tabRefs = useRef<(HTMLButtonElement | null)[]>([]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    let newIndex = active;
    if (e.key === 'ArrowRight') {
      newIndex = (active + 1) % stages.length;
      e.preventDefault();
    } else if (e.key === 'ArrowLeft') {
      newIndex = (active - 1 + stages.length) % stages.length;
      e.preventDefault();
    }
    if (newIndex !== active) {
      setActive(newIndex);
      tabRefs.current[newIndex]?.focus();
    }
  };

  return (
    <div className="card">
      <div className="grid" role="tablist" aria-label="Implementation stages" onKeyDown={handleKeyDown}>
        {stages.map((s, i) => (
          <button
            key={s.title}
            role="tab"
            aria-selected={i === active}
            aria-controls="journey-tabpanel"
            id={`journey-tab-${i}`}
            tabIndex={i === active ? 0 : -1}
            ref={(el) => { tabRefs.current[i] = el; }}
            onClick={() => setActive(i)}
            style={{
              padding: '1rem',
              borderRadius: '1rem',
              border: '1px solid var(--line)',
              background:
                i === active
                  ? 'linear-gradient(135deg,#6ee7f9,#95f985)'
                  : '#101827',
              color: i === active ? '#071018' : 'var(--text)',
              fontWeight: 800,
              cursor: 'pointer',
            }}
          >
            {i + 1}. {s.title}
          </button>
        ))}
      </div>
      <div
        role="tabpanel"
        id="journey-tabpanel"
        aria-labelledby={`journey-tab-${active}`}
        tabIndex={0}
      >
        <h3>{stage.title}</h3>
        <p className="lead" style={{ fontSize: '1rem' }}>
          {stage.detail}
        </p>
        <figure
          className="code-block"
          dangerouslySetInnerHTML={{ __html: stage.codeHtml }}
        />
      </div>
    </div>
  );
}
