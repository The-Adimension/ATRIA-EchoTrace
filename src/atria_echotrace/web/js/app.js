/**
 * ATRIA EchoTrace single-page workstation.
 *
 * Production form of the notebook's HITL interface (notebook_as_py.txt L1381-1574):
 * two synchronised frame editors, drag-to-edit vertices, and a save action that
 * produces JSON plus 4-panel figures. Everything here is wired to the real backend —
 * there is no mock data path.
 *
 * Buildless: Preact + hooks + htm are vendored ES modules resolved by the import map
 * in index.html (RESEARCH.md §4.2).
 */

import { h, render } from 'preact';
import { useCallback, useEffect, useRef, useState } from 'preact/hooks';
import htm from 'htm';

import { api, ApiError } from './api.js';
import { ContourEditor, PANEL } from './canvas-editor.js';

const html = htm.bind(h);
const INSTANTS = ['ED', 'ES'];
const PANEL_LABELS = [
  [PANEL.ORIGINAL, 'Original'],
  [PANEL.MODEL, 'Model'],
  [PANEL.REVISION, 'Revision'],
  [PANEL.OVERLAY, 'Overlay'],
];

const emptyTracing = () => ({ model: [], revision: [] });

/* ========================================================================== */
/*  Root                                                                       */
/* ========================================================================== */

function App() {
  const [caps, setCaps] = useState(null);
  const [capsError, setCapsError] = useState(null);
  const [cases, setCases] = useState(null);
  const [filters, setFilters] = useState({ source: '', view: '' });
  const [activeCase, setActiveCase] = useState(null);
  const [loadingCase, setLoadingCase] = useState(false);
  // Uploaded frames carry no view metadata, so the clinician declares it. '' means
  // "unknown", which keeps the generic prompt rather than asserting a view.
  const [uploadView, setUploadView] = useState('4CH');
  const [uploading, setUploading] = useState({ ED: false, ES: false });

  const [structure, setStructure] = useState('LV');
  const [adapter, setAdapter] = useState('camus');
  const [promptVariant, setPromptVariant] = useState('');
  const [modelStatus, setModelStatus] = useState(null);

  const [tracings, setTracings] = useState({ ED: emptyTracing(), ES: emptyTracing() });
  const [panels, setPanels] = useState({ ED: PANEL.REVISION, ES: PANEL.REVISION });
  const [predicting, setPredicting] = useState({ ED: false, ES: false });
  const [agreement, setAgreement] = useState({ ED: null, ES: null });

  const [metrics, setMetrics] = useState(null);
  const [spacing, setSpacing] = useState({ h: '', w: '' });
  const [notes, setNotes] = useState('');
  const [saving, setSaving] = useState(false);
  const [lastRevision, setLastRevision] = useState(null);
  const [revisions, setRevisions] = useState([]);

  const [toasts, setToasts] = useState([]);
  const [modal, setModal] = useState(null);
  const [disclaimers, setDisclaimers] = useState(null);

  const toastSeq = useRef(0);

  const pushToast = useCallback((body, kind = 'error', title = null) => {
    const id = (toastSeq.current += 1);
    const titles = { error: 'Error', info: 'Notice', success: 'Saved' };
    setToasts((current) => [
      ...current.slice(-3),
      { id, body: String(body), kind, title: title || titles[kind] || 'Notice' },
    ]);
    if (kind !== 'error') {
      setTimeout(() => setToasts((c) => c.filter((t) => t.id !== id)), 6000);
    }
  }, []);

  const dismissToast = useCallback((id) => {
    setToasts((current) => current.filter((t) => t.id !== id));
  }, []);

  /* ---------------------------------------------------------- initial load */
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const capabilities = await api.capabilities();
        if (!alive) return;
        setCaps(capabilities);
        setAdapter(capabilities.default_adapter || 'camus');
      } catch (error) {
        if (alive) setCapsError(error.message);
        return;
      }
      try {
        const status = await api.modelStatus();
        if (alive) setModelStatus(status);
      } catch (error) {
        /* status is advisory; the rest of the app works without it */
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  /* ------------------------------------------------------------ case list */
  useEffect(() => {
    let alive = true;
    setCases(null);
    api
      .cases({ source: filters.source || undefined, view: filters.view || undefined })
      .then((data) => alive && setCases(data))
      .catch((error) => {
        if (alive) {
          setCases({ count: 0, cases: [], sources: [], views: [] });
          pushToast(error.message);
        }
      });
    return () => {
      alive = false;
    };
  }, [filters.source, filters.view, pushToast]);

  /* ------------------------------------------------- keep model state honest */
  // Polls continuously, not only while loading: the model is process-wide state that
  // another tab — or a load started before this tab mounted — can change underneath
  // us. Without this the status pill can sit on a stale "unloaded" while the model is
  // in fact ready, and the Trace buttons stay wrongly disabled.
  useEffect(() => {
    if (!caps || !caps.tiers.ai) return undefined;
    const loading = modelStatus && modelStatus.state === 'loading';
    const timer = setInterval(
      async () => {
        try {
          const status = await api.modelStatus();
          setModelStatus((previous) => {
            if (
              status.state === 'error' &&
              status.error &&
              (!previous || previous.error !== status.error)
            ) {
              pushToast(status.error, 'error', 'Model load failed');
            }
            return status;
          });
        } catch (error) {
          /* transient; the next tick retries */
        }
      },
      loading ? 1500 : 10000
    );
    return () => clearInterval(timer);
  }, [caps, modelStatus, pushToast]);

  /* ------------------------------------------------------- case selection */
  const selectCase = useCallback(
    async (caseKey) => {
      setLoadingCase(true);
      try {
        const detail = await api.caseDetail(caseKey);
        setActiveCase(detail);
        setTracings({ ED: emptyTracing(), ES: emptyTracing() });
        setAgreement({ ED: null, ES: null });
        setPanels({ ED: PANEL.REVISION, ES: PANEL.REVISION });
        setMetrics(null);
        setLastRevision(null);
        setNotes('');
        setSpacing({ h: '', w: '' });
        // Match the adapter to the case's own dataset: the CAMUS adapter was tuned
        // on CAMUS frames and the EchoNet adapter on EchoNet frames.
        if (detail.source === 'camus' || detail.source === 'echonet') setAdapter(detail.source);
        if (!detail.has_la && structure === 'LA') setStructure('LV');
      } catch (error) {
        pushToast(error.message);
      } finally {
        setLoadingCase(false);
      }
    },
    [pushToast, structure]
  );

  /* ------------------------------------------------------- uploaded frames */
  // The notebook's HITL stage was upload-driven: two file inputs feeding two editors,
  // "Image 1 (e.g. End-Diastole)" and "Image 2 (e.g. End-Systole)"
  // (notebook_as_py.txt L1404-1422). The slot therefore supplies the instant, exactly
  // as `img_idx` did (L1293), and the only metadata left for the clinician is the view.
  //
  // No resizing happens anywhere: polygons are normalised to [0, norm_scale], so the
  // preprocessing contract is resolution-independent by construction.
  const uploadFrame = useCallback(
    async (instant, file) => {
      if (!file) return;
      setUploading((current) => ({ ...current, [instant]: true }));
      try {
        const result = await api.uploadFrame(file);
        const frame = {
          stem: null,
          upload_id: result.upload_id,
          image_url: result.image_url,
          image_h: result.image_h,
          image_w: result.image_w,
          // An upload has no reference trace; the clinician's own contour is the truth.
          lv_polygon: null,
          la_polygon: null,
          filename: result.filename,
        };
        setActiveCase((current) => {
          const base =
            current && current.is_upload
              ? current
              : {
                  is_upload: true,
                  // Deliberately null: no dataset case exists, so metrics and revisions
                  // resolve calibration as unknown rather than against a stranger's case.
                  case_key: null,
                  case_id: 'Uploaded frames',
                  source: 'upload',
                  has_la: false,
                  integrity_flags: [],
                  frames: {},
                };
          return { ...base, view: uploadView, frames: { ...base.frames, [instant]: frame } };
        });
        // A new image invalidates whatever was traced in that slot.
        setTracings((current) => ({ ...current, [instant]: emptyTracing() }));
        setAgreement((current) => ({ ...current, [instant]: null }));
        setPanels((current) => ({ ...current, [instant]: PANEL.REVISION }));
        setMetrics(null);
        setLastRevision(null);
        setStructure('LV');
        pushToast(
          `${instant}: ${result.image_w}×${result.image_h}. ${result.note}`,
          'info',
          'Frame uploaded'
        );
      } catch (error) {
        pushToast(error.message, 'error', 'Upload rejected');
      } finally {
        setUploading((current) => ({ ...current, [instant]: false }));
      }
    },
    [uploadView, pushToast]
  );

  const changeUploadView = useCallback((view) => {
    setUploadView(view);
    setActiveCase((current) => (current && current.is_upload ? { ...current, view } : current));
  }, []);

  /* --------------------------------------------------------- model loading */
  const loadModel = useCallback(async () => {
    try {
      const status = await api.loadModel(adapter);
      setModelStatus(status);
      pushToast(`Loading ${adapter}. This can take several minutes on first use.`, 'info');
    } catch (error) {
      pushToast(error.message, 'error', error.status === 503 ? 'AI tier unavailable' : 'Error');
    }
  }, [adapter, pushToast]);

  const unloadModel = useCallback(async () => {
    try {
      setModelStatus(await api.unloadModel());
    } catch (error) {
      pushToast(error.message);
    }
  }, [pushToast]);

  /* -------------------------------------------------------------- predict */
  const predict = useCallback(
    async (instant) => {
      const frame = activeCase && activeCase.frames && activeCase.frames[instant];
      if (!frame) return;
      setPredicting((current) => ({ ...current, [instant]: true }));
      try {
        const result = await api.predict({
          // Dataset frames infer view and instant server-side from the stem; uploads
          // cannot, so they are declared here. Supplying both selects the training
          // prompt template — the one the adapters were fine-tuned with.
          ...(frame.upload_id
            ? { upload_id: frame.upload_id, view: activeCase.view || undefined, instant }
            : { stem: frame.stem }),
          target_structure: structure,
          prompt_variant: promptVariant || undefined,
        });
        setTracings((current) => ({
          ...current,
          [instant]: { model: result.polygon, revision: result.polygon },
        }));
        setAgreement((current) => ({ ...current, [instant]: result.agreement || null }));
        pushToast(
          `${instant}: ${result.vertices} vertices in ${result.inference_seconds}s` +
            (result.agreement ? ` · Dice ${result.agreement.dice} vs reference` : ''),
          'success',
          'Prediction complete'
        );
      } catch (error) {
        const title =
          error.status === 409
            ? 'Model not loaded'
            : error.status === 503
              ? 'AI tier unavailable'
              : 'Prediction failed';
        pushToast(error.message, 'error', title);
      } finally {
        setPredicting((current) => ({ ...current, [instant]: false }));
      }
    },
    [activeCase, structure, promptVariant, pushToast]
  );

  const predictBoth = useCallback(async () => {
    for (const instant of INSTANTS) {
      if (activeCase && activeCase.frames && activeCase.frames[instant]) {
        // Sequential: the backend serialises generation on one device anyway, and
        // this keeps per-frame errors attributable.
        await predict(instant);
      }
    }
  }, [activeCase, predict]);

  /* --------------------------------------------------- live metric updates */
  const updateTracing = useCallback((instant, revision) => {
    setTracings((current) => ({
      ...current,
      [instant]: { ...current[instant], revision },
    }));
  }, []);

  const seedFromReference = useCallback(
    (instant) => {
      const frame = activeCase && activeCase.frames && activeCase.frames[instant];
      const reference = frame && frame[structure === 'LV' ? 'lv_polygon' : 'la_polygon'];
      if (!reference || !reference.length) return;
      setTracings((current) => ({
        ...current,
        [instant]: { ...current[instant], revision: reference.map((p) => [p[0], p[1]]) },
      }));
    },
    [activeCase, structure]
  );

  const hasAnyRevision = INSTANTS.some((i) => tracings[i].revision.length >= 3);

  useEffect(() => {
    if (!activeCase || !hasAnyRevision) {
      setMetrics(null);
      return undefined;
    }
    const reference = activeCase.frames.ED || activeCase.frames.ES;
    const payload = {
      ed_polygon: tracings.ED.revision,
      es_polygon: tracings.ES.revision,
      image_h: reference.image_h,
      image_w: reference.image_w,
      case_key: activeCase.case_key,
    };
    if (activeCase.frames.ES) {
      payload.es_image_h = activeCase.frames.ES.image_h;
      payload.es_image_w = activeCase.frames.ES.image_w;
    }
    const manualH = parseFloat(spacing.h);
    const manualW = parseFloat(spacing.w);
    if (manualH > 0 && manualW > 0) {
      payload.spacing_h = manualH;
      payload.spacing_w = manualW;
    }
    // Debounced so dragging a vertex does not issue a request per pointer move.
    const timer = setTimeout(() => {
      api
        .metrics(payload)
        .then(setMetrics)
        .catch((error) => {
          if (!(error instanceof ApiError) || error.status !== 422) pushToast(error.message);
        });
    }, 220);
    return () => clearTimeout(timer);
  }, [activeCase, tracings, spacing, hasAnyRevision, pushToast]);

  /* ------------------------------------------------------------------ save */
  const saveRevision = useCallback(async () => {
    if (!activeCase) return;
    const phases = INSTANTS.filter(
      (instant) => activeCase.frames[instant] && tracings[instant].revision.length >= 3
    ).map((instant) => {
      const frame = activeCase.frames[instant];
      return {
        instant,
        ...(frame.upload_id ? { upload_id: frame.upload_id } : { stem: frame.stem }),
        model_polygon: tracings[instant].model,
        user_polygon: tracings[instant].revision,
      };
    });
    if (!phases.length) {
      pushToast('Trace at least one frame with 3 or more vertices before saving.');
      return;
    }
    setSaving(true);
    try {
      const payload = {
        phases,
        case_key: activeCase.case_key,
        // Uploads have no dataset view to fall back on, so carry the declared one into
        // the record — an exported corpus needs it to rebuild the training prompt.
        view: activeCase.is_upload ? activeCase.view || undefined : undefined,
        target_structure: structure,
        adapter,
        prompt_variant: promptVariant || undefined,
        notes,
      };
      const manualH = parseFloat(spacing.h);
      const manualW = parseFloat(spacing.w);
      if (manualH > 0 && manualW > 0) {
        payload.spacing_h = manualH;
        payload.spacing_w = manualW;
      }
      const record = await api.createRevision(payload);
      setLastRevision(record);
      pushToast(`Revision ${record.revision_id} written with ${Object.keys(record.files).length} artefacts.`, 'success');
    } catch (error) {
      pushToast(error.message, 'error', 'Save failed');
    } finally {
      setSaving(false);
    }
  }, [activeCase, tracings, structure, adapter, promptVariant, notes, spacing, pushToast]);

  /* --------------------------------------------------------- past revisions */
  // Saved work was previously invisible after the session that created it, which made
  // the corpus export (`atria export-corpus`) hard to reason about: you could not see
  // what there was to export. Reloading one restores both polygons into the editor.
  const refreshRevisions = useCallback(() => {
    api
      .revisions()
      .then((data) => setRevisions((data && data.revisions) || []))
      .catch(() => setRevisions([]));
  }, []);

  useEffect(refreshRevisions, [refreshRevisions]);
  useEffect(() => {
    if (lastRevision) refreshRevisions();
  }, [lastRevision, refreshRevisions]);

  const openRevision = useCallback(
    async (revisionId) => {
      setLoadingCase(true);
      try {
        const record = await api.revision(revisionId);
        const phases = record.phases || {};
        const provenance = record.provenance || {};
        const frames = {};
        const restored = { ED: emptyTracing(), ES: emptyTracing() };
        for (const instant of INSTANTS) {
          const phase = phases[instant];
          if (!phase) continue;
          const upload = /^upload:(.+)$/.exec(phase.stem || '');
          frames[instant] = {
            stem: upload ? null : phase.stem,
            upload_id: upload ? upload[1] : undefined,
            image_url: upload
              ? `/api/dataset/uploads/${upload[1]}.png`
              : `/api/dataset/frames/${phase.stem}.png`,
            image_h: phase.image_h,
            image_w: phase.image_w,
            lv_polygon: phase.ground_truth_polygon_2d || null,
            la_polygon: null,
          };
          restored[instant] = {
            model: phase.model_polygon_2d || [],
            revision: phase.user_polygon_2d || [],
          };
        }
        setActiveCase({
          is_upload: Object.values(frames).some((f) => f.upload_id),
          case_key: provenance.case_key || null,
          case_id: record.case || revisionId,
          source: 'revision',
          view: (provenance.views || {}).ED || (provenance.views || {}).ES || '',
          has_la: false,
          integrity_flags: provenance.dataset_integrity_flags || [],
          frames,
        });
        setTracings(restored);
        setPanels({ ED: PANEL.REVISION, ES: PANEL.REVISION });
        setAgreement({ ED: null, ES: null });
        setStructure(provenance.target_structure === 'LA' ? 'LA' : 'LV');
        setNotes(record.notes || '');
        setSpacing(
          provenance.spacing_h && provenance.spacing_w
            ? { h: String(provenance.spacing_h), w: String(provenance.spacing_w) }
            : { h: '', w: '' }
        );
        setLastRevision(null);
        pushToast(`Reopened ${revisionId}.`, 'info', 'Revision loaded');
      } catch (error) {
        pushToast(error.message, 'error', 'Could not reopen');
      } finally {
        setLoadingCase(false);
      }
    },
    [pushToast]
  );

  /* --------------------------------------------------------------- about */
  const openAbout = useCallback(async () => {
    setModal('about');
    if (disclaimers) return;
    try {
      setDisclaimers(await api.disclaimers());
    } catch (error) {
      pushToast(error.message);
    }
  }, [disclaimers, pushToast]);

  /* ------------------------------------------------------------- rendering */
  if (capsError) {
    return html`
      <div class="boot">
        <p>Cannot reach the ATRIA EchoTrace backend.</p>
        <p style="font-family:var(--sans);text-transform:none;letter-spacing:0">${capsError}</p>
      </div>
    `;
  }
  if (!caps) {
    return html`<div class="boot">
      <div class="boot__mark"></div>
      <p>Connecting…</p>
    </div>`;
  }

  const aiReady = modelStatus && modelStatus.state === 'ready';

  return html`
    <div class="shell">
      <${TopBar}
        caps=${caps}
        modelStatus=${modelStatus}
        onLoadModel=${loadModel}
        onUnloadModel=${unloadModel}
        onAbout=${openAbout}
        onWeights=${() => setModal('weights')}
        onStages=${() => setModal('stages')}
        onEvaluation=${() => setModal('evaluation')}
      />
      <div class="oversight">
        <span class="oversight__tag">Human oversight required</span>
        <span
          >Research use only — not a medical device. Every AI-generated contour is a
          preliminary proposal that must be reviewed and revised by qualified
          personnel.</span
        >
        <button class="btn btn--sm btn--ghost" onClick=${openAbout}>Disclaimers</button>
      </div>

      <div class="workspace">
        <${CaseBrowser}
          cases=${cases}
          filters=${filters}
          onFilter=${setFilters}
          activeKey=${activeCase && activeCase.case_key}
          onSelect=${selectCase}
          uploadView=${uploadView}
          onUploadView=${changeUploadView}
          onUpload=${uploadFrame}
          uploading=${uploading}
          uploadFrames=${activeCase && activeCase.is_upload ? activeCase.frames : null}
          revisions=${revisions}
          onOpenRevision=${openRevision}
        />

        <${Stage}
          caps=${caps}
          activeCase=${activeCase}
          loadingCase=${loadingCase}
          structure=${structure}
          onStructure=${setStructure}
          adapter=${adapter}
          onAdapter=${setAdapter}
          promptVariant=${promptVariant}
          onPromptVariant=${setPromptVariant}
          tracings=${tracings}
          panels=${panels}
          onPanel=${(instant, panel) => setPanels((c) => ({ ...c, [instant]: panel }))}
          onTracingChange=${updateTracing}
          onPredict=${predict}
          onPredictBoth=${predictBoth}
          predicting=${predicting}
          agreement=${agreement}
          aiReady=${aiReady}
          onSeedReference=${seedFromReference}
        />

        <${MetricsRail}
          activeCase=${activeCase}
          metrics=${metrics}
          tracings=${tracings}
          agreement=${agreement}
          spacing=${spacing}
          onSpacing=${setSpacing}
          notes=${notes}
          onNotes=${setNotes}
          onSave=${saveRevision}
          saving=${saving}
          canSave=${Boolean(activeCase) && hasAnyRevision}
          lastRevision=${lastRevision}
        />
      </div>
    </div>

    <div class="toasts">
      ${toasts.map(
        (toast) => html`
          <div class="toast toast--${toast.kind}" key=${toast.id} role="status">
            <div>
              <div class="toast__title">${toast.title}</div>
              <div class="toast__body">${toast.body}</div>
            </div>
            <button
              class="toast__close"
              aria-label="Dismiss"
              onClick=${() => dismissToast(toast.id)}
            >
              ×
            </button>
          </div>
        `
      )}
    </div>

    ${modal === 'about'
      ? html`<${AboutModal} data=${disclaimers} caps=${caps} onClose=${() => setModal(null)} />`
      : null}
    ${modal === 'weights'
      ? html`<${WeightsModal} caps=${caps} onClose=${() => setModal(null)} />`
      : null}
    ${modal === 'stages'
      ? html`<${StagesModal} onClose=${() => setModal(null)} />`
      : null}
    ${modal === 'evaluation'
      ? html`<${EvaluationModal} caps=${caps} onClose=${() => setModal(null)} />`
      : null}
  `;
}

/* ========================================================================== */
/*  Top bar                                                                    */
/* ========================================================================== */

/** Human-readable summary of where the base weights come from. */
const WEIGHT_SOURCE_LABEL = {
  local: 'local files',
  cache: 'hugging face cache',
  hub: 'download required',
};

function TopBar({
  caps, modelStatus, onLoadModel, onUnloadModel, onAbout, onWeights, onStages, onEvaluation,
}) {
  const aiInstalled = Boolean(caps.tiers.ai);
  const state = (modelStatus && modelStatus.state) || 'unknown';
  const device = caps.device || {};
  const weights = caps.weights || {};
  const base = weights.base || {};

  // Only one pill when there is no AI tier: a second "model unavailable" alongside
  // "no ai" said the same thing twice and read like two separate faults.
  // A CPU-only torch wheel on a machine that HAS a GPU is the single most common
  // install fault, and it is invisible if the pill just says "cpu". Call it out.
  const cpuOnlyBuild = device.cpu_cause === 'cpu_only_build';
  const computeLabel = !device.available
    ? 'review tier · ai not installed'
    : cpuOnlyBuild
      ? 'cpu · gpu present but torch has no cuda'
      : `${device.device}${device.gpu_name ? ` · ${device.gpu_name}` : ''} · ${device.compute_dtype}${device.quantization ? ' · 4-bit nf4' : ''}`;

  const modelKind =
    state === 'ready' ? 'ready' : state === 'loading' ? 'busy' : state === 'error' ? 'error' : '';
  const modelLabel =
    state === 'loading' && modelStatus
      ? `loading ${Math.round((modelStatus.progress || 0) * 100)}%`
      : state === 'ready'
        ? `model ready · ${WEIGHT_SOURCE_LABEL[modelStatus.weights_source] || 'loaded'}`
        : state === 'error'
          ? 'model failed'
          : `model not loaded · ${WEIGHT_SOURCE_LABEL[base.source] || 'weights unknown'}`;

  return html`
    <header class="topbar">
      <div class="brand">
        <span class="brand__name">ATRIA</span>
        <span class="brand__sub">EchoTrace</span>
      </div>
      <div class="brand__rule"></div>
      <span
        class=${`pill${cpuOnlyBuild ? ' pill--error' : ''}`}
        title=${device.available
          ? `${device.reason || ''}${device.remedy ? `

${device.remedy}` : ''}`
          : 'The AI tier is not installed, so contours cannot be predicted. Everything else — browsing, manual tracing, measurements and exports — works. Install with: pip install -e ".[ai]"'}
      >
        <span class="pill__dot"></span>${computeLabel}
      </span>
      ${aiInstalled
        ? html`<span
            class="pill pill--${modelKind}"
            title=${(modelStatus && modelStatus.message) || ''}
          >
            <span class="pill__dot"></span>${modelLabel}
          </span>`
        : null}
      <div class="topbar__spacer"></div>
      <button
        class="btn btn--sm btn--ghost"
        onClick=${onWeights}
        title="Where the model weights are loaded from"
      >
        Weights${base.ready === false ? ' !' : ''}
      </button>
      ${aiInstalled
        ? html`
            <button class="btn btn--sm" onClick=${onLoadModel} disabled=${state === 'loading'}>
              ${state === 'loading' ? html`<span class="btn__spin"></span>` : null}
              ${state === 'ready' ? 'Reload model' : 'Load model'}
            </button>
            ${state === 'ready'
              ? html`<button class="btn btn--sm btn--ghost" onClick=${onUnloadModel}>
                  Unload
                </button>`
              : null}
          `
        : null}
      <button
        class="btn btn--sm btn--ghost"
        onClick=${onStages}
        title="The lifecycle stages, what each needs, and what exists on this machine"
      >
        Stages
      </button>
      <button
        class="btn btn--sm btn--ghost"
        onClick=${onEvaluation}
        title="Score a split for parse rate, Dice and IoU"
      >
        Evaluate
      </button>
      <button class="btn btn--sm btn--ghost" onClick=${onAbout}>About</button>
      <a class="btn btn--sm btn--ghost" href="/api/docs" target="_blank" rel="noopener">API</a>
    </header>
  `;
}

/* ========================================================================== */
/*  Stage launcher — the platform is a lifecycle, not one screen               */
/* ========================================================================== */

/**
 * The notebook is three sequential stages (fine-tuning, HITL exchange, adapter
 * transfer) sitting on top of the dataset preprocessing that feeds them. Only the HITL
 * stage has a screen, which left the other three undiscoverable.
 *
 * Progressive disclosure hides *detail*, never the existence of a capability, and the
 * step-by-step variant is for linear flows — the wrong shape here, because any stage is
 * a legitimate entry point. So every stage is always listed with its readiness; the
 * ones that belong on a command line say so and give the command.
 */
function StagesModal({ onClose }) {
  // Stages are fetched, never declared here: api/stages.py is the single source of truth
  // for every title, description, command and readiness rule. Adding a stage there makes
  // it appear here with no JavaScript change.
  const [stages, setStages] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const onKey = (e) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  useEffect(() => {
    api
      .stages()
      .then((data) => setStages(data.stages))
      .catch((err) => setError(err.message));
  }, []);

  return html`
    <div class="modal" onClick=${onClose} role="dialog" aria-modal="true" aria-label="Stages">
      <div class="modal__panel" onClick=${(e) => e.stopPropagation()}>
        <div class="modal__head">
          <h2>Lifecycle stages</h2>
          <button class="modal__x" onClick=${onClose} aria-label="Close">×</button>
        </div>
        <p class="note">
          Every stage is an entry point — start from raw data, from a prepared corpus,
          from an adapter, or from your own revisions. Nothing here requires the others
          to have run first. Each row shows what exists on this machine right now.
        </p>
        ${error ? html`<div class="note note--danger">${error}</div>` : null}
        ${!stages && !error ? html`<div class="skeleton"></div>` : null}
        <ol class="stages">
          ${(stages || []).map(
            (stage) => html`
              <li class="stage-row${stage.here ? ' is-here' : ''}" key=${stage.id}>
                <div class="stage-row__body">
                  <div class="stage-row__head">
                    <span class="stage-row__name">${stage.title}</span>
                    <span class="badge${stage.ready ? '' : ' badge--uncal'}">
                      ${stage.here ? 'you are here' : stage.ready ? 'ready' : stage.blocked}
                    </span>
                  </div>
                  <p class="stage-row__blurb">${stage.summary}</p>
                  <ul class="stage-row__state">
                    ${(stage.state || []).map(
                      (fact) => html`
                        <li class=${fact.present ? 'is-present' : ''} key=${fact.label}>
                          <span class="stage-row__dot">${fact.present ? '●' : '○'}</span>
                          <b>${fact.label}</b>${fact.detail ? ` — ${fact.detail}` : ''}
                        </li>
                      `
                    )}
                  </ul>
                  ${stage.command
                    ? html`<pre class="stage-row__cmd">${stage.command}</pre>`
                    : null}
                </div>
              </li>
            `
          )}
        </ol>
      </div>
    </div>
  `;
}

/* ========================================================================== */
/*  Evaluation — score a split without leaving the workstation                 */
/* ========================================================================== */

function EvaluationModal({ caps, onClose }) {
  const [runs, setRuns] = useState(null);
  const [current, setCurrent] = useState(null);
  const [split, setSplit] = useState('test');
  const [source, setSource] = useState('');
  const [maxSamples, setMaxSamples] = useState(10);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const onKey = (e) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  const refresh = useCallback(
    () =>
      api
        .evaluationRuns()
        .then((data) => {
          setRuns(data.runs || []);
          setCurrent(data.current || null);
          return data.current;
        })
        .catch((err) => setError(err.message)),
    []
  );

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Poll only while a run is live, so an idle panel costs nothing.
  useEffect(() => {
    if (!current || ['completed', 'failed'].includes(current.state)) return undefined;
    const id = setInterval(refresh, 2000);
    return () => clearInterval(id);
  }, [current, refresh]);

  const start = useCallback(() => {
    setBusy(true);
    setError(null);
    api
      .startEvaluation({
        split: split || null,
        source: source || null,
        max_samples: Number(maxSamples) || 10,
      })
      .then(() => refresh())
      .catch((err) => setError(err.message))
      .finally(() => setBusy(false));
  }, [split, source, maxSamples, refresh]);

  const aiReady = Boolean(caps.tiers && caps.tiers.ai);
  const modelReady = caps.model && caps.model.state === 'ready';

  return html`
    <div class="modal" onClick=${onClose} role="dialog" aria-modal="true" aria-label="Evaluation">
      <div class="modal__panel" onClick=${(e) => e.stopPropagation()}>
        <div class="modal__head">
          <h2>Evaluation</h2>
          <button class="modal__x" onClick=${onClose} aria-label="Close">×</button>
        </div>
        <p class="note">
          ${'Scores a split for parse rate, Dice and IoU. Runs in the background — this '}
          ${'panel polls until it finishes. The same job is available as '}
          <code>atria evaluate</code>
        </p>

        ${!aiReady
          ? html`<div class="note note--danger">
              The AI tier is not installed, so runs cannot be started here. Existing runs
              are still listed below.
            </div>`
          : !modelReady
            ? html`<div class="note note--info">
                Load a model from the top bar before starting a run.
              </div>`
            : null}

        <div class="eval-form">
          <label>Split
            <select class="select" value=${split} onChange=${(e) => setSplit(e.target.value)}>
              <option value="test">test</option>
              <option value="val">val</option>
              <option value="train">train</option>
              <option value="">all</option>
            </select>
          </label>
          <label>Source
            <select class="select" value=${source} onChange=${(e) => setSource(e.target.value)}>
              <option value="">any</option>
              <option value="camus">camus</option>
              <option value="echonet">echonet</option>
            </select>
          </label>
          <label>Max frames
            <input class="input" type="number" min="1" max="5000" value=${maxSamples}
              onInput=${(e) => setMaxSamples(e.target.value)} />
          </label>
          <button class="btn btn--primary" disabled=${!aiReady || !modelReady || busy}
            onClick=${start}>
            ${busy ? html`<span class="btn__spin"></span>` : null} Start run
          </button>
        </div>

        ${error ? html`<div class="note note--danger">${error}</div>` : null}

        ${current
          ? html`<div class="note note--info">
              <b>${current.run_id}</b> — ${current.state}
              ${current.progress != null ? ` · ${Math.round(current.progress * 100)}%` : ''}
              ${current.message ? ` · ${current.message}` : ''}
            </div>`
          : null}

        <h3 class="eval-h3">Past runs</h3>
        ${runs && runs.length
          ? html`<table class="eval-table">
              <thead>
                <tr>
                  <th>run</th><th>adapter</th><th>split</th>
                  <th>n</th><th>parse</th><th>Dice</th><th>IoU</th>
                </tr>
              </thead>
              <tbody>
                ${runs.map((r) => {
                  // Metrics live under `summary`; a still-running record has none yet.
                  const s = r.summary || {};
                  const dice = s.dice || {};
                  const iou = s.iou || {};
                  return html`<tr key=${r.run_id}>
                    <td>${r.run_id}</td>
                    <td>${r.adapter || '—'}</td>
                    <td>${r.split || 'all'}${r.source ? ` · ${r.source}` : ''}</td>
                    <td>${s.total_samples != null ? s.total_samples : '—'}</td>
                    <td>
                      ${s.parse_rate_percent != null ? `${s.parse_rate_percent}%` : '—'}
                    </td>
                    <td>
                      ${dice.mean != null
                        ? `${dice.mean.toFixed(3)}${dice.std != null ? ` ±${dice.std.toFixed(3)}` : ''}`
                        : '—'}
                    </td>
                    <td>${iou.mean != null ? iou.mean.toFixed(3) : '—'}</td>
                  </tr>`;
                })}
              </tbody>
            </table>`
          : html`<p class="note">No runs yet.</p>`}

        <p class="note">
          ${'A full 200-frame evaluation of the official CAMUS test split lives in '}
          <code>outputs/benchmark/full200/</code>
          ${' — 200/200 parsed, median 4.98 mm point-to-curve — with a browsable '}
          ${'200-image QA gallery at '}
          <code>outputs/benchmark/full200/overlays/index.html</code>
        </p>
      </div>
    </div>
  `;
}

/* ========================================================================== */
/*  Weights — where the model actually comes from                              */
/* ========================================================================== */

function WeightsModal({ caps, onClose }) {
  useEffect(() => {
    const onKey = (event) => event.key === 'Escape' && onClose();
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  const weights = caps.weights || {};
  const base = weights.base || {};
  const adapters = weights.adapters || [];

  const row = (label, entry) => html`
    <div class="weights__row">
      <span class="weights__name">${label}</span>
      <span class="badge ${entry.ready ? 'badge--ok' : 'badge--flag'}">
        ${entry.source === 'local'
          ? 'local files'
          : entry.source === 'cache'
            ? 'hf cache'
            : 'not present'}
      </span>
      <span class="weights__detail">${entry.detail}</span>
    </div>
  `;

  return html`
    <div class="modal" onClick=${onClose} role="dialog" aria-modal="true" aria-label="Weights">
      <div class="modal__panel" onClick=${(e) => e.stopPropagation()}>
        <div class="modal__head">
          <h2 class="modal__title">Model weights</h2>
          <div style="flex:1"></div>
          <button class="btn btn--sm btn--ghost" onClick=${onClose}>Close</button>
        </div>
        <div class="modal__body">
          ${!caps.tiers.ai
            ? html`<div class="note">
                ${'The AI tier is not installed, so no weights are loaded and contours '}
                ${'cannot be predicted. Install it with '}
                <code>pip install -e ".[ai]"</code>${' (or launch with '}
                <code>run.sh --ai</code> / <code>run.cmd --ai</code>${') and restart. '}
                ${'Everything below still describes where weights would be read from.'}
              </div>`
            : null}

          <h3>Resolved right now</h3>
          <div class="weights">
            ${row('Base model', base)} ${adapters.map((a) => row(`Adapter · ${a.id}`, a))}
          </div>

          <h3>How resolution works</h3>
          <p>
            For each weight the application takes the first of these that exists, so a
            fully offline install is possible and no token is needed once the files are
            on disk:
          </p>
          <ol class="weights__order">
            <li>
              <strong>A folder in this project.</strong>${' Base model in '}
              <code>${weights.models_dir}</code>${', adapters in '}
              <code>${weights.adapters_dir}</code>${'. This is checked first and needs '}
              ${'no token and no network.'}
            </li>
            <li>
              <strong>The Hugging Face cache</strong>${' — anything already fetched by '}
              <code>hf download</code>${' or a previous run.'}
            </li>
            <li>
              <strong>The Hugging Face Hub.</strong> Both the base model and the adapters
              are <em>gated</em>: you must accept their terms while signed in and provide
              a token (<code>hf auth login</code>, or the <code>HF_TOKEN</code>
              environment variable).
            </li>
          </ol>

          <h3>Placing files by hand</h3>
          <p>
            Download each repository and drop it in, keeping the folder names below.
            Paths are relative to the project directory.
          </p>
          <pre class="weights__tree">
${[
  `${weights.models_dir}`,
  '  medgemma-1.5-4b-it/          <-  google/medgemma-1.5-4b-it',
  `${weights.adapters_dir}`,
  '  atria-echotrace-camus/       <-  The-Adimension/EchoTrace-MedGemma-CAMUS',
  '  atria-echotrace-echonet/     <-  The-Adimension/EchoTrace-MedGemma-EchoNet',
].join('\n')}</pre
          >
          <p>
            A base-model folder is recognised by its <code>config.json</code>; an adapter
            folder by its <code>adapter_config.json</code>${'. The names '}
            <code>google--medgemma-1.5-4b-it</code>${' and '}
            <code>google/medgemma-1.5-4b-it</code> are accepted too. Any other checkpoint
            works via <code>POST /api/model/load</code> with an absolute path.
          </p>
          <p>
            Hugging Face token currently detected:
            <strong>${weights.has_token ? 'yes' : 'no'}</strong>${weights.has_token
              ? ''
              : ' — not needed while the files above are present.'}
          </p>
        </div>
      </div>
    </div>
  `;
}

/* ========================================================================== */
/*  Case browser                                                               */
/* ========================================================================== */

/**
 * Upload entry point for the HITL stage (notebook_as_py.txt L1404-1422).
 *
 * The native `<input type="file">` is the functional core — it stays in the tab order
 * (opacity, not `display:none`) so keyboard and screen-reader users get the standard
 * file picker. The drop zone is an enhancement layered over it via the DataTransfer
 * API, never the only route in.
 */
function UploadPanel({ view, onView, onUpload, uploading, frames }) {
  const [over, setOver] = useState(null);
  return html`
    <div class="section">
      Your own frame <span class="section__rule"></span>
    </div>
    <div class="upload">
      <select
        class="select"
        aria-label="Apical view for uploaded frames"
        value=${view}
        onChange=${(e) => onView(e.target.value)}
      >
        <option value="4CH">Apical 4-chamber</option>
        <option value="2CH">Apical 2-chamber</option>
        <option value="">View not specified</option>
      </select>
      ${INSTANTS.map((instant) => {
        const frame = frames && frames[instant];
        const busy = uploading[instant];
        return html`
          <label
            class="upload__drop${over === instant ? ' is-over' : ''}${frame ? ' is-set' : ''}"
            key=${instant}
            onDragOver=${(e) => {
              e.preventDefault();
              setOver(instant);
            }}
            onDragLeave=${() => setOver(null)}
            onDrop=${(e) => {
              e.preventDefault();
              setOver(null);
              onUpload(instant, e.dataTransfer.files && e.dataTransfer.files[0]);
            }}
          >
            <input
              type="file"
              accept="image/png,image/jpeg,image/bmp,image/tiff"
              aria-label=${`Upload the ${instant} frame`}
              disabled=${busy}
              onChange=${(e) => {
                onUpload(instant, e.target.files && e.target.files[0]);
                e.target.value = ''; // allow re-picking the same file
              }}
            />
            <span class="upload__instant">${instant}</span>
            <span class="upload__hint" role=${busy ? 'status' : null}>
              ${busy
                ? 'Uploading…'
                : frame
                  ? `${frame.image_w}×${frame.image_h} — replace`
                  : 'Choose a file or drop one here'}
            </span>
          </label>
        `;
      })}
      <p class="upload__note">
        Uploaded frames carry no pixel spacing, so areas stay in px² until you enter a
        spacing. Any resolution is accepted — contours are normalised, never resized.
      </p>
    </div>
  `;
}

function CaseBrowser(props) {
  const { cases, filters, onFilter, activeKey, onSelect } = props;
  return html`
    <aside class="rail rail--left">
      <${UploadPanel}
        view=${props.uploadView}
        onView=${props.onUploadView}
        onUpload=${props.onUpload}
        uploading=${props.uploading}
        frames=${props.uploadFrames}
      />
      <div class="section">
        Cases <span class="section__rule"></span>
        <span class="section__count">${cases ? cases.count : '—'}</span>
      </div>
      <div class="filters">
        <select
          class="select"
          aria-label="Filter by dataset"
          value=${filters.source}
          onChange=${(e) => onFilter({ ...filters, source: e.target.value })}
        >
          <option value="">All datasets</option>
          ${(cases ? cases.sources : []).map(
            (source) => html`<option value=${source}>${source}</option>`
          )}
        </select>
        <select
          class="select"
          aria-label="Filter by view"
          value=${filters.view}
          onChange=${(e) => onFilter({ ...filters, view: e.target.value })}
        >
          <option value="">All views</option>
          ${(cases ? cases.views : []).map((view) => html`<option value=${view}>${view}</option>`)}
        </select>
      </div>
      <div class="rail__body">
        ${!cases
          ? html`<div>
              ${[0, 1, 2, 3, 4, 5].map(() => html`<div class="skeleton"></div>`)}
            </div>`
          : cases.count === 0
            ? html`<div class="empty">
                <div class="empty__mark"></div>
                <div>No cases match these filters.</div>
              </div>`
            : html`<div class="caselist">
                ${cases.cases.map(
                  (item) => html`
                    <button
                      class="case"
                      key=${item.case_key}
                      aria-current=${String(item.case_key === activeKey)}
                      aria-label=${`${item.case_id}, ${item.source} ${item.view}, instants ${item.instants.join(' and ')}`}
                      onClick=${() => onSelect(item.case_key)}
                    >
                      <div class="case__thumbs">
                        ${INSTANTS.map((instant) =>
                          item.frames[instant]
                            ? html`<img
                                class="case__thumb"
                                src=${item.frames[instant].image_url}
                                alt="${item.case_key} ${instant}"
                                loading="lazy"
                              />`
                            : html`<span class="case__thumb"></span>`
                        )}
                      </div>
                      <div>
                        <div class="case__id">${item.case_id}</div>
                        <div class="case__meta">
                          <span class="badge badge--${item.source}">${item.source}</span>
                          <span class="badge">${item.view}</span>
                          ${item.calibration_source === 'unknown'
                            ? html`<span class="badge badge--uncal" title="No pixel spacing published"
                                >uncal</span
                              >`
                            : null}
                          ${item.has_la
                            ? html`<span
                                class="badge"
                                title="A left-atrium reference contour is also available for this case. The workspace traces the left ventricle by default; switch Structure to LA to use it."
                                >+la ref</span
                              >`
                            : null}
                          ${(item.integrity_flags || []).length
                            ? html`<span
                                class="badge badge--flag"
                                title="This case's ES trace encloses MORE area than its ED trace, which is physiologically impossible — the ED/ES labels for this case are almost certainly transposed in the source data. Reported as-is, never silently corrected."
                                >${'es > ed'}</span
                              >`
                            : null}
                          ${item.ef != null
                            ? html`<span class="badge">ef ${item.ef.toFixed(0)}</span>`
                            : null}
                        </div>
                      </div>
                    </button>
                  `
                )}
              </div>`}
      </div>
      <${RevisionList} revisions=${props.revisions} onOpen=${props.onOpenRevision} />
    </aside>
  `;
}

/**
 * Saved revisions, newest first. Reopening one restores both polygons into the editor,
 * and this is also the list `atria export-corpus` turns into a trainable dataset.
 */
function RevisionList({ revisions, onOpen }) {
  if (!revisions || !revisions.length) return null;
  return html`
    <div class="section">
      Saved revisions <span class="section__rule"></span>
      <span class="section__count">${revisions.length}</span>
    </div>
    <div class="revlist">
      ${revisions.slice(0, 12).map(
        (item) => html`
          <button
            class="rev"
            key=${item.revision_id}
            onClick=${() => onOpen(item.revision_id)}
            title=${`${item.revision_id}${item.notes ? ` — ${item.notes}` : ''}`}
          >
            <span class="rev__case">${item.case || item.revision_id}</span>
            <span class="rev__meta">
              <span>${(item.timestamp_utc || '').replace('T', ' ').replace('Z', '')}</span>
              <span class="badge"
                >${(item.provenance && item.provenance.target_structure) || 'LV'}</span
              >
              ${item.fac_percent != null
                ? html`<span class="badge">fac ${item.fac_percent.toFixed(0)}</span>`
                : null}
            </span>
          </button>
        `
      )}
    </div>
  `;
}

/* ========================================================================== */
/*  Centre stage                                                               */
/* ========================================================================== */

function Stage(props) {
  const {
    caps,
    activeCase,
    loadingCase,
    structure,
    onStructure,
    adapter,
    onAdapter,
    promptVariant,
    onPromptVariant,
    tracings,
    panels,
    onPanel,
    onTracingChange,
    onPredict,
    onPredictBoth,
    predicting,
    agreement,
    aiReady,
    onSeedReference,
  } = props;

  const busy = predicting.ED || predicting.ES;

  return html`
    <main class="stage">
      <div class="toolbar">
        <div class="toolbar__group">
          <span class="toolbar__label">Structure</span>
          <div class="seg">
            ${['LV', 'LA'].map(
              (value) => html`
                <button
                  aria-pressed=${String(structure === value)}
                  disabled=${value === 'LA' && activeCase && !activeCase.has_la}
                  title=${value === 'LA'
                    ? activeCase && !activeCase.has_la
                      ? 'This dataset provides no left-atrium reference for this case'
                      : 'The shipped adapters were tuned on LV only. LA is unsupported — any contour is out of distribution and must not be read as a validated prediction.'
                    : ''}
                  onClick=${() => onStructure(value)}
                >
                  ${value}
                </button>
              `
            )}
          </div>
          ${structure === 'LA'
            ? html`<span
                class="badge badge--flag"
                title="The shipped adapters were tuned on LV only. LA is unsupported — any contour is out of distribution and must not be read as a validated prediction."
                >LA unsupported · adapters tuned on LV only</span
              >`
            : null}
        </div>

        <div class="toolbar__group">
          <span class="toolbar__label">Adapter</span>
          <select class="select" value=${adapter} onChange=${(e) => onAdapter(e.target.value)}>
            ${caps.adapters.map(
              (item) => html`<option
                value=${item.id}
                title=${`${item.label}${item.available_locally ? ' — checkpoint present locally' : item.repo ? ' — gated download required' : ''}`}
              >
                ${item.id}${item.available_locally ? ' ·local' : ''}
              </option>`
            )}
          </select>
        </div>

        <div class="toolbar__group">
          <span class="toolbar__label">Prompt</span>
          <select
            class="select"
            value=${promptVariant}
            onChange=${(e) => onPromptVariant(e.target.value)}
            title="The adapters were fine-tuned with the training template, which names the view and cardiac instant."
          >
            <option value="">auto (training)</option>
            <option value="training">training</option>
            <option value="generic">generic</option>
          </select>
        </div>

        <div class="toolbar__spacer" style="flex:1"></div>

        <button
          class="btn btn--primary"
          onClick=${onPredictBoth}
          disabled=${!activeCase || busy || !aiReady}
          title=${!aiReady ? 'Load the model first' : 'Run inference on both frames'}
        >
          ${busy ? html`<span class="btn__spin"></span>` : null} Trace ED + ES
        </button>
      </div>

      ${!activeCase
        ? html`<div class="empty" style="flex:1">
            <div class="empty__mark"></div>
            <div>
              ${loadingCase ? 'Loading case…' : 'Select a case from the left to begin tracing.'}
            </div>
          </div>`
        : html`<div class="editors">
            ${INSTANTS.map((instant) =>
              activeCase.frames[instant]
                ? html`<${EditorPane}
                    key=${`${activeCase.case_key}:${instant}`}
                    instant=${instant}
                    frame=${activeCase.frames[instant]}
                    structure=${structure}
                    tracing=${tracings[instant]}
                    panel=${panels[instant]}
                    onPanel=${(panel) => onPanel(instant, panel)}
                    onChange=${(revision) => onTracingChange(instant, revision)}
                    onPredict=${() => onPredict(instant)}
                    predicting=${predicting[instant]}
                    agreement=${agreement[instant]}
                    aiReady=${aiReady}
                    onSeedReference=${() => onSeedReference(instant)}
                  />`
                : html`<div class="editor">
                    <div class="editor__head"><span class="editor__phase">${instant}</span></div>
                    <div class="empty" style="flex:1">
                      <div>No ${instant} frame for this case.</div>
                    </div>
                  </div>`
            )}
          </div>`}
    </main>
  `;
}

/* ========================================================================== */
/*  One frame editor                                                           */
/* ========================================================================== */

function EditorPane(props) {
  const {
    instant,
    frame,
    structure,
    tracing,
    panel,
    onPanel,
    onChange,
    onPredict,
    predicting,
    agreement,
    aiReady,
    onSeedReference,
  } = props;

  const canvasRef = useRef(null);
  const editorRef = useRef(null);
  const [cursor, setCursor] = useState(null);
  const [drawMode, setDrawMode] = useState(false);
  const [, forceRender] = useState(0);
  const [imageError, setImageError] = useState(null);

  const reference = structure === 'LV' ? frame.lv_polygon : frame.la_polygon;

  // Create the editor once per mounted pane.
  useEffect(() => {
    if (!canvasRef.current) return undefined;
    const editor = new ContourEditor(canvasRef.current, {
      onChange: (revision) => {
        onChange(revision);
        forceRender((n) => n + 1); // refresh undo/redo affordances
      },
      onCursor: setCursor,
      onSelect: () => forceRender((n) => n + 1),
    });
    editorRef.current = editor;
    editor
      .setImage(frame.image_url)
      .then(() => forceRender((n) => n + 1))
      .catch((error) => setImageError(error.message));
    return () => {
      editor.destroy();
      editorRef.current = null;
    };
    // eslint-disable-next-line
  }, [frame.image_url]);

  // Push polygon state down whenever it changes outside the canvas (predict, seed).
  useEffect(() => {
    const editor = editorRef.current;
    if (!editor) return;
    const current = JSON.stringify(editor.getRevision());
    const next = JSON.stringify(tracing.revision.map(([y, x]) => [Math.round(y), Math.round(x)]));
    editor.setPolygons(
      { model: tracing.model, revision: tracing.revision, groundTruth: reference || null },
      current !== next
    );
  }, [tracing.model, tracing.revision, reference]);

  useEffect(() => {
    if (editorRef.current) editorRef.current.setPanel(panel);
  }, [panel]);

  const editor = editorRef.current;
  const toggleDraw = () => {
    if (!editor) return;
    const next = !drawMode;
    // Vertices can only be placed on an editable panel, so entering draw mode from
    // Original or Model would otherwise silently do nothing.
    if (next && panel !== PANEL.REVISION && panel !== PANEL.OVERLAY) onPanel(PANEL.REVISION);
    setDrawMode(editor.setDrawMode(next));
  };

  return html`
    <section class="editor">
      <div class="editor__head">
        <span class="editor__phase">${instant}</span>
        <span class="editor__stem">${frame.stem}</span>
        <div class="btnrow">
          <button
            class="btn btn--sm"
            onClick=${onPredict}
            disabled=${predicting || !aiReady}
            title=${!aiReady ? 'Load the model first' : `Run inference on the ${instant} frame`}
          >
            ${predicting ? html`<span class="btn__spin"></span>` : null} Trace
          </button>
        </div>
      </div>

      <div class="viewport">
        <div class="viewport__ticks viewport__ticks--top"></div>
        <div class="viewport__ticks viewport__ticks--left"></div>
        <canvas class="viewport__canvas" ref=${canvasRef} aria-label="${instant} frame editor"></canvas>
        ${imageError
          ? html`<div class="viewport__empty"><div>${imageError}</div></div>`
          : drawMode
            ? html`<div class="viewport__hint">
                Click to place vertices · Draw to finish
              </div>`
            : !tracing.revision.length && panel !== PANEL.MODEL
              ? html`<div class="viewport__empty">
                  <div>No contour yet</div>
                  <div style="display:flex;gap:6px;flex-wrap:wrap;justify-content:center">
                    <button
                      class="btn btn--sm"
                      onClick=${onPredict}
                      disabled=${predicting || !aiReady}
                    >
                      Trace with AI
                    </button>
                    ${reference && reference.length
                      ? html`<button class="btn btn--sm" onClick=${onSeedReference}>
                          Start from reference
                        </button>`
                      : null}
                    <button class="btn btn--sm" onClick=${toggleDraw}>Draw manually</button>
                  </div>
                </div>`
              : null}
        <div class="viewport__hud">
          <span>${frame.image_w}×${frame.image_h}</span>
          <span>${tracing.revision.length} pts</span>
          ${cursor ? html`<span>y ${cursor.y} x ${cursor.x}</span>` : null}
          ${agreement ? html`<span>dice ${agreement.dice}</span>` : null}
        </div>
      </div>

      <div class="editor__foot">
        <div class="seg">
          ${PANEL_LABELS.map(
            ([value, label]) => html`
              <button aria-pressed=${String(panel === value)} onClick=${() => onPanel(value)}>
                ${label}
              </button>
            `
          )}
        </div>
      </div>

      <div class="editor__foot" style="border-top:0;padding-top:0">
        <button
          class="btn btn--sm"
          aria-pressed=${String(drawMode)}
          onClick=${toggleDraw}
          title="Click on the image to append vertices"
        >
          ${drawMode ? 'Drawing…' : 'Draw'}
        </button>
        <button
          class="btn btn--sm"
          onClick=${() => editor && editor.undo()}
          disabled=${!editor || !editor.canUndo}
          title="Undo (Ctrl+Z)"
        >
          Undo
        </button>
        <button
          class="btn btn--sm"
          onClick=${() => editor && editor.redo()}
          disabled=${!editor || !editor.canRedo}
          title="Redo (Ctrl+Shift+Z)"
        >
          Redo
        </button>
        <button
          class="btn btn--sm"
          onClick=${() => editor && editor.resetToModel()}
          disabled=${!tracing.model.length}
          title="Discard edits and restore the model proposal"
        >
          Reset
        </button>
        <button
          class="btn btn--sm btn--ghost"
          onClick=${() => editor && editor.resetView()}
          title="Fit to viewport (0)"
        >
          Fit
        </button>
      </div>

      <div class="legend">
        <span class="legend__item"
          ><span class="legend__swatch" style="background:var(--model)"></span>Model</span
        >
        <span class="legend__item"
          ><span class="legend__swatch" style="background:var(--user)"></span>Revision</span
        >
        ${reference && reference.length
          ? html`<span class="legend__item"
              ><span class="legend__swatch" style="background:var(--truth)"></span>Reference</span
            >`
          : null}
        <span style="margin-left:auto">drag · dbl-click add · alt-click remove · wheel zoom</span>
      </div>
    </section>
  `;
}

/* ========================================================================== */
/*  Metrics rail                                                               */
/* ========================================================================== */

function Metric({ label, value, unit, hero, na }) {
  return html`
    <div class="metric ${hero ? 'metric--hero' : ''} ${na ? 'metric--na' : ''}">
      <span class="metric__label">${label}</span>
      <span class="metric__value"
        >${value}${unit ? html`<span class="metric__unit">${unit}</span>` : null}</span
      >
    </div>
  `;
}

function MetricsRail(props) {
  const {
    activeCase,
    metrics,
    tracings,
    agreement,
    spacing,
    onSpacing,
    notes,
    onNotes,
    onSave,
    saving,
    canSave,
    lastRevision,
  } = props;

  const uncalibrated = metrics && metrics.calibration && metrics.calibration.source === 'unknown';
  const flags = (activeCase && activeCase.integrity_flags) || [];

  const fmt = (value, digits = 2) =>
    value == null ? '—' : Number(value).toFixed(digits);

  // "not calibrated" must mean exactly that. An untraced frame also has no physical
  // area, but for a different reason, and conflating the two misreads as a
  // calibration problem the clinician cannot fix.
  const physical = (phase, key, unit) => {
    if (!phase || phase.vertices < 3) return { value: '—', unit: '', na: true };
    if (phase[key] == null) return { value: 'not calibrated', unit: '', na: true };
    return { value: fmt(phase[key]), unit, na: false };
  };

  return html`
    <aside class="rail rail--right">
      <div class="rail__body">
        <div class="section">Measurements <span class="section__rule"></span></div>

        ${!metrics
          ? html`<div class="empty">
              <div>Trace a frame to see live measurements.</div>
            </div>`
          : html`
              <div class="metrics">
                <${Metric}
                  label="FAC"
                  hero
                  value=${metrics.fac_percent == null ? '—' : fmt(metrics.fac_percent, 1)}
                  unit=${metrics.fac_percent == null ? '' : '%'}
                />
                <${Metric} label="ED area" value=${fmt(metrics.ed.area_px, 0)} unit="px²" />
                <${Metric} label="ES area" value=${fmt(metrics.es.area_px, 0)} unit="px²" />
                <${Metric} label="ED physical" ...${physical(metrics.ed, 'area_cm2', 'cm²')} />
                <${Metric} label="ES physical" ...${physical(metrics.es, 'area_cm2', 'cm²')} />
                <${Metric}
                  label="ED perimeter"
                  ...${physical(metrics.ed, 'perimeter_cm', 'cm')}
                />
                <${Metric} label="ED vertices" value=${tracings.ED.revision.length} />
                <${Metric} label="ES vertices" value=${tracings.ES.revision.length} />
                ${INSTANTS.map((instant) =>
                  agreement[instant]
                    ? html`<${Metric}
                        label="${instant} dice vs ref"
                        value=${fmt(agreement[instant].dice, 3)}
                      />`
                    : null
                )}
              </div>
            `}

        ${uncalibrated
          ? html`<div class="note">
              ${metrics.calibration.note} Enter a pixel spacing below to compute physical areas.
            </div>`
          : null}
        ${flags.includes('es_area_exceeds_ed')
          ? html`<div class="note note--danger">
              This case's ES reference trace encloses more area than its ED trace, which is
              physiologically impossible and suggests the ED/ES labels are transposed in the
              source data. Measurements are reported exactly as computed and are not corrected.
            </div>`
          : null}

        <div class="section">Calibration <span class="section__rule"></span></div>
        <div class="filters">
          <div>
            <div class="field__label" style="padding-bottom:3px">spacing h mm/px</div>
            <input
              class="input input--mono"
              type="number"
              step="0.001"
              min="0"
              placeholder=${metrics && metrics.calibration.spacing_h != null
                ? Number(metrics.calibration.spacing_h).toFixed(4).replace(/0+$/, '')
                : 'unknown'}
              value=${spacing.h}
              onInput=${(e) => onSpacing({ ...spacing, h: e.target.value })}
            />
          </div>
          <div>
            <div class="field__label" style="padding-bottom:3px">spacing w mm/px</div>
            <input
              class="input input--mono"
              type="number"
              step="0.001"
              min="0"
              placeholder=${metrics && metrics.calibration.spacing_w != null
                ? Number(metrics.calibration.spacing_w).toFixed(4).replace(/0+$/, '')
                : 'unknown'}
              value=${spacing.w}
              onInput=${(e) => onSpacing({ ...spacing, w: e.target.value })}
            />
          </div>
        </div>

        <div class="section">Clinician notes <span class="section__rule"></span></div>
        <div class="field">
          <textarea
            class="textarea"
            placeholder="Findings, deviations from the model proposal, image quality…"
            value=${notes}
            onInput=${(e) => onNotes(e.target.value)}
          ></textarea>
        </div>

        <div class="btnrow">
          <button class="btn btn--commit btn--block" onClick=${onSave} disabled=${!canSave || saving}>
            ${saving ? html`<span class="btn__spin"></span>` : null} Save revision
          </button>
        </div>

        ${lastRevision
          ? html`
              <div class="section">
                Exports <span class="section__rule"></span>
                <span class="section__count">${lastRevision.revision_id}</span>
              </div>
              <div class="exports">
                ${Object.keys(lastRevision.files)
                  .sort()
                  .map((name) => {
                    const ext = name.split('.').pop();
                    const href =
                      ext === 'zip'
                        ? lastRevision.download_url
                        : `/api/revisions/${lastRevision.revision_id}/files/${name}`;
                    return html`<a class="export" href=${href} key=${name} download>
                      <span class="export__ext">${ext}</span>${name}
                    </a>`;
                  })}
              </div>
            `
          : null}
      </div>
    </aside>
  `;
}

/* ========================================================================== */
/*  About / disclaimers                                                        */
/* ========================================================================== */

function AboutModal({ data, caps, onClose }) {
  useEffect(() => {
    const onKey = (event) => event.key === 'Escape' && onClose();
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  return html`
    <div class="modal" onClick=${onClose} role="dialog" aria-modal="true" aria-label="About">
      <div class="modal__panel" onClick=${(e) => e.stopPropagation()}>
        <div class="modal__head">
          <h2 class="modal__title">ATRIA EchoTrace ${caps.version}</h2>
          <div style="flex:1"></div>
          <button class="btn btn--sm btn--ghost" onClick=${onClose}>Close</button>
        </div>
        <div class="modal__body">
          ${!data
            ? html`<p>Loading…</p>`
            : html`
                <h3>Base model &amp; adapters</h3>
                <p>
                  Base: <code>${caps.base_model_id}</code>. Adapters:
                  ${caps.adapters
                    .filter((a) => a.repo)
                    .map(
                      (a) => html`<span
                        >${' '}<a href="https://huggingface.co/${a.repo}" target="_blank" rel="noopener"
                          >${a.repo}</a
                        >${a.doi ? ` (DOI ${a.doi})` : ''}${' '}</span
                      >`
                    )}
                </p>

                ${data.disclaimers.map(
                  (item) => html`
                    <h3>${item.title}</h3>
                    <p>${item.text}</p>
                  `
                )}

                <h3>DEITY principles framework</h3>
                <div class="deity">
                  ${data.deity.map(
                    (row) => html`
                      <div class="deity__row">
                        <span class="deity__pillar">${row.pillar}</span>
                        <ul class="deity__points">
                          ${row.points.map((point) => html`<li>${point}</li>`)}
                        </ul>
                      </div>
                    `
                  )}
                </div>

                <h3>Citations</h3>
                ${data.citations.map(
                  (citation) => html`
                    <p>
                      <strong>${citation.label}.</strong> ${citation.text}${' '}
                      <a href=${citation.url} target="_blank" rel="noopener">${citation.url}</a>
                    </p>
                  `
                )}
              `}
        </div>
      </div>
    </div>
  `;
}

render(html`<${App} />`, document.getElementById('root'));
