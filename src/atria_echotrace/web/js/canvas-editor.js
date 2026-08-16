/**
 * Interactive contour editor on a 2D canvas.
 *
 * Production form of the notebook's HITL canvas (notebook_as_py.txt L1426-1544),
 * which drew the frame plus polygon and dragged the nearest vertex on mousedown.
 * This keeps that core interaction and adds what a clinical tool needs: zoom/pan,
 * vertex insert/delete, undo/redo, keyboard nudging, device-pixel-ratio-correct
 * rendering, and the four comparison panels.
 *
 * Coordinate systems, kept strictly separate:
 *   normalised  [y, x] in [0, NORM_SCALE]  — API/model/disk format
 *   image       (x, y) in image pixels
 *   screen      (x, y) in CSS pixels on the canvas
 *
 * Nothing here assumes a vertex count, a winding direction, or that the contour is
 * explicitly closed: the two published adapters differ on all three (RESEARCH.md §0.5).
 */

export const NORM_SCALE = 1000;

export const PANEL = {
  ORIGINAL: 'original',
  MODEL: 'model',
  REVISION: 'revision',
  OVERLAY: 'overlay',
};

const COLORS = {
  model: 'oklch(65% 0.203 22)',
  modelFill: 'oklch(65% 0.203 22 / 0.16)',
  user: 'oklch(74% 0.176 150)',
  userFill: 'oklch(74% 0.176 150 / 0.16)',
  truth: 'oklch(76% 0.13 233)',
  truthFill: 'oklch(76% 0.13 233 / 0.12)',
  handle: 'oklch(97% 0.004 252)',
  active: 'oklch(80% 0.148 78)',
};

const HANDLE_RADIUS = 4.5;
const HIT_RADIUS = 10;
const MAX_HISTORY = 60;

export class ContourEditor {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {{onChange?: Function, onCursor?: Function, onSelect?: Function}} handlers
   */
  constructor(canvas, handlers = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.handlers = handlers;

    this.image = null;
    this.imageW = 0;
    this.imageH = 0;

    this.polygons = { model: [], revision: [], groundTruth: null };
    this.panel = PANEL.REVISION;

    this.zoom = 1;
    this.panX = 0;
    this.panY = 0;
    this.fitScale = 1;

    this.activeIndex = -1;
    this.draggingIndex = -1;
    this.isPanning = false;
    this.panOrigin = null;
    /** When set, a plain click appends a vertex — used to trace a frame from scratch. */
    this.drawMode = false;

    this.history = [];
    this.future = [];

    this._bind();
    this._observeResize();
  }

  // ------------------------------------------------------------------ setup
  _bind() {
    this._onPointerDown = this._handlePointerDown.bind(this);
    this._onPointerMove = this._handlePointerMove.bind(this);
    this._onPointerUp = this._handlePointerUp.bind(this);
    this._onWheel = this._handleWheel.bind(this);
    this._onDblClick = this._handleDoubleClick.bind(this);
    this._onKeyDown = this._handleKeyDown.bind(this);
    this._onContextMenu = this._handleContextMenu.bind(this);
    this._onPointerLeave = this._handlePointerLeave.bind(this);

    this.canvas.addEventListener('pointerdown', this._onPointerDown);
    this.canvas.addEventListener('pointermove', this._onPointerMove);
    this.canvas.addEventListener('pointerup', this._onPointerUp);
    this.canvas.addEventListener('pointercancel', this._onPointerUp);
    this.canvas.addEventListener('pointerleave', this._onPointerLeave);
    this.canvas.addEventListener('wheel', this._onWheel, { passive: false });
    this.canvas.addEventListener('dblclick', this._onDblClick);
    this.canvas.addEventListener('keydown', this._onKeyDown);
    this.canvas.addEventListener('contextmenu', this._onContextMenu);
    this.canvas.tabIndex = 0;
    // Focusable without an accessible name announces as a bare "canvas". The editor is
    // keyboard-operable (Tab between vertices, arrows to nudge, Ctrl+Z), so say so.
    this.canvas.setAttribute('role', 'application');
    this.canvas.setAttribute(
      'aria-label',
      'Contour editor. Tab steps through vertices, arrow keys nudge the selected ' +
        'vertex (Shift for ten pixels), Ctrl+Z undoes.'
    );
  }

  _observeResize() {
    this._resizeObserver = new ResizeObserver(() => this._resize());
    this._resizeObserver.observe(this.canvas.parentElement || this.canvas);
    this._resize();
  }

  destroy() {
    this.canvas.removeEventListener('pointerdown', this._onPointerDown);
    this.canvas.removeEventListener('pointermove', this._onPointerMove);
    this.canvas.removeEventListener('pointerup', this._onPointerUp);
    this.canvas.removeEventListener('pointercancel', this._onPointerUp);
    this.canvas.removeEventListener('pointerleave', this._onPointerLeave);
    this.canvas.removeEventListener('wheel', this._onWheel);
    this.canvas.removeEventListener('dblclick', this._onDblClick);
    this.canvas.removeEventListener('keydown', this._onKeyDown);
    this.canvas.removeEventListener('contextmenu', this._onContextMenu);
    if (this._resizeObserver) this._resizeObserver.disconnect();
    this.image = null;
  }

  _resize() {
    const rect = this.canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const dpr = window.devicePixelRatio || 1;
    this.cssW = rect.width;
    this.cssH = rect.height;
    this.canvas.width = Math.round(rect.width * dpr);
    this.canvas.height = Math.round(rect.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this._computeFit();
    this.render();
  }

  _computeFit() {
    if (!this.imageW || !this.imageH || !this.cssW) {
      this.fitScale = 1;
      return;
    }
    this.fitScale = Math.min(this.cssW / this.imageW, this.cssH / this.imageH);
  }

  // ------------------------------------------------------------------ state
  /** Load a frame. Resolves once the bitmap is decoded and drawn. */
  async setImage(url) {
    if (!url) {
      this.image = null;
      this.imageW = this.imageH = 0;
      this.render();
      return;
    }
    const image = new Image();
    image.decoding = 'async';
    await new Promise((resolve, reject) => {
      image.onload = resolve;
      image.onerror = () => reject(new Error(`Could not load frame image: ${url}`));
      image.src = url;
    });
    this.image = image;
    this.imageW = image.naturalWidth;
    this.imageH = image.naturalHeight;
    this.resetView();
  }

  /**
   * Replace the polygons. `revision` is the editable one.
   * @param {{model?: Array, revision?: Array, groundTruth?: Array|null}} polygons
   * @param {boolean} clearHistory
   */
  setPolygons(polygons, clearHistory = true) {
    if (polygons.model !== undefined) this.polygons.model = clone(polygons.model);
    if (polygons.revision !== undefined) this.polygons.revision = clone(polygons.revision);
    if (polygons.groundTruth !== undefined) {
      this.polygons.groundTruth = polygons.groundTruth ? clone(polygons.groundTruth) : null;
    }
    if (clearHistory) {
      this.history = [];
      this.future = [];
    }
    this.activeIndex = -1;
    this.render();
  }

  setPanel(panel) {
    this.panel = panel;
    this.render();
  }

  /** The clinician's polygon, normalised and integer-rounded. */
  getRevision() {
    return this.polygons.revision.map(([y, x]) => [Math.round(y), Math.round(x)]);
  }

  get isEditable() {
    return this.panel === PANEL.REVISION || this.panel === PANEL.OVERLAY;
  }

  get vertexCount() {
    return this.polygons.revision.length;
  }

  resetView() {
    this.zoom = 1;
    this.panX = 0;
    this.panY = 0;
    this._computeFit();
    this.render();
  }

  // ---------------------------------------------------------------- history
  _snapshot() {
    this.history.push(clone(this.polygons.revision));
    if (this.history.length > MAX_HISTORY) this.history.shift();
    this.future = [];
  }

  undo() {
    if (!this.history.length) return false;
    this.future.push(clone(this.polygons.revision));
    this.polygons.revision = this.history.pop();
    this.activeIndex = -1;
    this.render();
    this._emitChange();
    return true;
  }

  redo() {
    if (!this.future.length) return false;
    this.history.push(clone(this.polygons.revision));
    this.polygons.revision = this.future.pop();
    this.activeIndex = -1;
    this.render();
    this._emitChange();
    return true;
  }

  get canUndo() {
    return this.history.length > 0;
  }
  get canRedo() {
    return this.future.length > 0;
  }

  /** Restore the model's proposal, discarding edits (kept undoable). */
  resetToModel() {
    if (!this.polygons.model.length) return false;
    this._snapshot();
    this.polygons.revision = clone(this.polygons.model);
    this.render();
    this._emitChange();
    return true;
  }

  /** Adopt the dataset reference trace as the starting point. */
  copyGroundTruth() {
    if (!this.polygons.groundTruth || !this.polygons.groundTruth.length) return false;
    this._snapshot();
    this.polygons.revision = clone(this.polygons.groundTruth);
    this.render();
    this._emitChange();
    return true;
  }

  _emitChange() {
    if (this.handlers.onChange) this.handlers.onChange(this.getRevision());
  }

  // ------------------------------------------------------------- transforms
  _origin() {
    const scale = this.fitScale * this.zoom;
    return {
      scale,
      x: (this.cssW - this.imageW * scale) / 2 + this.panX,
      y: (this.cssH - this.imageH * scale) / 2 + this.panY,
    };
  }

  /** normalised [y, x] -> screen {x, y} */
  _toScreen([y, x]) {
    const o = this._origin();
    return {
      x: o.x + (x / NORM_SCALE) * this.imageW * o.scale,
      y: o.y + (y / NORM_SCALE) * this.imageH * o.scale,
    };
  }

  /** screen {x, y} -> normalised [y, x], clamped into range */
  _toNormalised(sx, sy) {
    const o = this._origin();
    const imgX = (sx - o.x) / o.scale;
    const imgY = (sy - o.y) / o.scale;
    return [
      clamp((imgY / this.imageH) * NORM_SCALE, 0, NORM_SCALE),
      clamp((imgX / this.imageW) * NORM_SCALE, 0, NORM_SCALE),
    ];
  }

  _pointerPosition(event) {
    const rect = this.canvas.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  }

  // ----------------------------------------------------------- interactions
  _findVertex(sx, sy) {
    let best = -1;
    let bestDistance = HIT_RADIUS;
    this.polygons.revision.forEach((point, index) => {
      const screen = this._toScreen(point);
      const distance = Math.hypot(screen.x - sx, screen.y - sy);
      if (distance <= bestDistance) {
        bestDistance = distance;
        best = index;
      }
    });
    return best;
  }

  /** Nearest edge to a screen point, with the projected insertion position. */
  _findEdge(sx, sy) {
    const polygon = this.polygons.revision;
    if (polygon.length < 2) return null;
    let best = null;
    for (let i = 0; i < polygon.length; i += 1) {
      const a = this._toScreen(polygon[i]);
      const b = this._toScreen(polygon[(i + 1) % polygon.length]);
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const lengthSq = dx * dx + dy * dy;
      if (lengthSq === 0) continue;
      let t = ((sx - a.x) * dx + (sy - a.y) * dy) / lengthSq;
      t = clamp(t, 0, 1);
      const px = a.x + t * dx;
      const py = a.y + t * dy;
      const distance = Math.hypot(px - sx, py - sy);
      if (!best || distance < best.distance) {
        best = { index: i, distance, x: px, y: py };
      }
    }
    return best;
  }

  _handlePointerDown(event) {
    if (!this.image) return;
    this.canvas.focus({ preventScroll: true });

    // Middle button, or space held, pans.
    if (event.button === 1 || event.getModifierState?.(' ') || event.shiftKey) {
      this.isPanning = true;
      this.panOrigin = { ...this._pointerPosition(event), panX: this.panX, panY: this.panY };
      this.canvas.classList.add('is-panning');
      this.canvas.setPointerCapture(event.pointerId);
      event.preventDefault();
      return;
    }
    if (!this.isEditable) return;

    const { x, y } = this._pointerPosition(event);
    const index = this._findVertex(x, y);

    // Alt-click or right-click removes a vertex.
    if (index !== -1 && (event.altKey || event.button === 2)) {
      if (this.polygons.revision.length <= 3) return; // keep an area-bearing polygon
      this._snapshot();
      this.polygons.revision.splice(index, 1);
      this.activeIndex = -1;
      this.render();
      this._emitChange();
      event.preventDefault();
      return;
    }

    if (index !== -1) {
      this._snapshot();
      this.draggingIndex = index;
      this.activeIndex = index;
      this.canvas.setPointerCapture(event.pointerId);
      this.render();
      if (this.handlers.onSelect) this.handlers.onSelect(index);
      return;
    }

    // Draw mode: append a vertex. This is the only way to trace a frame that has
    // neither a model proposal nor a reference contour (e.g. an upload with the AI
    // tier unavailable), so the tool is never a dead end.
    if (this.drawMode) {
      this._snapshot();
      this.polygons.revision.push(this._toNormalised(x, y));
      this.activeIndex = this.polygons.revision.length - 1;
      this.render();
      this._emitChange();
    }
  }

  setDrawMode(enabled) {
    this.drawMode = Boolean(enabled);
    this.canvas.style.cursor = this.drawMode ? 'copy' : 'crosshair';
    return this.drawMode;
  }

  /** Discard the working contour entirely (undoable). */
  clearRevision() {
    if (!this.polygons.revision.length) return false;
    this._snapshot();
    this.polygons.revision = [];
    this.activeIndex = -1;
    this.render();
    this._emitChange();
    return true;
  }

  _handlePointerMove(event) {
    if (!this.image) return;
    const { x, y } = this._pointerPosition(event);

    if (this.isPanning && this.panOrigin) {
      this.panX = this.panOrigin.panX + (x - this.panOrigin.x);
      this.panY = this.panOrigin.panY + (y - this.panOrigin.y);
      this.render();
      return;
    }

    if (this.draggingIndex !== -1) {
      this.polygons.revision[this.draggingIndex] = this._toNormalised(x, y);
      this.render();
      this._emitChange();
    } else if (this.isEditable && !this.drawMode) {
      const hovered = this._findVertex(x, y);
      this.canvas.style.cursor = hovered === -1 ? 'crosshair' : 'grab';
    }

    if (this.handlers.onCursor) {
      const [ny, nx] = this._toNormalised(x, y);
      this.handlers.onCursor({ y: Math.round(ny), x: Math.round(nx) });
    }
  }

  _handlePointerUp(event) {
    if (this.canvas.hasPointerCapture?.(event.pointerId)) {
      this.canvas.releasePointerCapture(event.pointerId);
    }
    if (this.isPanning) {
      this.isPanning = false;
      this.panOrigin = null;
      this.canvas.classList.remove('is-panning');
    }
    if (this.draggingIndex !== -1) {
      this.draggingIndex = -1;
      this.render();
      this._emitChange();
    }
  }

  _handlePointerLeave() {
    if (this.handlers.onCursor) this.handlers.onCursor(null);
  }

  _handleContextMenu(event) {
    // Right-click is a vertex-delete gesture, so suppress the browser menu.
    if (this.isEditable) event.preventDefault();
  }

  _handleDoubleClick(event) {
    if (!this.image || !this.isEditable) return;
    const { x, y } = this._pointerPosition(event);
    if (this._findVertex(x, y) !== -1) return; // double-click on a handle is not an insert
    const edge = this._findEdge(x, y);
    if (!edge || edge.distance > 14) return;
    this._snapshot();
    this.polygons.revision.splice(edge.index + 1, 0, this._toNormalised(edge.x, edge.y));
    this.activeIndex = edge.index + 1;
    this.render();
    this._emitChange();
  }

  _handleWheel(event) {
    if (!this.image) return;
    event.preventDefault();
    const { x, y } = this._pointerPosition(event);
    const before = this._toNormalised(x, y);
    const factor = Math.exp(-event.deltaY * 0.0016);
    this.zoom = clamp(this.zoom * factor, 1, 14);
    // Keep the point under the cursor stationary.
    const after = this._toScreen(before);
    this.panX += x - after.x;
    this.panY += y - after.y;
    this.render();
  }

  _handleKeyDown(event) {
    if (!this.isEditable) return;
    const key = event.key;

    if ((event.ctrlKey || event.metaKey) && key.toLowerCase() === 'z') {
      event.preventDefault();
      if (event.shiftKey) this.redo();
      else this.undo();
      return;
    }
    if ((event.ctrlKey || event.metaKey) && key.toLowerCase() === 'y') {
      event.preventDefault();
      this.redo();
      return;
    }
    if (key === '0') {
      event.preventDefault();
      this.resetView();
      return;
    }
    if (key === 'Tab' && this.polygons.revision.length) {
      event.preventDefault();
      const step = event.shiftKey ? -1 : 1;
      const count = this.polygons.revision.length;
      this.activeIndex = (this.activeIndex + step + count) % count;
      this.render();
      if (this.handlers.onSelect) this.handlers.onSelect(this.activeIndex);
      return;
    }
    if ((key === 'Delete' || key === 'Backspace') && this.activeIndex !== -1) {
      event.preventDefault();
      if (this.polygons.revision.length <= 3) return;
      this._snapshot();
      this.polygons.revision.splice(this.activeIndex, 1);
      this.activeIndex = -1;
      this.render();
      this._emitChange();
      return;
    }

    const deltas = {
      ArrowUp: [-1, 0],
      ArrowDown: [1, 0],
      ArrowLeft: [0, -1],
      ArrowRight: [0, 1],
    };
    const delta = deltas[key];
    if (!delta || this.activeIndex === -1) return;
    event.preventDefault();
    // Shift multiplies the step; a plain arrow is one normalised unit, which at
    // NORM_SCALE=1000 is finer than a pixel on every frame in the sample dataset.
    const step = event.shiftKey ? 10 : 1;
    this._snapshot();
    const point = this.polygons.revision[this.activeIndex];
    point[0] = clamp(point[0] + delta[0] * step, 0, NORM_SCALE);
    point[1] = clamp(point[1] + delta[1] * step, 0, NORM_SCALE);
    this.render();
    this._emitChange();
  }

  // -------------------------------------------------------------- rendering
  render() {
    const ctx = this.ctx;
    if (!ctx || !this.cssW) return;

    ctx.clearRect(0, 0, this.cssW, this.cssH);
    if (!this.image) return;

    const o = this._origin();
    ctx.imageSmoothingEnabled = this.zoom < 3; // crisp pixels when zoomed right in
    ctx.drawImage(this.image, o.x, o.y, this.imageW * o.scale, this.imageH * o.scale);

    const { model, revision, groundTruth } = this.polygons;

    if (this.panel === PANEL.MODEL) {
      this._strokePolygon(model, COLORS.model, COLORS.modelFill);
    } else if (this.panel === PANEL.REVISION) {
      this._strokePolygon(revision, COLORS.user, COLORS.userFill);
      this._drawHandles(revision, COLORS.user);
    } else if (this.panel === PANEL.OVERLAY) {
      if (groundTruth) this._strokePolygon(groundTruth, COLORS.truth, COLORS.truthFill, [5, 4]);
      this._strokePolygon(model, COLORS.model, 'transparent');
      this._strokePolygon(revision, COLORS.user, COLORS.userFill);
      this._drawHandles(revision, COLORS.user);
    }
  }

  _strokePolygon(polygon, stroke, fill, dash) {
    if (!polygon || polygon.length < 2) return;
    const ctx = this.ctx;
    ctx.save();
    ctx.beginPath();
    polygon.forEach((point, index) => {
      const { x, y } = this._toScreen(point);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
    if (fill && fill !== 'transparent') {
      ctx.fillStyle = fill;
      ctx.fill();
    }
    ctx.setLineDash(dash || []);
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1.6;
    ctx.lineJoin = 'round';
    ctx.stroke();
    ctx.restore();
  }

  _drawHandles(polygon, color) {
    if (!polygon || !polygon.length) return;
    const ctx = this.ctx;
    ctx.save();
    polygon.forEach((point, index) => {
      const { x, y } = this._toScreen(point);
      const isActive = index === this.activeIndex;
      const radius = isActive ? HANDLE_RADIUS + 1.5 : HANDLE_RADIUS;
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fillStyle = isActive ? COLORS.active : COLORS.handle;
      ctx.fill();
      ctx.lineWidth = 1.4;
      ctx.strokeStyle = color;
      ctx.stroke();
    });
    ctx.restore();
  }
}

function clone(polygon) {
  return (polygon || []).map(([y, x]) => [y, x]);
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}
