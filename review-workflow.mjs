import { alignedDensityFrame, alignmentAt, applyDisplayAdjustment, interpolateKnots, inverseUvToDisplayParameters } from "/viewer-math.mjs";
import { fitLandmarkTransform, reviewAlignmentAt, validateReviewMatrix } from "/review-math.mjs";

const fields = ["scale_x", "scale_y", "tx", "ty", "rotation_deg"];
const modelFields = { scale_x: "x_scale", scale_y: "y_scale", tx: "x_position", ty: "y_position", rotation_deg: "rotation_deg" };
const allReviewedFields = ["z_position", "x_position", "y_position", "rotation_deg", "x_scale", "y_scale"];
const $ = (id) => document.querySelector(`#${id}`);
const clamp = (value, lo, hi) => Math.max(lo, Math.min(hi, value));

export function estimatedDensityFrame(profile, depth, maxFrame) {
  const mapped = alignedDensityFrame(profile, depth);
  if (mapped !== null) return mapped;
  const knots = profile?.depth_knots;
  if (!knots?.length) return depth * maxFrame;
  const endpoint = depth < knots[0].color_depth ? knots[0] : knots.at(-1);
  // Source frame coordinates already use color-plane units. Outside the fitted
  // interval this is an explicitly unverified preview, never a saved mapping.
  return endpoint.ct_frame + (depth - endpoint.color_depth) * maxFrame;
}

export class AlignmentReview {
  constructor({ state, render, setCompare, canvas }) {
    Object.assign(this, { state, render, setCompare, canvas, anchors: [], legacy: [], report: null, draft: null, matching: false, pending: null, loadSerial: 0 });
    const run = (fn) => (...args) => Promise.resolve().then(() => fn(...args)).catch((error) => { $("calibration-status").textContent = error.message; });
    $("edit-current-frame").onclick = run(() => this.edit(Math.round(state.depth * this.maxFrame)));
    $("review-color-frame").onchange = run(() => this.edit(Number($("review-color-frame").value)));
    $("review-ct-frame").oninput = run(() => {
      const input = $("review-ct-frame");
      if (!input.value.trim() || input.validity.badInput) return;
      this.changePlane(input.valueAsNumber, { preserveInput: true });
    });
    $("review-ct-frame").onblur = () => {
      if (this.atDraft()) $("review-ct-frame").value = this.draft.ct_frame.toFixed(2);
    };
    $("ct-previous").onclick = run(() => this.stepPlane(-1));
    $("ct-next").onclick = run(() => this.stepPlane(1));
    $("calibration-anchor").onchange = run(() => { const value = $("calibration-anchor").value; if (value) this.edit(Number(value.split(":")[1]), value.startsWith("legacy:")); });
    for (const field of fields) $("calibration-" + field).oninput = run(() => this.adjust());
    $("save-calibration").onclick = run(() => this.save());
    $("auto-align").onclick = run(() => this.autoAlign());
    $("undo-auto-align").onclick = run(() => this.undoAutoAlign());
    $("clear-calibration").onclick = run(() => this.clear());
    $("reset-calibration").onclick = run(() => this.reset());
    $("calibration-preview").onchange = () => { this.draft = null; this.autoUndo = null; this.matching = false; this.pending = null; this.updateSaveAction(); this.render(); };
    $("review-interpolate").onchange = () => this.render();
    $("scan-alignment").onclick = run(() => this.scan());
    $("suggest-alignment").onclick = run(() => this.suggest());
    $("problem-select").onchange = run(() => this.openProblem());
    $("problem-previous").onclick = run(() => this.nextProblem(-1));
    $("problem-next").onclick = run(() => this.nextProblem(1));
    $("match-landmarks").onclick = run(() => this.startMatching());
    $("fit-landmarks").onclick = run(() => this.fit());
    $("undo-landmark").onclick = () => { if (this.pending) this.pending = null; else this.draft?.landmarks.pop(); this.updatePoints(); };
    canvas.addEventListener("pointerdown", run((event) => this.pointerDown(event)));
    canvas.addEventListener("pointermove", (event) => this.pointerMove(event));
    canvas.addEventListener("pointerup", () => { this.drag = null; });
    canvas.addEventListener("pointercancel", () => { this.drag = null; });
    window.addEventListener("keydown", (event) => {
      const tab = document.body.dataset.activeTab;
      if (tab !== "alignment" && !(tab === "align3d" && event.target.closest(".calibration-panel"))) return;
      if (event.target.matches("input,select,button") || !this.atDraft() || this.matching || !this.state.registrationEnabled || !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) return;
      event.preventDefault(); const step = event.shiftKey ? .25 : 1;
      const a = this.draft.adjustment;
      if (event.key === "ArrowLeft") a.tx -= step; if (event.key === "ArrowRight") a.tx += step;
      if (event.key === "ArrowUp") a.ty -= step; if (event.key === "ArrowDown") a.ty += step;
      this.setInputs(); this.adjust();
    });
  }
  get maxFrame() { return Math.max(1, this.state.manifest?.colourPlanes - 1 || 1); }
  get profile() { return this.state.activeRegistration || this.state.manifest?.registration; }
  get sourceMatrix() {
    return { identity: [1,0,0,0,1,0], flip_y: [1,0,0,0,-1,1], flip_x: [-1,0,1,0,1,0], rotate_180: [-1,0,1,0,-1,1] }[this.profile?.orientation] || [1,0,0,0,1,0];
  }
  isProfileKnot(frame) {
    const knots = this.profile?.parameter_knots?.length ? this.profile.parameter_knots : this.profile?.spatial_knots || [];
    return knots.some((item) => (Number.isFinite(item.color_frame) ? item.color_frame : Math.round(item.color_depth * this.maxFrame)) === frame);
  }
  adjustmentForMatrix(matrix) {
    const value = inverseUvToDisplayParameters(matrix, this.profile?.orientation);
    if (!value) throw new Error("Could not read the knot transform.");
    return { scale_x:value.x_scale, scale_y:value.y_scale, tx:value.x_position, ty:value.y_position, rotation_deg:value.rotation_deg };
  }
  setDraftMatrix(matrix) {
    this.draft.base_matrix = [...this.sourceMatrix];
    this.draft.adjustment = this.adjustmentForMatrix(matrix);
  }
  currentMatrix() { return applyDisplayAdjustment(this.draft.base_matrix, this.draft.adjustment); }
  atDraft(depth = this.state.depth) { return this.draft && Math.abs(depth * this.maxFrame - this.draft.color_frame) <= .51; }
  async api(suffix, options) {
    const response = await fetch(`/api/review/${this.state.subject}/${suffix}`, options);
    const value = await response.json(); if (!response.ok) throw new Error(value.error || `HTTP ${response.status}`); return value;
  }
  async load() {
    const serial = ++this.loadSerial;
    $("scan-alignment").disabled = false;
    this.draft = null; this.autoUndo = null; this.anchors = []; this.legacy = []; this.report = null; this.matching = false; this.pending = null;
    this.updateSaveAction();
    if (!this.profile) { this.refreshSaved(); this.refreshProblems(); $("calibration-status").textContent = "A current registration profile is required for alignment review."; return; }
    const [saved, legacy, problems] = await Promise.all([this.api("frames"), this.api("calibration"), this.api("problems")]);
    if (serial !== this.loadSerial) return;
    this.inputs = saved.inputs; this.anchors = saved.anchors; this.legacy = legacy.anchors;
    this.report = problems.stale ? null : problems.report;
    this.refreshSaved(); this.refreshProblems();
    $("review-color-frame").max = String(this.maxFrame);
    const density = this.state.manifest.density.layers;
    $("review-ct-frame").min = String(density[0]?.frame ?? 0); $("review-ct-frame").max = String(density.at(-1)?.frame ?? 0);
    const candidateQueue = this.state.manifest.registrationCandidate?.control_review_queue || this.state.manifest.registrationCandidate?.z_review_queue;
    const zSuggestion = this.state.activeRegistration === this.state.manifest.registrationCandidate
      ? candidateQueue?.find((item) => item.status === "pending") : null;
    if (zSuggestion) this.edit(zSuggestion.color_frame);
    else if (this.report?.issues?.length) this.openProblem();
    else this.edit(this.legacy[0]?.color_frame ?? Math.round(.1 * this.maxFrame));
    if (problems.running) this.pollScan(problems.running, serial);
  }
  refreshSaved() {
    const selected = $("calibration-anchor").value;
    $("calibration-anchor").replaceChildren(new Option("Choose a saved frame…", ""),
      ...this.anchors.map((a) => new Option(`Reviewed · ${a.color_frame}${a.stale ? " · stale" : ""}`, `frame:${a.color_frame}`)),
      ...this.legacy.map((a) => new Option(`Original anchor ${a.slot} · ${a.color_frame}`, `legacy:${a.color_frame}`)));
    $("calibration-anchor").value = selected;
    $("calibration-count").textContent = `${this.anchors.length} saved`;
    window.dispatchEvent(new CustomEvent("human-anchors-updated"));
  }
  reviewsChanged() {
    const candidate = this.state.manifest?.registrationCandidate;
    if (!candidate) return;
    candidate.status = "needs_rebuild";
    candidate.qc ||= {}; candidate.qc.accepted = false;
    candidate.qc.promotion_blockers = [...new Set([...(candidate.qc.promotion_blockers || []), "reviewed_anchors_changed"])];
    window.dispatchEvent(new CustomEvent("candidate-review-updated"));
  }
  edit(frame, useLegacy = false, automatic = false) {
    if (!Number.isInteger(frame) || !this.state.manifest.rgb.layers.some((a) => a.frame === frame)) throw new Error("Choose an existing color frame.");
    clearInterval(this.state.flickerTimer); this.state.flickerTimer = null; this.state.reviewRaw = false; $("raw-flicker").checked = false;
    $("raw-flicker").disabled = false;
    $("suggestion-notes").hidden = true;
    const depth = frame / this.maxFrame;
    const saved = automatic ? null : this.anchors.find((a) => a.color_frame === frame && !a.stale);
    const legacy = useLegacy ? this.legacy.find((a) => a.color_frame === frame) : null;
    let matrix = saved?.inverse_uv || alignmentAt(this.profile, depth).inverseUv;
    if (legacy) matrix = applyDisplayAdjustment(alignmentAt(this.profile, legacy.color_depth).inverseUv, legacy);
    const ctFrames = this.state.manifest.density.layers.map((a) => a.frame);
    const estimate = estimatedDensityFrame(this.profile, depth, this.maxFrame);
    const ctFrame = saved?.ct_frame ?? ctFrames.reduce((best, x) => Math.abs(x - estimate) < Math.abs(best - estimate) ? x : best, ctFrames[0]);
    const adjustment = this.adjustmentForMatrix(matrix);
    this.draft = { color_frame: frame, color_depth: depth, ct_frame: ctFrame, base_matrix: [...this.sourceMatrix], adjustment,
      original: { ct_frame: ctFrame, adjustment: { ...adjustment } }, origin: saved ? "human" : this.isProfileKnot(frame) ? "profile" : "slice",
      reviewed_fields: saved?.reviewed_fields, landmarks: structuredClone(saved?.landmarks || []) };
    this.autoUndo = null; this.matching = false; this.pending = null; this.state.depth = depth;
    history.replaceState(null, "", `?subject=${encodeURIComponent(this.state.subject)}&frame=${frame}${location.hash}`);
    const issue = this.report?.issues.find((item) => frame >= item.start_frame && frame <= item.end_frame);
    if (issue) {
      $("problem-select").value = issue.id;
      $("problem-detail").textContent = `Frames ${issue.start_frame}–${issue.end_frame}: ${issue.reasons.join(". ")}`;
    } else $("problem-detail").textContent = `No automatic flag at frame ${frame}.`;
    this.state.registrationEnabled = Boolean(this.profile); $("registration-toggle").checked = this.state.registrationEnabled;
    this.state.blend = .5; $("blend-slider").value = "50";
    $("calibration-preview").checked = true;
    this.setInputs(); this.updatePoints();
    $("calibration-status").textContent = legacy ? "Original anchor loaded" : saved ? "Saved frame loaded" : "Unsaved";
    window.dispatchEvent(new CustomEvent("review-frame-selected", { detail: { color_frame: frame, saved: Boolean(saved) } }));
    this.render();
  }
  updateSaveAction() {
    const button = $("save-calibration"), allowed = this.draft && this.draft.origin !== "slice";
    button.disabled = Boolean(this.saving || this.autoRequest || !allowed);
    button.textContent = !allowed ? "Select a timeline knot" : this.draft.origin === "human" ? "Update knot correction" : "Correct this knot";
    button.title = allowed ? "Save a correction for this existing timeline knot" : "Use the bottom knot arrows or click a notch before saving";
    $("auto-align").disabled = Boolean(this.saving || this.autoRequest || !this.atDraft() || !this.profile || !this.inputs);
    $("auto-align").textContent = this.autoRequest ? "Aligning…" : "Auto Align";
    $("undo-auto-align").disabled = Boolean(this.saving || this.autoRequest || !this.autoUndo || !this.atDraft() || this.autoUndo.draft !== this.draft);
  }
  setInputs({ preserveCt = false } = {}) {
    if (!this.draft) return;
    $("review-color-frame").value = String(this.draft.color_frame);
    if (!preserveCt) $("review-ct-frame").value = this.draft.ct_frame.toFixed(2);
    for (const field of fields) $("calibration-" + field).value = this.draft.adjustment[field].toFixed(field.startsWith("scale") ? 3 : 2);
    this.updateSaveAction();
  }
  requireDraft() { if (!this.atDraft()) throw new Error("Click Edit current frame before adjusting this slice."); }
  reset() {
    const frame = this.draft?.color_frame ?? Math.round(this.state.depth * this.maxFrame);
    if (!window.visibleHumanAlignment?.select(frame)) this.edit(frame);
    $("calibration-status").textContent = this.draft?.origin === "human" ? "Saved knot restored" : "Working knot restored";
  }
  adjust() {
    this.requireDraft(); const adjustment = {};
    for (const field of fields) {
      const input = $("calibration-" + field);
      if (!input.value.trim() || input.validity.badInput) return;
      adjustment[field] = input.valueAsNumber;
      if (!Number.isFinite(adjustment[field])) return;
    }
    adjustment.scale_x = clamp(adjustment.scale_x, .25, 4); adjustment.scale_y = clamp(adjustment.scale_y, .25, 4);
    adjustment.tx = clamp(adjustment.tx, -512, 512); adjustment.ty = clamp(adjustment.ty, -304, 304); adjustment.rotation_deg = clamp(adjustment.rotation_deg, -20, 20);
    this.draft.adjustment = adjustment;
    $("calibration-status").textContent = "Unsaved changes"; this.render();
  }
  changePlane(frame, { preserveInput = false } = {}) {
    this.requireDraft();
    const layers = this.state.manifest.density.layers;
    if (!Number.isFinite(frame) || frame < layers[0].frame || frame > layers.at(-1).frame) throw new Error("CT frame is outside the available source.");
    this.draft.ct_frame = frame; this.draft.landmarks = []; this.pending = null;
    this.setInputs({ preserveCt: preserveInput }); this.updatePoints(); $("calibration-status").textContent = "Unsaved CT change"; this.render();
  }
  stepPlane(direction) {
    this.requireDraft(); const layers = this.state.manifest.density.layers;
    const item = direction > 0 ? layers.find((a) => a.frame > this.draft.ct_frame + .001) : layers.filter((a) => a.frame < this.draft.ct_frame - .001).at(-1);
    if (item) this.changePlane(item.frame);
  }
  selection(depth) {
    if (!$("calibration-preview").checked) return null;
    if (this.atDraft(depth)) return { inverse_uv: applyDisplayAdjustment(this.draft.base_matrix, this.draft.adjustment), ct_frame: this.draft.ct_frame, label: "frame preview" };
    const boundaries = this.state.manifest.boundaries.map((b) => b.modality === "density" ? interpolateKnots(this.profile.depth_knots, b.right, "ct_frame", "color_depth") : b.right / this.maxFrame).filter(Number.isFinite);
    const ctBoundaries = this.state.manifest.boundaries.filter((b) => b.modality === "density").map((b) => b.right);
    const reviewed = reviewAlignmentAt(this.profile, this.anchors.filter((a) => !a.stale), depth, { interpolate: $("review-interpolate").checked, maxFrame: this.maxFrame, maxGap: .04, boundaries, ctBoundaries });
    if (reviewed) return { ...reviewed, label: reviewed.interpolated ? "interpolated review preview" : "saved frame" };
    return null;
  }
  async save() {
    this.requireDraft();
    if (this.autoRequest) throw new Error("Wait for Auto Align to finish before saving.");
    if (this.draft.origin === "slice") throw new Error("This is an ordinary slice, not a knot. Select a bottom notch before saving.");
    const savingDraft = this.draft;
    this.saving = true; this.updateSaveAction();
    try {
      const inverse_uv = this.currentMatrix();
      const spatial = inverseUvToDisplayParameters(inverse_uv, this.profile.orientation);
      if (!spatial) throw new Error("Could not decompose this transform into the six-curve model.");
      const changed = [];
      if (Math.abs(this.draft.ct_frame - this.draft.original.ct_frame) > 1e-6) changed.push("z_position");
      const spatialChanged = fields.some((field) => Math.abs(this.draft.adjustment[field] - this.draft.original.adjustment[field]) > 1e-6);
      if (spatialChanged) changed.push("x_position", "y_position", "rotation_deg", "x_scale", "y_scale");
      let reviewed_fields;
      if (this.draft.origin === "human" && !Array.isArray(this.draft.reviewed_fields)) reviewed_fields = [...allReviewedFields];
      else reviewed_fields = [...new Set([...(this.draft.reviewed_fields || []), ...(changed.length ? changed : ["z_position"])])];
      const body = { inverse_uv, ct_frame: this.draft.ct_frame, parameters: { z_position: this.draft.ct_frame, ...spatial }, reviewed_fields,
        z_confirmed: reviewed_fields.includes("z_position"), landmarks: this.draft.landmarks, inputs: this.inputs };
      validateReviewMatrix(body.inverse_uv);
      const frame = this.draft.color_frame, subject = this.state.subject;
      $("calibration-status").textContent = "Saving to disk…";
      const anchor = await this.api(`frames/${frame}`, { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
      if (subject !== this.state.subject) return;
      this.anchors = this.anchors.filter((a) => a.color_frame !== frame).concat(anchor).sort((a,b) => a.color_frame - b.color_frame);
      const sameDraft = this.draft === savingDraft;
      if (sameDraft) {
        savingDraft.origin = "human"; savingDraft.reviewed_fields = anchor.reviewed_fields;
        savingDraft.original = { ct_frame: anchor.ct_frame, adjustment: this.adjustmentForMatrix(anchor.inverse_uv) };
        this.autoUndo = null;
      }
      const queues = [this.state.manifest.registrationCandidate?.z_review_queue, this.state.manifest.registrationCandidate?.control_review_queue];
      for (const queue of queues) { const suggestion = queue?.find((item) => item.color_frame === frame); if (suggestion) { suggestion.status = "confirmed"; suggestion.confirmed_ct_frame = anchor.ct_frame; } }
      this.reviewsChanged();
      this.refreshSaved(); this.refreshProblems();
      if (sameDraft) {
        $("calibration-status").textContent = `Saved to disk · RGB ${frame} → CT ${Number(anchor.ct_frame).toFixed(2)}`;
        window.dispatchEvent(new CustomEvent("review-frame-selected", { detail: { color_frame: frame, saved: true } }));
      }
      this.render();
    } finally { this.saving = false; this.updateSaveAction(); }
  }
  async clear() {
    this.requireDraft(); const frame = this.draft.color_frame, serial = this.loadSerial;
    const value = await this.api(`frames/${frame}`, { method: "DELETE" });
    if (serial !== this.loadSerial) return;
    this.anchors = value.anchors;
    for (const queue of [this.state.manifest?.registrationCandidate?.z_review_queue, this.state.manifest?.registrationCandidate?.control_review_queue]) {
      const item = queue?.find((entry) => entry.color_frame === frame);
      if (item) { item.status = "pending"; delete item.confirmed_ct_frame; }
    }
    this.reviewsChanged(); this.refreshSaved(); this.edit(frame, false, true);
    $("calibration-status").textContent = "Saved frame removed";
  }
  refreshProblems() {
    const report = this.report, selected = $("problem-select").value;
    const issues = report?.issues || [];
    $("problem-select").replaceChildren(...issues.map((issue, index) => new Option(`${index+1}. Frame ${issue.color_frame} · ${issue.reasons[0] || "review"}`, issue.id)));
    if (issues.some((issue) => issue.id === selected)) $("problem-select").value = selected;
    else if (issues.some((issue) => Number.isFinite(issue.ct_frame))) $("problem-select").value = issues.find((issue) => Number.isFinite(issue.ct_frame)).id;
    $("problem-summary").textContent = report ? `${issues.length} flagged regions` : "Not scanned";
    $("problem-previous").disabled = $("problem-next").disabled = !issues.length;
  }
  openProblem() {
    const issue = this.report?.issues.find((a) => a.id === $("problem-select").value); if (!issue) return;
    $("problem-detail").textContent = `Frames ${issue.start_frame}–${issue.end_frame}: ${issue.reasons.join(". ")}`;
    this.edit(issue.color_frame);
  }
  nextProblem(direction) {
    const select = $("problem-select"), length = select.options.length; if (!length) return;
    select.selectedIndex = (select.selectedIndex + direction + length) % length; this.openProblem();
  }
  async scan() {
    const job = await this.api("scan", { method: "POST" }); this.pollScan(job.id, this.loadSerial);
  }
  autoAlignFingerprint() {
    // Include uncommitted form text as well as full-precision draft values.
    // A delayed fit must never overwrite edits made while it was running.
    return JSON.stringify({ draft: this.draft, profile: this.profile, inputs: this.inputs, depth: this.state.depth,
      values: ["review-color-frame", "review-ct-frame", ...fields.map((field) => "calibration-" + field)].map((id) => $(id).value) });
  }
  autoAlignContextMatches(request) {
    return this.autoRequest === request && this.loadSerial === request.serial && this.state.subject === request.subject
      && this.draft === request.draft && this.profile === request.profile && this.atDraft();
  }
  showFitWarnings(warnings) {
    const notes = $("suggestion-notes"); notes.hidden = !warnings.length;
    notes.querySelector("div").replaceChildren(...warnings.map((warning) => {
      const p = document.createElement("p"); p.textContent = warning; return p;
    }));
  }
  async autoAlign() {
    this.requireDraft();
    if (this.autoRequest || this.saving) return;
    const inverse_uv = this.currentMatrix();
    validateReviewMatrix(inverse_uv);
    const request = { subject: this.state.subject, serial: this.loadSerial, profile: this.profile, draft: this.draft,
      before: structuredClone(this.draft), fingerprint: this.autoAlignFingerprint() };
    const body = { inverse_uv, ct_frame: this.draft.ct_frame, orientation: this.profile.orientation,
      landmarks: structuredClone(this.draft.landmarks), inputs: structuredClone(this.inputs) };
    this.autoRequest = request;
    this.matching = false; this.pending = null; this.drag = null;
    $("raw-flicker").disabled = false;
    this.updateSaveAction(); this.updatePoints();
    $("calibration-status").textContent = "Auto aligning RGB and CT…";
    try {
      const candidate = await this.api(`refine/${request.draft.color_frame}`, {
        method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body),
      });
      if (!this.autoAlignContextMatches(request)) return;
      if (this.autoAlignFingerprint() !== request.fingerprint) {
        $("calibration-status").textContent = "Auto Align discarded · values changed";
        return;
      }
      const warnings = Array.isArray(candidate.warnings) ? candidate.warnings.map(String) : [];
      if (candidate.accepted !== true) {
        this.showFitWarnings(warnings);
        $("calibration-status").textContent = "Auto Align · no reliable improvement";
        return;
      }
      validateReviewMatrix(candidate.inverse_uv, { referenceMatrix: inverse_uv });
      const layers = this.state.manifest.density.layers;
      if (!Number.isFinite(candidate.ct_frame) || candidate.ct_frame < layers[0].frame || candidate.ct_frame > layers.at(-1).frame) {
        throw new Error("Auto Align returned an unavailable CT frame.");
      }
      const adjustment = this.adjustmentForMatrix(candidate.inverse_uv);
      const displayedMatrix = applyDisplayAdjustment(this.sourceMatrix, adjustment);
      if (displayedMatrix.some((value, index) => Math.abs(value - candidate.inverse_uv[index]) > 1e-7)) {
        throw new Error("Auto Align returned a transform outside the supported scale/rotation model.");
      }
      const changedZ = Math.abs(candidate.ct_frame - request.before.ct_frame) > 1e-6;
      this.autoUndo = { draft: this.draft, before: request.before };
      // Keep original, origin and frame identity untouched: this is an unsaved
      // correction of the selected knot, never a new human-confirmed anchor.
      this.draft.base_matrix = [...this.sourceMatrix];
      this.draft.adjustment = adjustment;
      this.draft.ct_frame = candidate.ct_frame;
      this.matching = false; this.pending = null; this.drag = null;
      if (changedZ && this.draft.landmarks.length) {
        this.draft.landmarks = [];
        warnings.push("Landmarks cleared because the CT plane changed.");
      }
      clearInterval(this.state.flickerTimer); this.state.flickerTimer = null;
      this.state.reviewRaw = false; $("raw-flicker").checked = false;
      this.state.registrationEnabled = true; $("registration-toggle").checked = true;
      $("calibration-preview").checked = true;
      this.setInputs(); this.setCompare("blend"); this.updatePoints(); this.showFitWarnings(warnings);
      $("calibration-status").textContent = `Auto Align preview · unsaved${changedZ && request.before.landmarks.length ? " · landmarks cleared (Z changed)" : ""}`;
      this.render();
    } catch (error) {
      if (this.autoAlignContextMatches(request)) {
        $("calibration-status").textContent = this.autoAlignFingerprint() === request.fingerprint
          ? `Auto Align failed · ${error.message}` : "Auto Align discarded · values changed";
      }
    } finally {
      if (this.autoRequest === request) { this.autoRequest = null; this.updateSaveAction(); }
    }
  }
  undoAutoAlign() {
    this.requireDraft();
    if (this.autoRequest || this.saving || !this.autoUndo || this.autoUndo.draft !== this.draft) return;
    this.draft = structuredClone(this.autoUndo.before); this.autoUndo = null;
    this.matching = false; this.pending = null; this.drag = null;
    $("raw-flicker").disabled = false;
    this.setInputs(); this.updatePoints(); this.showFitWarnings([]);
    $("calibration-status").textContent = "Auto Align undone";
    this.render();
  }
  async suggest() {
    this.requireDraft(); const draft = this.draft, serial = this.loadSerial;
    $("suggest-alignment").disabled = true; $("calibration-status").textContent = "Fitting outlines with annotation and support filtering…";
    try {
      const candidate = await this.api(`suggest/${draft.color_frame}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ ct_frame: draft.ct_frame }) });
      if (serial !== this.loadSerial || this.draft !== draft) return;
      this.setDraftMatrix(candidate.inverse_uv); draft.ct_frame = candidate.ct_frame; draft.landmarks = [];
      this.pending = null; this.matching = false; this.setInputs(); this.updatePoints(); this.setCompare("blend");
      const notes = $("suggestion-notes"); notes.hidden = false;
      notes.querySelector("div").replaceChildren(...candidate.warnings.map((warning) => { const p = document.createElement("p"); p.textContent = warning; return p; }));
      const interiorWarning = candidate.warnings.some((warning) => warning.includes("increases interior"));
      $("calibration-status").textContent = interiorWarning ? "Outline preview · interior mismatch" : "Outline preview";
      this.render();
    } finally { $("suggest-alignment").disabled = false; }
  }
  async pollScan(id, serial) {
    $("scan-alignment").disabled = true;
    try {
      while (serial === this.loadSerial) {
        const response = await fetch(`/api/review/jobs/${id}`), job = await response.json();
        if (!response.ok || job.state === "failed") throw new Error(job.error || job.log?.at(-1) || "Scan failed");
        $("problem-summary").textContent = job.log?.at(-1)?.trim().slice(-180) || "Scanning size, coverage and interior edge disagreement…";
        if (job.state === "complete") { const value = await this.api("problems"); if (serial === this.loadSerial) { this.report = value.stale ? null : value.report; this.refreshProblems(); } break; }
        await new Promise((resolve) => setTimeout(resolve, 1200));
      }
    } catch (error) { if (serial === this.loadSerial) $("problem-summary").textContent = error.message; }
    finally { if (serial === this.loadSerial) $("scan-alignment").disabled = false; }
  }
  startMatching() {
    if (!this.atDraft()) this.edit(Math.round(this.state.depth * this.maxFrame));
    clearInterval(this.state.flickerTimer); this.state.flickerTimer = null; this.state.reviewRaw = false; $("raw-flicker").checked = false;
    $("raw-flicker").disabled = true;
    this.matching = true; this.pending = null; this.setCompare("split"); this.updatePoints(); this.render();
  }
  fit() {
    this.requireDraft();
    const fit = fitLandmarkTransform(this.draft.landmarks, { referenceMatrix: this.currentMatrix() });
    this.setDraftMatrix(fit.inverse_uv); this.matching = false; this.pending = null;
    $("raw-flicker").disabled = false;
    this.setInputs(); this.setCompare("blend"); this.updatePoints();
    $("calibration-status").textContent = `${fit.count} points · ${fit.rms_pixels.toFixed(2)} px residual`; this.render();
  }
  imageRect() {
    const rect = this.canvas.getBoundingClientRect(), split = this.state.compare === "split";
    const aspect = (split ? 2 : 1) * 512 / 304;
    const width = Math.min(rect.width, rect.height * aspect), height = width / aspect;
    return { left: rect.left + (rect.width-width)/2, top: rect.top + (rect.height-height)/2, width, height };
  }
  pointerDown(event) {
    if (event.button !== 0 || !this.state.registrationEnabled || !this.atDraft()) return;
    const r = this.imageRect(), u = (event.clientX-r.left)/r.width, v = (event.clientY-r.top)/r.height;
    if (u < 0 || u > 1 || v < 0 || v > 1) return;
    if (this.matching && this.state.compare === "split") {
      event.preventDefault();
      if (!this.pending) { if (u >= .5) throw new Error("Click the color image on the left first."); this.pending = [u*2, v]; }
      else {
        if (u < .5) throw new Error("Now click the same structure in CT on the right.");
        const point = [(u-.5)*2, v], m = this.sourceMatrix;
        this.draft.landmarks.push({ color: this.pending, density: [m[0]*point[0]+m[1]*point[1]+m[2], m[3]*point[0]+m[4]*point[1]+m[5]] }); this.pending = null;
      }
      this.updatePoints(); return;
    }
    if (this.state.compare === "split") return;
    this.drag = { x:event.clientX, y:event.clientY, tx:this.draft.adjustment.tx, ty:this.draft.adjustment.ty };
    this.canvas.setPointerCapture(event.pointerId); event.preventDefault();
  }
  pointerMove(event) {
    if (!this.drag) return; const r = this.imageRect();
    this.draft.adjustment.tx = this.drag.tx + (event.clientX-this.drag.x)*512/r.width;
    this.draft.adjustment.ty = this.drag.ty + (event.clientY-this.drag.y)*304/r.height;
    this.setInputs(); this.adjust();
  }
  updatePoints() {
    const points = this.draft?.landmarks || [];
    $("fit-landmarks").disabled = points.length < 3 || !this.atDraft();
    $("landmark-status").textContent = `${points.length} pairs${this.matching ? this.pending ? " · CT point next" : " · RGB point next" : ""}`;
    const holder = $("landmark-markers"); holder.replaceChildren();
    if (this.state.compare !== "split" || !this.atDraft()) return;
    const r = this.imageRect(), m = this.sourceMatrix;
    const mark = (point, index, side) => { const dot = document.createElement("span"); dot.textContent = String(index+1); dot.className = side ? "ct-point" : "color-point"; dot.style.left = `${r.left + (side+point[0])*r.width/2}px`; dot.style.top = `${r.top + point[1]*r.height}px`; holder.append(dot); };
    points.forEach((pair,index) => { mark(pair.color,index,0); mark([m[0]*(pair.density[0]-m[2]),m[4]*(pair.density[1]-m[5])],index,1); });
    if (this.pending) mark(this.pending,points.length,0);
  }
}
