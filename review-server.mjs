import { readFile, mkdir, writeFile, rename } from "node:fs/promises";
import { createHash, randomUUID } from "node:crypto";
import { spawn } from "node:child_process";
import path from "node:path";
import { validateReviewMatrix } from "./review-math.mjs";
import { applyDisplayAdjustment, inverseUvToDisplayParameters } from "./viewer-math.mjs";

export function alignmentReviewRoutes({ here, processedRoot, jobs, sendJson, requestBody }) {
  const activeScans = new Map(), activeOptimizers = new Map(), writes = new Map();
  let refinementRunning = false;
  const digest = (bytes) => createHash("sha256").update(bytes).digest("hex");
  async function readJson(file, fallback) {
    try { return JSON.parse(await readFile(file, "utf8")); }
    catch (error) { if (error.code === "ENOENT") return fallback; throw error; }
  }
  async function inputs(directory) {
    const [manifest, alignment] = await Promise.all([readFile(path.join(directory, "manifest.json")), readFile(path.join(directory, "alignment.json"))]);
    return { manifest_sha256: digest(manifest), alignment_sha256: digest(alignment) };
  }
  function sameInputs(a, b) { return a?.manifest_sha256 === b.manifest_sha256 && a?.alignment_sha256 === b.alignment_sha256; }
  function refinementParameters(matrix, orientation) {
    const spatial = inverseUvToDisplayParameters(matrix, orientation);
    if (!spatial || spatial.x_scale < .25 || spatial.x_scale > 4 || spatial.y_scale < .25 || spatial.y_scale > 4 || Math.abs(spatial.rotation_deg) > 20 || Math.abs(spatial.x_position) > 512 || Math.abs(spatial.y_position) > 304) throw new Error("Transform is outside the supported editor range");
    const source = { identity:[1,0,0,0,1,0], flip_y:[1,0,0,0,-1,1], flip_x:[-1,0,1,0,1,0], rotate_180:[-1,0,1,0,-1,1] }[orientation];
    const recomposed = applyDisplayAdjustment(source, { scale_x:spatial.x_scale, scale_y:spatial.y_scale, tx:spatial.x_position, ty:spatial.y_position, rotation_deg:spatial.rotation_deg });
    if (matrix.some((value, index) => Math.abs(value - recomposed[index]) > 1e-5)) throw new Error("Auto Align requires the fixed CT orientation and a transform without shear");
    return spatial;
  }
  async function atomicJson(file, value) {
    await mkdir(path.dirname(file), { recursive: true });
    const temporary = `${file}.${randomUUID()}.tmp`;
    await writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`); await rename(temporary, file);
  }
  return async function handle(req, res, parts) {
    const subject = parts[2], kind = parts[3];
    if (!["problems", "scan", "frames", "suggest", "refine", "candidate", "optimize"].includes(kind)) return false;
    if (!["male", "female"].includes(subject)) { sendJson(res, 400, { error: "Unknown subject" }); return true; }
    const directory = path.join(processedRoot, subject), reportFile = path.join(directory, "alignment-problems.json"), anchorFile = path.join(directory, "reviewed-alignment.json"), candidateFile = path.join(directory, "alignment-candidate-v2.json");
    const currentInputs = await inputs(directory);
    if (kind === "candidate") {
      const value = await readJson(candidateFile, null);
      if (req.method === "GET") { sendJson(res, 200, { candidate: value, running: activeOptimizers.get(subject) || null }); return true; }
      if (req.method === "PUT") {
        const body = await requestBody(req), frame = Number(body.color_frame);
        if (body.action === "promote") {
          if (!value?.qc?.accepted) { sendJson(res, 409, { error: `Candidate cannot be promoted: ${(value?.qc?.promotion_blockers || ["validation incomplete"]).join(", ")}` }); return true; }
          const baselineFile = path.join(directory, "alignment.json"), rollbackFile = path.join(directory, "alignment.previous.json");
          await writeFile(rollbackFile, await readFile(baselineFile));
          value.status = "promoted"; value.promoted_at = new Date().toISOString();
          await atomicJson(baselineFile, value); sendJson(res, 200, { promoted: true, rollback: "alignment.previous.json" }); return true;
        }
        if (body.action === "delete_knot") {
          if (!value || !Number.isFinite(frame)) { sendJson(res, 400, { error: "Choose an existing candidate knot" }); return true; }
          const parameterKnots = value.parameter_knots || [], orderedFrames = parameterKnots.map((item) => Number.isFinite(item.color_frame) ? item.color_frame : NaN).filter(Number.isFinite).sort((a,b) => a-b);
          const currentReviews = await readJson(anchorFile, { version: 1, subject, anchors: [] }), savedFrame = currentReviews.anchors.some((item) => item.color_frame === frame);
          if (!orderedFrames.includes(frame) && !savedFrame) { sendJson(res, 404, { error: `RGB ${frame} is not a working knot` }); return true; }
          if (orderedFrames.includes(frame) && (frame === orderedFrames[0] || frame === orderedFrames.at(-1))) { sendJson(res, 409, { error: "The first and last coverage knots cannot be deleted" }); return true; }
          const previousWrite = writes.get(subject) || Promise.resolve();
          const operation = previousWrite.catch(() => {}).then(async () => {
            const reviews = await readJson(anchorFile, { version: 1, subject, anchors: [] });
            const historyDirectory = path.join(directory, "review-history"); await mkdir(historyDirectory, { recursive: true });
            const historyName = `reviewed-alignment-${new Date().toISOString().replaceAll(":", "-")}-${randomUUID()}.json`;
            await writeFile(path.join(historyDirectory, historyName), `${JSON.stringify(reviews, null, 2)}\n`);
            reviews.anchors = reviews.anchors.filter((item) => item.color_frame !== frame);
            await atomicJson(anchorFile, reviews);
            value.deleted_knot_frames = [...new Set([...(value.deleted_knot_frames || []), frame])].sort((a,b) => a-b);
            value.status = "needs_rebuild"; value.qc ||= {}; value.qc.accepted = false;
            value.qc.promotion_blockers = [...new Set([...(value.qc.promotion_blockers || []), "reviewed_anchors_changed"] )];
            await atomicJson(candidateFile, value); return reviews;
          });
          writes.set(subject, operation); const reviews = await operation;
          sendJson(res, 200, { deleted_frame: frame, deleted_knot_frames: value.deleted_knot_frames,
            anchors: reviews.anchors.map((anchor) => ({ ...anchor, stale: !sameInputs(anchor.inputs, currentInputs) })) }); return true;
        }
        if (!value || !Number.isFinite(frame) || !["rejected", "pending"].includes(body.status)) { sendJson(res, 400, { error: "Invalid candidate review update" }); return true; }
        const item = value.z_review_queue?.find((entry) => entry.color_frame === frame);
        if (item) { item.status = body.status; item.reviewed_at = new Date().toISOString(); }
        value.control_review_skips = [...new Set(value.control_review_skips || [])].filter((entry) => entry !== frame);
        if (body.status === "rejected") value.control_review_skips.push(frame);
        await atomicJson(candidateFile, value); sendJson(res, 200, item || { color_frame: frame, status: body.status === "rejected" ? "skipped" : "pending" }); return true;
      }
      sendJson(res, 405, { error: "Method not allowed" }); return true;
    }
    if (kind === "optimize") {
      if (req.method !== "POST") { sendJson(res, 405, { error: "Method not allowed" }); return true; }
      if (subject !== "male") { sendJson(res, 409, { error: "Rebuild female CT geometry before optimizing female" }); return true; }
      if (activeOptimizers.has(subject)) { sendJson(res, 202, jobs.get(activeOptimizers.get(subject))); return true; }
      const body = await requestBody(req), limit = Number(body.memory_limit_gib ?? 4);
      if (!Number.isFinite(limit) || limit < .5 || limit > 8) { sendJson(res, 400, { error: "Memory limit must be .5–8 GiB" }); return true; }
      const id = `optimize-${randomUUID()}`, job = { id, subject, state: "running", progress: 0, log: [] }; jobs.set(id, job); activeOptimizers.set(subject, id);
      const python = process.env.VISIBLE_HUMAN_PYTHON || path.join(here, ".venv", "bin", "python");
      const child = spawn(python, [path.join(here, "optimize_alignment.py"), "--subject", subject, "--processed", directory, "--candidate", candidateFile, "--memory-limit-gib", String(limit)], { stdio: ["ignore", "pipe", "pipe"] });
      const collect = (chunk) => { job.log.push(chunk.toString().slice(-2000)); job.log = job.log.slice(-8); };
      child.stdout.on("data", collect); child.stderr.on("data", collect);
      child.once("error", (error) => { Object.assign(job, { state: "failed", error: error.message }); activeOptimizers.delete(subject); });
      child.once("close", (code) => { Object.assign(job, { state: code === 0 ? "complete" : "failed", progress: code === 0 ? 1 : 0, exitCode: code }); activeOptimizers.delete(subject); });
      sendJson(res, 202, job); return true;
    }
    if (kind === "problems" && req.method === "GET") {
      const report = await readJson(reportFile, null);
      sendJson(res, 200, { report, stale: Boolean(report && !sameInputs(report.inputs, currentInputs)), running: activeScans.get(subject) || null }); return true;
    }
    if (kind === "scan" && req.method === "POST") {
      if (activeScans.has(subject)) { sendJson(res, 202, jobs.get(activeScans.get(subject))); return true; }
      const id = `scan-${randomUUID()}`, job = { id, subject, state: "running", progress: 0, log: [] }; jobs.set(id, job); activeScans.set(subject, id);
      const python = process.env.VISIBLE_HUMAN_PYTHON || path.join(here, ".venv", "bin", "python");
      const child = spawn(python, [path.join(here, "scan_alignment.py"), "--processed", directory, "--subject", subject, "--report", reportFile, "--stride", "12"], { stdio: ["ignore", "pipe", "pipe"] });
      const collect = (chunk) => { job.log.push(chunk.toString().slice(-2000)); job.log = job.log.slice(-5); };
      child.stdout.on("data", collect); child.stderr.on("data", collect);
      child.once("error", (error) => { job.state = "failed"; job.error = error.message; activeScans.delete(subject); });
      child.once("close", (code) => { job.state = code === 0 ? "complete" : "failed"; job.progress = code === 0 ? 1 : 0; activeScans.delete(subject); });
      sendJson(res, 202, job); return true;
    }
    if (!["frames", "suggest", "refine"].includes(kind)) { sendJson(res, 405, { error: "Method not allowed" }); return true; }
    if (kind === "frames" && req.method === "GET") {
      const value = await readJson(anchorFile, { version: 1, subject, anchors: [] });
      sendJson(res, 200, { ...value, anchors: value.anchors.map((anchor) => ({ ...anchor, stale: !sameInputs(anchor.inputs, currentInputs) })), inputs: currentInputs }); return true;
    }
    const frame = /^\d+$/.test(parts[4] || "") ? Number(parts[4]) : NaN;
    const manifest = await readJson(path.join(directory, "manifest.json"));
    if (!manifest.rgb.layers.some((item) => item.frame === frame)) { sendJson(res, 400, { error: "Choose an existing color frame" }); return true; }
    if (kind === "refine") {
      if (req.method !== "POST") { sendJson(res, 405, { error: "Method not allowed" }); return true; }
      if (refinementRunning) { sendJson(res, 409, { error: "Auto Align is already running. Wait for the current preview." }); return true; }
      const body = await requestBody(req), orientation = body.orientation;
      const ctFrames = manifest.density.layers.map((item) => item.frame).sort((a,b) => a-b);
      const landmarks = body.landmarks || [];
      try {
        validateReviewMatrix(body.inverse_uv);
        if (!["identity", "flip_y", "flip_x", "rotate_180"].includes(orientation)) throw new Error("A fixed CT orientation is required");
        refinementParameters(body.inverse_uv, orientation);
        if (!Number.isFinite(body.ct_frame) || body.ct_frame < ctFrames[0] || body.ct_frame > ctFrames.at(-1)) throw new Error("Choose an available CT frame first");
        if (!Array.isArray(landmarks) || landmarks.length > 30 || !landmarks.every((pair) => [pair?.color, pair?.density].every((point) => Array.isArray(point) && point.length === 2 && point.every((x) => Number.isFinite(x) && x >= 0 && x <= 1)))) throw new Error("Invalid landmark coordinates");
      } catch (error) { sendJson(res, 400, { error: error.message }); return true; }
      if (!sameInputs(body.inputs, currentInputs)) { sendJson(res, 409, { error: "Registration changed; reload before Auto Align" }); return true; }
      const reviews = await readJson(anchorFile, { anchors: [] });
      const neighbors = reviews.anchors.filter((item) => item.color_frame !== frame && item.z_confirmed !== false && Number.isFinite(item.ct_frame) && sameInputs(item.inputs, currentInputs)).sort((a,b) => a.color_frame-b.color_frame);
      const before = neighbors.filter((item) => item.color_frame < frame).at(-1), after = neighbors.find((item) => item.color_frame > frame);
      let zBounds = [Math.max(ctFrames[0], before ? before.ct_frame + .001 : -Infinity), Math.min(ctFrames.at(-1), after ? after.ct_frame - .001 : Infinity)];
      if (subject === "male" && ctFrames.includes(7)) {
        zBounds[0] = Math.max(zBounds[0], frame === 0 ? 7 : 7.001);
        if (frame === 0) zBounds[1] = Math.min(zBounds[1], 7);
      }
      if (zBounds[0] > zBounds[1] || body.ct_frame < zBounds[0] || body.ct_frame > zBounds[1]) {
        sendJson(res, 409, { error: `Current Z must be between CT ${zBounds[0].toFixed(3)} and ${zBounds[1].toFixed(3)} to preserve confirmed depth order` }); return true;
      }
      if (landmarks.length) zBounds = [body.ct_frame, body.ct_frame];
      const snapshot = async () => JSON.stringify(await Promise.all(["manifest.json", "alignment.json", "alignment-candidate-v2.json", "reviewed-alignment.json", "corrections.json"].map(async (name) => {
        try { return digest(await readFile(path.join(directory, name))); } catch (error) { if (error.code === "ENOENT") return null; throw error; }
      })));
      const inputSnapshot = await snapshot();
      // Serialize local fits across tabs/subjects to keep the memory window bounded.
      if (refinementRunning) { sendJson(res, 409, { error: "Auto Align is already running. Wait for the current preview." }); return true; }
      refinementRunning = true;
      try {
        const result = await new Promise((resolve, reject) => {
          const python = process.env.VISIBLE_HUMAN_PYTHON || path.join(here, ".venv", "bin", "python");
          const child = spawn(python, [path.join(here, "refine_alignment.py"), "--processed", directory, "--color-frame", String(frame)], {
            stdio: ["pipe", "pipe", "pipe"], env: { ...process.env, OPENBLAS_NUM_THREADS: "1", OMP_NUM_THREADS: "1", VECLIB_MAXIMUM_THREADS: "1", PYTHONDONTWRITEBYTECODE: "1" },
          });
          let stdout = "", stderr = "", settled = false;
          const finish = (error, value) => { if (settled) return; settled = true; clearTimeout(timer); res.off("close", disconnect); error ? reject(error) : resolve(value); };
          const disconnect = () => { child.kill(); finish(new Error("Auto Align request was cancelled")); };
          const timer = setTimeout(() => { child.kill("SIGKILL"); finish(new Error("Auto Align timed out; current transform was retained")); }, 90000);
          res.once("close", disconnect);
          child.stdout.on("data", (chunk) => { stdout += chunk; if (stdout.length > 1024 * 1024) { child.kill(); finish(new Error("Auto Align returned an oversized result")); } });
          child.stderr.on("data", (chunk) => { stderr = (stderr + chunk).slice(-3000); });
          child.once("error", (error) => finish(error)); child.stdin.on("error", (error) => finish(error));
          child.once("close", (code) => {
            if (settled) return;
            try { const value = JSON.parse(stdout); if (code) finish(new Error(value.error || stderr || "Auto Align could not fit this slice")); else finish(null, value); }
            catch { finish(new Error(stderr || "Auto Align returned no usable result")); }
          });
          child.stdin.end(JSON.stringify({ ct_frame: body.ct_frame, inverse_uv: body.inverse_uv, orientation, landmarks, inputs: currentInputs, z_bounds: zBounds }));
        });
        if (res.destroyed) return true;
        if (inputSnapshot !== await snapshot()) { sendJson(res, 409, { error: "Alignment inputs changed during Auto Align; current edit was retained" }); return true; }
        if (typeof result.accepted !== "boolean" || !Number.isFinite(result.ct_frame) || result.ct_frame < zBounds[0] - 1e-8 || result.ct_frame > zBounds[1] + 1e-8) throw new Error("Auto Align returned an invalid CT correspondence");
        validateReviewMatrix(result.inverse_uv, { referenceMatrix: body.inverse_uv });
        if (!result.accepted) { result.inverse_uv = body.inverse_uv; result.ct_frame = body.ct_frame; }
        // Editor convention is authoritative for displayed pixels and saved values.
        const spatial = refinementParameters(result.inverse_uv, orientation);
        result.parameters = { z_position: result.ct_frame, ...spatial };
        sendJson(res, 200, { ...result, inputs: currentInputs, preview_only: true });
      } catch (error) { if (!res.destroyed) sendJson(res, 422, { error: error.message }); }
      finally { refinementRunning = false; }
      return true;
    }
    if (kind === "suggest") {
      if (req.method !== "POST") { sendJson(res, 405, { error: "Method not allowed" }); return true; }
      const body = await requestBody(req);
      if (!Number.isFinite(body.ct_frame)) { sendJson(res, 400, { error: "Choose a CT frame first" }); return true; }
      const python = process.env.VISIBLE_HUMAN_PYTHON || path.join(here, ".venv", "bin", "python");
      try {
        const result = await new Promise((resolve, reject) => {
          const child = spawn(python, [path.join(here, "suggest_alignment.py"), "--processed", directory, "--color-frame", String(frame), "--ct-frame", String(body.ct_frame)], { stdio: ["ignore", "pipe", "pipe"] });
          let stdout = "", stderr = "";
          const timer = setTimeout(() => { child.kill(); reject(new Error("Suggestion timed out")); }, 30000);
          child.stdout.on("data", (chunk) => { stdout += chunk; }); child.stderr.on("data", (chunk) => { stderr = (stderr + chunk).slice(-2000); });
          child.once("error", (error) => { clearTimeout(timer); reject(error); });
          child.once("close", (code) => { clearTimeout(timer); try { const value = JSON.parse(stdout); if (code) reject(new Error(value.error || stderr || "No usable outline fit")); else resolve(value); } catch (error) { reject(error); } });
        });
        validateReviewMatrix(result.inverse_uv);
        sendJson(res, 200, result);
      } catch (error) { sendJson(res, 422, { error: error.message }); }
      return true;
    }
    let anchor;
    if (req.method === "PUT") {
      const body = await requestBody(req), matrix = body.inverse_uv;
      const determinant = Array.isArray(matrix) ? matrix[0] * matrix[4] - matrix[1] * matrix[3] : NaN;
      try { validateReviewMatrix(matrix); } catch (error) { sendJson(res, 400, { error: error.message }); return true; }
      const ctFrames = manifest.density.layers.map((item) => item.frame), ctFrame = body.ct_frame;
      if (!Array.isArray(matrix) || matrix.length !== 6 || !matrix.every((x) => Number.isFinite(x) && Math.abs(x) <= 20) || !Number.isFinite(determinant) || Math.abs(determinant) < .01 || Math.abs(determinant) > 25 || !Number.isFinite(ctFrame) || ctFrame < Math.min(...ctFrames) || ctFrame > Math.max(...ctFrames)) {
        sendJson(res, 400, { error: "Invalid or singular transform / CT frame" }); return true;
      }
      const before = ctFrames.filter((x) => x <= ctFrame).at(-1), after = ctFrames.find((x) => x >= ctFrame);
      if (after - before > (subject === "female" ? 6 : 2)) { sendJson(res, 400, { error: "CT selection lies in a missing source interval" }); return true; }
      const landmarks = body.landmarks || [];
      if (!Array.isArray(landmarks) || landmarks.length > 30 || !landmarks.every((pair) => [pair?.color, pair?.density].every((point) => Array.isArray(point) && point.length === 2 && point.every((x) => Number.isFinite(x) && x >= 0 && x <= 1)))) {
        sendJson(res, 400, { error: "Invalid landmark coordinates" }); return true;
      }
      if (body.inputs && !sameInputs(body.inputs, currentInputs)) { sendJson(res, 409, { error: "Registration changed; reload before saving" }); return true; }
      const parameters = body.parameters;
      const parameterFields = ["z_position", "x_position", "y_position", "rotation_deg", "x_scale", "y_scale"];
      if (!parameters || !parameterFields.every((field) => Number.isFinite(parameters[field])) || Math.abs(parameters.z_position - ctFrame) > 1e-6 || parameters.x_scale < .25 || parameters.x_scale > 4 || parameters.y_scale < .25 || parameters.y_scale > 4 || Math.abs(parameters.rotation_deg) > 20 || Math.abs(parameters.x_position) > 512 || Math.abs(parameters.y_position) > 304) {
        sendJson(res, 400, { error: "Save all six finite registration parameters; Z must equal the selected CT frame" }); return true;
      }
      const reviewedFields = body.reviewed_fields === undefined ? parameterFields : body.reviewed_fields;
      if (!Array.isArray(reviewedFields) || !reviewedFields.length || !reviewedFields.every((field) => parameterFields.includes(field))) {
        sendJson(res, 400, { error: "Invalid reviewed parameter fields" }); return true;
      }
      const zConfirmed = body.z_confirmed === undefined ? reviewedFields.includes("z_position") : body.z_confirmed === true && reviewedFields.includes("z_position");
      const existing = await readJson(anchorFile, { anchors: [] });
      const neighbors = existing.anchors.filter((item) => item.color_frame !== frame && item.z_confirmed !== false && sameInputs(item.inputs, currentInputs)).sort((a,b) => a.color_frame-b.color_frame);
      const beforeAnchor = neighbors.filter((item) => item.color_frame < frame).at(-1), afterAnchor = neighbors.find((item) => item.color_frame > frame);
      if (zConfirmed && ((subject === "male" && frame === 0 && ctFrames.includes(7) && Math.abs(ctFrame - 7) > 1e-6) ||
          (beforeAnchor && beforeAnchor.ct_frame >= ctFrame) || (afterAnchor && afterAnchor.ct_frame <= ctFrame))) {
        sendJson(res, 400, { error: `Z must remain strictly increasing${beforeAnchor ? ` after RGB ${beforeAnchor.color_frame} → CT ${beforeAnchor.ct_frame}` : ""}${afterAnchor ? ` and before RGB ${afterAnchor.color_frame} → CT ${afterAnchor.ct_frame}` : ""}` }); return true;
      }
      anchor = { color_frame: frame, color_depth: frame / Math.max(1, ...manifest.rgb.layers.map((x) => x.frame)), ct_frame: ctFrame, z_confirmed: zConfirmed,
        reviewed_fields: [...new Set(reviewedFields)], parameters, inverse_uv: matrix, landmarks, inputs: currentInputs, updated_at: new Date().toISOString() };
    } else if (req.method !== "DELETE") { sendJson(res, 405, { error: "Method not allowed" }); return true; }
    const previous = writes.get(subject) || Promise.resolve();
    const operation = previous.catch(() => {}).then(async () => {
      const value = await readJson(anchorFile, { version: 1, subject, anchors: [] });
      const historyDirectory = path.join(directory, "review-history");
      await mkdir(historyDirectory, { recursive: true });
      const historyName = `reviewed-alignment-${new Date().toISOString().replaceAll(":", "-")}-${randomUUID()}.json`;
      await writeFile(path.join(historyDirectory, historyName), `${JSON.stringify(value, null, 2)}\n`);
      value.anchors = value.anchors.filter((item) => item.color_frame !== frame);
      if (anchor) value.anchors.push(anchor);
      value.anchors.sort((a, b) => a.color_frame - b.color_frame);
      await atomicJson(anchorFile, value); return value;
    });
    writes.set(subject, operation);
    const value = await operation;
    const candidate = await readJson(candidateFile, null);
    if (candidate) {
      const suggestion = candidate.z_review_queue?.find((item) => item.color_frame === frame);
      if (suggestion && anchor?.z_confirmed) {
        suggestion.status = "confirmed"; suggestion.confirmed_ct_frame = anchor.ct_frame; suggestion.reviewed_at = anchor.updated_at;
      } else if (suggestion) {
        suggestion.status = "pending"; delete suggestion.confirmed_ct_frame; delete suggestion.reviewed_at;
      }
      candidate.status = "needs_rebuild";
      candidate.qc ||= {}; candidate.qc.accepted = false;
      candidate.qc.promotion_blockers = [...new Set([...(candidate.qc.promotion_blockers || []), "reviewed_anchors_changed"])];
      await atomicJson(candidateFile, candidate);
    }
    sendJson(res, 200, anchor || { ...value, anchors: value.anchors.map((a) => ({ ...a, stale: !sameInputs(a.inputs, currentInputs) })) }); return true;
  };
}
