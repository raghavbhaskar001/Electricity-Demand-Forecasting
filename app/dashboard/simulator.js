/* ==========================================================================
   Midnight Grid — Interactive Electricity Demand Prediction Simulator
   Uses verified PyTorch checkpoints & exact 70%-train StandardScalers.
   Tries live Python PyTorch `/api/predict` when served over HTTP and falls
   back to exact IEEE-754 Float32 checkpoint execution (`SIMULATOR_BUNDLE`)
   when opened via `file://` or inside a static Streamlit iframe.
   ========================================================================== */
(() => {
  "use strict";

  const B = window.SIMULATOR_BUNDLE;
  const LOOKBACK = 16;

  const ARCH_LABEL = { "1D_CNN": "1D CNN", LSTM: "LSTM", CNN_LSTM: "CNN–LSTM" };
  const FEAT_LABEL = {
    consumption_only: "Consumption only",
    consumption_weather_retro: "+ Retrospective weather",
    consumption_weather_operational: "+ Operational weather",
  };
  const FEAT_SHORT = {
    consumption_only: "Consumption (1)",
    consumption_weather_retro: "Retro wx (10)",
    consumption_weather_operational: "Operational wx (10)",
  };
  const DIST_LABEL = { mathura: "Mathura", bareilly: "Bareilly" };
  const SCALE_LABEL = { aggregate: "Aggregate", individual: "Individual" };

  // ------------------------------------------------------------------
  // 1. Float32 Checkpoint Tensor Decoding & Exact Neural Forward Pass
  // ------------------------------------------------------------------
  const decodedCache = {};

  function decodeB64Float32(b64) {
    let bin;
    if (typeof atob === "function") {
      const str = atob(b64);
      const bytes = new Uint8Array(str.length);
      for (let i = 0; i < str.length; i++) bytes[i] = str.charCodeAt(i);
      bin = bytes.buffer;
    } else {
      const buf = Buffer.from(b64, "base64");
      bin = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    }
    return new Float32Array(bin);
  }

  function getDecodedTensors(expId) {
    if (decodedCache[expId]) return decodedCache[expId];
    const m = B && B.models && B.models[expId];
    if (!m) throw new Error(`Checkpoint weights for ${expId} are not available.`);
    const out = {};
    for (const [k, obj] of Object.entries(m.tensors)) {
      out[k] = { shape: obj.shape, data: decodeB64Float32(obj.b64) };
    }
    decodedCache[expId] = out;
    return out;
  }

  /** Conv1d(inCh, outCh, kernel=3, padding=1) + BatchNorm1d(eps=1e-5) + ReLU on (inCh, L). */
  function conv1dBnRelu(x, inCh, L, convW, convB, bnW, bnB, bnMean, bnVar) {
    const outCh = convB.length;
    const y = new Float32Array(outCh * L);
    for (let oc = 0; oc < outCh; oc++) {
      const invStd = 1.0 / Math.sqrt(bnVar[oc] + 1e-5);
      const gamma = bnW[oc], beta = bnB[oc], mu = bnMean[oc], bias = convB[oc];
      const wBaseOc = oc * inCh * 3;
      for (let t = 0; t < L; t++) {
        let sum = bias;
        for (let ic = 0; ic < inCh; ic++) {
          const wBase = wBaseOc + ic * 3;
          const xBase = ic * L;
          if (t > 0) sum += convW[wBase] * x[xBase + t - 1];
          sum += convW[wBase + 1] * x[xBase + t];
          if (t + 1 < L) sum += convW[wBase + 2] * x[xBase + t + 1];
        }
        const bn = (sum - mu) * invStd * gamma + beta;
        y[oc * L + t] = bn > 0 ? bn : 0;
      }
    }
    return y;
  }

  function sigmoid(v) {
    return 1.0 / (1.0 + Math.exp(-v));
  }

  /** Single LSTM layer on sequence `seq` of shape (L, inSize) -> (L, H). */
  function lstmLayer(seq, L, inSize, H, wIh, wHh, bIh, bHh) {
    const out = new Float32Array(L * H);
    const h = new Float32Array(H);
    const c = new Float32Array(H);
    const gates = new Float32Array(4 * H);

    for (let t = 0; t < L; t++) {
      const xOff = t * inSize;
      for (let g = 0; g < 4 * H; g++) {
        let s = bIh[g] + bHh[g];
        const ihOff = g * inSize;
        for (let k = 0; k < inSize; k++) s += wIh[ihOff + k] * seq[xOff + k];
        const hhOff = g * H;
        for (let k = 0; k < H; k++) s += wHh[hhOff + k] * h[k];
        gates[g] = s;
      }
      for (let j = 0; j < H; j++) {
        const iGate = sigmoid(gates[j]);
        const fGate = sigmoid(gates[H + j]);
        const gGate = Math.tanh(gates[2 * H + j]);
        const oGate = sigmoid(gates[3 * H + j]);
        const cj = fGate * c[j] + iGate * gGate;
        c[j] = cj;
        const hj = oGate * Math.tanh(cj);
        h[j] = hj;
        out[t * H + j] = hj;
      }
    }
    return out;
  }

  /** Linear(64, 32) -> ReLU -> Linear(32, 1) */
  function regressorHead(vec64, w0, b0, w3, b3) {
    const fc1 = new Float32Array(32);
    for (let i = 0; i < 32; i++) {
      let s = b0[i];
      const off = i * 64;
      for (let j = 0; j < 64; j++) s += w0[off + j] * vec64[j];
      fc1[i] = s > 0 ? s : 0;
    }
    let out = b3[0];
    for (let i = 0; i < 32; i++) out += w3[i] * fc1[i];
    return out;
  }

  /** Run exact eval() forward pass on scaled window `W_scaled` of shape (L=16, nFeats). */
  function forwardModel(expId, arch, nFeats, wScaledFlat) {
    const T = getDecodedTensors(expId);
    const L = LOOKBACK;

    if (arch === "1D_CNN") {
      // Permute (L, nFeats) -> (nFeats, L)
      const xPerm = new Float32Array(nFeats * L);
      for (let t = 0; t < L; t++) {
        for (let f = 0; f < nFeats; f++) xPerm[f * L + t] = wScaledFlat[t * nFeats + f];
      }
      const c1 = conv1dBnRelu(
        xPerm, nFeats, L,
        T["conv_stack.0.weight"].data, T["conv_stack.0.bias"].data,
        T["conv_stack.1.weight"].data, T["conv_stack.1.bias"].data,
        T["conv_stack.1.running_mean"].data, T["conv_stack.1.running_var"].data
      );
      const c2 = conv1dBnRelu(
        c1, 64, L,
        T["conv_stack.4.weight"].data, T["conv_stack.4.bias"].data,
        T["conv_stack.5.weight"].data, T["conv_stack.5.bias"].data,
        T["conv_stack.5.running_mean"].data, T["conv_stack.5.running_var"].data
      );
      const c3 = conv1dBnRelu(
        c2, 128, L,
        T["conv_stack.8.weight"].data, T["conv_stack.8.bias"].data,
        T["conv_stack.9.weight"].data, T["conv_stack.9.bias"].data,
        T["conv_stack.9.running_mean"].data, T["conv_stack.9.running_var"].data
      );
      const pooled = new Float32Array(64);
      for (let ch = 0; ch < 64; ch++) {
        let s = 0;
        for (let t = 0; t < L; t++) s += c3[ch * L + t];
        pooled[ch] = s / L;
      }
      return regressorHead(
        pooled,
        T["regressor.0.weight"].data, T["regressor.0.bias"].data,
        T["regressor.3.weight"].data, T["regressor.3.bias"].data
      );
    }

    if (arch === "LSTM") {
      const l0 = lstmLayer(
        wScaledFlat, L, nFeats, 64,
        T["lstm.weight_ih_l0"].data, T["lstm.weight_hh_l0"].data,
        T["lstm.bias_ih_l0"].data, T["lstm.bias_hh_l0"].data
      );
      const l1 = lstmLayer(
        l0, L, 64, 64,
        T["lstm.weight_ih_l1"].data, T["lstm.weight_hh_l1"].data,
        T["lstm.bias_ih_l1"].data, T["lstm.bias_hh_l1"].data
      );
      const lastStep = l1.subarray((L - 1) * 64, L * 64);
      return regressorHead(
        lastStep,
        T["regressor.0.weight"].data, T["regressor.0.bias"].data,
        T["regressor.3.weight"].data, T["regressor.3.bias"].data
      );
    }

    if (arch === "CNN_LSTM") {
      const xPerm = new Float32Array(nFeats * L);
      for (let t = 0; t < L; t++) {
        for (let f = 0; f < nFeats; f++) xPerm[f * L + t] = wScaledFlat[t * nFeats + f];
      }
      const c1 = conv1dBnRelu(
        xPerm, nFeats, L,
        T["cnn.0.weight"].data, T["cnn.0.bias"].data,
        T["cnn.1.weight"].data, T["cnn.1.bias"].data,
        T["cnn.1.running_mean"].data, T["cnn.1.running_var"].data
      );
      // Permute back (64, L) -> (L, 64)
      const seq64 = new Float32Array(L * 64);
      for (let t = 0; t < L; t++) {
        for (let ch = 0; ch < 64; ch++) seq64[t * 64 + ch] = c1[ch * L + t];
      }
      const l0 = lstmLayer(
        seq64, L, 64, 64,
        T["lstm.weight_ih_l0"].data, T["lstm.weight_hh_l0"].data,
        T["lstm.bias_ih_l0"].data, T["lstm.bias_hh_l0"].data
      );
      const l1 = lstmLayer(
        l0, L, 64, 64,
        T["lstm.weight_ih_l1"].data, T["lstm.weight_hh_l1"].data,
        T["lstm.bias_ih_l1"].data, T["lstm.bias_hh_l1"].data
      );
      const lastStep = l1.subarray((L - 1) * 64, L * 64);
      return regressorHead(
        lastStep,
        T["regressor.0.weight"].data, T["regressor.0.bias"].data,
        T["regressor.3.weight"].data, T["regressor.3.bias"].data
      );
    }

    throw new Error(`Unsupported architecture ${arch}`);
  }

  // ------------------------------------------------------------------
  // 2. Client Validation, Scenario Sampling & CSV Parsing
  // ------------------------------------------------------------------
  function parseFinite(raw, label) {
    if (raw === null || raw === undefined || typeof raw === "boolean") {
      throw new Error(`Missing value at ${label}.`);
    }
    const str = String(raw).trim();
    if (str === "") throw new Error(`Missing empty value at ${label}.`);
    const v = Number(str);
    if (!Number.isFinite(v)) throw new Error(`Non-numeric or infinite value "${raw}" at ${label}.`);
    return v;
  }

  function weatherPhysicalError(colName, v) {
    const base = colName.replace(/^retro_|^operational_/, "");
    if ((base === "relative_humidity_2m" || base === "cloud_cover") && (v < 0 || v > 100)) {
      return `${colName}=${v} outside physical [0, 100]% bounds`;
    }
    if ((base === "wind_speed_10m" || base === "direct_radiation" || base === "diffuse_radiation" || base === "precipitation") && v < 0) {
      return `${colName}=${v} cannot be negative`;
    }
    if (base === "surface_pressure" && (v <= 500 || v >= 1150)) {
      return `${colName}=${v} hPa outside physical atmospheric bounds (500–1150 hPa)`;
    }
    if ((base === "temperature_2m" || base === "dew_point_2m") && (v < -50 || v > 70)) {
      return `${colName}=${v} °C outside physical terrestrial bounds (-50 to 70 °C)`;
    }
    return null;
  }

  function validateInputsLocal(district, featureConfig, consumptionKw, weatherRows) {
    const distObj = B.districts[district];
    if (!distObj) throw new Error(`Unsupported district "${district}".`);
    const fCols = B.feature_sets[featureConfig];
    if (!fCols) throw new Error(`Unsupported feature configuration "${featureConfig}".`);

    if (!Array.isArray(consumptionKw) || consumptionKw.length !== LOOKBACK) {
      throw new Error(`Expected exactly ${LOOKBACK} consecutive consumption intervals (4 hours), received ${Array.isArray(consumptionKw) ? consumptionKw.length : 0}.`);
    }

    const cleanCons = new Float64Array(LOOKBACK);
    for (let i = 0; i < LOOKBACK; i++) {
      const v = parseFinite(consumptionKw[i], `consumption step ${i + 1} (t-${LOOKBACK - 1 - i})`);
      if (v < 0) throw new Error(`Invalid negative electricity demand (${v} kW) at step ${i + 1}; demand must be ≥ 0 kW.`);
      cleanCons[i] = v;
    }

    const nFeats = fCols.length;
    const W = new Float64Array(LOOKBACK * nFeats);
    for (let i = 0; i < LOOKBACK; i++) W[i * nFeats] = cleanCons[i];

    const wxCols = fCols.slice(1);
    if (wxCols.length > 0) {
      if (!Array.isArray(weatherRows)) {
        throw new Error(`Feature configuration "${FEAT_LABEL[featureConfig]}" requires exact 16-step weather inputs for all ${wxCols.length} meteorological variables. Select a pipeline historical weather window, generate a paired scenario, or upload a 10-column CSV.`);
      }
      if (weatherRows.length !== LOOKBACK) {
        throw new Error(`Weather input must contain ${LOOKBACK} consecutive rows, received ${weatherRows.length}.`);
      }
      for (let r = 0; r < LOOKBACK; r++) {
        const row = weatherRows[r];
        if (!Array.isArray(row) || row.length !== wxCols.length) {
          throw new Error(`Weather row ${r + 1} must contain all ${wxCols.length} trained weather variables.`);
        }
        for (let c = 0; c < wxCols.length; c++) {
          const val = parseFinite(row[c], `${wxCols[c]} at step ${r + 1}`);
          const pErr = weatherPhysicalError(wxCols[c], val);
          if (pErr) throw new Error(`Step ${r + 1}: ${pErr}.`);
          W[r * nFeats + c + 1] = val;
        }
      }
    }

    const statsCfg = distObj.feature_stats[featureConfig];
    const cStats = statsCfg.demand_target;
    const warnings = [];
    const belowMin = [], aboveMax = [], tailLow = [], tailHigh = [];
    for (let i = 0; i < LOOKBACK; i++) {
      const v = cleanCons[i];
      if (v < cStats.min) belowMin.push(i + 1);
      else if (v > cStats.max) aboveMax.push(i + 1);
      else if (v < cStats.p01) tailLow.push(i + 1);
      else if (v > cStats.p99) tailHigh.push(i + 1);
    }

    if (belowMin.length || aboveMax.length) {
      warnings.push({
        feature: "demand_target",
        severity: "high",
        code: "consumption_outside_training_range",
        message: `Consumption outside ${DIST_LABEL[district]} training range [${cStats.min.toFixed(3)}, ${cStats.max.toFixed(3)}] kW at step(s) ${[...belowMin, ...aboveMax].join(", ")}. Raw values preserved without clipping.`,
        steps_below_min: belowMin,
        steps_above_max: aboveMax,
      });
    } else if (tailLow.length || tailHigh.length) {
      warnings.push({
        feature: "demand_target",
        severity: "moderate",
        code: "consumption_in_extreme_tail",
        message: `Consumption enters ${DIST_LABEL[district]} 1%–99% training tail ([${cStats.p01.toFixed(3)}, ${cStats.p99.toFixed(3)}] kW) at step(s) ${[...tailLow, ...tailHigh].join(", ")}. Raw values preserved without clipping.`,
        steps_tail_low: tailLow,
        steps_tail_high: tailHigh,
      });
    }

    for (let c = 0; c < wxCols.length; c++) {
      const colName = wxCols[c], st = statsCfg[colName], outSteps = [];
      for (let r = 0; r < LOOKBACK; r++) {
        const val = W[r * nFeats + c + 1];
        if (val < st.min || val > st.max) outSteps.push(r + 1);
      }
      if (outSteps.length) {
        warnings.push({
          feature: colName,
          severity: "high",
          code: "weather_outside_training_range",
          message: `${colName} outside ${DIST_LABEL[district]} training range [${st.min.toFixed(2)}, ${st.max.toFixed(2)}] at step(s) ${outSteps.join(", ")}. Preserved without clipping.`,
          steps: outSteps,
        });
      }
    }

    let minC = Infinity, maxC = -Infinity, sumC = 0;
    for (let i = 0; i < LOOKBACK; i++) {
      if (cleanCons[i] < minC) minC = cleanCons[i];
      if (cleanCons[i] > maxC) maxC = cleanCons[i];
      sumC += cleanCons[i];
    }
    const meanC = sumC / LOOKBACK;
    let varC = 0;
    for (let i = 0; i < LOOKBACK; i++) varC += (cleanCons[i] - meanC) ** 2;

    return {
      W,
      report: {
        valid: true,
        status: warnings.length ? "valid_with_ood_warning" : "valid_in_distribution",
        clipped: false,
        lookback_steps: LOOKBACK,
        in_features: nFeats,
        feature_order: fCols,
        warnings,
        training_consumption_range_kw: cStats,
        input_summary_kw: {
          min: minC, max: maxC, mean: meanC, std: Math.sqrt(varC / LOOKBACK), last_t0: cleanCons[LOOKBACK - 1],
        },
      },
    };
  }

  function predictLocal(scale, district, arch, featureConfig, consumptionKw, weatherRows) {
    const prefix = scale === "aggregate" ? "agg" : "ind";
    const expId = `${prefix}_${district}_${arch}_${featureConfig}`;
    const avail = B.availability.find(r => r.exp_id === expId);
    if (!avail) throw new Error(`Unknown model configuration ${expId}.`);
    if (!avail.supported) throw new Error(`[${expId}] Unsupported: ${avail.reason}`);

    const { W, report } = validateInputsLocal(district, featureConfig, consumptionKw, weatherRows);
    const sc = B.districts[district].scalers[featureConfig];
    const nFeats = sc.feature_columns.length;
    const wScaled = new Float32Array(LOOKBACK * nFeats);
    for (let t = 0; t < LOOKBACK; t++) {
      for (let f = 0; f < nFeats; f++) {
        wScaled[t * nFeats + f] = (W[t * nFeats + f] - sc.feat_mean[f]) / sc.feat_scale[f];
      }
    }

    const predScaled = forwardModel(expId, arch, nFeats, wScaled);
    const predKw = predScaled * sc.targ_scale + sc.targ_mean;
    const persKw = W[(LOOKBACK - 1) * nFeats];
    const deltaKw = predKw - persKw;
    const deltaPct = persKw > 1e-9 ? (deltaKw / persKw) * 100.0 : null;
    const mMeta = B.models[expId];

    return {
      exp_id: expId,
      engine: "checkpoint_bundle_f32",
      scale,
      district,
      architecture: arch,
      feature_config: featureConfig,
      horizon_minutes: 15,
      lookback_intervals: LOOKBACK,
      predicted_demand_kw: predKw,
      naive_persistence_kw: persKw,
      delta_vs_persistence_kw: deltaKw,
      delta_vs_persistence_pct: deltaPct,
      historical_consumption_kw: Array.from({ length: LOOKBACK }, (_, i) => W[i * nFeats]),
      validation: report,
      checkpoint: {
        file: avail.checkpoint_file,
        sha256_16: avail.checkpoint_sha256_16,
        best_epoch: mMeta.epoch,
        val_loss: mMeta.val_loss,
        test_rmse_kw: avail.test_rmse_kw,
        test_r2: avail.test_r2,
      },
    };
  }

  /** Reproducible seeded PRNG (Mulberry32) + Box-Muller standard normals. */
  function mulberry32(seed) {
    let a = (Number(seed) || 42) >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function generateScenarioLocal(district, scenario, seed = 42, perturbScale = 0.12) {
    const distObj = B.districts[district];
    const reg = distObj && distObj.regimes[scenario];
    if (!reg) throw new Error(`Unknown scenario "${scenario}" for ${district}.`);
    const rand = mulberry32(seed * 2654435761 + scenario.length * 97);
    const pick = Math.floor(rand() * reg.pool_refs.length);
    const win = distObj.train_pool[reg.pool_refs[pick]];
    const cStats = distObj.feature_stats.consumption_only.demand_target;
    const floorKw = Math.max(0.05, cStats.min * 0.5);

    const z = new Float64Array(LOOKBACK);
    for (let i = 0; i < LOOKBACK; i += 2) {
      const u1 = Math.max(1e-12, rand()), u2 = rand();
      const mag = Math.sqrt(-2.0 * Math.log(u1));
      z[i] = mag * Math.cos(2 * Math.PI * u2);
      if (i + 1 < LOOKBACK) z[i + 1] = mag * Math.sin(2 * Math.PI * u2);
    }

    const cons = [];
    for (let i = 0; i < LOOKBACK; i++) {
      let delta = 0;
      for (let j = 0; j <= i; j++) delta += reg.chol[i][j] * z[j];
      const v = perturbScale > 0 ? Math.max(floorKw, win.c[i] + perturbScale * delta) : win.c[i];
      cons.push(+v.toFixed(4));
    }

    return {
      district,
      scenario,
      scenario_label: B.scenario_presets[scenario].label,
      scenario_description: B.scenario_presets[scenario].description,
      seed,
      source_train_window_index: win.train_idx,
      regime_pool_size: reg.full_pool_size,
      reference_end_timestamp: win.end_ts,
      consumption_kw: cons,
      paired_weather: {
        consumption_weather_retro: {
          columns: B.weather_columns.consumption_weather_retro,
          rows: win.retro.map(r => r.slice()),
        },
        consumption_weather_operational: {
          columns: B.weather_columns.consumption_weather_operational,
          rows: win.oper.map(r => r.slice()),
        },
      },
    };
  }

  function splitCsvLine(line) {
    const out = [];
    let cur = "", inQ = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (ch === '"') inQ = !inQ;
      else if (ch === "," && !inQ) { out.push(cur.trim()); cur = ""; }
      else cur += ch;
    }
    out.push(cur.trim());
    return out;
  }

  function parseCsvLocal(csvText, featureConfig = "consumption_only", rowOffset = 0) {
    if (!csvText || !csvText.trim()) throw new Error("Uploaded CSV is empty.");
    const lines = csvText.replace(/\r\n/g, "\n").replace(/\r/g, "\n")
      .split("\n").map(l => l.trim()).filter(Boolean);
    if (!lines.length) throw new Error("Uploaded CSV contains no non-empty rows.");

    const rawRows = lines.map(splitCsvLine);
    const first = rawRows[0];
    const hasHeader = first.some(c => Number.isNaN(Number(c)) || c === "");
    const wxCols = B.feature_sets[featureConfig].slice(1);

    let headers, dataRows;
    if (!hasHeader) {
      if (wxCols.length > 0) {
        throw new Error(`Weather-dependent configuration requires a CSV header row naming consumption and all 9 weather columns.`);
      }
      if (rawRows.every(r => r.length === 1)) {
        headers = ["demand_target"]; dataRows = rawRows;
      } else if (rawRows.length === 1 && rawRows[0].length === LOOKBACK) {
        headers = ["demand_target"]; dataRows = rawRows[0].map(v => [v]);
      } else {
        throw new Error("Headerless CSV must have either 1 column of 16 rows or 1 row of 16 values, or include a header row.");
      }
    } else {
      headers = first;
      dataRows = rawRows.slice(1);
    }

    if (dataRows.length < LOOKBACK) {
      throw new Error(`CSV must contain at least ${LOOKBACK} consecutive data rows; found ${dataRows.length}.`);
    }
    const start = rowOffset < 0 ? dataRows.length - LOOKBACK : Number(rowOffset) || 0;
    if (start < 0 || start + LOOKBACK > dataRows.length) {
      throw new Error(`Row offset ${rowOffset} out of range for ${dataRows.length} CSV rows.`);
    }
    const sel = dataRows.slice(start, start + LOOKBACK);
    const lowerMap = Object.fromEntries(headers.map((h, idx) => [h.toLowerCase(), idx]));
    const aliases = ["demand_target", "consumption_kw", "aggregate_demand_kw_primary", "actual_demand_kw", "demand_kw", "kw"];
    let consIdx = aliases.map(a => lowerMap[a]).find(i => i !== undefined);
    if (consIdx === undefined) {
      if (headers.length === 1 && wxCols.length === 0) consIdx = 0;
      else throw new Error(`CSV missing consumption column (expected one of: ${aliases.join(", ")}).`);
    }

    if (lowerMap.timestamp !== undefined) {
      const tsIdx = lowerMap.timestamp;
      const ms = sel.map(r => Date.parse(r[tsIdx].replace(" ", "T")));
      const seen = new Set();
      for (let i = 0; i < ms.length; i++) {
        if (!Number.isFinite(ms[i])) {
          throw new Error(`Invalid timestamp "${sel[i][tsIdx]}" at CSV row ${start + i + 1}.`);
        }
        if (seen.has(ms[i])) {
          throw new Error("CSV `timestamp` column contains duplicate timestamps across the selected 16 rows.");
        }
        seen.add(ms[i]);
        if (i > 0 && Math.abs(ms[i] - ms[i - 1] - 15 * 60e3) > 1000) {
          throw new Error("CSV `timestamp` column is not strictly consecutive at 15-minute intervals across the selected 16 rows.");
        }
      }
    }

    const consVals = sel.map((r, i) => parseFinite(r[consIdx], `CSV row ${start + i + 1} consumption`));
    let wxMatrix = null;
    if (wxCols.length > 0) {
      const missing = wxCols.filter(c => lowerMap[c.toLowerCase()] === undefined);
      if (missing.length) {
        throw new Error(`CSV missing required trained weather column(s): ${missing.join(", ")}.`);
      }
      const idxs = wxCols.map(c => lowerMap[c.toLowerCase()]);
      wxMatrix = sel.map((r, i) => idxs.map((ci, k) => parseFinite(r[ci], `CSV row ${start + i + 1} ${wxCols[k]}`)));
    }

    return {
      total_rows: dataRows.length,
      row_offset: start,
      headers,
      consumption_kw: consVals,
      weather_rows: wxMatrix,
    };
  }

  // Export engine for headless Node reconciliation tests
  window.SIMULATOR_ENGINE = {
    predictLocal,
    validateInputsLocal,
    generateScenarioLocal,
    parseCsvLocal,
  };

  // ------------------------------------------------------------------
  // 3. UI Integration for `#view-simulator`
  // ------------------------------------------------------------------
  window.SIMULATOR_INIT = function (helpers) {
    const { showTT, hideTT, ttRow, seg, esc, fx } = helpers;
    const $ = s => document.querySelector(s);

    const state = {
      mounted: false,
      scale: "aggregate",
      district: "mathura",
      arch: "LSTM",
      features: "consumption_only",
      mode: "scenario", // "manual" | "scenario" | "csv"
      scenario: "evening_peak",
      seed: 42,
      consumption: Array(LOOKBACK).fill(""),
      weatherSource: "paired", // "paired" | "historical" | "csv" | "manual" | "none"
      histIdx: 0,
      weatherRows: null, // 16 x 9 or null
      showWxEditor: false,
      csvText: "",
      csvName: "",
      csvTotalRows: 0,
      csvOffset: 0,
      scenarioMeta: null,
      result: null,
      errorMsg: null,
      matrixFilter: "all",
    };

    const canTryApi = () => typeof location !== "undefined" && /^https?:$/i.test(location.protocol) && typeof fetch === "function";

    async function callPredict(consArr, wxRows) {
      if (canTryApi()) {
        try {
          const resp = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              scale: state.scale,
              district: state.district,
              arch: state.arch,
              feature_config: state.features,
              consumption_kw: consArr,
              weather_rows: wxRows,
            }),
          });
          const data = await resp.json();
          if (!data.ok) throw new Error(data.error || "Prediction failed.");
          return data.result;
        } catch (err) {
          if (err && /Expected|Invalid|Missing|Non-numeric|outside physical|requires exact|Unsupported/i.test(err.message)) {
            throw err;
          }
          // Fall back to local Float32 checkpoint bundle if static HTTP server has no /api
        }
      }
      return predictLocal(state.scale, state.district, state.arch, state.features, consArr, wxRows);
    }

    async function callScenario() {
      if (canTryApi()) {
        try {
          const resp = await fetch("/api/scenario", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ district: state.district, scenario: state.scenario, seed: state.seed }),
          });
          const data = await resp.json();
          if (data.ok) return data.result;
        } catch (_) { /* fallback to bundle */ }
      }
      return generateScenarioLocal(state.district, state.scenario, state.seed);
    }

    function currentExpId() {
      return `${state.scale === "aggregate" ? "agg" : "ind"}_${state.district}_${state.arch}_${state.features}`;
    }

    function currentAvail() {
      return B.availability.find(r => r.exp_id === currentExpId());
    }

    function syncWeatherForConfig() {
      if (state.features === "consumption_only") return;
      if (state.weatherSource === "paired" && state.scenarioMeta && state.scenarioMeta.paired_weather) {
        const pw = state.scenarioMeta.paired_weather[state.features];
        if (pw) state.weatherRows = pw.rows.map(r => r.slice());
      } else if (state.weatherSource === "historical") {
        const wins = B.districts[state.district].historical_weather_windows;
        const w = wins[Math.min(state.histIdx, wins.length - 1)];
        if (w) state.weatherRows = w[state.features].map(r => r.slice());
      }
    }

    async function loadScenarioAndPredict() {
      try {
        const sc = await callScenario();
        state.scenarioMeta = sc;
        state.consumption = sc.consumption_kw.map(v => String(v));
        if (state.features !== "consumption_only" && (state.weatherSource === "paired" || !state.weatherRows)) {
          state.weatherSource = "paired";
          state.weatherRows = sc.paired_weather[state.features].rows.map(r => r.slice());
        }
        state.errorMsg = null;
        if (state.scale === "aggregate") {
          state.result = await callPredict(state.consumption, state.features === "consumption_only" ? null : state.weatherRows);
        } else {
          state.result = null;
        }
      } catch (e) {
        state.errorMsg = e.message || String(e);
        state.result = null;
      }
      renderWorkspace();
    }

    async function runPrediction() {
      state.errorMsg = null;
      try {
        const res = await callPredict(
          state.consumption,
          state.features === "consumption_only" ? null : state.weatherRows
        );
        state.result = res;
      } catch (e) {
        state.result = null;
        state.errorMsg = e.message || String(e);
      }
      renderResultPanel();
      highlightInputCells();
    }

    function buildCsvTemplate() {
      const wins = B.districts[state.district].historical_weather_windows;
      const w = wins[0];
      const fCols = B.feature_sets[state.features];
      const endMs = Date.parse(w.end_timestamp.replace(" ", "T"));
      const lines = [["timestamp", ...fCols].join(",")];
      for (let i = 0; i < LOOKBACK; i++) {
        const t = new Date(endMs - (LOOKBACK - 1 - i) * 15 * 60e3).toISOString().slice(0, 16).replace("T", " ");
        const wx = state.features === "consumption_only" ? [] : w[state.features][i];
        lines.push([t, w.consumption_kw[i], ...wx].join(","));
      }
      const blob = new Blob([lines.join("\n") + "\n"], { type: "text/csv;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `simulator_template_${state.district}_${state.features}.csv`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }

    function renderControls() {
      const ctl = $("#sim-controls");
      ctl.innerHTML = "";
      const left = document.createElement("div");
      left.style.cssText = "display:flex;flex-wrap:wrap;gap:14px 18px;align-items:flex-end";

      left.append(
        seg("Scale", "scale", [["aggregate", "Aggregate"], ["individual", "Individual"]], state, () => {
          state.errorMsg = null;
          if (state.scale === "individual") state.result = null;
          renderControls();
          renderWorkspace();
        }),
        seg("District", "district", [["mathura", "Mathura"], ["bareilly", "Bareilly"]], state, () => {
          state.histIdx = 0;
          renderControls();
          loadScenarioAndPredict();
        }),
        seg("Architecture", "arch", [["1D_CNN", "1D CNN"], ["LSTM", "LSTM"], ["CNN_LSTM", "CNN–LSTM"]], state, () => {
          renderControls();
          if (state.scale === "aggregate") runPrediction(); else renderWorkspace();
        }),
        seg("Features", "features", [
          ["consumption_only", FEAT_SHORT.consumption_only],
          ["consumption_weather_retro", FEAT_SHORT.consumption_weather_retro],
          ["consumption_weather_operational", FEAT_SHORT.consumption_weather_operational],
        ], state, () => {
          if (state.features !== "consumption_only" && !state.weatherRows) {
            if (state.scenarioMeta) {
              state.weatherSource = "paired";
              syncWeatherForConfig();
            } else {
              state.weatherSource = "historical";
              syncWeatherForConfig();
            }
          } else if (state.features !== "consumption_only") {
            syncWeatherForConfig();
          }
          renderControls();
          renderWorkspace();
          if (state.scale === "aggregate") runPrediction();
        })
      );

      const av = currentAvail();
      const right = document.createElement("div");
      right.style.cssText = "display:flex;align-items:center;gap:10px";
      right.innerHTML = av && av.supported
        ? `<span class="badge b-ok">Supported · ${esc(av.checkpoint_file)}</span>`
        : `<span class="badge b-err">Unsupported · Inference Blocked</span>`;
      ctl.append(left, right);
    }

    function highlightInputCells() {
      const cStats = B.districts[state.district].feature_stats.consumption_only.demand_target;
      document.querySelectorAll("#sim-grid-16 .sim-cell").forEach((cell, idx) => {
        cell.classList.remove("ood", "bad");
        const raw = String(state.consumption[idx] ?? "").trim();
        const v = Number(raw);
        if (raw === "" || !Number.isFinite(v) || v < 0) cell.classList.add("bad");
        else if (v < cStats.p01 || v > cStats.p99) cell.classList.add("ood");
      });
    }

    function renderWeatherSection() {
      if (state.features === "consumption_only") return "";
      const wxCols = B.feature_sets[state.features].slice(1);
      const wins = B.districts[state.district].historical_weather_windows;
      const hasWx = Array.isArray(state.weatherRows) && state.weatherRows.length === LOOKBACK;

      return `
        <div class="wx-box">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap">
            <div>
              <div style="font-size:12px;font-weight:600;color:var(--cyan)">Trained Meteorological Inputs · 9 variables × 16 steps</div>
              <div class="muted" style="font-size:11.5px">Weather values are never fabricated. Load a pipeline historical window, use paired scenario weather, upload a 10-column CSV, or edit below.</div>
            </div>
            <span class="badge ${hasWx ? "b-ok" : "b-err"}">${hasWx ? `16×9 loaded (${esc(state.weatherSource)})` : "Weather missing"}</span>
          </div>
          <div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:10px">
            <select class="sim-select" id="sim-wx-hist" aria-label="Historical weather window">
              ${wins.map((w, idx) => `<option value="${idx}" ${idx === state.histIdx ? "selected" : ""}>Pipeline test window · ${esc(w.summary)}</option>`).join("")}
            </select>
            <button type="button" class="btn-ghost" id="btn-load-hist-wx">Load pipeline weather</button>
            ${state.scenarioMeta ? `<button type="button" class="btn-ghost" id="btn-load-paired-wx">Use scenario paired weather</button>` : ""}
            <button type="button" class="btn-ghost" id="btn-toggle-wx-ed">${state.showWxEditor ? "Hide" : "Inspect / edit"} 16×9 table</button>
            <button type="button" class="btn-ghost" id="btn-clear-wx">Clear weather</button>
          </div>
          ${state.showWxEditor ? `
            <div class="table-wrap wx-table" style="margin-top:10px;max-height:240px">
              <table>
                <thead><tr><th>Step</th>${wxCols.map(c => {
                  const short = c.replace(/^retro_|^operational_/, "");
                  const unit = B.weather_units[short] || "";
                  return `<th class="r" title="${esc(c)}">${esc(short)}${unit ? `<br><span class="dim">${esc(unit)}</span>` : ""}</th>`;
                }).join("")}</tr></thead>
                <tbody>
                  ${Array.from({ length: LOOKBACK }, (_, r) => `<tr><td class="num dim">t${r - 15 === 0 ? "0" : r - 15}</td>` +
                    wxCols.map((_, c) => {
                      const val = hasWx && state.weatherRows[r] ? state.weatherRows[r][c] : "";
                      return `<td class="r"><input type="text" data-wr="${r}" data-wc="${c}" value="${esc(val)}" aria-label="Weather row ${r + 1} col ${c + 1}"/></td>`;
                    }).join("") + `</tr>`).join("")}
                </tbody>
              </table>
            </div>
          ` : ""}
        </div>`;
    }

    function renderWorkspace() {
      const body = $("#sim-body");
      const av = currentAvail();

      if (!av || !av.supported) {
        body.innerHTML = `
          <div class="glass panel">
            <div class="notice" style="border-color:rgba(239,68,68,0.42);background:rgba(239,68,68,0.08)">
              <svg viewBox="0 0 24 24" fill="none" stroke="#EF4444" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16h.01"/></svg>
              <div>
                <b>Unsupported configuration: <span class="num">${esc(av ? av.exp_id : currentExpId())}</span> (${SCALE_LABEL[state.scale]} · ${DIST_LABEL[state.district]} · ${ARCH_LABEL[state.arch]} · ${FEAT_LABEL[state.features]})</b>
                <span class="muted" style="margin-top:4px">${esc(av ? av.reason : "Model artifacts unavailable.")}</span>
                <span class="muted" style="margin-top:6px">Data-integrity enforcement: aggregate feeder checkpoints are never substituted for individual household inference, and interrupted partial checkpoints are never executed. Switch <b>Scale → Aggregate</b> above to run any of the 18 verified aggregate models, or inspect the full 36-combination availability matrix below.</span>
              </div>
            </div>
          </div>`;
        renderAvailabilityMatrix();
        return;
      }

      const cStats = B.districts[state.district].feature_stats.consumption_only.demand_target;

      let modeHtml = "";
      if (state.mode === "scenario") {
        modeHtml = `
          <div class="sim-presets">
            ${Object.entries(B.scenario_presets).map(([k, p]) => `
              <button type="button" class="sim-preset ${state.scenario === k ? "on" : ""}" data-sc="${esc(k)}">
                <div class="p-t">${esc(p.label)}</div>
                <div class="p-d">${esc(p.description)}</div>
              </button>
            `).join("")}
          </div>
          <div style="display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-bottom:10px">
            <label class="muted" style="font-size:12px">Seed
              <input type="number" id="sim-seed" class="sim-input" style="width:88px;margin-left:6px" value="${esc(state.seed)}"/>
            </label>
            <button type="button" class="btn-ghost" id="btn-next-seed">Resample (+1 seed)</button>
            <button type="button" class="btn-ghost" id="btn-regen-sc">Regenerate scenario</button>
            ${state.scenarioMeta ? `<span class="dim num" style="font-size:11px">70% train pool n=${state.scenarioMeta.regime_pool_size} · base win #${state.scenarioMeta.source_train_window_index}</span>` : ""}
          </div>`;
      } else if (state.mode === "manual") {
        modeHtml = `
          <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px">
            <input type="text" id="sim-paste" class="sim-input" style="flex:1;min-width:220px" placeholder="Quick paste 16 comma- or space-separated kW values…"/>
            <button type="button" class="btn-ghost" id="btn-apply-paste">Apply 16 values</button>
            <button type="button" class="btn-ghost" id="btn-clear-manual">Clear</button>
          </div>`;
      } else if (state.mode === "csv") {
        const maxOff = Math.max(0, state.csvTotalRows - LOOKBACK);
        modeHtml = `
          <div class="sim-drop" id="sim-drop">
            <div style="font-size:12.5px;font-weight:580">Upload 15-minute interval CSV (${state.features === "consumption_only" ? "consumption kW" : "consumption + 9 trained weather columns"})</div>
            <div class="muted" style="font-size:11.5px;margin:4px 0 10px">Requires ≥ 16 consecutive intervals (4 hours). Accepted demand headers: <span class="num">demand_target</span>, <span class="num">consumption_kw</span>, <span class="num">actual_demand_kw</span>.</div>
            <div style="display:flex;justify-content:center;gap:10px;flex-wrap:wrap;align-items:center">
              <label class="btn-ghost" style="cursor:pointer">
                Choose CSV file
                <input type="file" id="sim-csv-file" accept=".csv,.txt" style="display:none"/>
              </label>
              <button type="button" class="btn-ghost" id="btn-dl-tpl">Download valid 16-row CSV template</button>
            </div>
            ${state.csvName ? `<div class="num" style="margin-top:8px;font-size:11.5px;color:var(--cyan)">Loaded: ${esc(state.csvName)} (${state.csvTotalRows} data rows)</div>` : ""}
            ${state.csvTotalRows > LOOKBACK ? `
              <div style="margin-top:8px;display:flex;justify-content:center;align-items:center;gap:8px">
                <span class="muted" style="font-size:11.5px">16-row slice start (0..${maxOff}):</span>
                <input type="number" id="sim-csv-off" class="sim-input" style="width:82px" min="0" max="${maxOff}" value="${state.csvOffset}"/>
              </div>` : ""}
          </div>`;
      }

      body.innerHTML = `
        <div class="grid g-main">
          <div class="glass panel">
            <div class="panel-head">
              <div>
                <div class="panel-title">Lookback input · 16 consecutive 15-min intervals (4.0 hours, kW)</div>
                <div class="panel-sub">${DIST_LABEL[state.district]} training distribution: min <span class="num">${fx(cStats.min, 2)}</span> · p05–p95 <span class="num">${fx(cStats.p05, 2)}–${fx(cStats.p95, 2)}</span> · max <span class="num">${fx(cStats.max, 2)} kW</span></div>
              </div>
              <div class="seg" id="sim-mode-seg" role="group" aria-label="Input mode">
                <button type="button" data-m="manual" class="${state.mode === "manual" ? "on" : ""}">Manual entry</button>
                <button type="button" data-m="scenario" class="${state.mode === "scenario" ? "on" : ""}">Random scenario</button>
                <button type="button" data-m="csv" class="${state.mode === "csv" ? "on" : ""}">CSV upload</button>
              </div>
            </div>

            ${modeHtml}

            <div class="sim-grid-16" id="sim-grid-16">
              ${Array.from({ length: LOOKBACK }, (_, i) => {
                const lag = i - (LOOKBACK - 1);
                const mins = Math.abs(lag) * 15;
                const hh = Math.floor(mins / 60), mm = String(mins % 60).padStart(2, "0");
                const lbl = lag === 0 ? "t0 (now)" : `t${lag} (-${hh}:${mm})`;
                return `<div class="sim-cell">
                  <label><span>#${i + 1}</span><span>${lbl}</span></label>
                  <input type="text" inputmode="decimal" data-step="${i}" value="${esc(state.consumption[i] ?? "")}" placeholder="kW" aria-label="Consumption step ${i + 1} (${lbl}) in kW"/>
                </div>`;
              }).join("")}
            </div>

            ${renderWeatherSection()}

            <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;margin-top:14px;flex-wrap:wrap">
              <button type="button" class="btn-primary" id="btn-run-sim">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4"><path d="M13 2 4 14h7l-1 8 9-12h-7z"/></svg>
                Predict next 15-min demand (t+1)
              </button>
              <span class="dim" style="font-size:11.5px">Exact feature order: <span class="num">${esc(B.feature_sets[state.features].join(", "))}</span></span>
            </div>
          </div>

          <div class="glass panel" id="sim-result-panel"></div>
        </div>`;

      bindWorkspaceEvents();
      highlightInputCells();
      renderResultPanel();
      renderAvailabilityMatrix();
    }

    function bindWorkspaceEvents() {
      document.querySelectorAll("#sim-mode-seg button").forEach(b => b.addEventListener("click", () => {
        state.mode = b.dataset.m;
        renderWorkspace();
      }));

      document.querySelectorAll("#sim-grid-16 input").forEach(inp => inp.addEventListener("input", () => {
        state.consumption[+inp.dataset.step] = inp.value;
        highlightInputCells();
      }));

      const runBtn = $("#btn-run-sim");
      if (runBtn) runBtn.addEventListener("click", runPrediction);

      // Scenario mode controls
      document.querySelectorAll(".sim-preset").forEach(b => b.addEventListener("click", () => {
        state.scenario = b.dataset.sc;
        loadScenarioAndPredict();
      }));
      const seedInp = $("#sim-seed");
      if (seedInp) seedInp.addEventListener("change", () => { state.seed = parseInt(seedInp.value, 10) || 0; loadScenarioAndPredict(); });
      const nextSeedBtn = $("#btn-next-seed");
      if (nextSeedBtn) nextSeedBtn.addEventListener("click", () => { state.seed = (Number(state.seed) || 0) + 1; loadScenarioAndPredict(); });
      const regenBtn = $("#btn-regen-sc");
      if (regenBtn) regenBtn.addEventListener("click", loadScenarioAndPredict);

      // Manual paste controls
      const pasteBtn = $("#btn-apply-paste");
      if (pasteBtn) pasteBtn.addEventListener("click", () => {
        const raw = ($("#sim-paste").value || "").trim();
        const parts = raw.split(/[\s,;]+/).filter(Boolean);
        state.consumption = parts;
        if (parts.length === LOOKBACK) {
          renderWorkspace();
          runPrediction();
        } else {
          state.result = null;
          state.errorMsg = `Expected 16 values in quick-paste box, but received ${parts.length}.`;
          state.consumption = Array.from({ length: LOOKBACK }, (_, i) => parts[i] ?? "");
          renderWorkspace();
        }
      });
      const clearBtn = $("#btn-clear-manual");
      if (clearBtn) clearBtn.addEventListener("click", () => {
        state.consumption = Array(LOOKBACK).fill("");
        state.result = null;
        state.errorMsg = "Enter 16 consecutive consumption observations (kW) to run inference.";
        renderWorkspace();
      });

      // CSV controls
      const dlBtn = $("#btn-dl-tpl");
      if (dlBtn) dlBtn.addEventListener("click", buildCsvTemplate);
      const csvFile = $("#sim-csv-file");
      if (csvFile) csvFile.addEventListener("change", ev => {
        const file = ev.target.files && ev.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = () => {
          state.csvText = String(reader.result || "");
          state.csvName = file.name;
          state.csvOffset = 0;
          applyParsedCsv();
        };
        reader.readAsText(file);
      });
      const csvOff = $("#sim-csv-off");
      if (csvOff) csvOff.addEventListener("change", () => {
        state.csvOffset = parseInt(csvOff.value, 10) || 0;
        applyParsedCsv();
      });

      // Weather controls
      const histSel = $("#sim-wx-hist");
      if (histSel) histSel.addEventListener("change", () => { state.histIdx = +histSel.value; });
      const loadHistBtn = $("#btn-load-hist-wx");
      if (loadHistBtn) loadHistBtn.addEventListener("click", () => {
        state.weatherSource = "historical";
        syncWeatherForConfig();
        renderWorkspace();
        runPrediction();
      });
      const loadPairedBtn = $("#btn-load-paired-wx");
      if (loadPairedBtn) loadPairedBtn.addEventListener("click", () => {
        state.weatherSource = "paired";
        syncWeatherForConfig();
        renderWorkspace();
        runPrediction();
      });
      const toggleWxBtn = $("#btn-toggle-wx-ed");
      if (toggleWxBtn) toggleWxBtn.addEventListener("click", () => {
        if (!state.weatherRows) {
          state.weatherRows = Array.from({ length: LOOKBACK }, () => Array(9).fill(""));
          state.weatherSource = "manual";
        }
        state.showWxEditor = !state.showWxEditor;
        renderWorkspace();
      });
      const clearWxBtn = $("#btn-clear-wx");
      if (clearWxBtn) clearWxBtn.addEventListener("click", () => {
        state.weatherRows = null;
        state.weatherSource = "none";
        state.result = null;
        renderWorkspace();
        runPrediction();
      });
      document.querySelectorAll(".wx-table input").forEach(inp => inp.addEventListener("input", () => {
        const r = +inp.dataset.wr, c = +inp.dataset.wc;
        if (!state.weatherRows) state.weatherRows = Array.from({ length: LOOKBACK }, () => Array(9).fill(""));
        state.weatherRows[r][c] = inp.value;
        state.weatherSource = "manual";
      }));
    }

    function applyParsedCsv() {
      try {
        const parsed = parseCsvLocal(state.csvText, state.features, state.csvOffset);
        state.csvTotalRows = parsed.total_rows;
        state.consumption = parsed.consumption_kw.map(v => String(v));
        if (parsed.weather_rows) {
          state.weatherRows = parsed.weather_rows;
          state.weatherSource = "csv";
        }
        state.errorMsg = null;
        renderWorkspace();
        runPrediction();
      } catch (e) {
        state.result = null;
        state.errorMsg = e.message || String(e);
        renderWorkspace();
      }
    }

    function renderResultPanel() {
      const panel = $("#sim-result-panel");
      if (!panel) return;

      if (state.errorMsg || !state.result) {
        panel.innerHTML = `
          <div class="panel-head">
            <div><div class="panel-title">Prediction output · t+1 (+15 minutes)</div>
            <div class="panel-sub">${DIST_LABEL[state.district]} · ${ARCH_LABEL[state.arch]} · ${FEAT_LABEL[state.features]}</div></div>
            <span class="badge b-err">Validation Failed</span>
          </div>
          <div class="sim-status err">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#EF4444" stroke-width="2" style="flex:none;margin-top:1px"><circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16h.01"/></svg>
            <div>
              <b>Inference blocked due to invalid or incomplete input</b>
              <div class="muted" style="margin-top:3px">${esc(state.errorMsg || "Provide 16 valid observations and click Predict.")}</div>
            </div>
          </div>`;
        return;
      }

      const R = state.result;
      const V = R.validation;
      const warns = V.warnings || [];
      const deltaSign = R.delta_vs_persistence_kw >= 0 ? "+" : "";
      const pctStr = R.delta_vs_persistence_pct != null ? ` (${R.delta_vs_persistence_pct >= 0 ? "+" : ""}${R.delta_vs_persistence_pct.toFixed(2)}%)` : "";
      const engineBadge = R.engine === "pytorch_eval"
        ? '<span class="badge b-ok">PyTorch Live API · eval()</span>'
        : '<span class="badge b-info">Checkpoint Bundle · Float32 eval()</span>';

      panel.innerHTML = `
        <div class="panel-head">
          <div>
            <div class="panel-title">Next-15-minute forecast · t+1</div>
            <div class="panel-sub"><span class="num">${esc(R.exp_id)}</span> · epoch ${R.checkpoint.best_epoch}</div>
          </div>
          ${engineBadge}
        </div>

        <div class="sim-hero lift">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:10px">
            <div>
              <div class="eyebrow" style="color:var(--amber)">Predicted Demand (t+1 · +15 min)</div>
              <div class="sim-hero-val">${fx(R.predicted_demand_kw, 3)} <span style="font-size:15px;font-weight:500;color:var(--text-2)">kW</span></div>
            </div>
            <div style="text-align:right">
              <div class="kpi-label">vs Persistence y(t0)</div>
              <div class="num" style="font-size:14.5px;margin-top:4px;color:${R.delta_vs_persistence_kw >= 0 ? "var(--amber)" : "var(--cyan)"}">${deltaSign}${fx(R.delta_vs_persistence_kw, 3)} kW${esc(pctStr)}</div>
              <div class="dim num" style="font-size:11px">y(t0) = ${fx(R.naive_persistence_kw, 3)} kW</div>
            </div>
          </div>
        </div>

        <div class="stats" style="margin-top:10px">
          <div class="stat"><div class="k">District</div><div class="v" style="font-size:13.5px;font-family:var(--sans);font-weight:600">${DIST_LABEL[R.district]}</div><div class="s">Aggregate feeder scale</div></div>
          <div class="stat"><div class="k">Architecture</div><div class="v" style="font-size:13.5px;font-family:var(--sans);font-weight:600">${ARCH_LABEL[R.architecture]}</div><div class="s">Test RMSE ${fx(R.checkpoint.test_rmse_kw, 4)} kW</div></div>
          <div class="stat"><div class="k">Feature Config</div><div class="v" style="font-size:12.5px;font-family:var(--sans);font-weight:600">${FEAT_LABEL[R.feature_config]}</div><div class="s">${V.in_features} input channel${V.in_features > 1 ? "s" : ""}</div></div>
          <div class="stat"><div class="k">4h Window Mean</div><div class="v">${fx(V.input_summary_kw.mean, 3)}</div><div class="s">min ${fx(V.input_summary_kw.min, 2)} · max ${fx(V.input_summary_kw.max, 2)} kW</div></div>
        </div>

        ${warns.length === 0 ? `
          <div class="sim-status ok">
            <span class="badge b-ok">Valid</span>
            <div>All 16 intervals validated within ${DIST_LABEL[R.district]} training distribution (<span class="num">[${fx(V.training_consumption_range_kw.p01, 2)}, ${fx(V.training_consumption_range_kw.p99, 2)}] kW</span>).</div>
          </div>
        ` : `
          <div class="sim-status warn">
            <span class="badge b-warn">OOD Warning</span>
            <div>
              <b>Out-of-distribution input detected (no values clipped):</b>
              ${warns.map(w => `<div class="muted" style="margin-top:2px">${esc(w.message)}</div>`).join("")}
            </div>
          </div>
        `}

        <div class="panel-head" style="margin:14px 0 6px">
          <div class="panel-sub">16-step historical window (4h) + predicted t+1</div>
          <div class="legend">
            <span><i style="background:#00C8E8"></i>Historical (t-15…t0)</span>
            <span><i style="background:#F5B942"></i>Predicted t+1</span>
            <span style="color:#94A3B8"><i class="dash"></i>Persistence</span>
          </div>
        </div>
        <div class="chart" id="sim-chart"></div>
        <div class="panel-sub" style="margin-top:10px;padding-top:8px;border-top:1px solid var(--glass-border-soft)">
          <b>Simulation scope note:</b> Predictions generated from random scenarios or custom manual inputs represent exploratory model response simulations, not evaluations of real-world forecasting accuracy. Empirical test-set accuracy is reported in the <b>Forecast Analysis</b> and <b>Model Comparison</b> tabs.
        </div>`;

      drawSimulatorChart();
    }

    function drawSimulatorChart() {
      const el = $("#sim-chart");
      if (!el || !state.result) return;
      const R = state.result;
      const hist = R.historical_consumption_kw;
      const pred = R.predicted_demand_kw;
      const pers = R.naive_persistence_kw;
      const cStats = R.validation.training_consumption_range_kw;

      const W = el.clientWidth || 460, H = 245, P = { l: 48, r: 20, t: 14, b: 30 };
      const totalPts = LOOKBACK + 1; // indices 0..15 historical, 16 predicted t+1
      const allVals = [...hist, pred, pers, cStats.p05, cStats.p95];
      let lo = Math.min(...allVals), hi = Math.max(...allVals);
      const pad = (hi - lo) * 0.1 || 1;
      lo = Math.max(0, lo - pad); hi += pad;

      const x = i => P.l + (i / (totalPts - 1)) * (W - P.l - P.r);
      const y = v => H - P.b - ((v - lo) / ((hi - lo) || 1)) * (H - P.t - P.b);

      let svg = `<svg viewBox="0 0 ${W} ${H}" height="${H}" role="img" aria-label="16-step historical demand and next-step prediction">`;
      // Shaded training p05..p95 band
      const y95 = y(cStats.p95), y05 = y(cStats.p05);
      svg += `<rect x="${P.l}" y="${y95}" width="${W - P.l - P.r}" height="${Math.max(1, y05 - y95)}" fill="rgba(57,135,255,0.08)" stroke="rgba(57,135,255,0.18)" stroke-dasharray="3 3"/>`;
      svg += `<text x="${P.l + 6}" y="${y95 + 11}" fill="rgba(148,163,184,0.65)" font-size="9.5" font-family="var(--mono)">Train p05–p95</text>`;

      // Y gridlines
      svg += '<g class="axis">';
      for (let k = 0; k <= 4; k++) {
        const tv = lo + (k / 4) * (hi - lo);
        svg += `<line class="gridline" x1="${P.l}" x2="${W - P.r}" y1="${y(tv)}" y2="${y(tv)}"/><text x="${P.l - 7}" y="${y(tv) + 3.5}" text-anchor="end">${tv.toFixed(1)}</text>`;
      }
      for (const idx of [0, 4, 8, 12, 15, 16]) {
        const lbl = idx === 16 ? "t+1" : idx === 15 ? "t0" : `t${idx - 15}`;
        svg += `<text x="${x(idx)}" y="${H - 8}" text-anchor="middle" fill="${idx === 16 ? "#F5B942" : "#64748B"}">${lbl}</text>`;
      }
      svg += `<text class="axis-title" transform="translate(12 ${(H - P.b + P.t) / 2}) rotate(-90)" text-anchor="middle">kW</text></g>`;

      // Vertical separator at forecast origin t0 -> t+1
      svg += `<line x1="${x(15)}" x2="${x(15)}" y1="${P.t}" y2="${H - P.b}" stroke="rgba(245,185,66,0.28)" stroke-dasharray="3 3"/>`;

      // Historical cyan path
      let dHist = "";
      hist.forEach((v, i) => { dHist += (i === 0 ? "M" : "L") + x(i).toFixed(1) + " " + y(v).toFixed(1); });
      svg += `<path class="line" d="${dHist}" stroke="#00C8E8" stroke-width="2"/>`;

      // Persistence bridge & dot
      svg += `<line x1="${x(15)}" y1="${y(hist[15])}" x2="${x(16)}" y2="${y(pers)}" stroke="#94A3B8" stroke-width="1.3" stroke-dasharray="3 3"/>`;
      svg += `<circle cx="${x(16)}" cy="${y(pers)}" r="3.6" fill="#071321" stroke="#94A3B8" stroke-width="1.6"/>`;

      // Forecast bridge & predicted point
      svg += `<line x1="${x(15)}" y1="${y(hist[15])}" x2="${x(16)}" y2="${y(pred)}" stroke="#F5B942" stroke-width="2" stroke-dasharray="5 3"/>`;
      hist.forEach((v, i) => {
        svg += `<circle class="sim-pt" data-idx="${i}" cx="${x(i)}" cy="${y(v)}" r="3.2" fill="#00C8E8" stroke="#071321" stroke-width="1.3"/>`;
      });
      svg += `<circle cx="${x(16)}" cy="${y(pred)}" r="7" fill="rgba(245,185,66,0.22)"/>`;
      svg += `<circle class="sim-pt" data-idx="16" cx="${x(16)}" cy="${y(pred)}" r="4.6" fill="#F5B942" stroke="#071321" stroke-width="1.5"/>`;
      svg += `<rect x="${P.l}" y="${P.t}" width="${W - P.l - P.r}" height="${H - P.t - P.b}" fill="transparent"/></svg>`;
      el.innerHTML = svg;

      const root = el.querySelector("svg"), overlay = root.querySelector("rect:last-of-type");
      overlay.addEventListener("mousemove", ev => {
        const bb = root.getBoundingClientRect(), px = (ev.clientX - bb.left) * (W / bb.width);
        const idx = Math.max(0, Math.min(16, Math.round(((px - P.l) / (W - P.l - P.r)) * 16)));
        if (idx < 16) {
          showTT(`<div class="tt-h">Step #${idx + 1} · ${idx === 15 ? "t0 (0h00m)" : `t${idx - 15} (${(idx - 15) * 15} min)`}</div>` +
            ttRow("Observed input", fx(hist[idx], 3) + " kW", "#00C8E8"), ev.clientX, ev.clientY);
        } else {
          showTT(`<div class="tt-h">Forecast Horizon · t+1 (+15 min)</div>` +
            ttRow("Predicted", fx(pred, 3) + " kW", "#F5B942") +
            ttRow("Persistence", fx(pers, 3) + " kW", "#94A3B8"), ev.clientX, ev.clientY);
        }
      });
      overlay.addEventListener("mouseleave", hideTT);
    }

    function renderAvailabilityMatrix() {
      const box = $("#sim-matrix-panel");
      if (!box) return;
      const rows = B.availability.filter(r => state.matrixFilter === "all" || (state.matrixFilter === "supported" ? r.supported : !r.supported));
      const supCnt = B.availability.filter(r => r.supported).length;

      box.innerHTML = `
        <div class="panel-head">
          <div>
            <div class="panel-title">Model &amp; Preprocessing Availability Matrix (${supCnt} / ${B.availability.length} combinations runnable)</div>
            <div class="panel-sub">Audited against <span class="num">artifacts/checkpoints/</span>, <span class="num">checkpoint_integrity_manifest.csv</span>, and <span class="num">data/processed/</span>.</div>
          </div>
          <div class="seg" id="sim-mat-filter">
            <button type="button" data-f="all" class="${state.matrixFilter === "all" ? "on" : ""}">All (36)</button>
            <button type="button" data-f="supported" class="${state.matrixFilter === "supported" ? "on" : ""}">Supported (${supCnt})</button>
            <button type="button" data-f="unsupported" class="${state.matrixFilter === "unsupported" ? "on" : ""}">Unsupported (${B.availability.length - supCnt})</button>
          </div>
        </div>
        <div class="table-wrap" style="max-height:280px">
          <table>
            <thead>
              <tr>
                <th>Experiment ID</th><th>Scale</th><th>District</th><th>Architecture</th><th>Feature Config</th>
                <th>Inference Status</th><th>Audit Detail</th>
              </tr>
            </thead>
            <tbody>
              ${rows.map(r => `
                <tr>
                  <td class="id">${esc(r.exp_id)}</td>
                  <td>${SCALE_LABEL[r.scale]}</td>
                  <td>${DIST_LABEL[r.district]}</td>
                  <td>${ARCH_LABEL[r.arch]}</td>
                  <td>${FEAT_LABEL[r.feature_config]}</td>
                  <td>${r.supported
                    ? '<span class="badge b-ok">Supported</span>'
                    : r.status === "partial_checkpoint_excluded"
                      ? '<span class="badge b-warn">Excluded (Partial)</span>'
                      : '<span class="badge b-err">Unsupported</span>'}</td>
                  <td style="white-space:normal;min-width:260px;font-size:11.5px" class="muted">${esc(r.reason)}</td>
                </tr>`).join("")}
            </tbody>
          </table>
        </div>`;

      box.querySelectorAll("#sim-mat-filter button").forEach(b => b.addEventListener("click", () => {
        state.matrixFilter = b.dataset.f;
        renderAvailabilityMatrix();
      }));
    }

    return function renderSimulatorView() {
      if (!B) {
        $("#sim-body").innerHTML = `<div class="glass panel"><div class="notice"><div><b>Simulator bundle missing.</b><span class="muted">Run <span class="num">python app/build_simulator_bundle.py</span> to generate <span class="num">app/dashboard/simulator_bundle.js</span>.</span></div></div></div>`;
        return;
      }
      if (!state.mounted) {
        state.mounted = true;
        renderControls();
        loadScenarioAndPredict();
      } else {
        drawSimulatorChart();
      }
    };
  };
})();

