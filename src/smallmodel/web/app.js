let page = "classify",
  study = null;
const tabs = [
  ["classify", "Intent classifier"],
  ["study", "Compression study"],
  ["training", "Training and errors"],
  ["protocol", "Model card"],
];
function heading(t, d) {
  return `<div class="page-title"><div><h1>${t}</h1><p class="muted">${d}</p></div></div>`;
}
async function render(next = page) {
  page = next;
  setNav(tabs, page, guarded(render));
  if (!study) study = await api("/study");
  if (page === "classify") await classify();
  if (page === "study") comparison();
  if (page === "training") training();
  if (page === "protocol") protocol();
}
async function classify() {
  let m = await api("/model"),
    r = study.seeds["17"].models.distilled_student;
  $("#main").innerHTML =
    heading(
      "Small model. Real predictions.",
      "Classify banking support requests into 77 intents, with an explicit route to human review.",
    ) +
    `<div class="stats">${stat("Model parameters", fmt(r.parameters), "Mean-pooling student · 48-wide embeddings")}${stat("Test accuracy", pct(r.test.accuracy), "Release seed · 3,080 official test queries")}${stat("Review threshold", fmt(r.test.threshold, 3), "Chosen using validation data")}${stat("Runtime", m.ready ? (m.quantized ? "INT8" : "Float32") : "Not loaded", m.ready ? "Local CPU inference" : "Load the release artifact")}</div><div class="split"><section class="card"><h2>Try a support request</h2><label for="text">One request per line</label><textarea id="text" style="min-height:160px">My card still hasn't arrived after two weeks.
I don't recognise this card payment.</textarea><button class="primary" id="predict" ${m.ready ? "" : "disabled"}>Classify requests</button><div class="row" style="margin-top:15px">${["I forgot my PIN.", "The ATM gave me less cash than I requested.", "How do I close my account?"].map((t) => `<button class="small" data-example="${esc(t)}">${esc(t)}</button>`).join("")}</div>${!m.ready ? '<div class="notice">Run <code>python -m smallmodel.serve --demo --download-model</code> to load the checksummed release model.</div>' : ""}<p class="small muted" style="margin-top:18px">Confidence is calibrated on BANKING77. It does not detect every out-of-domain request. Suggestions do not change accounts or execute transactions.</p></section><section id="predictions" class="card">${empty("Predictions and their confidence will appear here.")}</section></div>`;
  document.querySelectorAll("[data-example]").forEach(
    (b) =>
      (b.onclick = () => {
        $("#text").value = b.dataset.example;
        $("#predict").click();
      }),
  );
  $("#predict").onclick = guarded(async () => {
    let button = $("#predict");
    button.disabled = true;
    try {
      let texts = $("#text")
          .value.split("\n")
          .filter((t) => t.trim()),
        d = await post("/predict", { texts });
      $("#predictions").innerHTML =
        `<div class="row spread"><h2>Classification results</h2><span class="small muted">${fmt(d.latency_ms, 2)} ms including tokenization</span></div>${d.results.map((r, i) => `<div style="margin:18px 0 28px"><p><strong>${esc(texts[i])}</strong></p><span class="badge ${r.abstained ? "warn" : "good"}">${r.abstained ? "Human review" : "Intent suggested"}</span>${r.suggestions.map((s, j) => `<div class="row spread" style="margin-top:12px"><span class="small ${j ? "muted" : ""}">${esc(s.intent.replaceAll("_", " "))}</span><span class="mono small">${pct(s.confidence)}</span></div><div class="bar"><span style="width:${100 * s.confidence}%"></span></div>`).join("")}</div>`).join("")}<details><summary>Artifact and runtime</summary><p class="mono small" style="overflow-wrap:anywhere">${esc(d.model_sha256)}</p><p class="small">${d.quantized ? "Quantized embedding and linear CPU kernels" : "Floating-point CPU model"}</p></details>`;
    } finally {
      button.disabled = false;
    }
  });
}
function comparison() {
  let seeds = Object.values(study.seeds),
    names = Object.keys(seeds[0].models),
    mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
  $("#main").innerHTML =
    heading(
      "Compression, with the tradeoffs intact.",
      "Three seeds. Six approaches. The same official test split, with checkpoints selected on validation accuracy.",
    ) +
    `<div class="card flush"><table><thead><tr><th>Approach</th><th>Mean accuracy</th><th>Seed range</th><th>Serialized weights</th><th>CPU p50 · seed 17</th></tr></thead><tbody>${names
      .map((n) => {
        let xs = seeds.map((s) => s.models[n].test.accuracy),
          m = seeds[0].models[n];
        return `<tr><td><strong>${esc(n.replaceAll("_", " "))}</strong></td><td>${pct(mean(xs))}<div class="bar"><span style="width:${100 * mean(xs)}%"></span></div></td><td>${pct(Math.min(...xs))}–${pct(Math.max(...xs))}</td><td>${m.serialized_bytes ? fmt(m.serialized_bytes / 1e6, 2) + " MB" : "—"}</td><td>${m.latency ? fmt(m.latency.p50_ms, 3) + " ms" : "Not measured"}</td></tr>`;
      })
      .join(
        "",
      )}</tbody></table></div><div class="grid two" style="margin-top:20px"><section class="card"><h2>What improved</h2><p>The distilled student outperformed its identically initialized supervised student in each reported seed. It also uses fewer parameters than the pretrained encoder.</p><p class="small muted">The lexical baseline is competitive. The LoRA configuration underperforms full tuning here; a small trainable parameter count does not guarantee comparable quality.</p></section><section class="card"><h2>What did not improve</h2><p>Quantized embedding and linear kernels reduced the saved weights by about 71%. They were slower than the floating-point student on this CPU.</p><p class="small muted">This is an intent classifier, not a generative assistant. Latency is a warmed single-example forward pass, excluding tokenization; live predictions above report total request processing.</p></section></div><div class="card" style="margin-top:20px"><h2>Selective prediction · release seed</h2><table><thead><tr><th>Model</th><th>Automatic coverage</th><th>Accuracy among accepted</th><th>Calibration error</th></tr></thead><tbody>${names
      .filter((n) => n !== "tfidf_logistic")
      .map((n) => {
        let t = seeds[0].models[n].test;
        return `<tr><td>${esc(n.replaceAll("_", " "))}</td><td>${pct(t.coverage)}</td><td>${pct(t.selective_accuracy || 0)}</td><td>${fmt(t.ece_10_bins, 3)}</td></tr>`;
      })
      .join(
        "",
      )}</tbody></table><p class="small muted" style="padding:15px">Thresholds target 95% accuracy on a validation subset of at least 50 examples. That target is not guaranteed on new data. ECE uses ten confidence bins.</p></div>`;
}
function training() {
  $("#main").innerHTML =
    heading(
      "Inspect the experiment, not just the headline.",
      "Training curves, held-out confusion pairs, and paired comparisons are retained for every seed.",
    ) +
    `<div class="toolbar"><label for="seed">Seed</label><select id="seed">${Object.keys(
      study.seeds,
    )
      .map((s) => "<option>" + s + "</option>")
      .join(
        "",
      )}</select><label for="model-select">Approach</label><select id="model-select">${Object.keys(
      study.seeds["17"].models,
    )
      .filter((n) => study.seeds["17"].models[n].history)
      .map(
        (n) =>
          `<option value="${n}" ${n === "distilled_student" ? "selected" : ""}>${n.replaceAll("_", " ")}</option>`,
      )
      .join("")}</select></div><div id="training-content"></div>`;
  const draw = () => {
    let seed = $("#seed").value,
      m = study.seeds[seed].models[$("#model-select").value];
    $("#training-content").innerHTML =
      `<div class="grid two"><section class="card"><h2>Validation learning curve</h2>${m.history.map((e) => `<div class="row spread"><span class="small mono">Epoch ${e.epoch}</span><span class="small">${pct(e.validation_accuracy)} · loss ${fmt(e.loss, 3)}</span></div><div class="bar" style="margin:5px 0 9px"><span style="width:${100 * e.validation_accuracy}%"></span></div>`).join("")}<p class="small muted">Selected epoch ${m.selected_epoch}; earliest tie. Training time ${fmt(m.train_seconds, 1)} seconds.</p></section><section class="card"><h2>Common test confusions</h2><table><thead><tr><th>True intent</th><th>Predicted intent</th><th>Count</th></tr></thead><tbody>${m.confusions.map((c) => `<tr><td class="small">${esc(c.truth.replaceAll("_", " "))}</td><td class="small">${esc(c.prediction.replaceAll("_", " "))}</td><td>${c.count}</td></tr>`).join("")}</tbody></table><details><summary>Paired accuracy differences</summary>${jsonView(study.seeds[seed].paired_accuracy)}</details></section></div>`;
  };
  $("#seed").onchange = draw;
  $("#model-select").onchange = draw;
  draw();
}
function protocol() {
  let r = study.seeds["17"];
  $("#main").innerHTML =
    heading(
      "A reproducible, bounded model.",
      "Pinned data and model revisions, audited splits, safe tensors, and explicit operating limits.",
    ) +
    `<div class="grid two"><section class="card"><h2>Training and evaluation</h2><p>BANKING77 contains 77 banking support intents. The study uses a stratified validation split from the official training data and retains the official 3,080-example test split.</p><p>Normalized duplicates are removed from training, including overlap with test text. Test labels are not used for training, threshold calibration, or checkpoint selection.</p><p>The teacher is a pretrained 4.4-million-parameter BERT encoder. The student is a 1.48-million-parameter classifier trained from scratch using labels and teacher logits.</p><p><a href="https://github.com/PolyAI-LDN/task-specific-datasets" target="_blank" rel="noreferrer">BANKING77 · PolyAI / Casanueva et al.</a> · CC BY 4.0</p><p><a href="https://huggingface.co/prajjwal1/bert-tiny" target="_blank" rel="noreferrer">Pretrained teacher: prajjwal1/bert-tiny</a></p></section><section class="card"><h2>Use and limitations</h2><p>Designed for studying low-cost intent suggestions. It does not understand account balances, verify identity, answer general financial questions, or execute account actions.</p><p>Mean pooling discards word order. Near-neighbor intents such as card arrival and delivery estimates remain a source of mistakes. Confidence on unrelated text may be misleading.</p><p>The three-seed study is exploratory. Bootstrap intervals describe test-example variation conditional on checkpoints; three seeds provide only limited evidence about training variability.</p><h3>Run locally</h3><pre class="code">pip install -e '.[research]'
python -m smallmodel.serve --demo --download-model</pre></section></div><details class="card" style="margin-top:20px"><summary>Dataset audit and experiment configuration</summary>${jsonView({ data: r.data, config: r.config })}</details>`;
}
window.addEventListener(
  "identity-changed",
  guarded(() => render()),
);
guarded(async () => {
  await initAuth();
  await render();
})();
