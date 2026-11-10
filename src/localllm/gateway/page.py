"""Desktop demo page. No frontend build step and no external assets."""

PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LocalLLM Ops</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #12140f;
      --panel: #1c2118;
      --line: #31382c;
      --text: #e7ecdf;
      --muted: #a3ad98;
      --accent: #d2f08a;
      --danger: #ffb4a2;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
      background: var(--bg);
      color: var(--text);
    }
    header, main { width: min(1100px, calc(100% - 48px)); margin: 0 auto; }
    header { padding: 36px 0 8px; }
    h1 { font-size: 40px; font-weight: 600; margin: 0; letter-spacing: -0.03em; }
    h1 span { color: var(--accent); }
    p.lede { color: var(--muted); font-size: 18px; max-width: 62ch; }
    form, .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px;
    }
    label { display: block; font-size: 13px; color: var(--muted); margin: 10px 0 4px; }
    input, select, textarea, button {
      font: 15px/1.4 ui-sans-serif, system-ui, sans-serif;
      color: var(--text);
      background: #10140e;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 10px;
      width: 100%;
    }
    textarea { min-height: 88px; resize: vertical; }
    .row { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }
    button {
      width: auto;
      margin-top: 14px;
      background: var(--accent);
      color: #1a2208;
      border: 0;
      font-weight: 700;
      cursor: pointer;
      padding: 10px 16px;
    }
    button:disabled { opacity: 0.6; cursor: wait; }
    .results { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 16px 0 40px; }
    .sql { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; white-space: pre-wrap; }
    table { width: 100%; border-collapse: collapse; font-family: ui-sans-serif, system-ui, sans-serif; font-size: 13px; }
    th, td { border-bottom: 1px solid var(--line); text-align: left; padding: 4px 6px; }
    .err { color: var(--danger); }
    .meta { color: var(--muted); font-family: ui-sans-serif, system-ui, sans-serif; font-size: 13px; }
  </style>
</head>
<body>
  <header>
    <h1>LocalLLM <span>Ops</span></h1>
    <p class="lede">Ask a question about a sample database. The gateway returns SQLite, checks that it is a single SELECT, and can run it read-only. The default backend is a deterministic keyword model so this page works with no GPU.</p>
  </header>
  <main>
    <form id="ask">
      <label for="key">API key (leave blank when the server has auth disabled)</label>
      <input id="key" type="password" autocomplete="off" />
      <label for="question">Question</label>
      <textarea id="question"></textarea>
      <div class="row">
        <div>
          <label for="db">Database</label>
          <select id="db"></select>
        </div>
        <div>
          <label for="left">Left backend</label>
          <select id="left"></select>
        </div>
        <div>
          <label for="right">Right backend</label>
          <select id="right"></select>
        </div>
      </div>
      <label><input id="execute" type="checkbox" checked style="width:auto" /> Run the SQL read-only</label>
      <button type="submit" id="go">Compare</button>
      <p id="status" class="meta"></p>
    </form>
    <section class="results">
      <article class="card" id="card-left"></article>
      <article class="card" id="card-right"></article>
    </section>
  </main>
  <script>
    const state = { examples: [], backends: [] };
    const question = document.querySelector("#question");
    const db = document.querySelector("#db");
    const left = document.querySelector("#left");
    const right = document.querySelector("#right");
    const status = document.querySelector("#status");
    const go = document.querySelector("#go");

    function fillQuestion() {
      const rows = state.examples.filter((row) => row.db_id === db.value);
      if (rows.length && !question.value) question.value = rows[0].question;
    }

    async function loadExamples() {
      status.textContent = "Loading sample databases…";
      try {
        const response = await fetch("/v1/examples");
        if (!response.ok) throw new Error("examples failed");
        const body = await response.json();
        state.examples = body.examples;
        state.backends = body.backends;
        const ids = [...new Set(body.examples.map((row) => row.db_id))];
        db.innerHTML = ids.map((id) => `<option>${id}</option>`).join("");
        const options = body.backends.map((name) => `<option>${name}</option>`).join("");
        left.innerHTML = options;
        right.innerHTML = options;
        if (body.backends.includes("tiny")) {
          left.value = "tiny";
          right.value = "tiny";
        }
        fillQuestion();
        status.textContent = "";
      } catch (error) {
        status.textContent = "Could not load examples.";
        status.className = "err";
      }
    }

    function render(card, body, httpStatus) {
      if (!body || body.detail) {
        card.innerHTML = `<h2>${card.dataset.backend}</h2><p class="err">${body ? body.detail : "no response"}</p>`;
        return;
      }
      const rows = body.rows || [];
      const columns = body.columns || [];
      const table = columns.length
        ? `<table><thead><tr>${columns.map((col) => `<th>${col}</th>`).join("")}</tr></thead><tbody>${
            rows.map((row) => `<tr>${row.map((cell) => `<td>${cell}</td>`).join("")}</tr>`).join("")
          }</tbody></table>`
        : "";
      const problem = body.valid ? "" : `<p class="err">${body.rejection_reason || "rejected"}</p>`;
      card.innerHTML = `<h2>${body.backend}</h2><p class="meta">${body.model} · ${body.latency_ms} ms · ${body.output_tokens} output tokens${body.fallback ? " · fallback" : ""}</p><pre class="sql">${body.sql || ""}</pre>${problem}${table}<p class="meta">HTTP ${httpStatus}</p>`;
    }

    async function callBackend(name) {
      const headers = { "content-type": "application/json" };
      const key = document.querySelector("#key").value.trim();
      if (key) headers.authorization = `Bearer ${key}`;
      const response = await fetch("/v1/sql", {
        method: "POST",
        headers,
        body: JSON.stringify({
          question: question.value,
          db_id: db.value,
          backend: name,
          execute: document.querySelector("#execute").checked,
        }),
      });
      let body = null;
      try { body = await response.json(); } catch (error) { body = { detail: "response was not JSON" }; }
      return { body, status: response.status };
    }

    document.querySelector("#ask").addEventListener("submit", async (event) => {
      event.preventDefault();
      if (!question.value.trim()) {
        status.textContent = "Write a question first.";
        return;
      }
      go.disabled = true;
      status.textContent = "Running…";
      const cardLeft = document.querySelector("#card-left");
      const cardRight = document.querySelector("#card-right");
      cardLeft.dataset.backend = left.value;
      cardRight.dataset.backend = right.value;
      try {
        const [a, b] = await Promise.all([callBackend(left.value), callBackend(right.value)]);
        render(cardLeft, a.body, a.status);
        render(cardRight, b.body, b.status);
        status.textContent = "Done.";
      } catch (error) {
        status.textContent = "The request failed.";
        status.className = "err";
      } finally {
        go.disabled = false;
      }
    });

    db.addEventListener("change", () => { question.value = ""; fillQuestion(); });
    loadExamples();
  </script>
</body>
</html>
"""
