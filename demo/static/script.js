// ─── MIA24 Pipeline Demo ────────────────────────────────────
// Handles drag-and-drop upload, step-by-step API calls,
// interactive clarification, and progressive UI updates.

const API = '';  // same origin
let imageId = null;

// ─── DOM refs ───────────────────────────────────────────────

const dropZone     = document.getElementById('dropZone');
const fileInput    = document.getElementById('fileInput');
const imagePreview = document.getElementById('imagePreview');
const previewImg   = document.getElementById('previewImg');
const resetBtn     = document.getElementById('resetBtn');

// ─── Helpers ────────────────────────────────────────────────

function setStepStatus(step, status) {
  document.getElementById(`step${step}`).dataset.status = status;
}

function setStepBody(step, html) {
  document.getElementById(`step${step}Body`).innerHTML = html;
}

function scrollToStep(step) {
  document.getElementById(`step${step}`).scrollIntoView({
    behavior: 'smooth',
    block: 'center',
  });
}

async function post(path, body = {}) {
  const res = await fetch(`${API}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json();
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// ─── File Upload ────────────────────────────────────────────

dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) {
    handleUpload(file);
  }
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleUpload(fileInput.files[0]);
});

resetBtn.addEventListener('click', resetPipeline);

async function handleUpload(file) {
  // Show preview immediately
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    dropZone.classList.add('hidden');
    imagePreview.classList.remove('hidden');
  };
  reader.readAsDataURL(file);

  // Upload to server
  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch(`${API}/api/upload`, {
      method: 'POST',
      body: formData,
    });
    const data = await res.json();
    imageId = data.image_id;

    // Start pipeline automatically
    runPhase1();
  } catch (err) {
    showError(1, `Upload failed: ${err.message}`);
  }
}

function resetPipeline() {
  imageId = null;
  previewImg.src = '';
  imagePreview.classList.add('hidden');
  dropZone.classList.remove('hidden');
  fileInput.value = '';

  for (let i = 1; i <= 8; i++) {
    setStepStatus(i, 'pending');
    setStepBody(i, '');
  }
}

function showError(step, msg) {
  setStepStatus(step, 'done');
  setStepBody(step, `<div class="error-msg">${escapeHtml(msg)}</div>`);
}

// ─── Phase 1: Describe + Clarify (automatic) ───────────────

async function runPhase1() {
  // Step 1: Describe
  setStepStatus(1, 'running');
  setStepBody(1, '<p class="step-loading">Analyzing food image...</p>');
  scrollToStep(1);

  try {
    const desc = await post('/api/pipeline/describe', { image_id: imageId });
    setStepStatus(1, 'done');
    setStepBody(1, `<p class="step-text">${escapeHtml(desc.description)}</p>`);
  } catch (err) {
    showError(1, err.message);
    return;
  }

  // Step 2: Clarify
  setStepStatus(2, 'running');
  setStepBody(2, '<p class="step-loading">Generating clarification questions...</p>');
  scrollToStep(2);

  try {
    const clar = await post('/api/pipeline/clarify', { image_id: imageId });
    setStepStatus(2, 'done');

    let html = '<ul class="question-list">';
    clar.questions.forEach((q, i) => {
      html += `<li><span class="question-num">Q${i + 1}</span>${escapeHtml(q)}</li>`;
    });
    html += '</ul>';
    setStepBody(2, html);

    // Show answer form in Step 3 with suggested answers
    showAnswerForm(clar.questions, clar.suggested_answers || []);
  } catch (err) {
    showError(2, err.message);
  }
}

// ─── Step 3: Interactive Answers ────────────────────────────

function showAnswerForm(questions, suggestedAnswers) {
  setStepStatus(3, 'waiting');
  scrollToStep(3);

  let html = '<div class="answer-form">';
  questions.forEach((q, i) => {
    const suggestions = suggestedAnswers[i] || [];
    html += `
      <div class="answer-group">
        <label>Q${i + 1}: ${escapeHtml(q)}</label>
        <textarea id="answer${i}" rows="2" placeholder="Type your answer or click a suggestion..."></textarea>`;
    if (suggestions.length > 0) {
      html += '<div class="suggestion-chips">';
      suggestions.forEach((s) => {
        html += `<span class="suggestion-chip" data-target="answer${i}">${escapeHtml(s)}</span>`;
      });
      html += '</div>';
    }
    html += '</div>';
  });
  html += '<button class="submit-btn" id="submitAnswers">Submit Answers</button>';
  html += '</div>';
  setStepBody(3, html);

  // Suggestion chip click handlers (click again to deselect)
  document.querySelectorAll('.suggestion-chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      const ta = document.getElementById(chip.dataset.target);
      const wasSelected = chip.classList.contains('selected');

      // Deselect all chips in this group
      chip.parentElement.querySelectorAll('.suggestion-chip').forEach(c => c.classList.remove('selected'));

      if (wasSelected) {
        // Toggle off: clear textarea
        if (ta) ta.value = '';
      } else {
        // Select: fill textarea
        if (ta) {
          ta.value = chip.textContent;
          ta.style.borderColor = 'var(--color-user)';
          setTimeout(() => { ta.style.borderColor = ''; }, 800);
        }
        chip.classList.add('selected');
      }
    });
  });

  // Focus first textarea
  setTimeout(() => {
    const first = document.getElementById('answer0');
    if (first) first.focus();
  }, 300);

  document.getElementById('submitAnswers').addEventListener('click', () => {
    const answers = questions.map((_, i) => {
      const ta = document.getElementById(`answer${i}`);
      return ta ? ta.value.trim() : '';
    });

    if (answers.some(a => !a)) {
      questions.forEach((_, i) => {
        const ta = document.getElementById(`answer${i}`);
        if (ta && !ta.value.trim()) {
          ta.style.borderColor = '#f87171';
          setTimeout(() => { ta.style.borderColor = ''; }, 1500);
        }
      });
      return;
    }

    document.getElementById('submitAnswers').disabled = true;
    setStepStatus(3, 'done');

    let ansHtml = '<ul class="question-list">';
    answers.forEach((a, i) => {
      ansHtml += `<li><span class="question-num" style="color:var(--color-user)">A${i + 1}</span>${escapeHtml(a)}</li>`;
    });
    ansHtml += '</ul>';
    setStepBody(3, ansHtml);

    runPhase2(answers);
  });
}

// ─── Phase 2: Expand + Retrieve + Select + Weight + Nutrition

async function runPhase2(answers) {
  // Step 4: Expand
  setStepStatus(4, 'running');
  setStepBody(4, '<p class="step-loading">Expanding queries with your answers...</p>');
  scrollToStep(4);

  let queries;
  try {
    const exp = await post('/api/pipeline/expand', {
      image_id: imageId,
      answers: answers,
    });
    queries = exp.queries;
    setStepStatus(4, 'done');

    let html = '<div class="section-label">Expanded description</div>';
    html += `<div class="expanded-desc">${escapeHtml(exp.expanded_description)}</div>`;
    html += '<div class="section-label">Ingredient queries</div>';
    html += '<div class="query-chips">';
    exp.queries.forEach((q) => {
      html += `<span class="query-chip">${escapeHtml(q)}</span>`;
    });
    html += '</div>';
    setStepBody(4, html);
  } catch (err) {
    showError(4, err.message);
    return;
  }

  // Step 5: Retrieve
  setStepStatus(5, 'running');
  setStepBody(5, '<p class="step-loading">Searching FNDDS database...</p>');
  scrollToStep(5);

  try {
    const ret = await post('/api/pipeline/retrieve', { image_id: imageId });
    setStepStatus(5, 'done');

    let html = `<div class="section-label">${ret.total_count} candidates across ${ret.groups.length} ingredients</div>`;
    ret.groups.forEach((group) => {
      html += '<div class="ingredient-group">';
      html += `<div class="ingredient-label">${escapeHtml(group.ingredient)}</div>`;

      const candidates = group.candidates;
      if (candidates.length === 0) {
        html += '<p class="step-text" style="padding:8px 14px;">No matches found</p>';
      } else {
        const minDist = Math.min(...candidates.map(c => c.distance || 999));
        const maxDist = Math.max(...candidates.map(c => c.distance || 0));
        const range = maxDist - minDist || 1;

        html += '<table class="candidates-table"><thead><tr>';
        html += '<th>#</th><th>Food Code</th><th>Food Name</th><th>Relevance</th>';
        html += '</tr></thead><tbody>';
        candidates.forEach((c, i) => {
          const pct = c.distance != null
            ? Math.max(10, 100 - ((c.distance - minDist) / range) * 80)
            : 50;
          html += `<tr${i === 0 ? ' class="top-match"' : ''}>
            <td>${i + 1}</td>
            <td><span class="code">${c.food_code}</span></td>
            <td>${escapeHtml(c.food_name)}</td>
            <td><div class="relevance-bar"><div class="relevance-fill" style="width:${pct}%"></div></div></td>
          </tr>`;
        });
        html += '</tbody></table>';
      }
      html += '</div>';
    });
    setStepBody(5, html);
  } catch (err) {
    showError(5, err.message);
    return;
  }

  // Step 6: Select
  setStepStatus(6, 'running');
  setStepBody(6, '<p class="step-loading">Selecting best food codes...</p>');
  scrollToStep(6);

  try {
    const sel = await post('/api/pipeline/select', { image_id: imageId });
    setStepStatus(6, 'done');

    let html = '<ul class="selected-list">';
    sel.selected.forEach((s) => {
      html += `<li class="selected-item">
        <span class="selected-check">\u2713</span>
        <span class="selected-code">${s.food_code}</span>
        <span class="selected-name">${escapeHtml(s.food_name)}</span>
      </li>`;
    });
    html += '</ul>';
    setStepBody(6, html);
  } catch (err) {
    showError(6, err.message);
    return;
  }

  // Step 7: Weight
  setStepStatus(7, 'running');
  setStepBody(7, '<p class="step-loading">Estimating food weights...</p>');
  scrollToStep(7);

  try {
    const wt = await post('/api/pipeline/weight', { image_id: imageId });
    setStepStatus(7, 'done');

    let html = '<table class="weight-table"><thead><tr>';
    html += '<th>Food Item</th><th>Code</th><th>Weight</th><th>Reasoning</th>';
    html += '</tr></thead><tbody>';
    wt.items.forEach((item) => {
      const displayWt = item.display_weight || `${item.weight_grams} g`;
      html += `<tr>
        <td>${escapeHtml(item.food_name)}</td>
        <td><span class="code">${item.food_code}</span></td>
        <td><span class="weight-value">${escapeHtml(displayWt)}</span></td>
        <td><span class="weight-reasoning">${escapeHtml(item.reasoning)}</span></td>
      </tr>`;
    });
    html += '</tbody></table>';
    setStepBody(7, html);
  } catch (err) {
    showError(7, err.message);
    return;
  }

  // Step 8: Nutrition
  setStepStatus(8, 'running');
  setStepBody(8, '<p class="step-loading">Calculating nutrition...</p>');
  scrollToStep(8);

  try {
    const nut = await post('/api/pipeline/nutrition', { image_id: imageId });
    setStepStatus(8, 'done');

    const metrics = [
      { key: 'mass_g',   label: 'Mass',     unit: 'g' },
      { key: 'calories', label: 'Energy',   unit: 'kcal' },
      { key: 'fat_g',    label: 'Fat',      unit: 'g' },
      { key: 'carb_g',   label: 'Carbs',    unit: 'g' },
      { key: 'protein_g',label: 'Protein',  unit: 'g' },
    ];

    let html = '<div class="nutrition-grid">';
    metrics.forEach((m) => {
      const val = nut.nutrition[m.key] ?? 0;
      html += `<div class="nutrient-card">
        <div class="nutrient-value">${val}</div>
        <div class="nutrient-unit">${m.unit}</div>
        <div class="nutrient-label">${m.label}</div>
      </div>`;
    });
    html += '</div>';
    setStepBody(8, html);

    scrollToStep(8);
  } catch (err) {
    showError(8, err.message);
  }
}
