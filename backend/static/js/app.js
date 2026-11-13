/**
 * ============================================================
 * AI-Powered Deepfake Detection Platform — Frontend Application
 * ============================================================
 * Handles file upload, API communication, results rendering,
 * history management, and dashboard visualization.
 */

const API_BASE = (window.location.protocol.startsWith('http') && window.location.port === '8000')
  ? window.location.origin
  : 'http://127.0.0.1:8000';

// ── State ──────────────────────────────────────────────────────
const state = {
  selectedFile: null,
  fileType: null, // 'image' or 'video'
  isAnalyzing: false,
  currentResult: null,
  history: [],
  backendOnline: false,
  activeTab: 'analyze',
  currentUser: null,
  authToken: localStorage.getItem('deepguard_token') || null,
};

// ── Authenticated Fetch Helper ─────────────────────────────────
function authFetch(url, options = {}) {
  const opts = { ...options };
  opts.headers = { ...(opts.headers || {}) };
  if (state.authToken) {
    opts.headers['Authorization'] = `Bearer ${state.authToken}`;
  }
  return fetch(url, opts);
}

// ── DOM Ready ──────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initApp();
});

function initApp() {
  setupNavigation();
  setupUploadZones();
  setupAnalyzeButton();
  setupAuthUI();
  restoreSession();
  checkBackendHealth();
  checkUrlForResetToken();

  // Periodically check backend health
  setInterval(checkBackendHealth, 30000);
}

// ── Navigation ─────────────────────────────────────────────────
function setupNavigation() {
  document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', () => {
      const tab = link.dataset.tab;
      switchTab(tab);
    });
  });
}

function switchTab(tabName) {
  state.activeTab = tabName;

  // Update nav links
  document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
  document.querySelector(`.nav-link[data-tab="${tabName}"]`)?.classList.add('active');

  // Update tab content
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.getElementById(`tab-${tabName}`)?.classList.add('active');

  // Lazy-load tab data
  if (tabName === 'history') loadHistory();
  if (tabName === 'dashboard') loadDashboard();
  if (tabName === 'admin') loadAdminData();
}

// ── Backend Health Check ───────────────────────────────────────
async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/v1/health`, { signal: AbortSignal.timeout(5000) });
    const data = await res.json();
    state.backendOnline = true;
    updateHealthUI(data);
  } catch {
    state.backendOnline = false;
    updateHealthUI(null);
  }
}

function updateHealthUI(data) {
  const badge = document.getElementById('status-badge');
  if (!badge) return;

  if (data) {
    badge.className = 'status-badge';
    badge.innerHTML = `<span class="status-dot"></span> Online · ${data.model_type || 'hybrid'}`;
  } else {
    badge.className = 'status-badge offline';
    badge.innerHTML = `<span class="status-dot"></span> Offline`;
  }
}

// ── Upload Zones ───────────────────────────────────────────────
function setupUploadZones() {
  const imageZone = document.getElementById('image-upload-zone');
  const videoZone = document.getElementById('video-upload-zone');
  const imageInput = document.getElementById('image-input');
  const videoInput = document.getElementById('video-input');

  // Image zone
  setupDropZone(imageZone, imageInput, 'image');
  // Video zone
  setupDropZone(videoZone, videoInput, 'video');
}

function setupDropZone(zone, input, type) {
  if (!zone || !input) return;

  zone.addEventListener('click', () => input.click());

  zone.addEventListener('dragover', (e) => {
    e.preventDefault();
    zone.classList.add('drag-over');
  });

  zone.addEventListener('dragleave', () => {
    zone.classList.remove('drag-over');
  });

  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFileSelect(files[0], type);
  });

  input.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFileSelect(e.target.files[0], type);
  });
}

function handleFileSelect(file, type) {
  // Reset other zone
  const otherType = type === 'image' ? 'video' : 'image';
  resetUploadZone(otherType);

  state.selectedFile = file;
  state.fileType = type;

  const zone = document.getElementById(`${type}-upload-zone`);
  zone.classList.add('has-file');

  // Show preview
  const previewContainer = document.getElementById(`${type}-preview`);
  if (previewContainer) {
    previewContainer.innerHTML = buildFilePreview(file, type);
    previewContainer.style.display = 'block';
  }

  // Enable analyze button
  updateAnalyzeButton();
}

function buildFilePreview(file, type) {
  const sizeStr = formatFileSize(file.size);
  let thumbHtml;

  if (type === 'image') {
    const url = URL.createObjectURL(file);
    thumbHtml = `<img class="file-preview-thumb" src="${url}" alt="Preview" />`;
  } else {
    thumbHtml = `<div class="file-preview-thumb" style="background:var(--bg-tertiary);display:flex;align-items:center;justify-content:center;font-size:20px;">🎬</div>`;
  }

  return `
    <div class="file-preview">
      ${thumbHtml}
      <div class="file-preview-info">
        <div class="file-preview-name">${escapeHtml(file.name)}</div>
        <div class="file-preview-size">${sizeStr}</div>
      </div>
      <button class="file-remove-btn" onclick="removeFile('${type}'); event.stopPropagation();">✕</button>
    </div>
  `;
}

function removeFile(type) {
  state.selectedFile = null;
  state.fileType = null;
  resetUploadZone(type);
  updateAnalyzeButton();
}

function resetUploadZone(type) {
  const zone = document.getElementById(`${type}-upload-zone`);
  if (zone) zone.classList.remove('has-file');

  const preview = document.getElementById(`${type}-preview`);
  if (preview) {
    preview.innerHTML = '';
    preview.style.display = 'none';
  }

  const input = document.getElementById(`${type}-input`);
  if (input) input.value = '';
}

// ── Analyze Button ─────────────────────────────────────────────
function setupAnalyzeButton() {
  const btn = document.getElementById('analyze-btn');
  if (btn) btn.addEventListener('click', startAnalysis);
}

function updateAnalyzeButton() {
  const btn = document.getElementById('analyze-btn');
  if (!btn) return;
  btn.disabled = !state.selectedFile || state.isAnalyzing;
}

async function startAnalysis() {
  if (!state.selectedFile || state.isAnalyzing) return;

  const btn = document.getElementById('analyze-btn');
  state.isAnalyzing = true;
  btn.disabled = true;
  btn.classList.add('loading');
  btn.innerHTML = '<span class="btn-spinner"></span> Analyzing...';

  // Hide previous results
  const resultsSection = document.getElementById('results-section');
  resultsSection.classList.remove('visible');

  try {
    const formData = new FormData();
    formData.append('file', state.selectedFile);

    const endpoint = state.fileType === 'video'
      ? `${API_BASE}/api/v1/analyze/video`
      : `${API_BASE}/api/v1/analyze/image`;

    const res = await authFetch(endpoint, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Analysis failed' }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const result = await res.json();
    state.currentResult = result;

    renderResults(result);
    showToast('success', `Analysis complete — ${result.label}`);
  } catch (err) {
    showToast('error', `Error: ${err.message}`);
  } finally {
    state.isAnalyzing = false;
    btn.disabled = false;
    btn.classList.remove('loading');
    btn.innerHTML = '🔍 Analyze Media';
    updateAnalyzeButton();
  }
}

// ── Render Results ─────────────────────────────────────────────
function renderResults(result) {
  const section = document.getElementById('results-section');
  const isVideo = result.media_type === 'video';
  const isReal = result.label === 'REAL';
  const verdictClass = isReal ? 'real' : 'fake';

  // Verdict Card
  const verdictCard = document.getElementById('verdict-card');
  verdictCard.className = `card verdict-card ${verdictClass}`;
  document.getElementById('verdict-result').textContent = result.label;
  document.getElementById('verdict-result').className = `verdict-result`;

  // Confidence Ring
  const confidence = result.confidence;
  const ringFill = document.getElementById('confidence-ring-fill');
  const circumference = 2 * Math.PI * 72; // r=72
  ringFill.style.strokeDasharray = `${circumference}`;
  // Animate after a brief delay
  requestAnimationFrame(() => {
    ringFill.style.strokeDashoffset = `${circumference * (1 - confidence / 100)}`;
  });

  document.getElementById('confidence-number').textContent = confidence.toFixed(1);

  // Probabilities
  const realProb = result.probabilities?.real || 0;
  const fakeProb = result.probabilities?.fake || 0;
  document.getElementById('prob-real-fill').style.width = `${(realProb * 100).toFixed(1)}%`;
  document.getElementById('prob-fake-fill').style.width = `${(fakeProb * 100).toFixed(1)}%`;
  document.getElementById('prob-real-value').textContent = `${(realProb * 100).toFixed(1)}%`;
  document.getElementById('prob-fake-value').textContent = `${(fakeProb * 100).toFixed(1)}%`;

  // Meta info
  document.getElementById('meta-time').textContent = `${result.inference_time_ms}ms`;
  document.getElementById('meta-type').textContent = result.media_type?.toUpperCase() || 'IMAGE';
  document.getElementById('meta-id').textContent = result.id || '—';

  // Artifact scores (image only)
  const artifactSection = document.getElementById('artifact-section');
  if (result.artifact_scores && !isVideo) {
    artifactSection.style.display = 'block';
    renderArtifactScores(result.artifact_scores);
  } else {
    artifactSection.style.display = 'none';
  }

  // Video-specific results
  const videoSection = document.getElementById('video-results-section');
  if (isVideo) {
    videoSection.style.display = 'block';
    renderVideoResults(result);
  } else {
    videoSection.style.display = 'none';
  }

  section.classList.add('visible');

  // Scroll to results
  section.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderArtifactScores(scores) {
  const items = [
    { key: 'blending_artifacts', label: 'Blending Artifacts', icon: '🔀' },
    { key: 'texture_inconsistency', label: 'Texture Inconsistency', icon: '🧶' },
    { key: 'eye_reflection_mismatch', label: 'Eye Reflection', icon: '👁️' },
    { key: 'boundary_edge_noise', label: 'Boundary Noise', icon: '📐' },
  ];

  const grid = document.getElementById('artifact-grid');
  grid.innerHTML = items.map(item => {
    const value = scores[item.key] || 0;
    const pct = (value * 100).toFixed(1);
    const level = value > 0.6 ? 'high' : value > 0.3 ? 'medium' : 'low';

    return `
      <div class="artifact-item">
        <div class="artifact-name">${item.icon} ${item.label}</div>
        <div class="artifact-score-bar">
          <div class="artifact-score-fill ${level}" style="width: 0%"
               data-target-width="${pct}%"></div>
        </div>
        <div class="artifact-value">${pct}%</div>
      </div>
    `;
  }).join('');

  // Animate bars
  requestAnimationFrame(() => {
    grid.querySelectorAll('.artifact-score-fill').forEach(bar => {
      bar.style.width = bar.dataset.targetWidth;
    });
  });
}

function renderVideoResults(result) {
  // Stats
  document.getElementById('video-total-frames').textContent = result.total_frames_analyzed || 0;
  document.getElementById('video-suspicious').textContent = result.suspicious_frame_count || 0;
  document.getElementById('video-time').textContent = `${(result.inference_time_ms / 1000).toFixed(1)}s`;
  document.getElementById('video-avg-score').textContent =
    result.per_frame_scores
      ? (result.per_frame_scores.reduce((a, b) => a + b, 0) / result.per_frame_scores.length * 100).toFixed(1) + '%'
      : '—';

  // Frame scores chart
  if (result.per_frame_scores && result.per_frame_scores.length > 0) {
    drawFrameScoresChart(result.per_frame_scores);
  }
}

function drawFrameScoresChart(scores) {
  const canvas = document.getElementById('frame-scores-canvas');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();

  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = rect.height;
  const padding = { top: 10, right: 10, bottom: 25, left: 40 };
  const chartW = w - padding.left - padding.right;
  const chartH = h - padding.top - padding.bottom;

  ctx.clearRect(0, 0, w, h);

  // Grid lines
  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = padding.top + (chartH / 4) * i;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(w - padding.right, y);
    ctx.stroke();
  }

  // Threshold line at 0.5
  const threshY = padding.top + chartH * 0.5;
  ctx.strokeStyle = 'rgba(239, 68, 68, 0.3)';
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(padding.left, threshY);
  ctx.lineTo(w - padding.right, threshY);
  ctx.stroke();
  ctx.setLineDash([]);

  // Labels
  ctx.fillStyle = '#64748b';
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const y = padding.top + (chartH / 4) * i;
    ctx.fillText(`${(100 - i * 25)}%`, padding.left - 6, y + 3);
  }

  ctx.textAlign = 'center';
  const step = Math.max(1, Math.floor(scores.length / 8));
  for (let i = 0; i < scores.length; i += step) {
    const x = padding.left + (chartW / (scores.length - 1)) * i;
    ctx.fillText(`F${i + 1}`, x, h - 4);
  }

  // Draw line
  const gradient = ctx.createLinearGradient(0, padding.top, 0, padding.top + chartH);
  gradient.addColorStop(0, '#ef4444');
  gradient.addColorStop(0.5, '#f59e0b');
  gradient.addColorStop(1, '#10b981');

  ctx.strokeStyle = gradient;
  ctx.lineWidth = 2;
  ctx.lineJoin = 'round';
  ctx.beginPath();

  scores.forEach((score, i) => {
    const x = padding.left + (chartW / Math.max(scores.length - 1, 1)) * i;
    const y = padding.top + chartH * (1 - score);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Draw points
  scores.forEach((score, i) => {
    const x = padding.left + (chartW / Math.max(scores.length - 1, 1)) * i;
    const y = padding.top + chartH * (1 - score);
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fillStyle = score >= 0.5 ? '#ef4444' : '#10b981';
    ctx.fill();
  });
}

// ── History ────────────────────────────────────────────────────
async function loadHistory() {
  try {
    const res = await authFetch(`${API_BASE}/api/v1/history?limit=50`);
    const data = await res.json();
    state.history = data.history || [];
    renderHistory(state.history, data);
  } catch {
    renderHistory([]);
  }
}

function renderHistory(items, meta = {}) {
  const container = document.getElementById('history-content');
  if (!container) return;

  const headerNotice = state.currentUser
    ? `<div style="display:flex;align-items:center;justify-content:space-between;padding:12px 16px;background:rgba(6,182,212,0.08);border:1px solid rgba(6,182,212,0.25);border-radius:var(--radius-md);margin-bottom:16px;">
        <div style="display:flex;align-items:center;gap:10px;">
          <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#10b981;"></span>
          <span style="font-size:13px;color:var(--text-secondary);">Logged in as <strong style="color:var(--text-primary);">${escapeHtml(state.currentUser.username)}</strong> (${escapeHtml(state.currentUser.email)}) &mdash; displaying your private scan history (${items.length} records)</span>
        </div>
        <button class="nav-btn-secondary" onclick="loadHistory()" style="font-size:12px;padding:4px 10px;">↻ Refresh</button>
      </div>`
    : `<div style="display:flex;align-items:center;justify-content:space-between;padding:12px 16px;background:rgba(99,102,241,0.08);border:1px dashed rgba(99,102,241,0.3);border-radius:var(--radius-md);margin-bottom:16px;">
        <div style="display:flex;align-items:center;gap:10px;">
          <span style="font-size:13px;color:var(--text-secondary);">Guest Mode &mdash; showing recent platform scans. <a href="javascript:void(0)" onclick="openAuthModal('signin')" style="color:var(--accent-cyan);font-weight:600;text-decoration:underline;">Sign In</a> or <a href="javascript:void(0)" onclick="openAuthModal('signup')" style="color:var(--accent-cyan);font-weight:600;text-decoration:underline;">Create Account</a> to record private scans to your account.</span>
        </div>
        <button class="nav-btn-secondary" onclick="loadHistory()" style="font-size:12px;padding:4px 10px;">↻ Refresh</button>
      </div>`;

  if (items.length === 0) {
    container.innerHTML = `
      ${headerNotice}
      <div class="empty-state">
        <div class="empty-state-icon">📋</div>
        <div class="empty-state-text">No analysis history yet</div>
      </div>
    `;
    return;
  }

  container.innerHTML = `
    ${headerNotice}
    <div class="history-table-container">
      <table class="history-table">
        <thead>
          <tr>
            <th>File</th>
            <th>Type</th>
            <th>Result</th>
            <th>Confidence</th>
            <th>Time</th>
            <th>Date</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          ${items.map(item => `
            <tr>
              <td style="font-weight:500;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">
                ${escapeHtml(item.filename || '—')}
              </td>
              <td>
                <span class="format-tag">${(item.media_type || '—').toUpperCase()}</span>
              </td>
              <td>
                <span class="history-label-badge ${item.label === 'REAL' ? 'real' : 'fake'}">
                  ${item.label === 'REAL' ? '✓' : '⚠'} ${item.label || '—'}
                </span>
              </td>
              <td class="confidence-pill">${item.confidence != null ? item.confidence.toFixed(1) + '%' : '—'}</td>
              <td style="color:var(--text-muted);font-family:'JetBrains Mono',monospace;font-size:12px;">
                ${item.inference_time_ms != null ? item.inference_time_ms + 'ms' : '—'}
              </td>
              <td style="color:var(--text-muted);font-size:12px;">${item.timestamp || '—'}</td>
              <td>
                <button class="delete-btn" onclick="deleteHistoryItem('${item.id}')" title="Delete">🗑</button>
              </td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    </div>
  `;
}

async function deleteHistoryItem(id) {
  try {
    const res = await authFetch(`${API_BASE}/api/v1/history/${id}`, { method: 'DELETE' });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Failed to delete record' }));
      throw new Error(err.detail);
    }
    showToast('success', 'Record deleted');
    loadHistory();
  } catch (err) {
    showToast('error', err.message || 'Failed to delete record');
  }
}

// ── Dashboard ──────────────────────────────────────────────────
async function loadDashboard() {
  // Load models info
  try {
    const res = await fetch(`${API_BASE}/api/v1/models`);
    const data = await res.json();
    renderModelCards(data);
  } catch {
    // fallback with static data
    renderModelCards(null);
  }

  // Load history stats
  try {
    const res = await fetch(`${API_BASE}/api/v1/history?limit=100`);
    const data = await res.json();
    renderDashboardStats(data.history || []);
  } catch {
    renderDashboardStats([]);
  }

  // Render training chart
  renderTrainingChart();
}

function renderModelCards(data) {
  const container = document.getElementById('model-cards');
  if (!container) return;

  const models = data?.models || [
    { id: 'hybrid', name: 'Hybrid CNN-Transformer', status: 'active', description: 'EfficientNet-B4 + Transformer Encoder — spatial & temporal modeling.' },
    { id: 'cnn', name: 'CNN Baseline (EfficientNet-B4)', status: 'ready', description: 'Convolutional network for local artifact detection.' },
    { id: 'transformer', name: 'Vision Transformer (ViT)', status: 'ready', description: 'Patch-based global spatial relationship modeling.' },
  ];

  container.innerHTML = models.map(m => `
    <div class="model-card ${m.status === 'active' ? 'active-model' : ''}">
      <span class="model-card-badge ${m.status}">${m.status}</span>
      <div class="model-card-name">${escapeHtml(m.name)}</div>
      <div class="model-card-desc">${escapeHtml(m.description)}</div>
    </div>
  `).join('');
}

function renderDashboardStats(history) {
  const total = history.length;
  const realCount = history.filter(h => h.label === 'REAL').length;
  const fakeCount = history.filter(h => h.label === 'DEEPFAKE' || h.label === 'FAKE').length;
  const avgConfidence = total > 0
    ? (history.reduce((s, h) => s + (h.confidence || 0), 0) / total).toFixed(1)
    : '—';

  const el = (id, val) => {
    const e = document.getElementById(id);
    if (e) e.textContent = val;
  };

  el('stat-total', total);
  el('stat-real', realCount);
  el('stat-fake', fakeCount);
  el('stat-confidence', avgConfidence !== '—' ? avgConfidence + '%' : '—');
}

function renderTrainingChart() {
  // Training history data from the model's actual training
  const trainingData = [
    { epoch: 1, val_acc: 0.813, val_auc: 0.652, train_loss: 0.164 },
    { epoch: 2, val_acc: 0.907, val_auc: 0.908, train_loss: 0.037 },
    { epoch: 3, val_acc: 0.920, val_auc: 0.848, train_loss: 0.054 },
    { epoch: 4, val_acc: 0.927, val_auc: 0.880, train_loss: 0.016 },
    { epoch: 5, val_acc: 0.947, val_auc: 0.894, train_loss: 0.000 },
    { epoch: 6, val_acc: 0.933, val_auc: 0.913, train_loss: 0.000 },
    { epoch: 7, val_acc: 0.907, val_auc: 0.879, train_loss: 0.012 },
    { epoch: 8, val_acc: 0.927, val_auc: 0.984, train_loss: 0.022 },
    { epoch: 9, val_acc: 0.947, val_auc: 0.895, train_loss: 0.001 },
    { epoch: 10, val_acc: 0.960, val_auc: 0.938, train_loss: 0.000 },
    { epoch: 11, val_acc: 0.947, val_auc: 0.950, train_loss: 0.007 },
    { epoch: 12, val_acc: 0.953, val_auc: 0.999, train_loss: 0.000 },
    { epoch: 13, val_acc: 0.947, val_auc: 0.950, train_loss: 0.002 },
    { epoch: 14, val_acc: 0.953, val_auc: 0.988, train_loss: 0.004 },
    { epoch: 15, val_acc: 0.947, val_auc: 0.958, train_loss: 0.000 },
    { epoch: 16, val_acc: 0.947, val_auc: 0.924, train_loss: 0.000 },
    { epoch: 17, val_acc: 0.933, val_auc: 0.870, train_loss: 0.000 },
    { epoch: 18, val_acc: 0.953, val_auc: 0.978, train_loss: 0.000 },
    { epoch: 19, val_acc: 0.967, val_auc: 0.969, train_loss: 0.000 },
    { epoch: 20, val_acc: 0.947, val_auc: 0.969, train_loss: 0.000 },
  ];

  const canvas = document.getElementById('training-chart-canvas');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();

  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = rect.height;
  const pad = { top: 20, right: 20, bottom: 35, left: 45 };
  const cw = w - pad.left - pad.right;
  const ch = h - pad.top - pad.bottom;

  ctx.clearRect(0, 0, w, h);

  // Grid
  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const y = pad.top + (ch / 5) * i;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(w - pad.right, y);
    ctx.stroke();
  }

  // Y-axis labels
  ctx.fillStyle = '#64748b';
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 5; i++) {
    const y = pad.top + (ch / 5) * i;
    const val = (1.0 - i * 0.2).toFixed(1);
    ctx.fillText(val, pad.left - 8, y + 3);
  }

  // X-axis labels
  ctx.textAlign = 'center';
  for (let i = 0; i < trainingData.length; i += 2) {
    const x = pad.left + (cw / (trainingData.length - 1)) * i;
    ctx.fillText(`E${trainingData[i].epoch}`, x, h - 8);
  }

  // Draw AUC line
  function drawLine(data, key, color, label) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.beginPath();

    data.forEach((d, i) => {
      const x = pad.left + (cw / (data.length - 1)) * i;
      const y = pad.top + ch * (1 - d[key]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Points
    data.forEach((d, i) => {
      const x = pad.left + (cw / (data.length - 1)) * i;
      const y = pad.top + ch * (1 - d[key]);
      ctx.beginPath();
      ctx.arc(x, y, 2.5, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    });
  }

  drawLine(trainingData, 'val_auc', '#06b6d4', 'Val AUC');
  drawLine(trainingData, 'val_acc', '#8b5cf6', 'Val Acc');

  // Legend
  const legendX = pad.left + 10;
  const legendY = pad.top + 10;

  [
    { color: '#06b6d4', label: 'Val AUC' },
    { color: '#8b5cf6', label: 'Val Accuracy' },
  ].forEach((item, i) => {
    const x = legendX + i * 110;
    ctx.fillStyle = item.color;
    ctx.beginPath();
    ctx.arc(x, legendY, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#94a3b8';
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(item.label, x + 10, legendY + 4);
  });
}

// ── Toast Notifications ────────────────────────────────────────
function showToast(type, message) {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const icons = { success: '✓', error: '✕', info: 'ℹ' };
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span>${icons[type] || ''}</span> ${escapeHtml(message)}`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.classList.add('fade-out');
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

// ── Utility Functions ──────────────────────────────────────────
function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function escapeHtml(str) {
  if (!str) return '';
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// ── Authentication UI & Session Management ─────────────────────

function setupAuthUI() {
  // Modal triggers
  const btnSignIn = document.getElementById('btn-open-signin');
  const btnSignUp = document.getElementById('btn-open-signup');
  const modalClose = document.getElementById('auth-modal-close');
  const modalOverlay = document.getElementById('auth-modal-overlay');

  if (btnSignIn) btnSignIn.addEventListener('click', () => openAuthModal('signin'));
  if (btnSignUp) btnSignUp.addEventListener('click', () => openAuthModal('signup'));
  if (modalClose) modalClose.addEventListener('click', closeAuthModal);

  // Close modal when clicking backdrop
  if (modalOverlay) {
    modalOverlay.addEventListener('click', (e) => {
      if (e.target === modalOverlay) closeAuthModal();
    });
  }

  // Close modal on Escape key
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeAuthModal();
  });

  // Tab switching inside modal
  const tabBtnSignIn = document.getElementById('tab-btn-signin');
  const tabBtnSignUp = document.getElementById('tab-btn-signup');
  const tabBtnForgot = document.getElementById('tab-btn-forgot');
  if (tabBtnSignIn) tabBtnSignIn.addEventListener('click', () => switchAuthTab('signin'));
  if (tabBtnSignUp) tabBtnSignUp.addEventListener('click', () => switchAuthTab('signup'));
  if (tabBtnForgot) tabBtnForgot.addEventListener('click', () => switchAuthTab('forgot'));

  // Password visibility toggles
  document.querySelectorAll('.password-toggle-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const targetId = btn.dataset.target;
      const input = document.getElementById(targetId);
      if (!input) return;
      if (input.type === 'password') {
        input.type = 'text';
        btn.textContent = '🙈';
      } else {
        input.type = 'password';
        btn.textContent = '👁';
      }
    });
  });

  // Forms
  const formSignIn = document.getElementById('form-signin');
  const formSignUp = document.getElementById('form-signup');
  const formForgotEmail = document.getElementById('form-forgot-email');
  const formCreateNewPassword = document.getElementById('form-create-new-password');

  if (formSignIn) formSignIn.addEventListener('submit', handleSignIn);
  if (formSignUp) formSignUp.addEventListener('submit', handleSignUp);
  if (formForgotEmail) formForgotEmail.addEventListener('submit', handleRequestResetLink);
  if (formCreateNewPassword) formCreateNewPassword.addEventListener('submit', handleConfirmCreatePassword);

  // Forgot password triggers
  const btnShowForgot = document.getElementById('btn-show-forgot');
  const btnShowForgotSub = document.getElementById('btn-show-forgot-sub');
  const btnCancelForgot = document.getElementById('btn-cancel-forgot');
  const btnCancelCreate = document.getElementById('btn-cancel-create');
  if (btnShowForgot) btnShowForgot.addEventListener('click', () => switchAuthTab('forgot'));
  if (btnShowForgotSub) btnShowForgotSub.addEventListener('click', () => switchAuthTab('forgot'));
  if (btnCancelForgot) btnCancelForgot.addEventListener('click', () => switchAuthTab('signin'));
  if (btnCancelCreate) btnCancelCreate.addEventListener('click', () => switchAuthTab('signin'));

  // Quick Demo Login
  const btnQuickDemo = document.getElementById('btn-quick-demo');
  if (btnQuickDemo) btnQuickDemo.addEventListener('click', loginWithDemoAccount);

  // User Profile Pill & Dropdown
  const userPill = document.getElementById('user-profile-pill');
  const userDropdown = document.getElementById('user-dropdown-menu');
  if (userPill && userDropdown) {
    userPill.addEventListener('click', (e) => {
      e.stopPropagation();
      const isExpanded = userPill.getAttribute('aria-expanded') === 'true';
      userPill.setAttribute('aria-expanded', !isExpanded);
      userDropdown.style.display = isExpanded ? 'none' : 'block';
    });

    document.addEventListener('click', (e) => {
      if (!userDropdown.contains(e.target) && !userPill.contains(e.target)) {
        userDropdown.style.display = 'none';
        userPill.setAttribute('aria-expanded', 'false');
      }
    });
  }

  // Dropdown menu buttons
  const menuScans = document.getElementById('menu-btn-scans');
  const menuAnalyze = document.getElementById('menu-btn-analyze');
  const menuLogout = document.getElementById('menu-btn-logout');

  if (menuScans) {
    menuScans.addEventListener('click', () => {
      if (userDropdown) userDropdown.style.display = 'none';
      if (userPill) userPill.setAttribute('aria-expanded', 'false');
      const appEntry = document.getElementById('btn-analyze-cta');
      if (appEntry) appEntry.click();
      switchTab('history');
    });
  }

  if (menuAnalyze) {
    menuAnalyze.addEventListener('click', () => {
      if (userDropdown) userDropdown.style.display = 'none';
      if (userPill) userPill.setAttribute('aria-expanded', 'false');
      const appEntry = document.getElementById('btn-analyze-cta');
      if (appEntry) appEntry.click();
      switchTab('analyze');
    });
  }

  if (menuLogout) {
    menuLogout.addEventListener('click', () => {
      if (userDropdown) userDropdown.style.display = 'none';
      if (userPill) userPill.setAttribute('aria-expanded', 'false');
      handleLogout();
    });
  }
}

function openAuthModal(initialTab = 'signin') {
  const overlay = document.getElementById('auth-modal-overlay');
  if (!overlay) return;
  overlay.style.display = 'flex';
  overlay.setAttribute('aria-hidden', 'false');
  clearAuthAlert();
  switchAuthTab(initialTab);
}

function closeAuthModal() {
  const overlay = document.getElementById('auth-modal-overlay');
  if (!overlay) return;
  overlay.style.display = 'none';
  overlay.setAttribute('aria-hidden', 'true');
  clearAuthAlert();
}

function switchAuthTab(tab) {
  const isSignIn = tab === 'signin';
  const isSignUp = tab === 'signup';
  const isForgot = tab === 'forgot';
  const isCreatePassword = tab === 'create_password';

  const tabSignIn = document.getElementById('tab-btn-signin');
  const tabSignUp = document.getElementById('tab-btn-signup');
  const tabForgot = document.getElementById('tab-btn-forgot');
  const authTabsContainer = document.querySelector('.auth-tabs');
  const formSignIn = document.getElementById('form-signin');
  const formSignUp = document.getElementById('form-signup');
  const formForgotEmail = document.getElementById('form-forgot-email');
  const formCreateNewPassword = document.getElementById('form-create-new-password');
  const modalTitle = document.getElementById('auth-modal-title');
  const modalSub = document.getElementById('auth-modal-subtitle');

  if (authTabsContainer) {
    authTabsContainer.style.display = isCreatePassword ? 'none' : 'grid';
  }

  if (tabSignIn) {
    tabSignIn.classList.toggle('active', isSignIn);
    tabSignIn.setAttribute('aria-selected', isSignIn);
  }
  if (tabSignUp) {
    tabSignUp.classList.toggle('active', isSignUp);
    tabSignUp.setAttribute('aria-selected', isSignUp);
  }
  if (tabForgot) {
    tabForgot.classList.toggle('active', isForgot);
    tabForgot.setAttribute('aria-selected', isForgot);
  }

  if (formSignIn) formSignIn.style.display = isSignIn ? 'flex' : 'none';
  if (formSignUp) formSignUp.style.display = isSignUp ? 'flex' : 'none';
  if (formForgotEmail) formForgotEmail.style.display = isForgot ? 'flex' : 'none';
  if (formCreateNewPassword) formCreateNewPassword.style.display = isCreatePassword ? 'flex' : 'none';

  if (modalTitle) {
    if (isForgot) modalTitle.textContent = 'Forgot Password';
    else if (isCreatePassword) modalTitle.textContent = 'Create New Password';
    else if (isSignIn) modalTitle.textContent = 'Sign In to DeepGuard';
    else modalTitle.textContent = 'Create an Account';
  }
  if (modalSub) {
    if (isForgot) modalSub.textContent = 'Enter your registered email address to receive a secure password reset link';
    else if (isCreatePassword) modalSub.textContent = 'Choose a new password for your DeepGuard AI account';
    else if (isSignIn) modalSub.textContent = 'Access forensic deepfake detection and personal scans';
    else modalSub.textContent = 'Register to verify media and preserve verifiable logs';
  }

  clearAuthAlert();

  // Focus input
  setTimeout(() => {
    let input = null;
    if (isForgot) input = document.getElementById('forgot-email');
    else if (isCreatePassword) input = document.getElementById('create-new-password');
    else if (isSignIn) input = document.getElementById('signin-identifier');
    else input = document.getElementById('signup-name');
    if (input) input.focus();
  }, 100);
}

// ── Email Reset Link Handlers ───────────────────────────────────

async function handleRequestResetLink(e) {
  e.preventDefault();
  clearAuthAlert();

  const emailInput = document.getElementById('forgot-email');
  const email = emailInput?.value.trim();
  const btn = document.getElementById('btn-submit-forgot-email');
  const previewBox = document.getElementById('forgot-link-preview');

  if (!email) {
    setAuthAlert('Please enter your registered email address.');
    return;
  }

  if (previewBox) previewBox.style.display = 'none';

  if (btn) {
    btn.disabled = true;
    btn.innerHTML = '<span class="btn-spinner"></span> Generating &amp; Sending Link...';
  }

  try {
    const res = await fetch(`${API_BASE}/api/v1/auth/forgot-password`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email }),
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || 'Could not send reset link. Please check your email address.');
    }

    if (previewBox) {
      if (data.email_sent) {
        previewBox.innerHTML = `
          <div style="background:rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.3); border-radius:8px; padding:10px 14px; margin-bottom:8px; color:#34d399;">
            <strong style="color:#34d399;">✓ Email Dispatched to Inbox!</strong>
            <p style="margin:4px 0 0; font-size:0.82rem; color:#cbd5e1;">
              A real email with the password reset link has been dispatched to <strong>${escapeHtml(email)}</strong>. Please check your email inbox and spam folder.
            </p>
          </div>
          <p style="margin: 0.4rem 0 0.6rem; color:#94a3b8; font-size:0.8rem;">
            You can also open the reset link immediately below:
          </p>
          <a href="${data.reset_url}" class="reset-link-btn" onclick="openCreatePasswordWithToken('${data.token}'); return false;">
            Open Link &amp; Create New Password →
          </a>
        `;
      } else {
        previewBox.innerHTML = `
          <div style="background:rgba(245,158,11,0.1); border:1px solid rgba(245,158,11,0.25); border-radius:8px; padding:10px 14px; margin-bottom:8px; color:#fbbf24;">
            <strong style="color:#fbbf24;">✓ Password Reset Link Ready</strong>
            <p style="margin:4px 0 0; font-size:0.82rem; color:#cbd5e1;">
              Reset link created for <strong>${escapeHtml(email)}</strong>. To deliver directly to external email servers, add your SMTP/Gmail credentials in the project's <code>.env</code> file.
            </p>
          </div>
          <p style="margin: 0.4rem 0 0.6rem; color:#94a3b8; font-size:0.8rem;">
            Click the button below to immediately open the link and create your new password:
          </p>
          <a href="${data.reset_url}" class="reset-link-btn" onclick="openCreatePasswordWithToken('${data.token}'); return false;">
            Open Link &amp; Create New Password →
          </a>
        `;
      }
      previewBox.style.display = 'block';
    }

    showToast('success', `Reset link sent for ${email}!`);
  } catch (err) {
    setAuthAlert(err.message, 'error');
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = '<span>Send Password Reset Link</span><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M5 12h14M12 5l7 7-7 7"/></svg>';
    }
  }
}

function openCreatePasswordWithToken(token) {
  openAuthModal('create_password');
  const tokenInput = document.getElementById('create-pwd-token');
  if (tokenInput) tokenInput.value = token;
  verifyResetTokenOnLoad(token);
}

async function verifyResetTokenOnLoad(token) {
  const statusPill = document.getElementById('reset-token-status');
  const statusText = document.getElementById('reset-token-user-email');

  if (statusPill && statusText) {
    statusPill.style.display = 'flex';
    statusText.textContent = 'Verifying security signature...';
  }

  try {
    const res = await fetch(`${API_BASE}/api/v1/auth/verify-reset-token?token=${encodeURIComponent(token)}`);
    const data = await res.json();
    if (res.ok && data.valid) {
      if (statusText) {
        statusText.innerHTML = `Verified account: <strong>${escapeHtml(data.username)}</strong> (${escapeHtml(data.email)})`;
      }
    } else {
      if (statusPill) statusPill.style.display = 'none';
      setAuthAlert(data.detail || 'This reset link has expired or is invalid. Please request a new one.', 'error');
    }
  } catch (err) {
    console.error('Verify token error:', err);
  }
}

async function handleConfirmCreatePassword(e) {
  e.preventDefault();
  clearAuthAlert();

  const token = document.getElementById('create-pwd-token')?.value.trim();
  const newPassword = document.getElementById('create-new-password')?.value;
  const confirmPassword = document.getElementById('create-confirm-password')?.value;
  const btn = document.getElementById('btn-submit-create-password');

  if (!token) {
    setAuthAlert('Missing or invalid reset token. Please request a new link.');
    return;
  }

  if (!newPassword || !confirmPassword) {
    setAuthAlert('Please fill in both password fields.');
    return;
  }

  if (newPassword.length < 6) {
    setAuthAlert('Password must be at least 6 characters long.');
    return;
  }

  if (newPassword !== confirmPassword) {
    setAuthAlert('Passwords do not match. Please verify.');
    return;
  }

  if (btn) {
    btn.disabled = true;
    btn.innerHTML = '<span class="btn-spinner"></span> Updating Password &amp; Signing In...';
  }

  try {
    const res = await fetch(`${API_BASE}/api/v1/auth/confirm-reset-password`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token, new_password: newPassword }),
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || 'Password reset failed.');
    }

    // Save token and auto-login
    state.authToken = data.access_token;
    state.currentUser = data.user;
    localStorage.setItem('deepguard_token', data.access_token);

    updateAuthNavUI();
    closeAuthModal();

    // Clean URL
    if (window.history.replaceState) {
      const cleanUrl = window.location.protocol + "//" + window.location.host + window.location.pathname;
      window.history.replaceState({ path: cleanUrl }, '', cleanUrl);
    }

    showToast('success', `Password updated successfully! Welcome back, ${data.user.username}.`);

    if (state.activeTab === 'history') loadHistory();
  } catch (err) {
    setAuthAlert(err.message, 'error');
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = '<span>Save New Password &amp; Log In</span><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M5 12h14M12 5l7 7-7 7"/></svg>';
    }
  }
}

function checkUrlForResetToken() {
  const urlParams = new URLSearchParams(window.location.search);
  const token = urlParams.get('reset_token');
  if (token) {
    openAuthModal('create_password');
    const tokenInput = document.getElementById('create-pwd-token');
    if (tokenInput) tokenInput.value = token;
    verifyResetTokenOnLoad(token);
  }
}

window.openCreatePasswordWithToken = openCreatePasswordWithToken;

function setAuthAlert(message, type = 'error') {
  const alertEl = document.getElementById('auth-alert');
  if (!alertEl) return;
  alertEl.className = `auth-alert ${type}`;
  alertEl.textContent = message;
  alertEl.style.display = 'block';
}

function clearAuthAlert() {
  const alertEl = document.getElementById('auth-alert');
  if (!alertEl) return;
  alertEl.style.display = 'none';
  alertEl.textContent = '';
}

async function handleSignIn(e) {
  e.preventDefault();
  clearAuthAlert();

  const identifier = document.getElementById('signin-identifier')?.value.trim();
  const password = document.getElementById('signin-password')?.value;
  const btn = document.getElementById('btn-submit-signin');

  if (!identifier || !password) {
    setAuthAlert('Please enter both your username/email and password.');
    return;
  }

  if (btn) {
    btn.disabled = true;
    btn.innerHTML = '<span class="btn-spinner"></span> Authenticating...';
  }

  try {
    const res = await fetch(`${API_BASE}/api/v1/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ identifier, password }),
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || 'Sign in failed. Check your credentials.');
    }

    // Save token and user
    state.authToken = data.access_token;
    state.currentUser = data.user;
    localStorage.setItem('deepguard_token', data.access_token);

    updateAuthNavUI();
    closeAuthModal();
    showToast('success', `Welcome back, ${data.user.full_name || data.user.username}!`);

    // Reload history if currently on history tab
    if (state.activeTab === 'history') loadHistory();
  } catch (err) {
    setAuthAlert(err.message, 'error');
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = '<span>Sign In</span> <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M5 12h14M12 5l7 7-7 7"/></svg>';
    }
  }
}

async function handleSignUp(e) {
  e.preventDefault();
  clearAuthAlert();

  const fullName = document.getElementById('signup-name')?.value.trim();
  const username = document.getElementById('signup-username')?.value.trim();
  const email = document.getElementById('signup-email')?.value.trim();
  const password = document.getElementById('signup-password')?.value;
  const confirmPassword = document.getElementById('signup-confirm-password')?.value;
  const btn = document.getElementById('btn-submit-signup');

  if (!username || !email || !password) {
    setAuthAlert('Please fill in all required fields.');
    return;
  }

  if (password.length < 6) {
    setAuthAlert('Password must be at least 6 characters long.');
    return;
  }

  if (password !== confirmPassword) {
    setAuthAlert('Passwords do not match.');
    return;
  }

  if (btn) {
    btn.disabled = true;
    btn.innerHTML = '<span class="btn-spinner"></span> Creating Account...';
  }

  try {
    const res = await fetch(`${API_BASE}/api/v1/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        username,
        email,
        password,
        full_name: fullName || username,
      }),
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || 'Registration failed.');
    }

    state.authToken = data.access_token;
    state.currentUser = data.user;
    localStorage.setItem('deepguard_token', data.access_token);

    updateAuthNavUI();
    closeAuthModal();
    showToast('success', `Account created successfully! Welcome, ${data.user.full_name || data.user.username}.`);

    if (state.activeTab === 'history') loadHistory();
  } catch (err) {
    setAuthAlert(err.message, 'error');
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = '<span>Create Account</span> <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M5 12h14M12 5l7 7-7 7"/></svg>';
    }
  }
}

async function loginWithDemoAccount() {
  const idInput = document.getElementById('signin-identifier');
  const pwdInput = document.getElementById('signin-password');
  if (idInput && pwdInput) {
    idInput.value = 'demo';
    pwdInput.value = 'demo12345';
  }
  const formSignIn = document.getElementById('form-signin');
  if (formSignIn) {
    formSignIn.dispatchEvent(new Event('submit', { cancelable: true }));
  }
}

function handleLogout() {
  state.authToken = null;
  state.currentUser = null;
  localStorage.removeItem('deepguard_token');

  // Notify server
  fetch(`${API_BASE}/api/v1/auth/logout`, { method: 'POST' }).catch(() => {});

  updateAuthNavUI();
  showToast('info', 'Signed out successfully.');

  // Refresh history to guest view
  if (state.activeTab === 'history') loadHistory();
}

async function restoreSession() {
  if (!state.authToken) {
    updateAuthNavUI();
    return;
  }

  try {
    const res = await authFetch(`${API_BASE}/api/v1/auth/me`);
    if (res.ok) {
      const user = await res.json();
      state.currentUser = user;
      updateAuthNavUI();
    } else {
      // Invalid or expired token
      state.authToken = null;
      state.currentUser = null;
      localStorage.removeItem('deepguard_token');
      updateAuthNavUI();
    }
  } catch {
    // Offline or network error; keep state optimistic or handle gracefully
    updateAuthNavUI();
  }
}

function updateAuthNavUI() {
  const loggedOutGroup = document.getElementById('auth-logged-out');
  const loggedInGroup = document.getElementById('auth-logged-in');
  const adminNavBtn = document.getElementById('nav-tab-admin');

  if (state.currentUser) {
    if (loggedOutGroup) loggedOutGroup.style.display = 'none';
    if (loggedInGroup) loggedInGroup.style.display = 'block';

    // Show Admin tab only for admin role
    if (adminNavBtn) {
      adminNavBtn.style.display = state.currentUser.role === 'admin' ? 'inline-flex' : 'none';
    }

    const displayName = state.currentUser.full_name || state.currentUser.username;
    const initial = (displayName.charAt(0) || 'U').toUpperCase();

    const navInitial = document.getElementById('nav-user-initial');
    const navName = document.getElementById('nav-user-name');
    const menuName = document.getElementById('menu-user-name');
    const menuEmail = document.getElementById('menu-user-email');
    const menuRole = document.getElementById('menu-user-role');

    if (navInitial) navInitial.textContent = initial;
    if (navName) navName.textContent = displayName;
    if (menuName) menuName.textContent = displayName;
    if (menuEmail) menuEmail.textContent = state.currentUser.email;
    if (menuRole) menuRole.textContent = state.currentUser.role || 'Analyst';
  } else {
    if (loggedOutGroup) loggedOutGroup.style.display = 'flex';
    if (loggedInGroup) loggedInGroup.style.display = 'none';
    if (adminNavBtn) adminNavBtn.style.display = 'none';
  }
}

// ── Expose globals for inline handlers ─────────────────────────
window.removeFile = removeFile;
window.deleteHistoryItem = deleteHistoryItem;
window.openAuthModal = openAuthModal;
window.handleLogout = handleLogout;

// ── Admin Panel ─────────────────────────────────────────────────

async function loadAdminData() {
  if (!state.currentUser || state.currentUser.role !== 'admin') {
    return; // Safety guard: non-admins cannot call this
  }

  // Reset loading states
  const usersTbody = document.getElementById('admin-users-tbody');
  const eventsTbody = document.getElementById('admin-events-tbody');
  const inquiriesTbody = document.getElementById('admin-inquiries-tbody');
  if (usersTbody) usersTbody.innerHTML = '<tr><td colspan="6" class="admin-loading">Loading…</td></tr>';
  if (eventsTbody) eventsTbody.innerHTML = '<tr><td colspan="6" class="admin-loading">Loading…</td></tr>';
  if (inquiriesTbody) inquiriesTbody.innerHTML = '<tr><td colspan="8" class="admin-loading">Loading inquiries…</td></tr>';

  try {
    // Fetch summary (stats + recent events)
    const summaryRes = await authFetch(`${API_BASE}/api/v1/admin/summary`);
    if (summaryRes.ok) {
      const summary = await summaryRes.json();
      const stats = summary.platform_stats || {};

      // Update summary cards
      const totalUsersEl = document.getElementById('admin-total-users');
      const totalAnalysesEl = document.getElementById('admin-total-analyses');
      const totalInquiriesEl = document.getElementById('admin-total-inquiries');
      if (totalUsersEl) totalUsersEl.textContent = summary.total_registered_users ?? '—';
      if (totalAnalysesEl) totalAnalysesEl.textContent = stats.total_analyses ?? '—';
      if (totalInquiriesEl) totalInquiriesEl.textContent = summary.total_inquiries ?? '—';
    }

    // Fetch all users
    const usersRes = await authFetch(`${API_BASE}/api/v1/admin/users`);
    if (usersRes.ok) {
      const data = await usersRes.json();
      renderAdminUsers(data.users || []);
    } else {
      if (usersTbody) usersTbody.innerHTML = '<tr><td colspan="6" class="admin-loading">Failed to load users.</td></tr>';
    }

    // Fetch login events
    const eventsRes = await authFetch(`${API_BASE}/api/v1/admin/events?limit=200`);
    if (eventsRes.ok) {
      const data = await eventsRes.json();
      renderAdminEvents(data.events || []);

      // Count logins vs registrations
      const logins = (data.events || []).filter(e => e.event_type === 'login').length;
      const regs = (data.events || []).filter(e => e.event_type === 'register').length;
      const totalLoginsEl = document.getElementById('admin-total-logins');
      const totalRegsEl = document.getElementById('admin-total-registrations');
      if (totalLoginsEl) totalLoginsEl.textContent = logins;
      if (totalRegsEl) totalRegsEl.textContent = regs;
    } else {
      if (eventsTbody) eventsTbody.innerHTML = '<tr><td colspan="6" class="admin-loading">Failed to load events.</td></tr>';
    }

    // Fetch inquiries ("Get in Touch")
    const inqRes = await authFetch(`${API_BASE}/api/v1/admin/inquiries?limit=100`);
    if (inqRes.ok) {
      const inqData = await inqRes.json();
      renderAdminInquiries(inqData.inquiries || []);
      const totalInquiriesEl = document.getElementById('admin-total-inquiries');
      if (totalInquiriesEl) totalInquiriesEl.textContent = (inqData.inquiries || []).length;
    } else {
      if (inquiriesTbody) inquiriesTbody.innerHTML = '<tr><td colspan="8" class="admin-loading">Failed to load inquiries.</td></tr>';
    }

  } catch (err) {
    console.error('Admin data load error:', err);
    if (usersTbody) usersTbody.innerHTML = '<tr><td colspan="6" class="admin-loading">Network error. Is the server running?</td></tr>';
  }
}

function renderAdminUsers(users) {
  const tbody = document.getElementById('admin-users-tbody');
  if (!tbody) return;

  if (!users.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="admin-loading">No users found.</td></tr>';
    return;
  }

  tbody.innerHTML = users.map((u, i) => `
    <tr>
      <td style="color:#64748b">${i + 1}</td>
      <td><strong style="color:#e2e8f0">${escapeHtml(u.username)}</strong></td>
      <td>${escapeHtml(u.full_name || '—')}</td>
      <td style="color:#94a3b8">${escapeHtml(u.email)}</td>
      <td><span class="${u.role === 'admin' ? 'badge-admin' : 'badge-analyst'}">${escapeHtml(u.role)}</span></td>
      <td style="color:#64748b;white-space:nowrap">${escapeHtml(u.created_at)}</td>
      <td>
        <button onclick="promptAdminResetPassword(${u.id}, '${escapeHtml(u.username)}')" style="background:rgba(239,68,68,0.12); border:1px solid rgba(239,68,68,0.25); color:#f87171; border-radius:6px; padding:0.25rem 0.6rem; font-size:0.75rem; font-weight:600; cursor:pointer;">
          🔑 Reset Pwd
        </button>
      </td>
    </tr>
  `).join('');
}

function renderAdminEvents(events) {
  const tbody = document.getElementById('admin-events-tbody');
  if (!tbody) return;

  if (!events.length) {
    tbody.innerHTML = '<tr><td colspan="6" class="admin-loading">No events recorded yet. Try logging in or registering a new account.</td></tr>';
    return;
  }

  tbody.innerHTML = events.map((e, i) => {
    let badgeHtml = '';
    if (e.event_type === 'register') {
      badgeHtml = '<span class="badge-register">✍️ Register</span>';
    } else if (e.event_type === 'login') {
      badgeHtml = '<span class="badge-login">🔐 Login</span>';
    } else if (e.event_type === 'password_reset_link_generated') {
      badgeHtml = '<span class="badge-inquiry">🔗 Link Generated</span>';
    } else if (e.event_type === 'password_reset_completed' || e.event_type === 'password_reset') {
      badgeHtml = '<span class="badge-reset">🔑 Pwd Changed</span>';
    } else if (e.event_type === 'admin_password_reset') {
      badgeHtml = '<span class="badge-reset">🛡️ Admin Reset</span>';
    } else {
      badgeHtml = `<span class="badge-analyst">${escapeHtml(e.event_type)}</span>`;
    }

    return `
    <tr>
      <td style="color:#64748b">${i + 1}</td>
      <td>${badgeHtml}</td>
      <td><strong style="color:#e2e8f0">${escapeHtml(e.username)}</strong></td>
      <td style="color:#94a3b8">${escapeHtml(e.email)}</td>
      <td style="color:#64748b;font-family:monospace;font-size:0.8rem">${escapeHtml(e.ip_address || 'unknown')}</td>
      <td style="color:#64748b;white-space:nowrap">${escapeHtml(e.timestamp)}</td>
    </tr>
  `;
  }).join('');
}

async function promptAdminResetPassword(userId, username) {
  const newPwd = prompt(`Enter new password for user '${username}' (minimum 6 characters):`, 'User12345');
  if (!newPwd || newPwd.trim().length < 6) {
    if (newPwd !== null) alert('Password must be at least 6 characters long.');
    return;
  }

  try {
    const res = await authFetch(`${API_BASE}/api/v1/admin/users/${userId}/reset-password`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ new_password: newPwd.trim() }),
    });

    const data = await res.json();
    if (res.ok) {
      alert(`Success! Password for '${username}' has been updated to: ${newPwd.trim()}`);
      loadAdminData();
    } else {
      alert(data.detail || 'Could not reset password.');
    }
  } catch (err) {
    console.error('Admin password reset error:', err);
    alert('Network error while resetting password.');
  }
}

window.promptAdminResetPassword = promptAdminResetPassword;

function renderAdminInquiries(inquiries) {
  const tbody = document.getElementById('admin-inquiries-tbody');
  if (!tbody) return;

  if (!inquiries.length) {
    tbody.innerHTML = '<tr><td colspan="8" class="admin-loading">No inquiries submitted yet. Visitors who use "Get in Touch" will appear here.</td></tr>';
    return;
  }

  tbody.innerHTML = inquiries.map((inq, i) => `
    <tr>
      <td style="color:#64748b">${i + 1}</td>
      <td><strong style="color:#e2e8f0">${escapeHtml(inq.full_name)}</strong></td>
      <td><a href="mailto:${escapeHtml(inq.email)}" style="color:#06b6d4;text-decoration:none;">${escapeHtml(inq.email)}</a></td>
      <td style="color:#cbd5e1">${escapeHtml(inq.organization || '—')}</td>
      <td><span class="badge-inquiry">${escapeHtml(inq.interest || 'General')}</span></td>
      <td style="color:#94a3b8;font-size:0.82rem;max-width:280px;word-break:break-word;">${escapeHtml(inq.message)}</td>
      <td style="color:#64748b;font-family:monospace;font-size:0.8rem">${escapeHtml(inq.ip_address || 'unknown')}</td>
      <td style="color:#64748b;white-space:nowrap">${escapeHtml(inq.timestamp)}</td>
    </tr>
  `).join('');
}

// ── "Get in Touch" Contact Form Submission ──────────────────────
async function handleContactSubmit(event) {
  event.preventDefault();
  const form = document.getElementById('contact-form');
  const btn = document.getElementById('btn-submit-contact');
  const feedback = document.getElementById('contact-form-feedback');

  if (!form || !btn) return;

  const fullName = document.getElementById('contact-name')?.value.trim();
  const email = document.getElementById('contact-email')?.value.trim();
  const org = document.getElementById('contact-org')?.value.trim();
  const interest = document.getElementById('contact-interest')?.value;
  const message = document.getElementById('contact-message')?.value.trim();

  if (!fullName || !email || !message) {
    if (feedback) {
      feedback.className = 'cfc-feedback error';
      feedback.textContent = 'Please fill in all required fields (Name, Email, Message).';
      feedback.style.display = 'block';
    }
    return;
  }

  // Loading state
  const originalText = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = '<span>Transmitting Inquiry...</span>';
  if (feedback) feedback.style.display = 'none';

  try {
    const res = await fetch(`${API_BASE}/api/v1/contact`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        full_name: fullName,
        email: email,
        organization: org,
        interest: interest,
        message: message,
      }),
    });

    const data = await res.json();

    if (res.ok) {
      if (feedback) {
        feedback.className = 'cfc-feedback success';
        feedback.innerHTML = `✓ <strong>Inquiry Received!</strong> Thank you ${escapeHtml(fullName)}. Your request has been securely logged and forwarded to Sunil Kumar. Our forensics team will reach out to <em>${escapeHtml(email)}</em> shortly.`;
        feedback.style.display = 'block';
      }
      form.reset();
      showToast('Forensic inquiry transmitted successfully!', 'success');
    } else {
      if (feedback) {
        feedback.className = 'cfc-feedback error';
        feedback.textContent = data.detail || 'Could not submit inquiry. Please try again.';
        feedback.style.display = 'block';
      }
    }
  } catch (err) {
    console.error('Contact submission error:', err);
    if (feedback) {
      feedback.className = 'cfc-feedback error';
      feedback.textContent = 'Network error. Please verify the platform server is running.';
      feedback.style.display = 'block';
    }
  } finally {
    btn.disabled = false;
    btn.innerHTML = originalText;
  }
}

// Expose handlers globally
window.loadAdminData = loadAdminData;
window.handleContactSubmit = handleContactSubmit;
window.loadAdminData = loadAdminData;
