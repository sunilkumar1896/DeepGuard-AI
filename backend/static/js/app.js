/**
 * ============================================================
 * AI-Powered Deepfake Detection Platform — Frontend Application
 * ============================================================
 * Handles file upload, API communication, results rendering,
 * history management, and dashboard visualization.
 */

const API_BASE = window.location.origin;

// ── State ──────────────────────────────────────────────────────
const state = {
  selectedFile: null,
  fileType: null, // 'image' or 'video'
  isAnalyzing: false,
  currentResult: null,
  history: [],
  backendOnline: false,
  activeTab: 'analyze',
};

// ── DOM Ready ──────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initApp();
});

function initApp() {
  setupNavigation();
  setupUploadZones();
  setupAnalyzeButton();
  checkBackendHealth();

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

    const res = await fetch(endpoint, {
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
    const res = await fetch(`${API_BASE}/api/v1/history?limit=50`);
    const data = await res.json();
    state.history = data.history || [];
    renderHistory(state.history);
  } catch {
    renderHistory([]);
  }
}

function renderHistory(items) {
  const container = document.getElementById('history-content');
  if (!container) return;

  if (items.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">📋</div>
        <div class="empty-state-text">No analysis history yet</div>
      </div>
    `;
    return;
  }

  container.innerHTML = `
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
    await fetch(`${API_BASE}/api/v1/history/${id}`, { method: 'DELETE' });
    showToast('success', 'Record deleted');
    loadHistory();
  } catch {
    showToast('error', 'Failed to delete record');
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

// ── Expose globals for inline handlers ─────────────────────────
window.removeFile = removeFile;
window.deleteHistoryItem = deleteHistoryItem;
