// =============================================================================
// Training Dashboard - Enhanced JavaScript
// =============================================================================

// Global state
let charts = {};
let currentData = [];
let currentRunPath = null;
let currentLogPath = null;
let autoRefreshInterval = null;
let statusInterval = null;

// API base URL (detect if running on server or file://)
const API_BASE = window.location.protocol === 'file:' ? 'http://localhost:8050' : '';

// =============================================================================
// Initialization
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    loadRuns();
    loadStatus();

    // Start status polling
    statusInterval = setInterval(loadStatus, 30000);

    // Bind file selector
    document.getElementById('file-selector').addEventListener('change', (e) => {
        if (e.target.value) {
            currentRunPath = e.target.value;
            loadTrainingData(e.target.value);
            loadAnalysis(e.target.value);
        }
    });
});

// =============================================================================
// Tab Navigation
// =============================================================================

function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    // Show selected tab
    document.getElementById(tabName + '-tab').classList.add('active');
    event.target.classList.add('active');

    // Load tab-specific data
    if (tabName === 'checkpoints') {
        loadCheckpoints();
    } else if (tabName === 'logs') {
        loadLogFiles();
    }
}

// =============================================================================
// API: Load Training Runs
// =============================================================================

async function loadRuns() {
    const selector = document.getElementById('file-selector');
    selector.innerHTML = '<option value="">Loading...</option>';

    try {
        const response = await fetch(API_BASE + '/api/runs');
        const data = await response.json();

        selector.innerHTML = '<option value="">-- Select Training Run --</option>';

        if (!data.runs || data.runs.length === 0) {
            selector.innerHTML += '<option disabled>No training runs found</option>';
            return;
        }

        data.runs.forEach(run => {
            const option = document.createElement('option');
            option.value = run.path;
            option.textContent = `${run.name} (${formatDate(run.modified)})`;
            selector.appendChild(option);
        });

        // Auto-select most recent
        if (data.runs.length > 0) {
            selector.value = data.runs[0].path;
            currentRunPath = data.runs[0].path;
            loadTrainingData(data.runs[0].path);
            loadAnalysis(data.runs[0].path);
        }

    } catch (e) {
        console.error("Failed to load runs:", e);
        selector.innerHTML = '<option value="">Error loading runs - Is server running?</option>';
    }
}

// =============================================================================
// API: Load Training Data
// =============================================================================

async function loadTrainingData(csvPath) {
    const btn = document.getElementById('refresh-btn');
    const originalText = btn.innerText;
    btn.innerText = "Loading...";

    try {
        const response = await fetch(API_BASE + '/api/training/' + csvPath);
        const result = await response.json();

        if (result.error) {
            console.error("Error:", result.error);
            return;
        }

        // Calculate derived metrics if not present
        currentData = result.data.map((d, i, arr) => {
            // Calculate action percentages if not present
            if (d.pct_buy === undefined && d.n_buy !== undefined) {
                const total = (d.n_buy || 0) + (d.n_hold || 0) + (d.n_sell || 0);
                if (total > 0) {
                    d.pct_buy = (d.n_buy / total) * 100;
                    d.pct_hold = (d.n_hold / total) * 100;
                    d.pct_sell = (d.n_sell / total) * 100;
                }
            }

            // Calculate reward_ma20 if not present
            if (d.reward_ma20 === undefined && d.reward !== undefined) {
                const start = Math.max(0, i - 19);
                const slice = arr.slice(start, i + 1).map(x => x.reward);
                d.reward_ma20 = slice.reduce((a, b) => a + b, 0) / slice.length;
            }

            return d;
        });

        updateDashboard();

    } catch (e) {
        console.error("Load failed:", e);
    } finally {
        btn.innerText = originalText;
    }
}

// =============================================================================
// API: Load Analysis
// =============================================================================

async function loadAnalysis(csvPath) {
    try {
        const response = await fetch(API_BASE + '/api/analysis/' + csvPath);
        const analysis = await response.json();

        if (analysis.error) {
            console.error("Analysis error:", analysis.error);
            return;
        }

        updateAnalysisTab(analysis);

    } catch (e) {
        console.error("Analysis load failed:", e);
    }
}

function updateAnalysisTab(analysis) {
    // Update insights
    const insightsList = document.getElementById('insights-list');
    if (analysis.insights && analysis.insights.length > 0) {
        insightsList.innerHTML = analysis.insights.map(i => `<li>${i}</li>`).join('');
    } else {
        insightsList.innerHTML = '<li class="empty">No insights yet</li>';
    }

    // Update warnings
    const warningsList = document.getElementById('warnings-list');
    if (analysis.warnings && analysis.warnings.length > 0) {
        warningsList.innerHTML = analysis.warnings.map(w => `<li>${w}</li>`).join('');
    } else {
        warningsList.innerHTML = '<li class="empty">No warnings</li>';
    }

    // Update detailed metrics
    const metricsDiv = document.getElementById('detailed-metrics');
    if (analysis.metrics) {
        const m = analysis.metrics;

        // Determine mastery threshold based on synthetic level
        let threshold = 2.5;
        if (m.synthetic_level !== undefined && m.synthetic_level !== null) {
            threshold = [2.5, 2.0, 1.5, 1.0][m.synthetic_level] || 2.5;
        }

        metricsDiv.innerHTML = `
            <div class="metrics-table">
                <div class="metric-row header">Configuration</div>
                <div class="metric-row">
                    <span class="metric-label">Total Epochs:</span>
                    <span class="metric-value">${m.total_epochs || 'N/A'}</span>
                </div>
                ${m.curriculum_name ? `<div class="metric-row">
                    <span class="metric-label">Curriculum:</span>
                    <span class="metric-value">${m.curriculum_name} (L${m.synthetic_level})</span>
                </div>` : ''}
                ${m.model_size ? `<div class="metric-row">
                    <span class="metric-label">Model Size:</span>
                    <span class="metric-value">${m.model_size}</span>
                </div>` : ''}

                <div class="metric-row header">Reward Performance</div>
                <div class="metric-row">
                    <span class="metric-label">Average Reward:</span>
                    <span class="metric-value">${m.avg_reward?.toFixed(3) || 'N/A'}%</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Final MA(50):</span>
                    <span class="metric-value">${m.final_reward_ma50?.toFixed(3) || 'N/A'}%</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Mastery Threshold:</span>
                    <span class="metric-value">${threshold}%</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Reward Trend:</span>
                    <span class="metric-value ${m.reward_trend_pct > 0 ? 'positive' : 'negative'}">${m.reward_trend_pct?.toFixed(1) || 'N/A'}%</span>
                </div>

                <div class="metric-row header">Value Calibration</div>
                <div class="metric-row">
                    <span class="metric-label">Value-Reward Corr:</span>
                    <span class="metric-value ${(m.value_reward_correlation || 0) > 0.7 ? 'positive' : (m.value_reward_correlation || 0) < 0.3 ? 'negative' : ''}">${m.value_reward_correlation?.toFixed(3) || 'N/A'}</span>
                </div>
                ${m.final_value_mse !== undefined ? `<div class="metric-row">
                    <span class="metric-label">Value MSE:</span>
                    <span class="metric-value ${m.final_value_mse < 5 ? 'positive' : m.final_value_mse > 10 ? 'negative' : ''}">${m.final_value_mse?.toFixed(2)}</span>
                </div>` : ''}
                ${m.final_value_std !== undefined ? `<div class="metric-row">
                    <span class="metric-label">Value Std:</span>
                    <span class="metric-value ${m.final_value_std > 1 ? 'positive' : m.final_value_std < 0.5 ? 'negative' : ''}">${m.final_value_std?.toFixed(2)}</span>
                </div>` : ''}
                ${m.final_value_scale !== undefined ? `<div class="metric-row">
                    <span class="metric-label">Value Scale:</span>
                    <span class="metric-value ${Math.abs(m.final_value_scale - 1.0) < 0.1 ? 'positive' : ''}">${m.final_value_scale?.toFixed(3)}</span>
                </div>` : ''}
                ${m.final_value_shift !== undefined ? `<div class="metric-row">
                    <span class="metric-label">Value Shift:</span>
                    <span class="metric-value ${Math.abs(m.final_value_shift) < 0.5 ? 'positive' : ''}">${m.final_value_shift?.toFixed(3)}</span>
                </div>` : ''}
                ${m.mse_reduction_pct !== undefined ? `<div class="metric-row">
                    <span class="metric-label">MSE Reduction:</span>
                    <span class="metric-value ${m.mse_reduction_pct > 0 ? 'positive' : 'negative'}">${m.mse_reduction_pct?.toFixed(1)}%</span>
                </div>` : ''}

                <div class="metric-row header">Policy Health</div>
                <div class="metric-row">
                    <span class="metric-label">Exploration State:</span>
                    <span class="metric-value">${m.exploration_state || 'N/A'}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Entropy Change:</span>
                    <span class="metric-value ${m.entropy_change_pct < -30 ? 'negative' : ''}">${m.entropy_change_pct?.toFixed(1) || 'N/A'}%</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Loss Reduction:</span>
                    <span class="metric-value ${(m.loss_reduction_pct || 0) > 0 ? 'positive' : ''}">${m.loss_reduction_pct?.toFixed(1) || 'N/A'}%</span>
                </div>

                <div class="metric-row header">Action Distribution (Recent)</div>
                <div class="metric-row">
                    <span class="metric-label">Buy:</span>
                    <span class="metric-value buy">${(m.recent_pct_buy ?? m.recent_buy_pct)?.toFixed(1) || 'N/A'}%</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Hold:</span>
                    <span class="metric-value hold">${(m.recent_pct_hold ?? m.recent_hold_pct)?.toFixed(1) || 'N/A'}%</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Sell:</span>
                    <span class="metric-value sell">${(m.recent_pct_sell ?? m.recent_sell_pct)?.toFixed(1) || 'N/A'}%</span>
                </div>
            </div>
        `;
    }

    // Update phases
    const phasesDiv = document.getElementById('phases-timeline');
    if (analysis.phases && analysis.phases.length > 0) {
        phasesDiv.innerHTML = analysis.phases.map((phase, i) => {
            const typeClass = phase.type;
            const duration = phase.end - phase.start;
            return `
                <div class="phase-item ${typeClass}">
                    <span class="phase-label">${phase.type.charAt(0).toUpperCase() + phase.type.slice(1)}</span>
                    <span class="phase-range">Epochs ${phase.start} - ${phase.end} (${duration} epochs)</span>
                </div>
            `;
        }).join('');
    } else {
        phasesDiv.innerHTML = '<p>Not enough data for phase detection</p>';
    }
}

// =============================================================================
// API: Load System Status
// =============================================================================

async function loadStatus() {
    try {
        const response = await fetch(API_BASE + '/api/status');
        const status = await response.json();

        // Training status badge
        const trainingBadge = document.getElementById('training-status');
        if (status.training_active) {
            trainingBadge.textContent = 'TRAINING';
            trainingBadge.className = 'status-badge active';
        } else {
            trainingBadge.textContent = 'IDLE';
            trainingBadge.className = 'status-badge idle';
        }

        // GPU status badge
        const gpuBadge = document.getElementById('gpu-status');
        if (status.gpu) {
            const g = status.gpu;
            gpuBadge.textContent = `GPU: ${g.utilization_pct}% | ${g.memory_used_mb}/${g.memory_total_mb}MB | ${g.temperature_c}C`;
            gpuBadge.title = g.name;
            gpuBadge.className = 'gpu-badge ' + (g.utilization_pct > 80 ? 'high' : g.utilization_pct > 30 ? 'medium' : 'low');
        } else {
            gpuBadge.textContent = 'GPU: N/A';
            gpuBadge.className = 'gpu-badge';
        }

    } catch (e) {
        // Server might not be running
        document.getElementById('training-status').textContent = '--';
        document.getElementById('gpu-status').textContent = '--';
    }
}

// =============================================================================
// API: Load Checkpoints
// =============================================================================

let checkpointProgressionChart = null;

async function loadCheckpoints() {
    const tbody = document.getElementById('checkpoints-body');
    tbody.innerHTML = '<tr><td colspan="5">Loading...</td></tr>';

    try {
        const response = await fetch(API_BASE + '/api/checkpoints');
        const data = await response.json();

        if (!data.checkpoints || data.checkpoints.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5">No checkpoints found</td></tr>';
            return;
        }

        // Group by experiment for comparison button
        const experiments = [...new Set(data.checkpoints.map(c => c.experiment))];

        tbody.innerHTML = data.checkpoints.map(ckpt => `
            <tr>
                <td>${ckpt.experiment}</td>
                <td>${ckpt.step}</td>
                <td>${ckpt.size_mb} MB</td>
                <td>${formatDate(ckpt.modified)}</td>
                <td>
                    <button class="btn-small" onclick="analyzeCheckpoint('${ckpt.path}')">Details</button>
                    <button class="btn-small" onclick="compareExperiment('${ckpt.experiment}')">Compare</button>
                </td>
            </tr>
        `).join('');

    } catch (e) {
        tbody.innerHTML = '<tr><td colspan="5">Error loading checkpoints</td></tr>';
    }
}

async function analyzeCheckpoint(ckptPath) {
    const detailsCard = document.getElementById('checkpoint-details-card');
    const content = document.getElementById('checkpoint-details-content');

    detailsCard.style.display = 'block';
    content.innerHTML = '<p>Loading analysis...</p>';

    try {
        const response = await fetch(API_BASE + '/api/checkpoint-analysis/' + ckptPath);
        const data = await response.json();

        if (data.error) {
            content.innerHTML = `<p class="error">Error: ${data.error}</p>`;
            return;
        }

        content.innerHTML = `
            <div class="checkpoint-info">
                <div class="info-section">
                    <h4>Overview</h4>
                    <div class="info-grid">
                        <div class="info-item">
                            <span class="info-label">Step:</span>
                            <span class="info-value">${data.step || 'N/A'}</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Total Size:</span>
                            <span class="info-value">${data.total_size_mb} MB</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Est. Parameters:</span>
                            <span class="info-value">${data.parameter_count_formatted || 'N/A'}</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Files:</span>
                            <span class="info-value">${data.files.length}</span>
                        </div>
                    </div>
                </div>

                ${data.checkpoint_metadata ? `
                <div class="info-section">
                    <h4>Metadata</h4>
                    <div class="info-grid">
                        <div class="info-item">
                            <span class="info-label">Handler:</span>
                            <span class="info-value" style="font-size: 11px;">${data.checkpoint_metadata.item_handlers || 'N/A'}</span>
                        </div>
                    </div>
                </div>
                ` : ''}

                <div class="info-section">
                    <h4>File Structure</h4>
                    <div class="file-list">
                        ${data.files.slice(0, 15).map(f => `
                            <div class="file-item">
                                <span class="file-name">${f.name}</span>
                                <span class="file-size">${f.size_kb} KB</span>
                            </div>
                        `).join('')}
                        ${data.files.length > 15 ? `<div class="file-item muted">... and ${data.files.length - 15} more files</div>` : ''}
                    </div>
                </div>
            </div>
        `;

    } catch (e) {
        content.innerHTML = `<p class="error">Error loading analysis: ${e.message}</p>`;
    }
}

async function compareExperiment(experiment) {
    const compCard = document.getElementById('checkpoint-comparison-card');
    const content = document.getElementById('checkpoint-comparison-content');

    compCard.style.display = 'block';
    content.innerHTML = '<p>Loading comparison...</p>';

    try {
        const response = await fetch(API_BASE + '/api/checkpoint-comparison?experiment=' + encodeURIComponent(experiment));
        const data = await response.json();

        if (data.error) {
            content.innerHTML = `<p class="error">Error: ${data.error}</p>`;
            return;
        }

        let speedInfo = '';
        if (data.training_speed) {
            speedInfo = `
                <div class="info-section">
                    <h4>Training Speed</h4>
                    <div class="info-grid">
                        <div class="info-item">
                            <span class="info-label">Steps/Hour:</span>
                            <span class="info-value">${data.training_speed.steps_per_hour}</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Total Hours:</span>
                            <span class="info-value">${data.training_speed.total_training_hours}</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Total Steps:</span>
                            <span class="info-value">${data.training_speed.total_steps}</span>
                        </div>
                    </div>
                </div>
            `;
        }

        content.innerHTML = `
            <div class="checkpoint-info">
                <div class="info-section">
                    <h4>Experiment: ${data.experiment}</h4>
                    <div class="info-grid">
                        <div class="info-item">
                            <span class="info-label">Checkpoints:</span>
                            <span class="info-value">${data.checkpoints.length}</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Latest Step:</span>
                            <span class="info-value">${data.checkpoints.length > 0 ? data.checkpoints[data.checkpoints.length - 1].step : 'N/A'}</span>
                        </div>
                    </div>
                </div>
                ${speedInfo}
            </div>
        `;

        // Update progression chart
        updateCheckpointProgressionChart(data);

    } catch (e) {
        content.innerHTML = `<p class="error">Error loading comparison: ${e.message}</p>`;
    }
}

function updateCheckpointProgressionChart(data) {
    const ctx = document.getElementById('checkpoint-progression-chart');

    if (checkpointProgressionChart) {
        checkpointProgressionChart.destroy();
    }

    if (!data.progression || data.progression.steps.length === 0) {
        return;
    }

    checkpointProgressionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.progression.steps,
            datasets: [{
                label: 'Checkpoint Size (MB)',
                data: data.progression.sizes_mb,
                borderColor: '#3498db',
                backgroundColor: 'rgba(52, 152, 219, 0.2)',
                fill: true,
                tension: 0.3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { title: { display: true, text: 'Step' } },
                y: { title: { display: true, text: 'Size (MB)' }, beginAtZero: true }
            }
        }
    });
}

function hideCheckpointDetails() {
    document.getElementById('checkpoint-details-card').style.display = 'none';
}

function hideCheckpointComparison() {
    document.getElementById('checkpoint-comparison-card').style.display = 'none';
}

// =============================================================================
// API: Load Log Files
// =============================================================================

async function loadLogFiles() {
    const list = document.getElementById('log-files-list');
    list.innerHTML = '<li>Loading...</li>';

    try {
        const response = await fetch(API_BASE + '/api/log-files');
        const data = await response.json();

        if (!data.files || data.files.length === 0) {
            list.innerHTML = '<li>No log files found</li>';
            return;
        }

        list.innerHTML = data.files.map(file => `
            <li onclick="loadLogContent('${file.path}')" class="log-file-item">
                <span class="log-name">${file.name}</span>
                <span class="log-meta">${formatSize(file.size)}</span>
            </li>
        `).join('');

    } catch (e) {
        list.innerHTML = '<li>Error loading log files</li>';
    }
}

async function loadLogContent(logPath) {
    currentLogPath = logPath;
    const lines = parseInt(document.getElementById('log-lines').value) || 200;

    document.getElementById('current-log-name').textContent = logPath.split('/').pop();
    document.getElementById('log-content').textContent = 'Loading...';

    // Highlight selected file
    document.querySelectorAll('.log-file-item').forEach(item => {
        item.classList.remove('selected');
        if (item.textContent.includes(logPath.split('/').pop())) {
            item.classList.add('selected');
        }
    });

    try {
        const response = await fetch(API_BASE + `/api/logs/${logPath}?lines=${lines}`);
        const data = await response.json();

        if (data.error) {
            document.getElementById('log-content').textContent = 'Error: ' + data.error;
            return;
        }

        document.getElementById('log-content').textContent = data.content;
        // Scroll to bottom
        const pre = document.getElementById('log-content');
        pre.scrollTop = pre.scrollHeight;

    } catch (e) {
        document.getElementById('log-content').textContent = 'Error loading log: ' + e.message;
    }
}

function refreshCurrentLog() {
    if (currentLogPath) {
        loadLogContent(currentLogPath);
    }
}

// =============================================================================
// Dashboard Updates
// =============================================================================

function updateDashboard() {
    const skip = parseInt(document.getElementById('skip-epochs').value) || 0;
    const filteredData = currentData.filter(d => d.epoch > skip);
    updateStats(filteredData);
    updateCharts(filteredData);
}

function updateStats(data) {
    if (!data || data.length === 0) return;

    const latest = data[data.length - 1];
    const avgReward50 = movingAverage(data.map(d => d.reward), 50).pop();

    // Calculate correlation
    let valCorr = 'N/A';
    if (data.length > 5 && data[0].avg_value !== undefined) {
        valCorr = calculateCorrelation(
            data.map(d => d.reward),
            data.map(d => d.avg_value)
        ).toFixed(3);
    }

    const container = document.getElementById('stats-content');

    // Determine mastery threshold based on synthetic level
    let threshold = 2.5;
    const synLevel = latest.synthetic_level;
    if (synLevel !== undefined) {
        threshold = [2.5, 2.0, 1.5, 1.0][synLevel] || 2.5;
    } else if (currentRunPath) {
        if (currentRunPath.includes('L0')) threshold = 2.5;
        else if (currentRunPath.includes('L1')) threshold = 2.0;
        else if (currentRunPath.includes('L2')) threshold = 1.5;
        else if (currentRunPath.includes('L3')) threshold = 1.0;
    }

    const isMastery = avgReward50 >= threshold;
    const statusColor = isMastery ? '#2ecc71' : (avgReward50 > 0 ? '#f1c40f' : '#e74c3c');

    // Entropy health indicator
    const entropy = latest.policy_entropy;
    let entropyStatus = '';
    let entropyColor = '';
    if (entropy !== undefined) {
        if (entropy >= 3.0 && entropy <= 4.5) {
            entropyStatus = 'Healthy';
            entropyColor = '#2ecc71';
        } else if (entropy < 3.0) {
            entropyStatus = 'Low!';
            entropyColor = '#e74c3c';
        } else {
            entropyStatus = 'High';
            entropyColor = '#f1c40f';
        }
    }

    // Model info
    const modelInfo = latest.model_size ? `${latest.model_size}` : '';
    const levelInfo = latest.synthetic_level !== undefined ? `L${latest.synthetic_level}` : '';
    const configInfo = [levelInfo, modelInfo].filter(x => x).join(' / ') || 'N/A';

    // Calculate recent action percentages (last 10 epochs)
    const recent10 = data.slice(-10);
    const hasLimitOrders = recent10.length > 0 && recent10[0].pct_limit_buy !== undefined;

    let actionDistHtml = '';
    let sellWarning = '';

    if (hasLimitOrders) {
        const recentBuy = recent10.reduce((s, d) => s + (d.pct_buy || 0), 0) / recent10.length;
        const recentHold = recent10.reduce((s, d) => s + (d.pct_hold || 0), 0) / recent10.length;
        const recentSell = recent10.reduce((s, d) => s + (d.pct_sell || 0), 0) / recent10.length;
        const recentLimitBuy = recent10.reduce((s, d) => s + (d.pct_limit_buy || 0), 0) / recent10.length;
        const recentLimitSell = recent10.reduce((s, d) => s + (d.pct_limit_sell || 0), 0) / recent10.length;
        const totalSell = recentSell + recentLimitSell;

        actionDistHtml = `
            <span style="color:#2ecc71" title="Market Buy">B:${recentBuy.toFixed(0)}%</span>
            <span style="color:#95a5a6" title="Hold">H:${recentHold.toFixed(0)}%</span>
            <span style="color:#e74c3c" title="Market Sell">S:${recentSell.toFixed(0)}%</span>
            <span style="color:#3498db" title="Limit Buy">LB:${recentLimitBuy.toFixed(0)}%</span>
            <span style="color:#c0392b" title="Limit Sell">LS:${recentLimitSell.toFixed(0)}%</span>
        `;
        sellWarning = totalSell < 10 ? 'Warning: Low sells!' : (totalSell < 20 ? 'Low sells' : 'Balanced');

    } else {
        const recentBuy = recent10.reduce((s, d) => s + (d.pct_buy || 0), 0) / recent10.length;
        const recentHold = recent10.reduce((s, d) => s + (d.pct_hold || 0), 0) / recent10.length;
        const recentSell = recent10.reduce((s, d) => s + (d.pct_sell || 0), 0) / recent10.length;
        
        actionDistHtml = `
            <span style="color:#2ecc71">B:${recentBuy.toFixed(0)}%</span>
            <span style="color:#95a5a6">H:${recentHold.toFixed(0)}%</span>
            <span style="color:#e74c3c">S:${recentSell.toFixed(0)}%</span>
        `;
        sellWarning = recentSell < 10 ? 'Warning: Low sells!' : (recentSell < 20 ? 'Low sells' : 'Balanced');
    }

    container.innerHTML = `
        <div class="stat-box" style="border-left-color: ${statusColor}">
            <div class="stat-label">Status</div>
            <div class="stat-value" style="color: ${statusColor}">
                ${isMastery ? 'MASTERY' : 'TRAINING'}
            </div>
            <div class="stat-sub">Epoch ${latest.epoch}</div>
        </div>

        <div class="stat-box">
            <div class="stat-label">Reward</div>
            <div class="stat-value">${avgReward50.toFixed(2)}%</div>
            <div class="stat-sub">MA20: ${latest.reward_ma20?.toFixed(2)}% | Target: ${threshold}%</div>
        </div>

        <div class="stat-box">
            <div class="stat-label">Value Calibration</div>
            <div class="stat-value">${valCorr}</div>
            <div class="stat-sub">Correlation (>0.7 good)</div>
        </div>

        <div class="stat-box" style="border-left-color: ${entropyColor || 'var(--accent-blue)'}">
            <div class="stat-label">Policy Entropy</div>
            <div class="stat-value">${entropy?.toFixed(2) || 'N/A'} <span style="font-size:12px;color:${entropyColor}">${entropyStatus}</span></div>
            <div class="stat-sub">3.0-4.5 = healthy exploration</div>
        </div>

        <div class="stat-box">
            <div class="stat-label">Actions (Recent 10)</div>
            <div class="stat-value" style="font-size:16px">
                ${actionDistHtml}
            </div>
            <div class="stat-sub">${sellWarning}</div>
        </div>

        <div class="stat-box">
            <div class="stat-label">Speed</div>
            <div class="stat-value">${((latest.train_time || 0) + (latest.self_play_time || 0)).toFixed(1)}s</div>
            <div class="stat-sub">per epoch</div>
        </div>
    `;

    // Update Progress Bar
    const progress = Math.min(100, Math.max(0, (avgReward50 / threshold) * 100));
    document.getElementById('mastery-section').style.display = 'block';
    const bar = document.getElementById('mastery-bar');
    bar.style.width = `${progress}%`;
    bar.style.backgroundColor = statusColor;
    document.getElementById('mastery-text').innerText =
        `${progress.toFixed(1)}% to mastery (${avgReward50.toFixed(2)}% / ${threshold}%)`;
}

function updateCharts(data) {
    if (!data || data.length === 0) return;

    const labels = data.map(d => d.epoch);

    // 1. Reward Chart
    const rewards = data.map(d => d.reward);
    const ma50 = movingAverage(rewards, 50);

    charts.reward.data.labels = labels;
    charts.reward.data.datasets[0].data = rewards;
    charts.reward.data.datasets[1].data = ma50;
    charts.reward.update();

    // 2. Loss Chart
    const isLog = document.getElementById('log-scale-loss').checked;
    charts.loss.options.scales.y.type = isLog ? 'logarithmic' : 'linear';

    charts.loss.data.labels = labels;
    charts.loss.data.datasets[0].data = data.map(d => d.total_loss);
    charts.loss.data.datasets[1].data = data.map(d => d.value_loss);
    charts.loss.data.datasets[2].data = data.map(d => d.policy_loss);
    charts.loss.data.datasets[3].data = data.map(d => d.reward_loss);
    charts.loss.update();

    // 3. Action Chart
    const isStacked = document.getElementById('stack-actions').checked;
    const isNormalized = document.getElementById('normalize-actions').checked;
    const hasLimitActions = data.length > 0 && data[0].n_limit_buy !== undefined;

    let datasets = [];
    let buyData = data.map(d => d.n_buy || 0);
    let holdData = data.map(d => d.n_hold || 0);
    let sellData = data.map(d => d.n_sell || 0);
    let limitBuyData = hasLimitActions ? data.map(d => d.n_limit_buy || 0) : [];
    let limitSellData = hasLimitActions ? data.map(d => d.n_limit_sell || 0) : [];

    if (isNormalized) {
        const totals = data.map(d => (d.n_buy || 0) + (d.n_hold || 0) + (d.n_sell || 0) + (d.n_limit_buy || 0) + (d.n_limit_sell || 0));
        buyData = buyData.map((v, i) => (totals[i] > 0 ? (v / totals[i]) * 100 : 0));
        holdData = holdData.map((v, i) => (totals[i] > 0 ? (v / totals[i]) * 100 : 0));
        sellData = sellData.map((v, i) => (totals[i] > 0 ? (v / totals[i]) * 100 : 0));
        if (hasLimitActions) {
            limitBuyData = limitBuyData.map((v, i) => (totals[i] > 0 ? (v / totals[i]) * 100 : 0));
            limitSellData = limitSellData.map((v, i) => (totals[i] > 0 ? (v / totals[i]) * 100 : 0));
        }
    }

    datasets.push({ label: 'Buy', data: buyData, backgroundColor: 'rgba(46, 204, 113, 0.5)', borderColor: '#2ecc71', fill: isStacked, pointRadius: 0 });
    datasets.push({ label: 'Hold', data: holdData, backgroundColor: 'rgba(149, 165, 166, 0.5)', borderColor: '#95a5a6', fill: isStacked, pointRadius: 0 });
    datasets.push({ label: 'Sell', data: sellData, backgroundColor: 'rgba(231, 76, 60, 0.5)', borderColor: '#e74c3c', fill: isStacked, pointRadius: 0 });
    
    if (hasLimitActions) {
        datasets.push({ label: 'Limit Buy', data: limitBuyData, backgroundColor: 'rgba(52, 152, 219, 0.5)', borderColor: '#3498db', fill: isStacked, pointRadius: 0 });
        datasets.push({ label: 'Limit Sell', data: limitSellData, backgroundColor: 'rgba(192, 57, 43, 0.5)', borderColor: '#c0392b', fill: isStacked, pointRadius: 0 });
    }

    charts.action.options.scales.y.stacked = isStacked;
    charts.action.options.scales.y.max = isNormalized ? 100 : undefined;
    charts.action.data.labels = labels;
    charts.action.data.datasets = datasets;
    charts.action.update();

    // 4. Speed Chart (with outlier filtering for warmup artifacts)
    const selfPlayTimes = data.map(d => d.self_play_time || 0);
    const trainTimes = data.map(d => d.train_time || 0);
    const totalTimes = data.map(d => (d.self_play_time || 0) + (d.train_time || 0));

    // Filter outliers: replace values >30% deviation from median with null
    const filterOutliers = (arr) => {
        const validVals = arr.filter(v => v > 0);
        if (validVals.length < 5) return arr; // Not enough data to filter

        const sorted = [...validVals].sort((a, b) => a - b);
        const median = sorted[Math.floor(sorted.length / 2)];
        const threshold = median * 0.3; // 30% deviation

        return arr.map(v => {
            if (v === 0) return 0; // Keep zeros (warmup)
            if (Math.abs(v - median) > threshold * 3) return null; // Extreme outlier
            return v;
        });
    };

    charts.speed.data.labels = labels;
    charts.speed.data.datasets[0].data = filterOutliers(selfPlayTimes);
    charts.speed.data.datasets[1].data = filterOutliers(trainTimes);
    charts.speed.data.datasets[2].data = filterOutliers(totalTimes);
    charts.speed.options.spanGaps = true; // Connect lines across null values
    charts.speed.update();

    // 5. Entropy Chart
    if (data[0].policy_entropy !== undefined) {
        charts.entropy.data.labels = labels;
        charts.entropy.data.datasets[0].data = data.map(d => d.policy_entropy);
        charts.entropy.update();
    }

    // 6. Calibration Scatter Chart
    if (data[0].avg_value !== undefined) {
        const scatterData = data.map(d => ({
            x: d.reward,
            y: d.avg_value
        }));

        // Calculate min/max for perfect calibration line
        const allVals = [...data.map(d => d.reward), ...data.map(d => d.avg_value)];
        const minVal = Math.min(...allVals);
        const maxVal = Math.max(...allVals);

        charts.calibration.data.datasets[0].data = scatterData;
        charts.calibration.data.datasets[1].data = [
            { x: minVal, y: minVal },
            { x: maxVal, y: maxVal }
        ];
        charts.calibration.update();
    }

    // 7. Value MSE + Std Chart (only show if data available)
    const mseContainer = document.getElementById('value-mse-container');
    if (data[0].value_mse !== undefined) {
        mseContainer.style.display = 'block';
        charts.valueMse.data.labels = labels;
        charts.valueMse.data.datasets[0].data = data.map(d => d.value_mse || 0);
        charts.valueMse.data.datasets[1].data = data.map(d => d.value_std || 0);
        charts.valueMse.update();
    } else {
        mseContainer.style.display = 'none';
    }

    // 8. Value Calibration Scale + Shift Chart (only show if data available)
    const calibrationContainer = document.getElementById('value-calibration-container');
    if (data[0].value_scale !== undefined || data[0].value_shift !== undefined) {
        calibrationContainer.style.display = 'block';
        charts.valueCalibration.data.labels = labels;
        charts.valueCalibration.data.datasets[0].data = data.map(d => d.value_scale ?? 1.0);
        charts.valueCalibration.data.datasets[1].data = data.map(d => d.value_shift ?? 0.0);
        charts.valueCalibration.update();
    } else {
        calibrationContainer.style.display = 'none';
    }

    // 9. Exploration vs Sells Chart (colored by reward) - uses n_sell like original Python
    if (data[0].policy_entropy !== undefined && data[0].n_sell !== undefined) {
        // Create scatter data with color based on reward
        const scatterData = data.map(d => ({
            x: d.policy_entropy || 0,
            y: d.n_sell || 0,
            reward: d.reward
        }));

        // Calculate colors based on reward (green = positive, red = negative)
        const rewards = data.map(d => d.reward);
        const minReward = Math.min(...rewards);
        const maxReward = Math.max(...rewards);

        const colors = rewards.map(r => {
            // Create color: red (negative) -> yellow (zero) -> green (positive)
            if (r < 0) {
                // Red to yellow
                const intensity = Math.min(1, Math.abs(r / (Math.abs(minReward) || 1)));
                return `rgba(231, ${Math.floor(76 + 180 * (1 - intensity))}, 60, 0.6)`;
            } else {
                // Yellow to green
                const intensity = Math.min(1, r / (maxReward || 1));
                return `rgba(${Math.floor(241 - 195 * intensity)}, ${Math.floor(196 + 8 * intensity)}, ${Math.floor(15 + 98 * intensity)}, 0.6)`;
            }
        });

        charts.explorationSells.data.datasets[0].data = scatterData;
        charts.explorationSells.data.datasets[0].backgroundColor = colors;
        charts.explorationSells.data.datasets[0].borderColor = colors.map(c => c.replace('0.6', '1'));
        charts.explorationSells.update();
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

function movingAverage(data, window) {
    let result = [];
    for (let i = 0; i < data.length; i++) {
        const start = Math.max(0, i - window + 1);
        const slice = data.slice(start, i + 1);
        const sum = slice.reduce((a, b) => a + b, 0);
        result.push(sum / slice.length);
    }
    return result;
}

function calculateCorrelation(x, y) {
    const n = x.length;
    if (n === 0) return 0;
    const sum_x = x.reduce((a, b) => a + b, 0);
    const sum_y = y.reduce((a, b) => a + b, 0);
    const sum_xy = x.reduce((a, b, i) => a + b * y[i], 0);
    const sum_x2 = x.reduce((a, b) => a + b * b, 0);
    const sum_y2 = y.reduce((a, b) => a + b * b, 0);

    const numerator = n * sum_xy - sum_x * sum_y;
    const denominator = Math.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));

    return denominator === 0 ? 0 : numerator / denominator;
}

function formatDate(isoString) {
    const d = new Date(isoString);
    return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
}

function formatSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1024 / 1024).toFixed(1) + ' MB';
}

function refreshData() {
    if (currentRunPath) {
        loadTrainingData(currentRunPath);
        loadAnalysis(currentRunPath);
    }
}

function toggleAutoRefresh() {
    const cb = document.getElementById('auto-refresh');
    if (cb.checked) {
        autoRefreshInterval = setInterval(refreshData, 10000);
    } else {
        clearInterval(autoRefreshInterval);
    }
}

// =============================================================================
// Chart Initialization
// =============================================================================

function initCharts() {
    // Reward Chart
    charts.reward = new Chart(document.getElementById('reward-chart'), {
        type: 'line',
        data: {
            datasets: [{
                label: 'Reward',
                borderColor: 'rgba(52, 152, 219, 0.3)',
                borderWidth: 1,
                pointRadius: 0,
                tension: 0.1
            }, {
                label: 'MA(50)',
                borderColor: '#2980b9',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                annotation: {
                    annotations: {
                        mastery: {
                            type: 'line',
                            yMin: 2.5,
                            yMax: 2.5,
                            borderColor: 'rgb(46, 204, 113)',
                            borderWidth: 2,
                            borderDash: [6, 6],
                            label: { content: 'Mastery', enabled: true }
                        },
                        zero: {
                            type: 'line',
                            yMin: 0,
                            yMax: 0,
                            borderColor: 'rgba(127, 140, 141, 0.5)',
                            borderWidth: 1
                        }
                    }
                }
            }
        }
    });

    // Loss Chart
    charts.loss = new Chart(document.getElementById('loss-chart'), {
        type: 'line',
        data: {
            datasets: [
                { label: 'Total', borderColor: 'black', borderWidth: 2, pointRadius: 0 },
                { label: 'Value', borderColor: '#3498db', borderWidth: 1, pointRadius: 0 },
                { label: 'Policy', borderColor: '#e67e22', borderWidth: 1, pointRadius: 0 },
                { label: 'Reward', borderColor: '#2ecc71', borderWidth: 1, pointRadius: 0 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            scales: { y: { type: 'linear' } }
        }
    });

    // Action Chart (Stacked Area)
    charts.action = new Chart(document.getElementById('action-chart'), {
        type: 'line',
        data: {
            datasets: [
                { label: 'Buy', backgroundColor: 'rgba(46, 204, 113, 0.5)', borderColor: '#2ecc71', fill: true, pointRadius: 0 },
                { label: 'Hold', backgroundColor: 'rgba(149, 165, 166, 0.5)', borderColor: '#95a5a6', fill: true, pointRadius: 0 },
                { label: 'Sell', backgroundColor: 'rgba(231, 76, 60, 0.5)', borderColor: '#e74c3c', fill: true, pointRadius: 0 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            elements: { line: { tension: 0.4 } },
            scales: { y: { stacked: true, beginAtZero: true } }
        }
    });

    // Speed Chart
    charts.speed = new Chart(document.getElementById('speed-chart'), {
        type: 'line',
        data: {
            datasets: [
                { label: 'Self-Play', borderColor: '#9b59b6', borderWidth: 1, pointRadius: 0 },
                { label: 'Train', borderColor: '#f1c40f', borderWidth: 1, pointRadius: 0 },
                { label: 'Total', borderColor: '#34495e', borderWidth: 2, borderDash: [5,5], pointRadius: 0 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            scales: { y: { beginAtZero: true, title: { display: true, text: 'Seconds' } } }
        }
    });

    // Entropy Chart
    charts.entropy = new Chart(document.getElementById('entropy-chart'), {
        type: 'line',
        data: {
            datasets: [{
                label: 'Entropy',
                borderColor: '#8e44ad',
                backgroundColor: 'rgba(142, 68, 173, 0.2)',
                borderWidth: 2,
                fill: true,
                pointRadius: 0,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            scales: { y: { beginAtZero: true } },
            plugins: {
                annotation: {
                    annotations: {
                        high: {
                            type: 'line',
                            yMin: 4.0,
                            yMax: 4.0,
                            borderColor: 'rgba(46, 204, 113, 0.5)',
                            borderDash: [5, 5],
                            label: { content: 'High Exploration', display: false }
                        },
                        balanced: {
                            type: 'line',
                            yMin: 2.5,
                            yMax: 2.5,
                            borderColor: 'rgba(241, 196, 15, 0.5)',
                            borderDash: [5, 5],
                            label: { content: 'Balanced', display: false }
                        },
                        low: {
                            type: 'line',
                            yMin: 1.5,
                            yMax: 1.5,
                            borderColor: 'rgba(231, 76, 60, 0.5)',
                            borderDash: [5, 5],
                            label: { content: 'Deterministic', display: false }
                        }
                    }
                }
            }
        }
    });

    // Value Calibration Scatter Chart
    charts.calibration = new Chart(document.getElementById('calibration-scatter'), {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Pred vs Actual',
                backgroundColor: 'rgba(52, 152, 219, 0.5)',
                borderColor: '#3498db',
                pointRadius: 4
            }, {
                label: 'Perfect Calibration',
                type: 'line',
                borderColor: '#e74c3c',
                borderWidth: 2,
                borderDash: [5, 5],
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { title: { display: true, text: 'Actual Reward (%)' } },
                y: { title: { display: true, text: 'Predicted Value (%)' } }
            }
        }
    });

    // Value MSE + Std Chart
    charts.valueMse = new Chart(document.getElementById('value-mse-chart'), {
        type: 'line',
        data: {
            datasets: [
                {
                    label: 'Value MSE',
                    borderColor: '#e74c3c',
                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    pointRadius: 0,
                    tension: 0.4,
                    yAxisID: 'y'
                },
                {
                    label: 'Value Std',
                    borderColor: '#3498db',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.4,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
                y: {
                    type: 'linear',
                    position: 'left',
                    title: { display: true, text: 'MSE' },
                    beginAtZero: true
                },
                y1: {
                    type: 'linear',
                    position: 'right',
                    title: { display: true, text: 'Std' },
                    beginAtZero: true,
                    grid: { drawOnChartArea: false }
                }
            },
            plugins: {
                annotation: {
                    annotations: {
                        goodMse: {
                            type: 'line',
                            yMin: 5,
                            yMax: 5,
                            borderColor: 'rgba(46, 204, 113, 0.5)',
                            borderDash: [5, 5],
                            label: { content: 'Good MSE', display: false }
                        }
                    }
                }
            }
        }
    });

    // Value Calibration Scale + Shift Chart
    charts.valueCalibration = new Chart(document.getElementById('value-calibration-chart'), {
        type: 'line',
        data: {
            datasets: [
                {
                    label: 'Scale',
                    borderColor: '#9b59b6',
                    backgroundColor: 'rgba(155, 89, 182, 0.1)',
                    borderWidth: 2,
                    fill: false,
                    pointRadius: 0,
                    tension: 0.4,
                    yAxisID: 'y'
                },
                {
                    label: 'Shift',
                    borderColor: '#e67e22',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.4,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
                y: {
                    type: 'linear',
                    position: 'left',
                    title: { display: true, text: 'Scale' },
                    suggestedMin: 0.5,
                    suggestedMax: 1.5
                },
                y1: {
                    type: 'linear',
                    position: 'right',
                    title: { display: true, text: 'Shift' },
                    grid: { drawOnChartArea: false }
                }
            },
            plugins: {
                annotation: {
                    annotations: {
                        defaultScale: {
                            type: 'line',
                            yMin: 1.0,
                            yMax: 1.0,
                            borderColor: 'rgba(46, 204, 113, 0.5)',
                            borderDash: [5, 5],
                            label: { content: 'Default Scale (1.0)', display: false }
                        }
                    }
                }
            }
        }
    });

    // Exploration vs Sells Chart (colored by reward) - matches Python original
    charts.explorationSells = new Chart(document.getElementById('exploration-sells-chart'), {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Entropy vs Sell Count',
                data: [],
                pointRadius: 6,
                pointHoverRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: { display: true, text: 'Policy Entropy (Exploration)' },
                    min: 0,
                    max: 6
                },
                y: {
                    title: { display: true, text: 'Sell Count' },
                    beginAtZero: true
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const point = context.raw;
                            return `Entropy: ${point.x.toFixed(2)}, Sells: ${point.y}, Reward: ${point.reward?.toFixed(2) || 'N/A'}%`;
                        }
                    }
                },
                annotation: {
                    annotations: {
                        highExploration: {
                            type: 'line',
                            xMin: 4.0,
                            xMax: 4.0,
                            borderColor: 'rgba(46, 204, 113, 0.5)',
                            borderDash: [5, 5],
                            label: { content: 'High Exploration', display: false }
                        },
                        balanced: {
                            type: 'line',
                            xMin: 2.5,
                            xMax: 2.5,
                            borderColor: 'rgba(241, 196, 15, 0.5)',
                            borderDash: [5, 5],
                            label: { content: 'Balanced', display: false }
                        },
                        deterministic: {
                            type: 'line',
                            xMin: 1.5,
                            xMax: 1.5,
                            borderColor: 'rgba(231, 76, 60, 0.5)',
                            borderDash: [5, 5],
                            label: { content: 'Deterministic', display: false }
                        }
                    }
                }
            }
        }
    });
}
