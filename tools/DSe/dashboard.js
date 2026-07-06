/**
 * DSe - Easy Training Dashboard
 * JavaScript for training visualization (supports both e-dojo and direct easy training)
 */

// ============================================================================
// State
// ============================================================================

const state = {
    currentTab: 'training',
    liveData: null,
    charts: {},
    checkpoints: null,
    selectedCheckpoint: null,
    autoRefresh: true,
    refreshInterval: null,
    epochFilter: 200,  // 0 = show all
    normalize: false
};

// Colors
const COLORS = {
    easy: '#2ecc71',       // Green
    easyVal: '#27ae60',    // Darker green (val)
    branchA: '#e74c3c',    // Red
    branchAVal: '#c0392b',
    branchB: '#3498db',    // Blue
    branchBVal: '#2980b9',
};

// ============================================================================
// Utilities
// ============================================================================

function formatTime(seconds) {
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
    return `${(seconds / 3600).toFixed(1)}h`;
}

function formatPercent(value) {
    if (value === null || value === undefined) return '--';
    return `${(value * 100).toFixed(1)}%`;
}

function getStatusClass(running) {
    return running ? 'status-running' : 'status-stopped';
}

function filterEpochs(data, limit) {
    if (!data || data.length === 0) return data;
    if (limit === 0) return data;  // 0 = show all
    return data.slice(-limit);
}

function hasData(arr) {
    return arr && arr.length > 0;
}

function normalizeArray(arr) {
    if (!arr || arr.length === 0) return arr;
    const valid = arr.filter(v => v !== null && v !== undefined && !isNaN(v));
    if (valid.length === 0) return arr;
    const min = Math.min(...valid);
    const max = Math.max(...valid);
    const range = max - min;
    if (range < 1e-10) return arr.map(v => (v !== null && v !== undefined) ? 50 : v);
    return arr.map(v => (v !== null && v !== undefined && !isNaN(v)) ? ((v - min) / range) * 100 : v);
}

function maybeNormalize(arr) {
    return state.normalize ? normalizeArray(arr) : arr;
}

// ============================================================================
// API Calls
// ============================================================================

async function fetchLiveData() {
    try {
        const response = await fetch('/api/live');
        return await response.json();
    } catch (error) {
        console.error('Error fetching live data:', error);
        return null;
    }
}

async function fetchCheckpoints() {
    try {
        const response = await fetch('/api/checkpoints');
        return await response.json();
    } catch (error) {
        console.error('Error fetching checkpoints:', error);
        return null;
    }
}

async function fetchCheckpointDetails(path) {
    try {
        const response = await fetch(`/api/checkpoint/${path}`);
        return await response.json();
    } catch (error) {
        console.error('Error fetching checkpoint details:', error);
        return null;
    }
}

async function fetchHeatmap(path, layer) {
    try {
        const response = await fetch(`/api/checkpoint/${path}/heatmap/${encodeURIComponent(layer)}`);
        return await response.json();
    } catch (error) {
        console.error('Error fetching heatmap:', error);
        return null;
    }
}

async function fetchLogs(branch) {
    try {
        const response = await fetch(`/api/logs/${branch}?lines=100`);
        return await response.json();
    } catch (error) {
        console.error('Error fetching logs:', error);
        return null;
    }
}

// ============================================================================
// Status Bar
// ============================================================================

function updateStatusBar(data) {
    if (!data) return;

    const hasEasy = hasData(data.easy?.data);
    const hasBranches = hasData(data.branch_a?.data) || hasData(data.branch_b?.data);

    // Easy training status
    const easyEl = document.getElementById('status-easy');
    if (hasEasy || data.easy?.running) {
        easyEl.style.display = '';
        const eEpoch = data.easy.data?.length || 0;
        const eAgree = data.easy.data?.[eEpoch - 1]?.oracle_agreement;
        const eTrade = data.easy.data?.[eEpoch - 1]?.agree_trade;
        document.getElementById('status-easy-value').textContent =
            `E${eEpoch} | ${formatPercent(eAgree)}` + (eTrade !== undefined ? ` | T:${formatPercent(eTrade)}` : '');
        easyEl.className = `status-item ${getStatusClass(data.easy.running)}`;
    } else {
        easyEl.style.display = 'none';
    }

    // Branch A
    const aEl = document.getElementById('status-a');
    if (hasBranches) {
        aEl.style.display = '';
        const aEpoch = data.branch_a.data?.length || 0;
        const aAgree = data.branch_a.data?.[aEpoch - 1]?.oracle_agreement;
        document.getElementById('status-a-value').textContent =
            `E${aEpoch} | ${formatPercent(aAgree)}`;
        aEl.className = `status-item ${getStatusClass(data.branch_a.running)}`;
    } else {
        aEl.style.display = 'none';
    }

    // Branch B
    const bEl = document.getElementById('status-b');
    if (hasBranches) {
        bEl.style.display = '';
        const bEpoch = data.branch_b.data?.length || 0;
        const bAgree = data.branch_b.data?.[bEpoch - 1]?.oracle_agreement;
        document.getElementById('status-b-value').textContent =
            `E${bEpoch} | ${formatPercent(bAgree)}`;
        bEl.className = `status-item ${getStatusClass(data.branch_b.running)}`;
    } else {
        bEl.style.display = 'none';
    }

    // Merges & Level (only relevant for e-dojo)
    const mergeEl = document.getElementById('status-merge');
    const levelEl = document.getElementById('status-level');
    if (hasBranches) {
        mergeEl.style.display = '';
        levelEl.style.display = '';
        const mergeCount = data.state?.merge_count || 0;
        document.getElementById('status-merge-value').textContent = mergeCount;
        const level = data.state?.current_level || 0;
        document.getElementById('status-level-value').textContent = `L${level}`;
    } else {
        mergeEl.style.display = 'none';
        levelEl.style.display = 'none';
    }

    // Last update
    document.getElementById('last-update').textContent =
        new Date().toLocaleTimeString();
}

// ============================================================================
// Training Tab Charts
// ============================================================================

function createChart(id, config) {
    const ctx = document.getElementById(id)?.getContext('2d');
    if (!ctx) return null;

    if (state.charts[id]) {
        state.charts[id].data = config.data;
        if (config.options) {
            state.charts[id].options = config.options;
        }
        state.charts[id].update('none');
        return state.charts[id];
    }

    state.charts[id] = new Chart(ctx, config);
    return state.charts[id];
}

function updateTrainingCharts(data) {
    if (!data) return;

    const N = state.normalize;
    const nTag = N ? ' (normalized)' : '';

    const rawEasy = data.easy?.data || [];
    const rawA = data.branch_a?.data || [];
    const rawB = data.branch_b?.data || [];

    // Apply epoch filter
    const dataEasy = filterEpochs(rawEasy, state.epochFilter);
    const dataA = filterEpochs(rawA, state.epochFilter);
    const dataB = filterEpochs(rawB, state.epochFilter);
    const merges = data.merges || [];

    const hasEasy = dataEasy.length > 0;
    const hasBranches = dataA.length > 0 || dataB.length > 0;

    // Determine the longest dataset for x-axis labels
    const maxLen = Math.max(dataEasy.length, dataA.length, dataB.length);
    const totalMax = Math.max(rawEasy.length, rawA.length, rawB.length);
    const offset = state.epochFilter > 0 ? Math.max(0, totalMax - state.epochFilter) : 0;
    const labels = Array.from({length: maxLen}, (_, i) => i + offset);

    // --- Agreement chart ---
    const agreeDatasets = [];
    if (hasEasy) {
        agreeDatasets.push({
            label: 'Easy Train',
            data: maybeNormalize(dataEasy.map(d => d.oracle_agreement * 100)),
            borderColor: COLORS.easy,
            backgroundColor: 'rgba(46, 204, 113, 0.1)',
            fill: true,
            tension: 0.1
        });
        agreeDatasets.push({
            label: 'Easy Val',
            data: maybeNormalize(dataEasy.map(d => d.val_oracle_agreement * 100)),
            borderColor: COLORS.easyVal,
            borderDash: [5, 5],
            fill: false,
            tension: 0.1
        });
    }
    if (hasBranches) {
        agreeDatasets.push({
            label: 'Branch A',
            data: maybeNormalize(dataA.map(d => d.oracle_agreement * 100)),
            borderColor: COLORS.branchA,
            backgroundColor: 'rgba(231, 76, 60, 0.1)',
            fill: true,
            tension: 0.1
        });
        agreeDatasets.push({
            label: 'Val A',
            data: maybeNormalize(dataA.map(d => d.val_oracle_agreement * 100)),
            borderColor: COLORS.branchAVal,
            borderDash: [5, 5],
            fill: false,
            tension: 0.1
        });
        agreeDatasets.push({
            label: 'Branch B',
            data: maybeNormalize(dataB.map(d => d.oracle_agreement * 100)),
            borderColor: COLORS.branchB,
            backgroundColor: 'rgba(52, 152, 219, 0.1)',
            fill: true,
            tension: 0.1
        });
        agreeDatasets.push({
            label: 'Val B',
            data: maybeNormalize(dataB.map(d => d.val_oracle_agreement * 100)),
            borderColor: COLORS.branchBVal,
            borderDash: [5, 5],
            fill: false,
            tension: 0.1
        });
    }

    const agreeYOpts = N
        ? { beginAtZero: true, max: 100, title: { display: true, text: 'Agreement %' + nTag } }
        : { beginAtZero: true, max: 100, title: { display: true, text: 'Agreement %' } };

    createChart('chart-agreement', {
        type: 'line',
        data: { labels, datasets: agreeDatasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: agreeYOpts,
                x: { title: { display: true, text: 'Epoch' } }
            },
            plugins: { legend: { position: 'top' } }
        }
    });

    // --- Trade Agreement chart (critical metric) ---
    const tradeDatasets = [];
    if (hasEasy) {
        tradeDatasets.push({
            label: 'Easy Train',
            data: maybeNormalize(dataEasy.map(d => (d.agree_trade || 0) * 100)),
            borderColor: COLORS.easy,
            backgroundColor: 'rgba(46, 204, 113, 0.1)',
            fill: true,
            tension: 0.1
        });
        tradeDatasets.push({
            label: 'Easy Val',
            data: maybeNormalize(dataEasy.map(d => (d.val_agree_trade || 0) * 100)),
            borderColor: COLORS.easyVal,
            borderDash: [5, 5],
            fill: false,
            tension: 0.1
        });
    }
    if (hasBranches) {
        if (dataA.length > 0 && dataA[0].agree_trade !== undefined) {
            tradeDatasets.push({
                label: 'Branch A',
                data: maybeNormalize(dataA.map(d => (d.agree_trade || 0) * 100)),
                borderColor: COLORS.branchA,
                fill: false,
                tension: 0.1
            });
        }
        if (dataB.length > 0 && dataB[0].agree_trade !== undefined) {
            tradeDatasets.push({
                label: 'Branch B',
                data: maybeNormalize(dataB.map(d => (d.agree_trade || 0) * 100)),
                borderColor: COLORS.branchB,
                fill: false,
                tension: 0.1
            });
        }
    }

    createChart('chart-trade-agree', {
        type: 'line',
        data: { labels, datasets: tradeDatasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: true, title: { display: true, text: 'Trade Agreement %' + nTag } },
                x: { title: { display: true, text: 'Epoch' } }
            },
            plugins: { legend: { position: 'top' } }
        }
    });

    // --- Loss chart ---
    const lossDatasets = [];
    if (hasEasy) {
        lossDatasets.push({
            label: 'Easy Total',
            data: maybeNormalize(dataEasy.map(d => d.total_loss)),
            borderColor: COLORS.easy,
            fill: false
        });
        lossDatasets.push({
            label: 'Easy Policy',
            data: maybeNormalize(dataEasy.map(d => d.policy_loss)),
            borderColor: COLORS.easyVal,
            borderDash: [3, 3],
            fill: false
        });
    }
    if (hasBranches) {
        lossDatasets.push({
            label: 'A Total',
            data: maybeNormalize(dataA.map(d => d.total_loss)),
            borderColor: COLORS.branchA,
            fill: false
        });
        lossDatasets.push({
            label: 'B Total',
            data: maybeNormalize(dataB.map(d => d.total_loss)),
            borderColor: COLORS.branchB,
            fill: false
        });
    }

    createChart('chart-loss', {
        type: 'line',
        data: { labels, datasets: lossDatasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { title: { display: true, text: 'Loss' + nTag } },
                x: { title: { display: true, text: 'Epoch' } }
            }
        }
    });

    // --- Validation chart ---
    const valDatasets = [];
    if (hasEasy) {
        valDatasets.push({
            label: 'Easy Val MSE',
            data: maybeNormalize(dataEasy.map(d => d.val_value_mse)),
            borderColor: COLORS.easy,
            fill: false
        });
        valDatasets.push({
            label: 'Easy Val Loss',
            data: maybeNormalize(dataEasy.map(d => d.val_value_loss)),
            borderColor: COLORS.easyVal,
            borderDash: [3, 3],
            fill: false
        });
    }
    if (hasBranches) {
        valDatasets.push({
            label: 'A Val MSE',
            data: maybeNormalize(dataA.map(d => d.val_value_mse)),
            borderColor: COLORS.branchA,
            fill: false
        });
        valDatasets.push({
            label: 'B Val MSE',
            data: maybeNormalize(dataB.map(d => d.val_value_mse)),
            borderColor: COLORS.branchB,
            fill: false
        });
    }

    createChart('chart-validation', {
        type: 'line',
        data: { labels, datasets: valDatasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { title: { display: true, text: 'Value MSE / Loss' + nTag } },
                x: { title: { display: true, text: 'Epoch' } }
            }
        }
    });

    // --- Gradient norm chart ---
    const gradDatasets = [];
    if (hasEasy) {
        gradDatasets.push({
            label: 'Easy',
            data: maybeNormalize(dataEasy.map(d => d.grad_norm)),
            borderColor: COLORS.easy,
            fill: false
        });
    }
    if (hasBranches) {
        gradDatasets.push({
            label: 'Branch A',
            data: maybeNormalize(dataA.map(d => d.grad_norm)),
            borderColor: COLORS.branchA,
            fill: false
        });
        gradDatasets.push({
            label: 'Branch B',
            data: maybeNormalize(dataB.map(d => d.grad_norm)),
            borderColor: COLORS.branchB,
            fill: false
        });
    }

    createChart('chart-gradnorm', {
        type: 'line',
        data: { labels, datasets: gradDatasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { title: { display: true, text: 'Grad Norm' + nTag } },
                x: { title: { display: true, text: 'Epoch' } }
            }
        }
    });

    // --- Epoch time chart ---
    const timeDatasets = [];
    if (hasEasy) {
        timeDatasets.push({
            label: 'Easy',
            data: maybeNormalize(dataEasy.map(d => d.epoch_time_s)),
            borderColor: COLORS.easy,
            fill: false
        });
    }
    if (hasBranches) {
        timeDatasets.push({
            label: 'Branch A',
            data: maybeNormalize(dataA.map(d => d.epoch_time_s)),
            borderColor: COLORS.branchA,
            fill: false
        });
        timeDatasets.push({
            label: 'Branch B',
            data: maybeNormalize(dataB.map(d => d.epoch_time_s)),
            borderColor: COLORS.branchB,
            fill: false
        });
    }

    createChart('chart-time', {
        type: 'line',
        data: { labels, datasets: timeDatasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { title: { display: true, text: 'Seconds' + nTag } },
                x: { title: { display: true, text: 'Epoch' } }
            }
        }
    });
}

// ============================================================================
// Divergence Tab
// ============================================================================

function updateDivergenceTab(data) {
    if (!data?.state) return;

    const cosine = data.state.last_cosine || 0;
    const barrier = data.state.last_barrier || 0;

    // Cosine gauge
    const cosineColor = cosine > 0.95 ? '#2ecc71' : cosine > 0.90 ? '#f39c12' : '#e74c3c';
    Plotly.react('gauge-cosine', [{
        type: 'indicator',
        mode: 'gauge+number',
        value: cosine,
        gauge: {
            axis: { range: [0, 1], tickwidth: 1 },
            bar: { color: cosineColor },
            steps: [
                { range: [0, 0.90], color: 'rgba(231, 76, 60, 0.2)' },
                { range: [0.90, 0.95], color: 'rgba(243, 156, 18, 0.2)' },
                { range: [0.95, 1], color: 'rgba(46, 204, 113, 0.2)' }
            ],
            threshold: {
                line: { color: 'white', width: 2 },
                value: 0.95
            }
        }
    }], {
        margin: { t: 0, b: 0, l: 30, r: 30 },
        height: 150,
        paper_bgcolor: 'transparent',
        font: { color: '#ecf0f1' }
    }, { displayModeBar: false });

    document.getElementById('cosine-label').textContent =
        cosine > 0.95 ? 'Safe' : cosine > 0.90 ? 'Warning' : 'Danger';
    document.getElementById('cosine-label').className =
        `gauge-label ${cosine > 0.95 ? 'safe' : cosine > 0.90 ? 'warning' : 'danger'}`;

    // Barrier gauge
    const barrierColor = barrier < 0.02 ? '#2ecc71' : barrier < 0.05 ? '#f39c12' : '#e74c3c';
    Plotly.react('gauge-barrier', [{
        type: 'indicator',
        mode: 'gauge+number',
        value: barrier,
        number: { valueformat: '.4f' },
        gauge: {
            axis: { range: [0, 0.1], tickwidth: 1 },
            bar: { color: barrierColor },
            steps: [
                { range: [0, 0.02], color: 'rgba(46, 204, 113, 0.2)' },
                { range: [0.02, 0.05], color: 'rgba(243, 156, 18, 0.2)' },
                { range: [0.05, 0.1], color: 'rgba(231, 76, 60, 0.2)' }
            ],
            threshold: {
                line: { color: 'white', width: 2 },
                value: 0.02
            }
        }
    }], {
        margin: { t: 0, b: 0, l: 30, r: 30 },
        height: 150,
        paper_bgcolor: 'transparent',
        font: { color: '#ecf0f1' }
    }, { displayModeBar: false });

    document.getElementById('barrier-label').textContent =
        barrier < 0.02 ? 'Safe' : barrier < 0.05 ? 'Warning' : 'Danger';
    document.getElementById('barrier-label').className =
        `gauge-label ${barrier < 0.02 ? 'safe' : barrier < 0.05 ? 'warning' : 'danger'}`;

    // Merge timeline
    const merges = data.merges || [];
    const timelineHtml = merges.map((m, i) => `
        <div class="merge-event ${m.method}" title="${m.reason || 'merge'}">
            <span class="merge-num">${i + 1}</span>
            <span class="merge-method">${m.method}</span>
            <span class="merge-cosine">${(m.cosine || 0).toFixed(3)}</span>
        </div>
    `).join('');
    document.getElementById('merge-timeline').innerHTML =
        timelineHtml || '<p class="placeholder">No merges yet</p>';

    // Effectiveness chart
    if (merges.length > 0) {
        createChart('chart-effectiveness', {
            type: 'bar',
            data: {
                labels: merges.map((_, i) => `Merge ${i + 1}`),
                datasets: [
                    {
                        label: 'Agreement A Before',
                        data: merges.map(m => (m.agreement_a || 0) * 100),
                        backgroundColor: 'rgba(231, 76, 60, 0.5)'
                    },
                    {
                        label: 'Agreement B Before',
                        data: merges.map(m => (m.agreement_b || 0) * 100),
                        backgroundColor: 'rgba(52, 152, 219, 0.5)'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Agreement %' }
                    }
                }
            }
        });
    }

    // Merge decision log
    const logHtml = merges.slice().reverse().map(m => `
        <div class="log-entry">
            <span class="log-time">${m.timestamp || '--'}</span>
            <span class="log-reason ${m.reason?.includes('emergency') ? 'emergency' : ''}">${m.reason || 'unknown'}</span>
            <span class="log-detail">cos=${(m.cosine || 0).toFixed(3)}, barrier=${(m.barrier || 0).toFixed(4)}</span>
        </div>
    `).join('');
    document.getElementById('merge-log').innerHTML =
        logHtml || '<p class="placeholder">No merge decisions recorded</p>';
}

// ============================================================================
// Checkpoints Tab
// ============================================================================

function updateCheckpointsTab(checkpoints) {
    if (!checkpoints) return;

    const allCheckpoints = [
        ...checkpoints.branch_a.map(c => ({ ...c, source: 'branch_a' })),
        ...checkpoints.branch_b.map(c => ({ ...c, source: 'branch_b' })),
        ...checkpoints.merged.map(c => ({ ...c, source: 'merged' }))
    ].sort((a, b) => new Date(b.modified) - new Date(a.modified));

    state.checkpoints = allCheckpoints;
    renderCheckpointList(allCheckpoints);
}

function renderCheckpointList(checkpoints, filter = 'all') {
    const filtered = filter === 'all'
        ? checkpoints
        : checkpoints.filter(c => c.source === filter);

    const html = filtered.map(c => `
        <div class="checkpoint-item" data-path="${c.path}" data-source="${c.source}">
            <span class="ckpt-name">${c.name}</span>
            <span class="ckpt-source ${c.source}">${c.source}</span>
            <span class="ckpt-size">${c.size_mb.toFixed(1)} MB</span>
            <span class="ckpt-time">${new Date(c.modified).toLocaleString()}</span>
        </div>
    `).join('');

    document.getElementById('checkpoint-list').innerHTML =
        html || '<p class="placeholder">No checkpoints found</p>';

    // Add click handlers
    document.querySelectorAll('.checkpoint-item').forEach(el => {
        el.addEventListener('click', () => loadCheckpointDetails(el.dataset.path));
    });
}

async function loadCheckpointDetails(path) {
    state.selectedCheckpoint = path;
    document.getElementById('checkpoint-info').innerHTML = '<p>Loading...</p>';

    const details = await fetchCheckpointDetails(path);
    if (!details || details.error) {
        document.getElementById('checkpoint-info').innerHTML =
            `<p class="error">${details?.error || 'Failed to load'}</p>`;
        return;
    }

    // Update info panel
    const m = details.metrics || {};
    let metricsHtml = `
        <div class="ckpt-detail">
            <span>Epoch:</span><span>${details.epoch}</span>
        </div>
        <div class="ckpt-detail">
            <span>Total Params:</span><span>${(details.total_params / 1e6).toFixed(2)}M</span>
        </div>
    `;

    if (m.oracle_agreement !== undefined) {
        metricsHtml += `<div class="ckpt-detail"><span>Oracle Agreement:</span><span>${formatPercent(m.oracle_agreement)}</span></div>`;
    }
    if (m.val_oracle_agreement !== undefined) {
        metricsHtml += `<div class="ckpt-detail"><span>Val Agreement:</span><span>${formatPercent(m.val_oracle_agreement)}</span></div>`;
    }
    if (m.source === 'merged') {
        metricsHtml += `<div class="ckpt-detail"><span>Source:</span><span style="color:#f39c12">MERGED</span></div>`;
        if (m.method) metricsHtml += `<div class="ckpt-detail"><span>Merge Method:</span><span>${m.method}</span></div>`;
        if (m.merge_count !== undefined) metricsHtml += `<div class="ckpt-detail"><span>Merge #:</span><span>${m.merge_count}</span></div>`;
        if (m.a_agreement !== undefined) metricsHtml += `<div class="ckpt-detail"><span>A Agreement:</span><span>${formatPercent(m.a_agreement)}</span></div>`;
        if (m.b_agreement !== undefined) metricsHtml += `<div class="ckpt-detail"><span>B Agreement:</span><span>${formatPercent(m.b_agreement)}</span></div>`;
    }
    if (m.total_loss !== undefined) metricsHtml += `<div class="ckpt-detail"><span>Total Loss:</span><span>${m.total_loss.toFixed(4)}</span></div>`;
    if (m.policy_loss !== undefined) metricsHtml += `<div class="ckpt-detail"><span>Policy Loss:</span><span>${m.policy_loss.toFixed(4)}</span></div>`;
    if (m.value_loss !== undefined) metricsHtml += `<div class="ckpt-detail"><span>Value Loss:</span><span>${m.value_loss.toFixed(4)}</span></div>`;
    if (m.level !== undefined) metricsHtml += `<div class="ckpt-detail"><span>Curriculum Level:</span><span>L${m.level}</span></div>`;

    document.getElementById('checkpoint-info').innerHTML = metricsHtml;

    // Layer stats table
    const statsHtml = details.layer_stats.map(s => `
        <tr>
            <td title="${s.name}">${s.name.split('/').slice(-2).join('/')}</td>
            <td>${s.shape.join('x')}</td>
            <td>${s.mean.toFixed(4)}</td>
            <td>${s.std.toFixed(4)}</td>
            <td>${(s.sparsity * 100).toFixed(1)}%</td>
        </tr>
    `).join('');

    document.getElementById('layer-stats').innerHTML = `
        <h4 style="color: var(--text-secondary); margin-bottom: 0.5rem;">${details.layer_stats.length} Layers</h4>
        <table class="stats-table">
            <thead>
                <tr><th>Layer</th><th>Shape</th><th>Mean</th><th>Std</th><th>Sparsity</th></tr>
            </thead>
            <tbody>${statsHtml}</tbody>
        </table>
    `;

    // Layer select for heatmap
    const selectHtml = details.layer_stats
        .filter(s => s.size > 100)
        .map(s => `<option value="${s.name}">${s.name} (${s.shape.join('x')})</option>`)
        .join('');
    document.getElementById('layer-select').innerHTML =
        '<option value="">Select a layer...</option>' + selectHtml;
}

async function loadHeatmap(layer) {
    if (!state.selectedCheckpoint || !layer) return;

    const heatmap = await fetchHeatmap(state.selectedCheckpoint, layer);
    if (!heatmap || heatmap.error) {
        document.getElementById('heatmap-container').innerHTML =
            `<p class="error">${heatmap?.error || 'Failed to load heatmap'}</p>`;
        return;
    }

    Plotly.react('heatmap-container', [{
        z: heatmap.data,
        type: 'heatmap',
        colorscale: 'RdBu',
        zmid: 0
    }], {
        title: `${layer} (${heatmap.shape.join('x')})`,
        margin: { t: 40, b: 40, l: 40, r: 40 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: '#ecf0f1' }
    });
}

// ============================================================================
// Logs Tab
// ============================================================================

async function updateLogsTab() {
    const [logsA, logsB] = await Promise.all([
        fetchLogs('branch_A_bella'),
        fetchLogs('branch_B_synth')
    ]);

    const formatLine = line => {
        if (line.includes('ERROR') || line.includes('error')) {
            return `<span class="log-error">${escapeHtml(line)}</span>`;
        }
        if (line.includes('WARNING') || line.includes('warning')) {
            return `<span class="log-warning">${escapeHtml(line)}</span>`;
        }
        if (line.includes('Epoch') || line.includes('epoch')) {
            return `<span class="log-epoch">${escapeHtml(line)}</span>`;
        }
        return escapeHtml(line);
    };

    const escapeHtml = text => text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

    const logAEl = document.getElementById('log-a');
    const logBEl = document.getElementById('log-b');

    if (logsA?.lines) {
        logAEl.innerHTML = logsA.lines.map(formatLine).join('\n');
        if (document.getElementById('log-a-follow').checked) {
            logAEl.scrollTop = logAEl.scrollHeight;
        }
    }

    if (logsB?.lines) {
        logBEl.innerHTML = logsB.lines.map(formatLine).join('\n');
        if (document.getElementById('log-b-follow').checked) {
            logBEl.scrollTop = logBEl.scrollHeight;
        }
    }
}

// ============================================================================
// Tab Handling
// ============================================================================

function switchTab(tabId) {
    state.currentTab = tabId;

    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.tab === tabId);
    });

    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.toggle('active', content.id === `tab-${tabId}`);
    });

    if (tabId === 'checkpoints' && !state.checkpoints) {
        fetchCheckpoints().then(updateCheckpointsTab);
    }
    if (tabId === 'logs') {
        updateLogsTab();
    }
}

// ============================================================================
// Polling & Initialization
// ============================================================================

async function refresh() {
    const data = await fetchLiveData();
    if (data) {
        state.liveData = data;
        updateStatusBar(data);

        if (state.currentTab === 'training') {
            updateTrainingCharts(data);
        } else if (state.currentTab === 'divergence') {
            updateDivergenceTab(data);
        }
    }

    if (state.currentTab === 'logs') {
        updateLogsTab();
    }
}

function startPolling() {
    if (state.refreshInterval) {
        clearInterval(state.refreshInterval);
    }
    state.refreshInterval = setInterval(() => {
        if (state.autoRefresh) {
            refresh();
        }
    }, 10000);
}

function init() {
    // Tab click handlers
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => switchTab(tab.dataset.tab));
    });

    // Auto-refresh toggle
    document.getElementById('auto-refresh').addEventListener('change', e => {
        state.autoRefresh = e.target.checked;
    });

    // Normalize toggle
    document.getElementById('normalize-toggle').addEventListener('change', e => {
        state.normalize = e.target.checked;
        if (state.liveData) {
            updateTrainingCharts(state.liveData);
        }
    });

    // Epoch filter buttons
    document.querySelectorAll('.epoch-btn[data-epochs]').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.epoch-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.epochFilter = parseInt(btn.dataset.epochs, 10);
            if (state.liveData) {
                updateTrainingCharts(state.liveData);
            }
        });
    });

    // Custom epoch filter
    document.getElementById('custom-epochs-btn').addEventListener('click', () => {
        const input = document.getElementById('custom-epochs');
        const value = parseInt(input.value, 10);
        if (value && value >= 10) {
            document.querySelectorAll('.epoch-btn').forEach(b => b.classList.remove('active'));
            state.epochFilter = value;
            if (state.liveData) {
                updateTrainingCharts(state.liveData);
            }
        }
    });

    // Enter key on custom input
    document.getElementById('custom-epochs').addEventListener('keypress', e => {
        if (e.key === 'Enter') {
            document.getElementById('custom-epochs-btn').click();
        }
    });

    // Checkpoint filter buttons
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            if (state.checkpoints) {
                renderCheckpointList(state.checkpoints, btn.dataset.filter);
            }
        });
    });

    // Layer select for heatmap
    document.getElementById('layer-select').addEventListener('change', e => {
        if (e.target.value) {
            loadHeatmap(e.target.value);
        }
    });

    // Visibility change - refresh when tab becomes visible
    document.addEventListener('visibilitychange', () => {
        if (!document.hidden && state.autoRefresh) {
            refresh();
        }
    });

    // Initial load
    refresh();
    startPolling();
}

// Start
document.addEventListener('DOMContentLoaded', init);
