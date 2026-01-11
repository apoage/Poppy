/**
 * DS24 - Enhanced Training Metrics Dashboard
 *
 * Visualization for TRM+MuZero Trading Agent with:
 * - MuZero metrics (value/policy/reward)
 * - TRM recursive reasoning dynamics
 * - Neural ODE physics components
 * - Multi-component interaction analysis
 */

// =============================================================================
// Global State
// =============================================================================

const state = {
    currentRun: null,
    trainingData: null,
    filteredData: null,  // Filtered view of trainingData
    analysisData: null,
    autoRefresh: false,
    refreshInterval: null,
    charts: {},

    // Epoch filter state
    filter: {
        type: 'all',      // all, last50, last100, last250, skip5, skip10, custom
        from: 0,
        to: null          // null means end
    }
};

// Chart.js defaults
Chart.defaults.color = '#e0e0e0';
Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.1)';
Chart.defaults.font.family = "'JetBrains Mono', 'Fira Code', monospace";

// Color palette
const COLORS = {
    primary: '#00d4aa',
    secondary: '#7c4dff',
    warning: '#ffab00',
    danger: '#ff5252',
    info: '#448aff',
    success: '#69f0ae',
    muted: '#607d8b',

    // Chart colors
    reward: '#00d4aa',
    rewardMA: '#ffffff',
    valueLoss: '#ff6384',
    policyLoss: '#36a2eb',
    rewardLoss: '#ffce56',
    predLoss: '#9966ff',
    physicsReg: '#ff9f40',

    // Action colors
    buy: '#00e676',
    hold: '#78909c',
    sell: '#ff5252',
    limitBuy: '#00bfa5',
    limitSell: '#ff8a80',

    // Market baseline colors
    alphaPositive: '#00e676',  // Green for beating market
    alphaNegative: '#ff5252',  // Red for underperforming
    alphaZero: '#78909c',      // Gray for neutral
    marketReturn: '#448aff',   // Blue for market
    agentReturn: '#00d4aa',    // Teal for agent

    // Physics colors
    momentum: '#ff6384',
    reversion: '#36a2eb',
    volatility: '#ffce56',
    correlation: '#9966ff',

    // TRM cycle colors
    cycle0: '#ff6384',
    cycle1: '#36a2eb',
    cycle2: '#ffce56',
    cycle3: '#4bc0c0'
};

// =============================================================================
// Initialization
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initCharts();
    initEpochFilter();
    loadRuns();
    loadStatus();

    // Event listeners
    document.getElementById('run-selector').addEventListener('change', onRunSelect);
    document.getElementById('auto-refresh-toggle').addEventListener('change', onAutoRefreshToggle);
    document.getElementById('log-scale-toggle').addEventListener('change', onLogScaleToggle);
    document.getElementById('refresh-checkpoints').addEventListener('click', loadCheckpoints);
    document.getElementById('refresh-log').addEventListener('click', refreshLog);

    // Status polling
    setInterval(loadStatus, 30000);

    // Refresh immediately when tab regains focus (browsers throttle setInterval in background)
    document.addEventListener('visibilitychange', () => {
        if (!document.hidden && state.autoRefresh && state.currentRun) {
            console.log('Tab visible - refreshing data');
            loadTrainingData(state.currentRun);
        }
    });

    // Also refresh on window focus (catches alt-tab back)
    window.addEventListener('focus', () => {
        if (state.autoRefresh && state.currentRun) {
            loadTrainingData(state.currentRun);
        }
    });
});

// =============================================================================
// Tab Navigation
// =============================================================================

function initTabs() {
    const tabs = document.querySelectorAll('.tab');
    const panels = document.querySelectorAll('.tab-panel');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const targetId = tab.dataset.tab;

            tabs.forEach(t => t.classList.remove('active'));
            panels.forEach(p => p.classList.remove('active'));

            tab.classList.add('active');
            document.getElementById(`${targetId}-tab`).classList.add('active');

            // Load tab-specific data if needed
            if (targetId === 'checkpoints') loadCheckpoints();
            if (targetId === 'logs') loadLogFiles();
        });
    });
}

// =============================================================================
// Epoch Filter
// =============================================================================

function initEpochFilter() {
    // Preset buttons
    document.querySelectorAll('.filter-preset').forEach(btn => {
        btn.addEventListener('click', () => onFilterPresetClick(btn));
    });

    // Custom range apply button
    document.getElementById('apply-filter').addEventListener('click', onApplyCustomFilter);

    // Enter key in custom inputs
    document.getElementById('filter-from').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') onApplyCustomFilter();
    });
    document.getElementById('filter-to').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') onApplyCustomFilter();
    });

    // Load saved filter from localStorage
    loadFilterState();
}

function onFilterPresetClick(btn) {
    const filterType = btn.dataset.filter;

    // Update active state
    document.querySelectorAll('.filter-preset').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    // Show/hide custom inputs
    const customPanel = document.getElementById('filter-custom');
    if (filterType === 'custom') {
        customPanel.style.display = 'flex';
        return; // Don't apply yet, wait for user input
    } else {
        customPanel.style.display = 'none';
    }

    // Apply preset filter
    state.filter.type = filterType;
    applyFilter();
    saveFilterState();
}

function onApplyCustomFilter() {
    const fromVal = document.getElementById('filter-from').value;
    const toVal = document.getElementById('filter-to').value;

    state.filter.type = 'custom';
    state.filter.from = fromVal ? parseInt(fromVal) : 0;
    state.filter.to = toVal ? parseInt(toVal) : null;

    applyFilter();
    saveFilterState();
}

function applyFilter() {
    if (!state.trainingData) return;

    const data = state.trainingData;
    const totalEpochs = data.epoch ? data.epoch.length : (data.reward ? data.reward.length : 0);

    if (totalEpochs === 0) {
        state.filteredData = data;
        updateFilterDisplay(0, 0, 0);
        return;
    }

    let fromIdx = 0;
    let toIdx = totalEpochs;

    // Calculate indices based on filter type
    switch (state.filter.type) {
        case 'all':
            fromIdx = 0;
            toIdx = totalEpochs;
            break;
        case 'last50':
            fromIdx = Math.max(0, totalEpochs - 50);
            toIdx = totalEpochs;
            break;
        case 'last100':
            fromIdx = Math.max(0, totalEpochs - 100);
            toIdx = totalEpochs;
            break;
        case 'last250':
            fromIdx = Math.max(0, totalEpochs - 250);
            toIdx = totalEpochs;
            break;
        case 'skip5':
            fromIdx = Math.min(5, totalEpochs);
            toIdx = totalEpochs;
            break;
        case 'skip10':
            fromIdx = Math.min(10, totalEpochs);
            toIdx = totalEpochs;
            break;
        case 'custom':
            fromIdx = Math.min(state.filter.from || 0, totalEpochs);
            toIdx = state.filter.to ? Math.min(state.filter.to, totalEpochs) : totalEpochs;
            break;
    }

    // Ensure valid range
    fromIdx = Math.max(0, fromIdx);
    toIdx = Math.max(fromIdx, toIdx);

    // Slice all arrays in the data
    state.filteredData = {};
    for (const [key, arr] of Object.entries(data)) {
        if (Array.isArray(arr)) {
            state.filteredData[key] = arr.slice(fromIdx, toIdx);
        } else {
            state.filteredData[key] = arr;
        }
    }

    updateFilterDisplay(fromIdx, toIdx, totalEpochs);

    // Refresh all visualizations with filtered data
    updateMetricsCharts();
    updateStats();
    updateTrainingConfig();
    updateReasoningTab();
    updatePhysicsTab();
    updatePlanningTab();
    updateInteractionsTab();
    updateRedoATC();
}

function updateFilterDisplay(fromIdx, toIdx, total) {
    const count = toIdx - fromIdx;
    const statusEl = document.getElementById('filter-status');
    const rangeEl = document.getElementById('epoch-range-display');

    if (state.filter.type === 'all') {
        statusEl.textContent = `Showing all ${total} epochs`;
        rangeEl.textContent = '';
    } else {
        statusEl.textContent = `Showing ${count} of ${total} epochs`;
        rangeEl.textContent = `[${fromIdx} - ${toIdx - 1}]`;
    }
}

function saveFilterState() {
    try {
        localStorage.setItem('ds24_filter', JSON.stringify(state.filter));
    } catch (e) {
        // localStorage not available
    }
}

function loadFilterState() {
    try {
        const saved = localStorage.getItem('ds24_filter');
        if (saved) {
            const parsed = JSON.parse(saved);
            state.filter = { ...state.filter, ...parsed };

            // Update UI to match saved state
            document.querySelectorAll('.filter-preset').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.filter === state.filter.type);
            });

            if (state.filter.type === 'custom') {
                document.getElementById('filter-custom').style.display = 'flex';
                document.getElementById('filter-from').value = state.filter.from || 0;
                document.getElementById('filter-to').value = state.filter.to || '';
            }
        }
    } catch (e) {
        // localStorage not available
    }
}

// =============================================================================
// Chart Initialization
// =============================================================================

function initCharts() {
    // === Metrics Tab Charts ===
    state.charts.reward = createLineChart('reward-chart', {
        datasets: [
            { label: 'Reward', borderColor: COLORS.reward, fill: false },
            { label: 'MA(50)', borderColor: COLORS.rewardMA, borderDash: [5, 5], fill: false }
        ]
    });

    // Market Baseline (Alpha) chart
    state.charts.alpha = createLineChart('alpha-chart', {
        datasets: [
            { label: 'Alpha (%)', borderColor: COLORS.alphaPositive, fill: false },
            { label: 'Alpha MA30', borderColor: '#ffffff', borderWidth: 3, fill: false },
            { label: 'Market Return', borderColor: COLORS.marketReturn, borderDash: [3, 3], fill: false },
            { label: 'Agent Return', borderColor: COLORS.agentReturn, borderDash: [3, 3], fill: false }
        ]
    });

    state.charts.loss = createLineChart('loss-chart', {
        datasets: [
            { label: 'Total Loss', borderColor: '#ffffff', borderWidth: 3, fill: false },
            { label: 'Value Loss', borderColor: COLORS.valueLoss, fill: false },
            { label: 'Policy Loss', borderColor: COLORS.policyLoss, fill: false },
            { label: 'Reward Loss', borderColor: COLORS.rewardLoss, fill: false }
        ]
    });

    state.charts.action = createStackedAreaChart('action-chart', {
        datasets: [
            { label: 'Buy', backgroundColor: COLORS.buy },
            { label: 'Hold', backgroundColor: COLORS.hold },
            { label: 'Sell', backgroundColor: COLORS.sell },
            { label: 'Limit Buy', backgroundColor: COLORS.limitBuy },
            { label: 'Limit Sell', backgroundColor: COLORS.limitSell }
        ]
    });

    state.charts.valueCal = createDualAxisChart('value-cal-chart',
        { label: 'Value MSE', borderColor: COLORS.valueLoss },
        { label: 'Value Std', borderColor: COLORS.policyLoss }
    );

    state.charts.entropy = createAreaChart('entropy-chart', {
        label: 'Policy Entropy',
        borderColor: COLORS.info,
        backgroundColor: 'rgba(68, 138, 255, 0.2)'
    });

    state.charts.speed = createStackedAreaChart('speed-chart', {
        datasets: [
            { label: 'Self-Play', backgroundColor: 'rgba(0, 212, 170, 0.6)' },
            { label: 'Training', backgroundColor: 'rgba(124, 77, 255, 0.6)' }
        ]
    });

    // === Reasoning Tab Charts ===
    state.charts.crystallization = createLineChart('crystallization-chart', {
        datasets: [
            { label: 'H0', borderColor: COLORS.cycle0, fill: false },
            { label: 'H1', borderColor: COLORS.cycle1, fill: false },
            { label: 'H2', borderColor: COLORS.cycle2, fill: false },
            { label: 'H3', borderColor: COLORS.cycle3, fill: false }
        ]
    });

    state.charts.latentNorm = createAreaChart('latent-norm-chart', {
        label: 'Latent Final Norm',
        borderColor: COLORS.primary,
        backgroundColor: 'rgba(0, 212, 170, 0.2)'
    });

    state.charts.cycleContribution = createStackedAreaChart('cycle-contribution-chart', {
        datasets: [
            { label: 'Cycle 0', backgroundColor: 'rgba(255, 99, 132, 0.6)' },
            { label: 'Cycle 1', backgroundColor: 'rgba(54, 162, 235, 0.6)' },
            { label: 'Cycle 2', backgroundColor: 'rgba(255, 206, 86, 0.6)' },
            { label: 'Cycle 3', backgroundColor: 'rgba(75, 192, 192, 0.6)' }
        ]
    });

    state.charts.trmUtilization = createAreaChart('trm-utilization-chart', {
        label: 'Utilization %',
        borderColor: COLORS.success,
        backgroundColor: 'rgba(105, 240, 174, 0.2)'
    });

    // === Physics Tab Charts ===
    // Use line chart with log scale for physics components (values range 1e-7 to 0.1)
    state.charts.physicsComponents = createLineChart('physics-components-chart', {
        datasets: [
            { label: 'G_momentum', borderColor: 'rgba(255, 99, 132, 1)', fill: false },
            { label: 'G_reversion', borderColor: 'rgba(54, 162, 235, 1)', fill: false },
            { label: 'G_volatility', borderColor: 'rgba(255, 206, 86, 1)', fill: false },
            { label: 'G_correlation', borderColor: 'rgba(153, 102, 255, 1)', fill: false }
        ],
        options: {
            scales: {
                y: {
                    type: 'logarithmic',
                    min: 1e-8,
                    title: { display: true, text: 'Magnitude (log scale)' }
                }
            }
        }
    });

    state.charts.physicsParams = createLineChart('physics-params-chart', {
        datasets: [
            { label: 'α (momentum)', borderColor: COLORS.momentum, fill: false },
            { label: 'β (reversion)', borderColor: COLORS.reversion, fill: false }
        ]
    });

    state.charts.garch = createLineChart('garch-chart', {
        datasets: [
            { label: 'Persistence (α+β)', borderColor: COLORS.warning, fill: false }
        ],
        options: {
            scales: {
                y: {
                    min: 0,
                    max: 1.1,
                    ticks: { stepSize: 0.2 }
                }
            },
            plugins: {
                annotation: {
                    annotations: {
                        threshold: {
                            type: 'line',
                            yMin: 1.0,
                            yMax: 1.0,
                            borderColor: COLORS.danger,
                            borderWidth: 2,
                            borderDash: [5, 5],
                            label: { content: 'Non-stationary' }
                        }
                    }
                }
            }
        }
    });

    // === Planning Tab Charts ===
    state.charts.horizon = createLineChart('horizon-chart', {
        datasets: [
            { label: 'Horizon 1', borderColor: COLORS.success, fill: false },
            { label: 'Horizon 5', borderColor: COLORS.warning, fill: false },
            { label: 'Horizon 10', borderColor: COLORS.danger, fill: false }
        ]
    });

    state.charts.mctsCalibration = createScatterChart('mcts-calibration-chart', {
        label: 'Predicted vs Actual',
        borderColor: COLORS.primary,
        backgroundColor: 'rgba(0, 212, 170, 0.5)'
    });

    state.charts.visitEntropy = createAreaChart('visit-entropy-chart', {
        label: 'Visit Entropy',
        borderColor: COLORS.secondary,
        backgroundColor: 'rgba(124, 77, 255, 0.2)'
    });

    state.charts.simROI = createLineChart('sim-roi-chart', {
        datasets: [
            { label: 'Reward per Sim', borderColor: COLORS.primary, fill: false }
        ]
    });

    // === Interactions Tab Charts ===
    state.charts.lossContribution = createStackedAreaChart('loss-contribution-chart', {
        datasets: [
            { label: 'Value', backgroundColor: 'rgba(255, 99, 132, 0.7)' },
            { label: 'Policy', backgroundColor: 'rgba(54, 162, 235, 0.7)' },
            { label: 'Reward', backgroundColor: 'rgba(255, 206, 86, 0.7)' },
            { label: 'Prediction', backgroundColor: 'rgba(153, 102, 255, 0.7)' },
            { label: 'Physics', backgroundColor: 'rgba(255, 159, 64, 0.7)' }
        ]
    });

    state.charts.gradientNorms = createLineChart('gradient-norms-chart', {
        datasets: [
            { label: 'Value Head', borderColor: COLORS.valueLoss, fill: false },
            { label: 'Policy Head', borderColor: COLORS.policyLoss, fill: false },
            { label: 'Dynamics', borderColor: COLORS.predLoss, fill: false }
        ]
    });

    state.charts.alignment = createDualAxisChart('alignment-chart',
        { label: 'Latent Norm', borderColor: COLORS.primary },
        { label: 'Physics Reg', borderColor: COLORS.secondary }
    );

    // === ReDo and ATC Charts (TASK_042, TASK_043) ===
    state.charts.redo = createLineChart('redo-chart', {
        datasets: [
            { label: 'Dormant %', borderColor: COLORS.warning, fill: false },
            { label: 'Recycled', borderColor: COLORS.success, fill: false }
        ]
    });

    state.charts.atc = createLineChart('atc-chart', {
        datasets: [
            { label: 'ATC Loss', borderColor: COLORS.secondary, fill: false },
            { label: 'Pos Sim', borderColor: COLORS.success, fill: false },
            { label: 'Neg Sim', borderColor: COLORS.danger, fill: false }
        ]
    });
}

// =============================================================================
// Chart Factory Functions
// =============================================================================

function createLineChart(canvasId, config) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    // Deep merge scales with custom options
    const customScales = config.options?.scales || {};
    const defaultScales = {
        x: { grid: { color: 'rgba(255,255,255,0.05)' } },
        y: { grid: { color: 'rgba(255,255,255,0.05)' } }
    };
    const mergedScales = {
        x: { ...defaultScales.x, ...customScales.x },
        y: { ...defaultScales.y, ...customScales.y }
    };

    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: config.datasets.map(d => ({
                ...d,
                tension: 0.1,
                pointRadius: 0,
                borderWidth: 2,
                spanGaps: true  // Connect lines across null/missing values
            }))
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { intersect: false, mode: 'index' },
            plugins: { legend: { position: 'top' } },
            scales: mergedScales,
            ...(config.options ? { ...config.options, scales: undefined } : {})
        }
    });
}

function createAreaChart(canvasId, config) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: config.label,
                borderColor: config.borderColor,
                backgroundColor: config.backgroundColor,
                fill: true,
                tension: 0.1,
                pointRadius: 0,
                borderWidth: 2,
                spanGaps: true  // Connect lines across null/missing values
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { intersect: false, mode: 'index' },
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.05)' } },
                y: { grid: { color: 'rgba(255,255,255,0.05)' } }
            }
        }
    });
}

function createStackedAreaChart(canvasId, config) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: config.datasets.map(d => ({
                ...d,
                fill: true,
                tension: 0.1,
                pointRadius: 0,
                borderWidth: 0,
                spanGaps: true  // Connect lines across null/missing values
            }))
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { intersect: false, mode: 'index' },
            plugins: { legend: { position: 'top' } },
            scales: {
                x: { stacked: true, grid: { color: 'rgba(255,255,255,0.05)' } },
                y: { stacked: true, grid: { color: 'rgba(255,255,255,0.05)' } }
            }
        }
    });
}

function createDualAxisChart(canvasId, leftConfig, rightConfig) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    ...leftConfig,
                    yAxisID: 'y',
                    fill: false,
                    tension: 0.1,
                    pointRadius: 0,
                    borderWidth: 2,
                    spanGaps: true  // Connect lines across null/missing values
                },
                {
                    ...rightConfig,
                    yAxisID: 'y1',
                    fill: false,
                    tension: 0.1,
                    pointRadius: 0,
                    borderWidth: 2,
                    spanGaps: true  // Connect lines across null/missing values
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { intersect: false, mode: 'index' },
            plugins: { legend: { position: 'top' } },
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.05)' } },
                y: {
                    type: 'linear',
                    position: 'left',
                    grid: { color: 'rgba(255,255,255,0.05)' }
                },
                y1: {
                    type: 'linear',
                    position: 'right',
                    grid: { drawOnChartArea: false }
                }
            }
        }
    });
}

function createScatterChart(canvasId, config) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    return new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: config.label,
                borderColor: config.borderColor,
                backgroundColor: config.backgroundColor,
                pointRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { title: { display: true, text: 'Predicted' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                y: { title: { display: true, text: 'Actual' }, grid: { color: 'rgba(255,255,255,0.05)' } }
            }
        }
    });
}

// =============================================================================
// Data Loading
// =============================================================================

async function loadRuns() {
    try {
        const response = await fetch('/api/runs');
        const runs = await response.json();

        const selector = document.getElementById('run-selector');
        selector.innerHTML = '<option value="">Select Training Run...</option>';

        runs.forEach(run => {
            const option = document.createElement('option');
            option.value = run.path;
            option.textContent = `${run.name} (${formatBytes(run.size)})`;
            selector.appendChild(option);
        });
    } catch (error) {
        console.error('Failed to load runs:', error);
    }
}

async function onRunSelect(event) {
    const path = event.target.value;
    if (!path) return;

    state.currentRun = path;
    await loadTrainingData(path);
    await loadAnalysis(path);

    // Auto-enable refresh when a run is selected
    if (!state.autoRefresh) {
        const toggle = document.getElementById('auto-refresh-toggle');
        toggle.checked = true;
        state.autoRefresh = true;
        state.refreshInterval = setInterval(() => {
            if (state.currentRun) {
                loadTrainingData(state.currentRun);
            }
        }, 10000);
    }
}

async function loadTrainingData(path) {
    try {
        const response = await fetch(`/api/training/${path}`);
        const data = await response.json();

        if (data.error) {
            console.error('Error loading training data:', data.error);
            return;
        }

        state.trainingData = data;

        // Apply current filter (this will update all charts)
        applyFilter();

        document.getElementById('last-update').textContent = `Last update: ${new Date().toLocaleTimeString()}`;

    } catch (error) {
        console.error('Failed to load training data:', error);
    }
}

async function loadAnalysis(path) {
    try {
        const response = await fetch(`/api/analysis/${path}`);
        const data = await response.json();

        if (data.error) {
            console.error('Error loading analysis:', data.error);
            return;
        }

        state.analysisData = data;
        updateAnalysisTab();

    } catch (error) {
        console.error('Failed to load analysis:', error);
    }
}

async function loadStatus() {
    try {
        const response = await fetch('/api/status');
        const status = await response.json();

        // Training status badge
        const trainingBadge = document.getElementById('training-status');
        if (status.training_active) {
            trainingBadge.textContent = 'TRAINING';
            trainingBadge.className = 'badge badge-training';
        } else {
            trainingBadge.textContent = 'IDLE';
            trainingBadge.className = 'badge badge-idle';
        }

        // GPU status badge
        const gpuBadge = document.getElementById('gpu-status');
        if (status.gpu) {
            gpuBadge.textContent = `GPU: ${status.gpu.utilization}% | ${status.gpu.memory_used}/${status.gpu.memory_total}MB`;

            if (status.gpu.utilization > 80) {
                gpuBadge.className = 'badge badge-gpu-high';
            } else if (status.gpu.utilization > 30) {
                gpuBadge.className = 'badge badge-gpu-med';
            } else {
                gpuBadge.className = 'badge badge-gpu-low';
            }
        } else {
            gpuBadge.textContent = 'GPU: N/A';
            gpuBadge.className = 'badge badge-gpu';
        }

    } catch (error) {
        console.error('Failed to load status:', error);
    }
}

// =============================================================================
// Chart Updates
// =============================================================================

function updateMetricsCharts() {
    const data = state.filteredData || state.trainingData;
    if (!data) return;

    const epochs = data.epoch || Array.from({ length: data.reward?.length || 0 }, (_, i) => i);

    // Reward chart
    if (state.charts.reward && data.reward) {
        const ma50 = movingAverage(data.reward, 50);
        state.charts.reward.data.labels = epochs;
        state.charts.reward.data.datasets[0].data = data.reward;
        state.charts.reward.data.datasets[1].data = ma50;
        state.charts.reward.update('none');
    }

    // Market Baseline (Alpha) chart
    if (state.charts.alpha && data.alpha) {
        // Calculate MA30 for alpha
        const alphaMa30 = movingAverage(data.alpha, 30);

        state.charts.alpha.data.labels = epochs;
        state.charts.alpha.data.datasets[0].data = data.alpha;
        state.charts.alpha.data.datasets[1].data = alphaMa30;
        state.charts.alpha.data.datasets[2].data = data.market_return || [];
        state.charts.alpha.data.datasets[3].data = data.agent_return || [];
        state.charts.alpha.update('none');
    }

    // Loss chart
    if (state.charts.loss) {
        state.charts.loss.data.labels = epochs;
        state.charts.loss.data.datasets[0].data = data.total_loss || [];
        state.charts.loss.data.datasets[1].data = data.value_loss || [];
        state.charts.loss.data.datasets[2].data = data.policy_loss || [];
        state.charts.loss.data.datasets[3].data = data.reward_loss || [];
        state.charts.loss.update('none');
    }

    // Action distribution
    if (state.charts.action) {
        state.charts.action.data.labels = epochs;
        state.charts.action.data.datasets[0].data = data.pct_buy || data.n_buy || [];
        state.charts.action.data.datasets[1].data = data.pct_hold || data.n_hold || [];
        state.charts.action.data.datasets[2].data = data.pct_sell || data.n_sell || [];
        state.charts.action.data.datasets[3].data = data.pct_limit_buy || data.n_limit_buy || [];
        state.charts.action.data.datasets[4].data = data.pct_limit_sell || data.n_limit_sell || [];
        state.charts.action.update('none');
    }

    // Value calibration
    if (state.charts.valueCal && data.value_mse) {
        state.charts.valueCal.data.labels = epochs;
        state.charts.valueCal.data.datasets[0].data = data.value_mse;
        state.charts.valueCal.data.datasets[1].data = data.value_std || [];
        state.charts.valueCal.update('none');
    }

    // Entropy
    if (state.charts.entropy && data.policy_entropy) {
        state.charts.entropy.data.labels = epochs;
        state.charts.entropy.data.datasets[0].data = data.policy_entropy;
        state.charts.entropy.update('none');
    }

    // Training speed
    if (state.charts.speed && data.self_play_time) {
        state.charts.speed.data.labels = epochs;
        state.charts.speed.data.datasets[0].data = data.self_play_time;
        state.charts.speed.data.datasets[1].data = data.train_time || [];
        state.charts.speed.update('none');
    }
}

function updateTrainingConfig() {
    const data = state.filteredData || state.trainingData;
    if (!data) return;

    // Get latest values
    const getLatest = (arr) => arr && arr.length > 0 ? arr[arr.length - 1] : null;

    // GAE
    const useGae = getLatest(data.ds24_use_gae);
    const gaeLambda = getLatest(data.ds24_gae_lambda);
    const gaeEl = document.getElementById('config-gae');
    const gaeLambdaEl = document.getElementById('config-gae-lambda');
    if (gaeEl) {
        gaeEl.textContent = useGae ? 'ON' : 'OFF';
        gaeEl.className = 'config-value ' + (useGae ? 'enabled' : 'disabled');
    }
    if (gaeLambdaEl) {
        gaeLambdaEl.textContent = gaeLambda != null ? gaeLambda.toFixed(2) : '-';
    }

    // Per-head clipping
    const perHeadClip = getLatest(data.ds24_per_head_clipping);
    const clipEl = document.getElementById('config-per-head-clip');
    if (clipEl) {
        clipEl.textContent = perHeadClip ? 'ON' : 'OFF';
        clipEl.className = 'config-value ' + (perHeadClip ? 'enabled' : 'disabled');
    }

    // Epsilon-greedy
    const epsilon = getLatest(data.ds24_epsilon_greedy);
    const epsilonEl = document.getElementById('config-epsilon');
    if (epsilonEl) {
        if (epsilon != null && epsilon > 0) {
            epsilonEl.textContent = (epsilon * 100).toFixed(0) + '%';
            epsilonEl.className = 'config-value enabled';
        } else {
            epsilonEl.textContent = 'OFF';
            epsilonEl.className = 'config-value disabled';
        }
    }
}

function updateRedoATC() {
    const data = state.filteredData || state.trainingData;
    if (!data) return;

    const epochs = data.epoch || Array.from({ length: Object.values(data)[0]?.length || 0 }, (_, i) => i);
    const getLatest = (arr) => arr && arr.length > 0 ? arr[arr.length - 1] : null;

    // ReDo chart
    if (state.charts.redo && data.redo_dormant_pct) {
        state.charts.redo.data.labels = epochs;
        state.charts.redo.data.datasets[0].data = data.redo_dormant_pct || [];
        state.charts.redo.data.datasets[1].data = data.redo_recycled_count || [];
        state.charts.redo.update('none');

        // Update metric values
        const dormantCount = getLatest(data.redo_dormant_neurons);
        const dormantPct = getLatest(data.redo_dormant_pct);
        const recycledCount = getLatest(data.redo_recycled_count);

        const dormantCountEl = document.getElementById('redo-dormant-count');
        const dormantPctEl = document.getElementById('redo-dormant-pct');
        const recycledCountEl = document.getElementById('redo-recycled-count');

        if (dormantCountEl) dormantCountEl.textContent = dormantCount != null ? dormantCount : '-';
        if (dormantPctEl) dormantPctEl.textContent = dormantPct != null ? `(${dormantPct.toFixed(1)}%)` : '(-)';
        if (recycledCountEl) recycledCountEl.textContent = recycledCount != null ? recycledCount : '-';
    }

    // ATC chart
    if (state.charts.atc && data.atc_loss) {
        state.charts.atc.data.labels = epochs;
        state.charts.atc.data.datasets[0].data = data.atc_loss || [];
        state.charts.atc.data.datasets[1].data = data.atc_pos_sim || [];
        state.charts.atc.data.datasets[2].data = data.atc_neg_sim || [];
        state.charts.atc.update('none');

        // Update metric values
        const posSim = getLatest(data.atc_pos_sim);
        const negSim = getLatest(data.atc_neg_sim);

        const posSimEl = document.getElementById('atc-pos-sim');
        const negSimEl = document.getElementById('atc-neg-sim');

        if (posSimEl) posSimEl.textContent = posSim != null ? posSim.toFixed(3) : '-';
        if (negSimEl) negSimEl.textContent = negSim != null ? negSim.toFixed(3) : '-';
    }
}

function updateStats() {
    const data = state.filteredData || state.trainingData;
    if (!data) return;

    // Epochs
    document.getElementById('stat-epochs').textContent = data.epoch ? data.epoch.length : '--';

    // Reward
    if (data.reward && data.reward.length > 0) {
        const ma50 = movingAverage(data.reward, 50);
        const currentMA = ma50[ma50.length - 1];
        document.getElementById('stat-reward').textContent = currentMA?.toFixed(2) || '--';
    }

    // Alpha (market baseline)
    const alphaEl = document.getElementById('stat-alpha');
    const alphaMa30El = document.getElementById('stat-alpha-ma30');
    if (data.alpha && data.alpha.length > 0) {
        const currentAlpha = data.alpha[data.alpha.length - 1];
        if (currentAlpha != null) {
            alphaEl.textContent = (currentAlpha >= 0 ? '+' : '') + currentAlpha.toFixed(2) + '%';
            // Color based on positive/negative
            if (currentAlpha > 0.5) {
                alphaEl.style.color = COLORS.alphaPositive;
            } else if (currentAlpha < -0.5) {
                alphaEl.style.color = COLORS.alphaNegative;
            } else {
                alphaEl.style.color = COLORS.alphaZero;
            }
        } else {
            alphaEl.textContent = '--';
            alphaEl.style.color = '';
        }

        // Alpha MA30 calculation
        const alphaValues = data.alpha.filter(v => v != null);
        if (alphaValues.length >= 5) {
            const window = Math.min(30, alphaValues.length);
            const recent = alphaValues.slice(-window);
            const ma30 = recent.reduce((a, b) => a + b, 0) / recent.length;
            alphaMa30El.textContent = (ma30 >= 0 ? '+' : '') + ma30.toFixed(2) + '%';
            // Color based on MA trend
            if (ma30 > 0.5) {
                alphaMa30El.style.color = COLORS.alphaPositive;
            } else if (ma30 < -0.5) {
                alphaMa30El.style.color = COLORS.alphaNegative;
            } else {
                alphaMa30El.style.color = COLORS.alphaZero;
            }
        } else {
            alphaMa30El.textContent = '--';
            alphaMa30El.style.color = '';
        }
    } else {
        alphaEl.textContent = '--';
        alphaEl.style.color = '';
        alphaMa30El.textContent = '--';
        alphaMa30El.style.color = '';
    }

    // Value MSE
    if (data.value_mse && data.value_mse.length > 0) {
        const current = data.value_mse[data.value_mse.length - 1];
        document.getElementById('stat-value-mse').textContent = current?.toFixed(3) || '--';
    }

    // Policy Entropy
    if (data.policy_entropy && data.policy_entropy.length > 0) {
        const current = data.policy_entropy[data.policy_entropy.length - 1];
        document.getElementById('stat-entropy').textContent = current?.toFixed(2) || '--';
    }

    // Status (based on reward trend)
    if (data.reward && data.reward.length >= 20) {
        const recent = data.reward.slice(-20);
        const early = data.reward.slice(0, 20);
        const recentMean = recent.reduce((a, b) => a + (b || 0), 0) / recent.length;
        const earlyMean = early.reduce((a, b) => a + (b || 0), 0) / early.length;

        const statusEl = document.getElementById('stat-status');
        if (recentMean > earlyMean * 1.2) {
            statusEl.textContent = 'IMPROVING';
            statusEl.style.color = COLORS.success;
        } else if (recentMean < earlyMean * 0.8) {
            statusEl.textContent = 'DECLINING';
            statusEl.style.color = COLORS.danger;
        } else {
            statusEl.textContent = 'STABLE';
            statusEl.style.color = COLORS.info;
        }
    }
}

// =============================================================================
// Reasoning Tab
// =============================================================================

function updateReasoningTab() {
    const data = state.filteredData || state.trainingData;
    if (!data) return;

    // Check if TRM data available
    const hasTRMData = Object.keys(data).some(k =>
        k.startsWith('latent_delta_') || k.startsWith('output_delta_')
    );

    if (!hasTRMData) {
        document.getElementById('reasoning-no-data').style.display = 'block';
        document.getElementById('reasoning-info').style.display = 'none';
        return;
    }

    document.getElementById('reasoning-no-data').style.display = 'none';
    document.getElementById('reasoning-info').style.display = 'block';

    const epochs = data.epoch || Array.from({ length: Object.values(data)[0]?.length || 0 }, (_, i) => i);

    // TRM Heatmap (Plotly)
    updateTRMHeatmap(data, epochs);

    // Crystallization chart
    if (state.charts.crystallization) {
        state.charts.crystallization.data.labels = epochs;
        for (let h = 0; h < 4; h++) {
            const key = `output_delta_norm_h${h}`;
            state.charts.crystallization.data.datasets[h].data = data[key] || [];
        }
        state.charts.crystallization.update('none');
    }

    // Latent norm
    if (state.charts.latentNorm && data.latent_final_norm) {
        state.charts.latentNorm.data.labels = epochs;
        state.charts.latentNorm.data.datasets[0].data = data.latent_final_norm;
        state.charts.latentNorm.update('none');
    }

    // TRM Utilization
    if (state.charts.trmUtilization && data.trm_utilization) {
        state.charts.trmUtilization.data.labels = epochs;
        state.charts.trmUtilization.data.datasets[0].data = data.trm_utilization;
        state.charts.trmUtilization.update('none');
    }

    // Cycle Contribution Analysis (fraction of total delta per H-cycle)
    if (state.charts.cycleContribution) {
        const h0 = data.latent_delta_norm_h0 || [];
        const h1 = data.latent_delta_norm_h1 || [];
        const h2 = data.latent_delta_norm_h2 || [];
        const h3 = data.latent_delta_norm_h3 || [];

        // Compute normalized contributions (percentage of total)
        const contrib0 = [];
        const contrib1 = [];
        const contrib2 = [];
        const contrib3 = [];

        for (let i = 0; i < epochs.length; i++) {
            const total = (h0[i] || 0) + (h1[i] || 0) + (h2[i] || 0) + (h3[i] || 0);
            if (total > 0) {
                contrib0.push(((h0[i] || 0) / total) * 100);
                contrib1.push(((h1[i] || 0) / total) * 100);
                contrib2.push(((h2[i] || 0) / total) * 100);
                contrib3.push(((h3[i] || 0) / total) * 100);
            } else {
                contrib0.push(25);
                contrib1.push(25);
                contrib2.push(25);
                contrib3.push(25);
            }
        }

        state.charts.cycleContribution.data.labels = epochs;
        state.charts.cycleContribution.data.datasets[0].data = contrib0;
        state.charts.cycleContribution.data.datasets[1].data = contrib1;
        state.charts.cycleContribution.data.datasets[2].data = contrib2;
        state.charts.cycleContribution.data.datasets[3].data = contrib3;
        state.charts.cycleContribution.update('none');
    }
}

function updateTRMHeatmap(data, epochs) {
    // Build heatmap data from latent_delta columns
    const hCycles = 4;
    const lCycles = 6;

    // Get latest epoch data for heatmap
    const latestIdx = epochs.length - 1;
    const heatmapData = [];

    for (let h = 0; h < hCycles; h++) {
        const row = [];
        for (let l = 0; l < lCycles; l++) {
            const key = `latent_delta_norm_h${h}_l${l}`;
            if (data[key] && data[key][latestIdx] !== undefined) {
                row.push(data[key][latestIdx]);
            } else {
                // Fallback: use h-level data
                const hKey = `latent_delta_norm_h${h}`;
                row.push(data[hKey]?.[latestIdx] || 0);
            }
        }
        heatmapData.push(row);
    }

    const plotData = [{
        z: heatmapData,
        x: Array.from({ length: lCycles }, (_, i) => `L${i}`),
        y: Array.from({ length: hCycles }, (_, i) => `H${i}`),
        type: 'heatmap',
        colorscale: 'Viridis',
        showscale: true,
        colorbar: { title: 'Delta Norm' }
    }];

    const layout = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: '#e0e0e0' },
        margin: { l: 50, r: 50, t: 30, b: 50 },
        xaxis: { title: 'L Cycle' },
        yaxis: { title: 'H Cycle' }
    };

    Plotly.newPlot('trm-heatmap', plotData, layout, { responsive: true });
}

// =============================================================================
// Physics Tab
// =============================================================================

function updatePhysicsTab() {
    const data = state.filteredData || state.trainingData;
    if (!data) return;

    // Check if physics data available
    const hasPhysicsData = ['G_momentum_mean', 'G_reversion_mean', 'G_volatility_mean', 'G_correlation_mean']
        .some(k => data[k]);

    if (!hasPhysicsData) {
        document.getElementById('physics-no-data').style.display = 'block';
        document.getElementById('physics-info').style.display = 'none';
        return;
    }

    document.getElementById('physics-no-data').style.display = 'none';
    document.getElementById('physics-info').style.display = 'block';

    const epochs = data.epoch || Array.from({ length: Object.values(data)[0]?.length || 0 }, (_, i) => i);

    // Physics components stacked area
    if (state.charts.physicsComponents) {
        // Use absolute values for stacking
        const absData = (arr) => (arr || []).map(v => Math.abs(v || 0));

        state.charts.physicsComponents.data.labels = epochs;
        state.charts.physicsComponents.data.datasets[0].data = absData(data.G_momentum_mean);
        state.charts.physicsComponents.data.datasets[1].data = absData(data.G_reversion_mean);
        state.charts.physicsComponents.data.datasets[2].data = absData(data.G_volatility_mean);
        state.charts.physicsComponents.data.datasets[3].data = absData(data.G_correlation_mean);
        state.charts.physicsComponents.update('none');
    }

    // Physics parameters
    if (state.charts.physicsParams) {
        state.charts.physicsParams.data.labels = epochs;
        state.charts.physicsParams.data.datasets[0].data = data.physics_alpha_mean || [];
        state.charts.physicsParams.data.datasets[1].data = data.physics_beta_mean || [];
        state.charts.physicsParams.update('none');
    }

    // GARCH persistence
    if (state.charts.garch && data.garch_persistence) {
        state.charts.garch.data.labels = epochs;
        state.charts.garch.data.datasets[0].data = data.garch_persistence;
        state.charts.garch.update('none');
    }

    // Correlation heatmap (placeholder - would need snapshot data)
    updateCorrelationHeatmap();
}

function updateCorrelationHeatmap() {
    // Placeholder - would load from tensor snapshots
    const n = 64;
    const data = [];

    for (let i = 0; i < n; i++) {
        const row = [];
        for (let j = 0; j < n; j++) {
            if (i === j) {
                row.push(1.0);
            } else {
                row.push(Math.random() * 0.3);
            }
        }
        data.push(row);
    }

    const plotData = [{
        z: data,
        type: 'heatmap',
        colorscale: 'RdBu',
        zmin: -1,
        zmax: 1,
        showscale: true
    }];

    const layout = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: '#e0e0e0' },
        margin: { l: 50, r: 50, t: 30, b: 50 },
        xaxis: { title: 'Asset', showticklabels: false },
        yaxis: { title: 'Asset', showticklabels: false }
    };

    Plotly.newPlot('correlation-heatmap', plotData, layout, { responsive: true });
}

// =============================================================================
// Planning Tab
// =============================================================================

function updatePlanningTab() {
    const data = state.filteredData || state.trainingData;
    if (!data) return;

    // Check if planning data available
    const hasPlanningData = ['mcts_root_value', 'value_horizon_1_error']
        .some(k => data[k]);

    if (!hasPlanningData) {
        document.getElementById('planning-no-data').style.display = 'block';
        document.getElementById('planning-info').style.display = 'none';
        return;
    }

    document.getElementById('planning-no-data').style.display = 'none';
    document.getElementById('planning-info').style.display = 'block';

    const epochs = data.epoch || Array.from({ length: Object.values(data)[0]?.length || 0 }, (_, i) => i);

    // Horizon accuracy
    if (state.charts.horizon) {
        state.charts.horizon.data.labels = epochs;
        state.charts.horizon.data.datasets[0].data = data.value_horizon_1_error || [];
        state.charts.horizon.data.datasets[1].data = data.value_horizon_5_error || [];
        state.charts.horizon.data.datasets[2].data = data.value_horizon_10_error || [];
        state.charts.horizon.update('none');
    }

    // MCTS calibration scatter
    if (state.charts.mctsCalibration && data.mcts_root_value && data.reward) {
        const scatterData = [];
        const n = Math.min(data.mcts_root_value.length, data.reward.length);
        for (let i = 0; i < n; i++) {
            if (data.mcts_root_value[i] != null && data.reward[i] != null) {
                scatterData.push({ x: data.mcts_root_value[i], y: data.reward[i] });
            }
        }
        state.charts.mctsCalibration.data.datasets[0].data = scatterData;
        state.charts.mctsCalibration.update('none');
    }

    // Visit entropy
    if (state.charts.visitEntropy && data.mcts_visit_entropy) {
        state.charts.visitEntropy.data.labels = epochs;
        state.charts.visitEntropy.data.datasets[0].data = data.mcts_visit_entropy;
        state.charts.visitEntropy.update('none');
    }
}

// =============================================================================
// Interactions Tab
// =============================================================================

function updateInteractionsTab() {
    const data = state.filteredData || state.trainingData;
    if (!data || !data.total_loss) return;

    const epochs = data.epoch || Array.from({ length: data.total_loss.length }, (_, i) => i);

    // Loss contribution (normalized)
    if (state.charts.lossContribution) {
        const total = data.total_loss.map(v => Math.max(v || 1, 0.001));

        const normalize = (arr) => {
            if (!arr) return [];
            return arr.map((v, i) => ((v || 0) / total[i]) * 100);
        };

        state.charts.lossContribution.data.labels = epochs;
        state.charts.lossContribution.data.datasets[0].data = normalize(data.value_loss);
        state.charts.lossContribution.data.datasets[1].data = normalize(data.policy_loss);
        state.charts.lossContribution.data.datasets[2].data = normalize(data.reward_loss);
        state.charts.lossContribution.data.datasets[3].data = normalize(data.pred_loss);
        state.charts.lossContribution.data.datasets[4].data = normalize(data.physics_reg);
        state.charts.lossContribution.update('none');
    }

    // Alignment (TRM latent vs physics)
    if (state.charts.alignment) {
        state.charts.alignment.data.labels = epochs;
        state.charts.alignment.data.datasets[0].data = data.latent_final_norm || [];
        state.charts.alignment.data.datasets[1].data = data.physics_reg || [];
        state.charts.alignment.update('none');
    }

    // Gradient Norms by Head
    if (state.charts.gradientNorms) {
        state.charts.gradientNorms.data.labels = epochs;
        state.charts.gradientNorms.data.datasets[0].data = data.grad_norm_value || [];
        state.charts.gradientNorms.data.datasets[1].data = data.grad_norm_policy || [];
        state.charts.gradientNorms.data.datasets[2].data = data.grad_norm_dynamics || [];
        state.charts.gradientNorms.update('none');
    }

    // Head correlation heatmap (placeholder)
    updateHeadCorrelationHeatmap();
}

function updateHeadCorrelationHeatmap() {
    const heads = ['Value', 'Policy', 'Reward', 'Dynamics'];
    const n = heads.length;

    // Placeholder correlation matrix
    const data = [
        [1.0, 0.3, 0.5, 0.2],
        [0.3, 1.0, 0.4, 0.1],
        [0.5, 0.4, 1.0, 0.3],
        [0.2, 0.1, 0.3, 1.0]
    ];

    const plotData = [{
        z: data,
        x: heads,
        y: heads,
        type: 'heatmap',
        colorscale: 'RdBu',
        zmin: -1,
        zmax: 1,
        showscale: true,
        text: data.map(row => row.map(v => v.toFixed(2))),
        texttemplate: '%{text}',
        textfont: { color: 'white' }
    }];

    const layout = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: '#e0e0e0' },
        margin: { l: 80, r: 50, t: 30, b: 80 }
    };

    Plotly.newPlot('head-correlation-heatmap', plotData, layout, { responsive: true });
}

// =============================================================================
// Analysis Tab
// =============================================================================

function updateAnalysisTab() {
    const analysis = state.analysisData;
    if (!analysis) return;

    // Insights
    const insightsList = document.getElementById('insights-list');
    insightsList.innerHTML = '';
    (analysis.insights || []).forEach(insight => {
        const li = document.createElement('li');
        li.className = 'insight-item';
        li.textContent = insight;
        insightsList.appendChild(li);
    });

    if (analysis.insights?.length === 0) {
        insightsList.innerHTML = '<li class="no-items">No insights yet</li>';
    }

    // Warnings
    const warningsList = document.getElementById('warnings-list');
    warningsList.innerHTML = '';
    (analysis.warnings || []).forEach(warning => {
        const li = document.createElement('li');
        li.className = 'warning-item';
        li.textContent = warning;
        warningsList.appendChild(li);
    });

    if (analysis.warnings?.length === 0) {
        warningsList.innerHTML = '<li class="no-items">No warnings</li>';
    }

    // Phases timeline
    const phasesEl = document.getElementById('phases-timeline');
    phasesEl.innerHTML = '';
    (analysis.phases || []).forEach(phase => {
        const div = document.createElement('div');
        div.className = `phase-block phase-${phase.type}`;
        div.innerHTML = `
            <span class="phase-type">${phase.type.toUpperCase()}</span>
            <span class="phase-range">Epochs ${phase.start}-${phase.end}</span>
            <span class="phase-reward">Avg: ${phase.mean_reward?.toFixed(2) || '--'}</span>
        `;
        phasesEl.appendChild(div);
    });

    // Detailed metrics
    const metricsEl = document.getElementById('detailed-metrics');
    metricsEl.innerHTML = '';
    const metrics = analysis.metrics || {};

    Object.entries(metrics).forEach(([key, value]) => {
        const div = document.createElement('div');
        div.className = 'metric-item';
        div.innerHTML = `
            <span class="metric-key">${formatMetricKey(key)}</span>
            <span class="metric-value">${formatMetricValue(value)}</span>
        `;
        metricsEl.appendChild(div);
    });

    // Health summary
    const healthEl = document.getElementById('health-summary');
    healthEl.innerHTML = '';

    const healthItems = [
        { name: 'TRM Dynamics', available: analysis.trm_dynamics?.available },
        { name: 'Physics Model', available: analysis.physics_analysis?.available },
        { name: 'Planning', available: analysis.planning_analysis?.available },
        { name: 'Market Baseline', available: analysis.market_baseline?.available }
    ];

    healthItems.forEach(item => {
        const div = document.createElement('div');
        div.className = `health-item ${item.available ? 'health-ok' : 'health-na'}`;
        div.innerHTML = `
            <span class="health-name">${item.name}</span>
            <span class="health-status">${item.available ? 'Active' : 'Not Available'}</span>
        `;
        healthEl.appendChild(div);
    });

    // Add market baseline details if available
    if (analysis.market_baseline?.available) {
        const mb = analysis.market_baseline;
        const mbDiv = document.createElement('div');
        mbDiv.className = 'health-item health-details';
        const alphaColor = mb.alpha_current > 0 ? COLORS.alphaPositive : (mb.alpha_current < 0 ? COLORS.alphaNegative : COLORS.alphaZero);
        mbDiv.innerHTML = `
            <span class="health-name">Alpha Performance</span>
            <span class="health-status" style="color: ${alphaColor}">
                ${mb.alpha_current >= 0 ? '+' : ''}${mb.alpha_current?.toFixed(2) || '--'}%
                (${mb.positive_alpha_rate?.toFixed(0) || '--'}% positive)
            </span>
        `;
        healthEl.appendChild(mbDiv);
    }
}

// =============================================================================
// Checkpoints Tab
// =============================================================================

async function loadCheckpoints() {
    try {
        const response = await fetch('/api/checkpoints');
        const checkpoints = await response.json();

        const listEl = document.getElementById('checkpoints-list');
        listEl.innerHTML = '';

        if (checkpoints.length === 0) {
            listEl.innerHTML = '<div class="no-items">No checkpoints found</div>';
            return;
        }

        checkpoints.forEach(ckpt => {
            const div = document.createElement('div');
            div.className = 'checkpoint-item';
            div.innerHTML = `
                <div class="checkpoint-info">
                    <span class="checkpoint-name">${ckpt.experiment}</span>
                    <span class="checkpoint-step">Step ${ckpt.step}</span>
                </div>
                <div class="checkpoint-meta">
                    <span>${ckpt.size_mb} MB</span>
                    <span>${new Date(ckpt.modified).toLocaleString()}</span>
                </div>
            `;
            listEl.appendChild(div);
        });

    } catch (error) {
        console.error('Failed to load checkpoints:', error);
    }
}

// =============================================================================
// Logs Tab
// =============================================================================

async function loadLogFiles() {
    try {
        const response = await fetch('/api/log-files');
        const files = await response.json();

        const listEl = document.getElementById('log-files-list');
        listEl.innerHTML = '';

        files.forEach(file => {
            const li = document.createElement('li');
            li.className = 'log-file-item';
            li.textContent = file.name;
            li.addEventListener('click', () => loadLogContent(file.path, file.name));
            listEl.appendChild(li);
        });

    } catch (error) {
        console.error('Failed to load log files:', error);
    }
}

async function loadLogContent(path, name) {
    try {
        const lines = document.getElementById('log-lines').value;
        const response = await fetch(`/api/logs/${path}?lines=${lines}`);
        const data = await response.json();

        document.getElementById('current-log-name').textContent = name;
        document.getElementById('log-content').textContent = data.content || 'No content';

        // Highlight active file
        document.querySelectorAll('.log-file-item').forEach(el => {
            el.classList.remove('active');
            if (el.textContent === name) el.classList.add('active');
        });

        // Store current path for refresh
        state.currentLogPath = path;

    } catch (error) {
        console.error('Failed to load log content:', error);
    }
}

function refreshLog() {
    if (state.currentLogPath) {
        const name = document.getElementById('current-log-name').textContent;
        loadLogContent(state.currentLogPath, name);
    }
}

// =============================================================================
// Event Handlers
// =============================================================================

function onAutoRefreshToggle(event) {
    state.autoRefresh = event.target.checked;

    if (state.autoRefresh) {
        state.refreshInterval = setInterval(() => {
            if (state.currentRun) {
                loadTrainingData(state.currentRun);
            }
        }, 10000);
    } else {
        if (state.refreshInterval) {
            clearInterval(state.refreshInterval);
            state.refreshInterval = null;
        }
    }
}

function onLogScaleToggle(event) {
    if (state.charts.loss) {
        state.charts.loss.options.scales.y.type = event.target.checked ? 'logarithmic' : 'linear';
        state.charts.loss.update();
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

function movingAverage(data, window) {
    const result = [];
    for (let i = 0; i < data.length; i++) {
        const start = Math.max(0, i - window + 1);
        const slice = data.slice(start, i + 1).filter(v => v != null);
        result.push(slice.length > 0 ? slice.reduce((a, b) => a + b, 0) / slice.length : null);
    }
    return result;
}

function formatBytes(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1024 / 1024).toFixed(1) + ' MB';
}

function formatMetricKey(key) {
    return key
        .replace(/_/g, ' ')
        .replace(/\b\w/g, c => c.toUpperCase());
}

function formatMetricValue(value) {
    if (typeof value === 'number') {
        return Math.abs(value) < 0.01 ? value.toExponential(2) : value.toFixed(3);
    }
    return String(value);
}
