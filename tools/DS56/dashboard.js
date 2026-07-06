/**
 * DS56 - Advanced Dynamics Visualization Dashboard
 *
 * Level 5: Vector Fields, Phase Space, Energy Landscapes, Divergence
 * Level 6: TDA, Multifractal Analysis, Strange Attractors, Lyapunov
 */

// =============================================================================
// Global State
// =============================================================================

const state = {
    currentRun: null,
    currentEpoch: null,
    dynamicsData: null,
    charts: {}
};

// Plotly defaults for dark theme
const PLOTLY_LAYOUT = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { color: '#e0e0e0', family: 'JetBrains Mono, monospace' },
    margin: { l: 50, r: 30, t: 30, b: 50 }
};

const PLOTLY_CONFIG = {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ['lasso2d', 'select2d']
};

// Colors
const COLORS = {
    primary: '#7c4dff',
    secondary: '#00d4aa',
    warning: '#ffab00',
    danger: '#ff5252',
    info: '#448aff',
    trajectory: '#00d4aa',
    vectors: '#7c4dff'
};

// =============================================================================
// Initialization
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initCharts();
    loadSnapshots();
    loadStatus();

    // Event listeners
    document.getElementById('run-selector').addEventListener('change', onRunSelect);
    document.getElementById('analyze-btn').addEventListener('click', onAnalyze);

    // Status polling
    setInterval(loadStatus, 30000);
});

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
        });
    });
}

function initCharts() {
    // Rolling Hurst Chart (Chart.js)
    const hurstCtx = document.getElementById('rolling-hurst-chart');
    if (hurstCtx) {
        state.charts.rollingHurst = new Chart(hurstCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Hurst Exponent',
                        borderColor: COLORS.secondary,
                        backgroundColor: 'rgba(0, 212, 170, 0.1)',
                        fill: true,
                        tension: 0.2,
                        pointRadius: 0,
                        borderWidth: 2,
                        data: []
                    },
                    {
                        label: 'Random Walk (H=0.5)',
                        borderColor: COLORS.warning,
                        borderDash: [5, 5],
                        pointRadius: 0,
                        borderWidth: 1,
                        data: []
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                aspectRatio: 4,  // Width:Height ratio
                plugins: {
                    legend: { position: 'top' }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { maxTicksLimit: 20 }
                    },
                    y: {
                        min: 0,
                        max: 1,
                        grid: { color: 'rgba(255,255,255,0.05)' }
                    }
                },
                animation: false  // Prevent animation issues
            }
        });
    }

    // Multifractal Spectrum Chart (Chart.js)
    const mfCtx = document.getElementById('multifractal-chart');
    if (mfCtx) {
        state.charts.multifractal = new Chart(mfCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'h(q)',
                    borderColor: COLORS.primary,
                    backgroundColor: 'rgba(124, 77, 255, 0.2)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 4,
                    borderWidth: 2,
                    data: []
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                aspectRatio: 1.5,  // Width:Height ratio
                plugins: { legend: { position: 'top' } },
                scales: {
                    x: {
                        title: { display: true, text: 'q (moment order)' },
                        grid: { color: 'rgba(255,255,255,0.05)' }
                    },
                    y: {
                        title: { display: true, text: 'h(q) (generalized Hurst)' },
                        grid: { color: 'rgba(255,255,255,0.05)' }
                    }
                },
                animation: false
            }
        });
    }
}

// =============================================================================
// Data Loading
// =============================================================================

async function loadSnapshots() {
    try {
        const response = await fetch('/api/snapshots');
        const snapshots = await response.json();

        const selector = document.getElementById('run-selector');
        selector.innerHTML = '<option value="">Select Tensor Snapshot...</option>';

        snapshots.forEach(snap => {
            const option = document.createElement('option');
            option.value = snap.run_name;
            option.textContent = `${snap.run_name} (${snap.num_snapshots} snapshots)`;
            option.dataset.epochs = JSON.stringify(snap.epochs);
            selector.appendChild(option);
        });

        updateConnectionStatus(true);

    } catch (error) {
        console.error('Failed to load snapshots:', error);
        updateConnectionStatus(false);
    }
}

function onRunSelect(event) {
    const runName = event.target.value;
    state.currentRun = runName;

    const epochSelector = document.getElementById('epoch-selector');
    epochSelector.innerHTML = '<option value="">Epoch...</option>';

    if (!runName) return;

    // Get epochs from selected option
    const option = event.target.selectedOptions[0];
    const epochs = JSON.parse(option.dataset.epochs || '[]');

    epochs.forEach(epoch => {
        const opt = document.createElement('option');
        opt.value = epoch;
        opt.textContent = `Epoch ${epoch}`;
        epochSelector.appendChild(opt);
    });

    // Select latest by default
    if (epochs.length > 0) {
        epochSelector.value = epochs[epochs.length - 1];
        state.currentEpoch = epochs[epochs.length - 1];
    }
}

async function onAnalyze() {
    const runName = state.currentRun;
    const epoch = document.getElementById('epoch-selector').value || '';

    if (!runName) {
        alert('Please select a run first');
        return;
    }

    const btn = document.getElementById('analyze-btn');
    btn.textContent = 'Analyzing...';
    btn.disabled = true;

    try {
        const url = epoch
            ? `/api/dynamics/${runName}?epoch=${epoch}`
            : `/api/dynamics/${runName}`;

        const response = await fetch(url);
        const data = await response.json();

        if (data.error) {
            alert(`Error: ${data.error}`);
            return;
        }

        state.dynamicsData = data;
        updateAllVisualizations();
        updateLastUpdate();

    } catch (error) {
        console.error('Analysis failed:', error);
        alert('Analysis failed. Check console for details.');
    } finally {
        btn.textContent = 'Analyze';
        btn.disabled = false;
    }
}

async function loadStatus() {
    try {
        const response = await fetch('/api/status');
        const status = await response.json();

        // GPU status
        const gpuBadge = document.getElementById('gpu-status');
        if (status.gpu) {
            gpuBadge.textContent = `GPU: ${status.gpu.utilization}%`;
        } else {
            gpuBadge.textContent = 'GPU: N/A';
        }

    } catch (error) {
        console.error('Failed to load status:', error);
    }
}

// =============================================================================
// Visualization Updates
// =============================================================================

function updateAllVisualizations() {
    const data = state.dynamicsData;
    if (!data) return;

    // Level 5
    updateVectorField(data);
    updatePhaseSpace(data);
    updateEnergyLandscape(data);
    updateDivergenceField(data);
    updateRollingHurst(data);
    updateMetrics(data);

    // Level 6
    updateAttractor(data);
    updateMultifractal(data);
    updateSummary(data);
}

function updateVectorField(data) {
    const container = document.getElementById('vector-field-chart');
    if (!container || !data.vector_field) return;

    const vf = data.vector_field;
    const traces = [];

    // Vector field as quiver (using cone plot approximation)
    const X = vf.X.flat();
    const Y = vf.Y.flat();
    const U = vf.U.flat();
    const V = vf.V.flat();

    // Create arrows using scatter with line segments
    const arrowTrace = {
        type: 'scatter',
        mode: 'markers',
        x: X,
        y: Y,
        marker: {
            size: 8,
            color: U.map((u, i) => Math.sqrt(u*u + V[i]*V[i])),
            colorscale: 'Viridis',
            showscale: true,
            colorbar: { title: '|v|' }
        },
        hovertemplate: 'x: %{x:.2f}<br>y: %{y:.2f}<br>|v|: %{marker.color:.3f}<extra></extra>'
    };
    traces.push(arrowTrace);

    // Trajectory overlay
    if (vf.trajectory_x && vf.trajectory_y) {
        traces.push({
            type: 'scatter',
            mode: 'lines',
            x: vf.trajectory_x,
            y: vf.trajectory_y,
            line: { color: COLORS.trajectory, width: 1.5 },
            opacity: 0.7,
            name: 'Trajectory'
        });
    }

    const layout = {
        ...PLOTLY_LAYOUT,
        xaxis: { title: 'PC1', gridcolor: 'rgba(255,255,255,0.05)' },
        yaxis: { title: 'PC2', gridcolor: 'rgba(255,255,255,0.05)', scaleanchor: 'x' },
        showlegend: false
    };

    Plotly.newPlot(container, traces, layout, PLOTLY_CONFIG);
}

function updatePhaseSpace(data) {
    const container = document.getElementById('phase-space-chart');
    if (!container || !data.vector_field) return;

    const vf = data.vector_field;
    if (!vf.trajectory_x || !vf.trajectory_y) return;

    const n = vf.trajectory_x.length;
    const colors = Array.from({ length: n }, (_, i) => i / n);

    const trace = {
        type: 'scatter',
        mode: 'lines+markers',
        x: vf.trajectory_x,
        y: vf.trajectory_y,
        marker: {
            size: 4,
            color: colors,
            colorscale: 'Portland',
            showscale: true,
            colorbar: { title: 'Time' }
        },
        line: {
            color: 'rgba(0, 212, 170, 0.3)',
            width: 1
        },
        hovertemplate: 'PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>t: %{marker.color:.0%}<extra></extra>'
    };

    const layout = {
        ...PLOTLY_LAYOUT,
        xaxis: { title: 'PC1', gridcolor: 'rgba(255,255,255,0.05)' },
        yaxis: { title: 'PC2', gridcolor: 'rgba(255,255,255,0.05)', scaleanchor: 'x' }
    };

    Plotly.newPlot(container, [trace], layout, PLOTLY_CONFIG);
}

function updateEnergyLandscape(data) {
    const container = document.getElementById('energy-landscape-chart');
    if (!container || !data.energy_landscape) return;

    const el = data.energy_landscape;

    const trace = {
        type: 'surface',
        x: el.X,
        y: el.Y,
        z: el.Z,
        colorscale: 'Viridis',
        reversescale: true,
        showscale: true,
        colorbar: { title: 'Energy' }
    };

    const layout = {
        ...PLOTLY_LAYOUT,
        scene: {
            xaxis: { title: 'PC1', gridcolor: 'rgba(255,255,255,0.1)' },
            yaxis: { title: 'PC2', gridcolor: 'rgba(255,255,255,0.1)' },
            zaxis: { title: 'E', gridcolor: 'rgba(255,255,255,0.1)' },
            bgcolor: 'transparent',
            camera: { eye: { x: 1.5, y: 1.5, z: 1.2 } }
        }
    };

    Plotly.newPlot(container, [trace], layout, PLOTLY_CONFIG);
}

function updateDivergenceField(data) {
    const container = document.getElementById('divergence-chart');
    if (!container || !data.divergence) return;

    const div = data.divergence;

    const trace = {
        type: 'heatmap',
        z: div.grid,
        colorscale: 'RdBu',
        zmid: 0,
        showscale: true,
        colorbar: { title: 'div(F)' }
    };

    const layout = {
        ...PLOTLY_LAYOUT,
        xaxis: { title: 'x', gridcolor: 'rgba(255,255,255,0.05)' },
        yaxis: { title: 'y', gridcolor: 'rgba(255,255,255,0.05)' }
    };

    Plotly.newPlot(container, [trace], layout, PLOTLY_CONFIG);
}

function updateRollingHurst(data) {
    if (!state.charts.rollingHurst || !data.hurst_rolling) return;

    const chart = state.charts.rollingHurst;
    const hurstData = data.hurst_rolling;
    const labels = Array.from({ length: hurstData.length }, (_, i) => i);
    const baseline = hurstData.map(() => 0.5);

    chart.data.labels = labels;
    chart.data.datasets[0].data = hurstData;
    chart.data.datasets[1].data = baseline;
    chart.update('none');
}

function updateMetrics(data) {
    // Hurst
    const hurst = data.hurst;
    const hurstEl = document.getElementById('hurst-value');
    const hurstHint = document.getElementById('hurst-hint');
    if (hurst !== undefined) {
        hurstEl.textContent = hurst.toFixed(3);
        hurstEl.className = 'metric-value';
        if (hurst > 0.6) {
            hurstEl.classList.add('info');
            hurstHint.textContent = 'Trending (persistent)';
        } else if (hurst < 0.4) {
            hurstEl.classList.add('warning');
            hurstHint.textContent = 'Mean-reverting';
        } else {
            hurstHint.textContent = 'Random walk';
        }
    }

    // Lyapunov
    const lyap = data.lyapunov;
    const lyapEl = document.getElementById('lyapunov-value');
    const lyapHint = document.getElementById('lyapunov-hint');
    if (lyap !== undefined) {
        lyapEl.textContent = lyap.toFixed(4);
        lyapEl.className = 'metric-value';
        if (lyap > 0) {
            lyapEl.classList.add('danger');
            lyapHint.textContent = 'Chaotic (sensitive)';
        } else {
            lyapEl.classList.add('info');
            lyapHint.textContent = 'Stable dynamics';
        }
    }

    // Divergence
    const div = data.divergence;
    const divEl = document.getElementById('divergence-value');
    const divHint = document.getElementById('divergence-hint');
    if (div) {
        divEl.textContent = div.mean.toFixed(4);
        divEl.className = 'metric-value';
        if (div.mean < -0.1) {
            divEl.classList.add('info');
            divHint.textContent = 'Converging (sink)';
        } else if (div.mean > 0.1) {
            divEl.classList.add('warning');
            divHint.textContent = 'Diverging (source)';
        } else {
            divHint.textContent = 'Near-conservative';
        }
    }

    // PCA variance
    const vf = data.vector_field;
    const pcaEl = document.getElementById('pca-value');
    if (vf && vf.explained_variance) {
        const total = vf.explained_variance[0] + vf.explained_variance[1];
        pcaEl.textContent = (total * 100).toFixed(1) + '%';
    }
}

function updateAttractor(data) {
    const container = document.getElementById('attractor-chart');
    if (!container || !data.attractor) return;

    const att = data.attractor;

    const trace = {
        type: 'scatter3d',
        mode: 'lines',
        x: att.x,
        y: att.y,
        z: att.z,
        line: {
            color: Array.from({ length: att.x.length }, (_, i) => i),
            colorscale: 'Viridis',
            width: 2
        },
        hovertemplate: 'x(t): %{x:.3f}<br>x(t-d): %{y:.3f}<br>x(t-2d): %{z:.3f}<extra></extra>'
    };

    const layout = {
        ...PLOTLY_LAYOUT,
        scene: {
            xaxis: { title: 'x(t)', gridcolor: 'rgba(255,255,255,0.1)' },
            yaxis: { title: 'x(t-d)', gridcolor: 'rgba(255,255,255,0.1)' },
            zaxis: { title: 'x(t-2d)', gridcolor: 'rgba(255,255,255,0.1)' },
            bgcolor: 'transparent',
            camera: { eye: { x: 1.5, y: 1.5, z: 1.2 } }
        }
    };

    Plotly.newPlot(container, [trace], layout, PLOTLY_CONFIG);

    // Update attractor points metric
    document.getElementById('attractor-points-value').textContent = att.x.length;
}

function updateMultifractal(data) {
    if (!state.charts.multifractal || !data.multifractal) return;

    const mf = data.multifractal;
    const chart = state.charts.multifractal;

    chart.data.labels = mf.q_values;
    chart.data.datasets[0].data = mf.h_values;
    chart.update('none');

    // Update width metric
    document.getElementById('mf-width-value').textContent = mf.width.toFixed(3);

    // Determine regime
    const regimeEl = document.getElementById('regime-value');
    const hurst = data.hurst || 0.5;
    const lyap = data.lyapunov || 0;

    if (lyap > 0.1) {
        regimeEl.textContent = 'Chaotic';
        regimeEl.className = 'metric-value danger';
    } else if (hurst > 0.6) {
        regimeEl.textContent = 'Trending';
        regimeEl.className = 'metric-value info';
    } else if (hurst < 0.4) {
        regimeEl.textContent = 'Mean-Revert';
        regimeEl.className = 'metric-value warning';
    } else {
        regimeEl.textContent = 'Random';
        regimeEl.className = 'metric-value';
    }
}

function updateSummary(data) {
    const summaryEl = document.getElementById('dynamics-summary');
    if (!summaryEl) return;

    const hurst = data.hurst !== undefined ? data.hurst.toFixed(3) : 'N/A';
    const lyap = data.lyapunov !== undefined ? data.lyapunov.toFixed(4) : 'N/A';
    const div = data.divergence ? data.divergence.mean.toFixed(4) : 'N/A';
    const mfWidth = data.multifractal ? data.multifractal.width.toFixed(3) : 'N/A';

    summaryEl.innerHTML = `
        <table class="summary-table">
            <tr><td>Run</td><td><strong>${data.run_name}</strong></td></tr>
            <tr><td>Epoch</td><td><strong>${data.epoch}</strong></td></tr>
            <tr><td>Hurst Exponent</td><td><strong>${hurst}</strong></td></tr>
            <tr><td>Lyapunov Exponent</td><td><strong>${lyap}</strong></td></tr>
            <tr><td>Mean Divergence</td><td><strong>${div}</strong></td></tr>
            <tr><td>Multifractal Width</td><td><strong>${mfWidth}</strong></td></tr>
        </table>
    `;
}

// =============================================================================
// Utility Functions
// =============================================================================

function updateConnectionStatus(connected) {
    const badge = document.getElementById('connection-status');
    if (connected) {
        badge.textContent = 'Connected';
        badge.className = 'badge badge-connected';
    } else {
        badge.textContent = 'Disconnected';
        badge.className = 'badge badge-idle';
    }
}

function updateLastUpdate() {
    document.getElementById('last-update').textContent =
        `Last update: ${new Date().toLocaleTimeString()}`;
}
