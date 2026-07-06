/**
 * DS78 - Shroom Engine Dashboard
 * Level 7-8: Volumetric & Fractal Visualization
 */

// =============================================================================
// Global State
// =============================================================================

const state = {
    currentRun: null,
    currentEpoch: null,
    volumeData: null,
    fractalData: null
};

const PLOTLY_LAYOUT = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { color: '#e8e8f0', family: 'JetBrains Mono, monospace' },
    margin: { l: 50, r: 30, t: 30, b: 50 }
};

const PLOTLY_CONFIG = {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ['lasso2d', 'select2d']
};

const COLORS = {
    primary: '#ff00ff',
    secondary: '#00ffaa',
    tertiary: '#ffaa00'
};

// =============================================================================
// Initialization
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    loadSnapshots();

    // Event listeners
    document.getElementById('run-selector').addEventListener('change', onRunSelect);
    document.getElementById('render-btn').addEventListener('click', onRender);
    document.getElementById('regenerate-clifford').addEventListener('click', loadAttractors);

    // Load reference attractors
    loadAttractors();
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

// =============================================================================
// Data Loading
// =============================================================================

async function loadSnapshots() {
    try {
        const response = await fetch('/api/snapshots');
        const snapshots = await response.json();

        const selector = document.getElementById('run-selector');
        selector.innerHTML = '<option value="">Select Run...</option>';

        snapshots.forEach(snap => {
            const option = document.createElement('option');
            option.value = snap.run_name;
            option.textContent = `${snap.run_name} (${snap.num_snapshots} epochs)`;
            option.dataset.epochs = JSON.stringify(snap.epochs);
            selector.appendChild(option);
        });

    } catch (error) {
        console.error('Failed to load snapshots:', error);
    }
}

function onRunSelect(event) {
    const runName = event.target.value;
    state.currentRun = runName;

    const epochSelector = document.getElementById('epoch-selector');
    epochSelector.innerHTML = '<option value="">Epoch...</option>';

    if (!runName) return;

    const option = event.target.selectedOptions[0];
    const epochs = JSON.parse(option.dataset.epochs || '[]');

    epochs.forEach(epoch => {
        const opt = document.createElement('option');
        opt.value = epoch;
        opt.textContent = `Epoch ${epoch}`;
        epochSelector.appendChild(opt);
    });

    if (epochs.length > 0) {
        epochSelector.value = epochs[epochs.length - 1];
        state.currentEpoch = epochs[epochs.length - 1];
    }
}

async function onRender() {
    const runName = state.currentRun;
    const epoch = document.getElementById('epoch-selector').value || '';

    if (!runName) {
        alert('Please select a run first');
        return;
    }

    const btn = document.getElementById('render-btn');
    btn.textContent = 'Rendering...';
    btn.disabled = true;

    try {
        // Fetch volume data
        const volumeUrl = epoch
            ? `/api/volume/${runName}?epoch=${epoch}`
            : `/api/volume/${runName}`;
        const volumeResponse = await fetch(volumeUrl);
        state.volumeData = await volumeResponse.json();

        // Fetch fractal data
        const fractalUrl = epoch
            ? `/api/fractal/${runName}?epoch=${epoch}`
            : `/api/fractal/${runName}`;
        const fractalResponse = await fetch(fractalUrl);
        state.fractalData = await fractalResponse.json();

        updateAllVisualizations();
        updateLastUpdate();

    } catch (error) {
        console.error('Render failed:', error);
        alert('Render failed. Check console for details.');
    } finally {
        btn.textContent = 'Render';
        btn.disabled = false;
    }
}

async function loadAttractors() {
    try {
        const a = document.getElementById('clifford-a').value;
        const b = document.getElementById('clifford-b').value;
        const c = document.getElementById('clifford-c').value;
        const d = document.getElementById('clifford-d').value;

        const response = await fetch(`/api/reference-attractors?a=${a}&b=${b}&c=${c}&d=${d}`);
        const data = await response.json();

        updateCliffordAttractor(data.clifford);
        updateLorenzAttractor(data.lorenz);

    } catch (error) {
        console.error('Failed to load attractors:', error);
    }
}

// =============================================================================
// Visualization Updates
// =============================================================================

function updateAllVisualizations() {
    if (state.volumeData) {
        updateVolumeViewer(state.volumeData);
        updateSplatsViewer(state.volumeData);
        updateVolumeMetrics(state.volumeData);
    }

    if (state.fractalData) {
        updateCorrelationDimension(state.fractalData);
        updateBoxCounting(state.fractalData);
        updateRecurrencePlot(state.fractalData);
        updateFractalMetrics(state.fractalData);
    }
}

function updateVolumeViewer(data) {
    const container = document.getElementById('volume-viewer');
    if (!container || !data.density_field) return;

    const df = data.density_field;
    if (!df.trajectory_3d || df.trajectory_3d.length === 0) return;

    const trajectory = df.trajectory_3d;
    const x = trajectory.map(p => p[0]);
    const y = trajectory.map(p => p[1]);
    const z = trajectory.map(p => p[2]);
    const colors = Array.from({ length: x.length }, (_, i) => i / x.length);

    // 3D trajectory with volume indication
    const trace = {
        type: 'scatter3d',
        mode: 'lines+markers',
        x: x,
        y: y,
        z: z,
        marker: {
            size: 3,
            color: colors,
            colorscale: 'Viridis',
            opacity: 0.8
        },
        line: {
            color: colors,
            colorscale: 'Viridis',
            width: 2
        },
        hovertemplate: 'PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<extra></extra>'
    };

    const layout = {
        ...PLOTLY_LAYOUT,
        scene: {
            xaxis: { title: 'PC1', gridcolor: 'rgba(255,0,255,0.1)', zerolinecolor: 'rgba(255,0,255,0.3)' },
            yaxis: { title: 'PC2', gridcolor: 'rgba(0,255,170,0.1)', zerolinecolor: 'rgba(0,255,170,0.3)' },
            zaxis: { title: 'PC3', gridcolor: 'rgba(255,170,0,0.1)', zerolinecolor: 'rgba(255,170,0,0.3)' },
            bgcolor: 'transparent',
            camera: { eye: { x: 1.5, y: 1.5, z: 1.2 } }
        }
    };

    Plotly.newPlot(container, [trace], layout, PLOTLY_CONFIG);
}

function updateSplatsViewer(data) {
    const container = document.getElementById('splats-viewer');
    if (!container || !data.gaussian_splats) return;

    const splats = data.gaussian_splats;
    if (!splats.positions || splats.positions.length === 0) return;

    const positions = splats.positions;
    const x = positions.map(p => p[0]);
    const y = positions.map(p => p[1]);
    const z = positions.map(p => p[2]);
    const sizes = splats.scales.map(s => Math.max(s * 30, 5));
    const colors = splats.colors;

    const trace = {
        type: 'scatter3d',
        mode: 'markers',
        x: x,
        y: y,
        z: z,
        marker: {
            size: sizes,
            color: colors,
            colorscale: [
                [0, '#ff00ff'],
                [0.5, '#00ffaa'],
                [1, '#ffaa00']
            ],
            opacity: 0.7,
            line: { width: 0 }
        },
        hovertemplate: 'x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>'
    };

    const layout = {
        ...PLOTLY_LAYOUT,
        scene: {
            xaxis: { title: 'PC1', gridcolor: 'rgba(255,0,255,0.1)' },
            yaxis: { title: 'PC2', gridcolor: 'rgba(0,255,170,0.1)' },
            zaxis: { title: 'PC3', gridcolor: 'rgba(255,170,0,0.1)' },
            bgcolor: 'transparent',
            camera: { eye: { x: 1.8, y: 1.8, z: 1.5 } }
        }
    };

    Plotly.newPlot(container, [trace], layout, PLOTLY_CONFIG);
}

function updateVolumeMetrics(data) {
    const df = data.density_field;
    if (df && df.explained_variance) {
        const ev = df.explained_variance;
        document.getElementById('pc1-var').textContent = (ev[0] * 100).toFixed(1) + '%';
        document.getElementById('pc2-var').textContent = (ev[1] * 100).toFixed(1) + '%';
        document.getElementById('pc3-var').textContent = (ev[2] * 100).toFixed(1) + '%';
    }

    if (data.gaussian_splats) {
        document.getElementById('n-splats').textContent = data.gaussian_splats.n_splats;
    }
}

function updateCorrelationDimension(data) {
    const container = document.getElementById('corr-dim-chart');
    if (!container || !data.correlation_dimension) return;

    const cd = data.correlation_dimension;
    if (!cd.log_radii || cd.log_radii.length === 0) return;

    const trace = {
        type: 'scatter',
        mode: 'lines+markers',
        x: cd.log_radii,
        y: cd.log_correlations,
        marker: { color: COLORS.primary, size: 6 },
        line: { color: COLORS.primary, width: 2 },
        name: 'C(r)'
    };

    // Fit line
    const n = cd.log_radii.length;
    const midStart = Math.floor(n / 4);
    const midEnd = Math.floor(3 * n / 4);

    if (cd.dimension !== null) {
        const fitX = [cd.log_radii[midStart], cd.log_radii[midEnd]];
        const fitY = fitX.map(x => cd.dimension * x + (cd.log_correlations[midStart] - cd.dimension * cd.log_radii[midStart]));

        const fitTrace = {
            type: 'scatter',
            mode: 'lines',
            x: fitX,
            y: fitY,
            line: { color: COLORS.secondary, width: 2, dash: 'dash' },
            name: `D2 = ${cd.dimension.toFixed(2)}`
        };

        Plotly.newPlot(container, [trace, fitTrace], {
            ...PLOTLY_LAYOUT,
            xaxis: { title: 'log(r)', gridcolor: 'rgba(255,255,255,0.05)' },
            yaxis: { title: 'log(C(r))', gridcolor: 'rgba(255,255,255,0.05)' },
            showlegend: true,
            legend: { x: 0.02, y: 0.98 }
        }, PLOTLY_CONFIG);
    }
}

function updateBoxCounting(data) {
    const container = document.getElementById('box-dim-chart');
    if (!container || !data.box_counting_dimension) return;

    const bc = data.box_counting_dimension;
    if (!bc.log_inv_scales || bc.log_inv_scales.length === 0) return;

    const trace = {
        type: 'scatter',
        mode: 'lines+markers',
        x: bc.log_inv_scales,
        y: bc.log_counts,
        marker: { color: COLORS.secondary, size: 6 },
        line: { color: COLORS.secondary, width: 2 },
        name: 'N(e)'
    };

    const layout = {
        ...PLOTLY_LAYOUT,
        xaxis: { title: 'log(1/e)', gridcolor: 'rgba(255,255,255,0.05)' },
        yaxis: { title: 'log(N)', gridcolor: 'rgba(255,255,255,0.05)' },
        showlegend: true,
        legend: { x: 0.02, y: 0.98 }
    };

    Plotly.newPlot(container, [trace], layout, PLOTLY_CONFIG);
}

function updateRecurrencePlot(data) {
    const container = document.getElementById('recurrence-plot');
    if (!container || !data.recurrence_plot) return;

    const rp = data.recurrence_plot;
    if (!rp.matrix || rp.matrix.length === 0) return;

    const trace = {
        type: 'heatmap',
        z: rp.matrix,
        colorscale: [
            [0, '#050508'],
            [1, '#ff00ff']
        ],
        showscale: false
    };

    const layout = {
        ...PLOTLY_LAYOUT,
        xaxis: { title: 'i', gridcolor: 'rgba(255,255,255,0.05)' },
        yaxis: { title: 'j', gridcolor: 'rgba(255,255,255,0.05)', scaleanchor: 'x' }
    };

    Plotly.newPlot(container, [trace], layout, PLOTLY_CONFIG);
}

function updateFractalMetrics(data) {
    const cd = data.correlation_dimension;
    if (cd && cd.dimension !== null) {
        document.getElementById('corr-dim-value').textContent = cd.dimension.toFixed(3);
    }

    const bc = data.box_counting_dimension;
    if (bc && bc.dimension !== null) {
        document.getElementById('box-dim-value').textContent = bc.dimension.toFixed(3);
    }

    const rp = data.recurrence_plot;
    if (rp) {
        if (rp.recurrence_rate !== undefined) {
            document.getElementById('rr-value').textContent = (rp.recurrence_rate * 100).toFixed(1) + '%';
        }
        if (rp.determinism !== undefined) {
            document.getElementById('det-value').textContent = (rp.determinism * 100).toFixed(1) + '%';
        }
    }
}

function updateCliffordAttractor(data) {
    const container = document.getElementById('clifford-chart');
    if (!container || !data) return;

    const trace = {
        type: 'scattergl',
        mode: 'markers',
        x: data.x,
        y: data.y,
        marker: {
            size: 1,
            color: Array.from({ length: data.x.length }, (_, i) => i),
            colorscale: [
                [0, '#ff00ff'],
                [0.5, '#00ffaa'],
                [1, '#ffaa00']
            ],
            opacity: 0.5
        },
        hoverinfo: 'skip'
    };

    const layout = {
        ...PLOTLY_LAYOUT,
        xaxis: { gridcolor: 'rgba(255,255,255,0.05)', zeroline: false },
        yaxis: { gridcolor: 'rgba(255,255,255,0.05)', zeroline: false, scaleanchor: 'x' }
    };

    Plotly.newPlot(container, [trace], layout, PLOTLY_CONFIG);
}

function updateLorenzAttractor(data) {
    const container = document.getElementById('lorenz-chart');
    if (!container || !data) return;

    const trace = {
        type: 'scatter3d',
        mode: 'lines',
        x: data.x,
        y: data.y,
        z: data.z,
        line: {
            color: Array.from({ length: data.x.length }, (_, i) => i),
            colorscale: [
                [0, '#ff00ff'],
                [0.5, '#00ffaa'],
                [1, '#ffaa00']
            ],
            width: 2
        },
        hoverinfo: 'skip'
    };

    const layout = {
        ...PLOTLY_LAYOUT,
        scene: {
            xaxis: { title: 'x', gridcolor: 'rgba(255,0,255,0.1)' },
            yaxis: { title: 'y', gridcolor: 'rgba(0,255,170,0.1)' },
            zaxis: { title: 'z', gridcolor: 'rgba(255,170,0,0.1)' },
            bgcolor: 'transparent',
            camera: { eye: { x: 1.5, y: 1.5, z: 1.0 } }
        }
    };

    Plotly.newPlot(container, [trace], layout, PLOTLY_CONFIG);
}

// =============================================================================
// Utility
// =============================================================================

function updateLastUpdate() {
    document.getElementById('last-update').textContent =
        `Last render: ${new Date().toLocaleTimeString()}`;
}
