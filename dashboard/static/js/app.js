// =================================================================
// DROUGHT EARLY WARNING DASHBOARD - Main JS
// =================================================================

// Helper: Fetch API with error handling
async function fetchAPI(endpoint) {
    try {
        const response = await fetch(endpoint);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json();
    } catch (error) {
        console.error(`Error fetching ${endpoint}:`, error);
        return null;
    }
}

// Helper: Plotly dark layout base
const plotlyBase = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#e2e8f0', family: 'Inter, sans-serif', size: 12 },
    margin: { t: 30, l: 55, r: 30, b: 50 },
    showlegend: true,
    legend: {
        orientation: 'h',
        y: -0.25,
        font: { size: 11 }
    },
    xaxis: {
        gridcolor: 'rgba(255,255,255,0.05)',
        linecolor: 'rgba(255,255,255,0.1)',
        tickfont: { size: 11 }
    },
    yaxis: {
        gridcolor: 'rgba(255,255,255,0.05)',
        linecolor: 'rgba(255,255,255,0.1)',
        tickfont: { size: 11 }
    }
};

const plotlyConfig = {
    responsive: true,
    displayModeBar: false
};

// =================================================================
// HERO PARTICLE CANVAS ANIMATION
// =================================================================
function initHeroParticles() {
    const canvas = document.getElementById('hero-canvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;

    const particles = [];
    const NUM = 60;

    for (let i = 0; i < NUM; i++) {
        particles.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            r: Math.random() * 2 + 0.5,
            vx: (Math.random() - 0.5) * 0.3,
            vy: (Math.random() - 0.5) * 0.3,
            alpha: Math.random() * 0.5 + 0.1
        });
    }

    function drawFrame() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        particles.forEach(p => {
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(79, 142, 247, ${p.alpha})`;
            ctx.fill();

            p.x += p.vx;
            p.y += p.vy;

            if (p.x < 0) p.x = canvas.width;
            if (p.x > canvas.width) p.x = 0;
            if (p.y < 0) p.y = canvas.height;
            if (p.y > canvas.height) p.y = 0;
        });

        // Draw connections
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 100) {
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = `rgba(79, 142, 247, ${0.1 * (1 - dist / 100)})`;
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            }
        }

        requestAnimationFrame(drawFrame);
    }

    drawFrame();

    window.addEventListener('resize', () => {
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
    });
}

// =================================================================
// ANIMATED COUNTER
// =================================================================
function animateCounter(el, target, duration = 1500) {
    const start = performance.now();
    const isFloat = typeof target === 'number' && target % 1 !== 0;

    function update(timestamp) {
        const elapsed = timestamp - start;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = eased * target;
        el.textContent = isFloat ? current.toFixed(2) : Math.floor(current);
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

// =================================================================
// MAP INITIALIZATION (Fixed - no API key required)
// =================================================================
let mapInstance;

async function initMap() {
    const mapEl = document.getElementById('drought-map');
    if (!mapEl) return;

    // Center on Oromia / Arsi-Bale pilot zone
    mapInstance = L.map('drought-map', {
        center: [7.8, 39.8],
        zoom: 7,
        zoomControl: true,
        attributionControl: true
    });

    // --- Base layers (all free, no API key) ---

    // 1. ESRI World Imagery – true-color satellite
    const esriSatellite = L.tileLayer(
        'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        {
            attribution: 'Tiles &copy; Esri &mdash; Source: Esri, DigitalGlobe, GeoEye, Earthstar Geographics, CNES/Airbus DS, USDA, USGS, AeroGRID, IGN',
            maxZoom: 19
        }
    );

    // 2. ESRI Reference labels overlay (roads, cities, boundaries)
    const esriLabels = L.tileLayer(
        'https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
        { attribution: '', maxZoom: 19, opacity: 0.85 }
    );

    // 3. OpenStreetMap (street view fallback)
    const osmStreet = L.tileLayer(
        'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
        {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
            maxZoom: 19
        }
    );

    // 4. ESRI hybrid = satellite + labels (default view)
    const hybridGroup = L.layerGroup([esriSatellite, esriLabels]);
    hybridGroup.addTo(mapInstance);

    // Layer switcher control
    const baseMaps = {
        '🛰️ Satellite': esriSatellite,
        '🛰️ Hybrid (Satellite + Labels)': hybridGroup,
        '🗺️ Street Map': osmStreet
    };
    L.control.layers(baseMaps, {}, { position: 'topright', collapsed: false }).addTo(mapInstance);

    const data = await fetchAPI('/api/drought-data');
    if (!data) {
        mapEl.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#7c8fad;flex-direction:column;gap:1rem;"><i class="fa-solid fa-exclamation-triangle" style="font-size:2rem;color:#f59e0b;"></i><span>Could not load map data</span></div>';
        return;
    }

    let severeCount = 0, warningCount = 0, watchCount = 0, normalCount = 0;
    const zones = new Set();
    const layersByZone = {};
    const allLayers = [];

    const colorMap = {
        'Normal': '#2ecc71',
        'Watch': '#f59e0b',
        'Warning': '#f97316',
        'Severe': '#ef4444'
    };

    const glowMap = {
        'Normal': 'rgba(46,204,113,0.4)',
        'Watch': 'rgba(245,158,11,0.4)',
        'Warning': 'rgba(249,115,22,0.4)',
        'Severe': 'rgba(239,68,68,0.4)'
    };

    data.forEach(point => {
        zones.add(point.zone);
        const cls = point.risk_class || 'Normal';
        const color = colorMap[cls] || '#2ecc71';

        if (cls === 'Severe') severeCount++;
        else if (cls === 'Warning') warningCount++;
        else if (cls === 'Watch') watchCount++;
        else normalCount++;

        const circle = L.circleMarker([point.lat, point.lon], {
            radius: cls === 'Severe' ? 10 : cls === 'Warning' ? 8 : 7,
            fillColor: color,
            color: 'rgba(0,0,0,0.3)',
            weight: 1,
            opacity: 1,
            fillOpacity: 0.85
        }).addTo(mapInstance);

        // Rich popup
        circle.bindPopup(`
            <div style="font-family:Inter,sans-serif; min-width:180px; background:#0a0a1f; color:#e2e8f0; border-radius:8px; padding:0.75rem;">
                <div style="font-weight:700; font-size:0.95rem; margin-bottom:0.5rem; color:${color};">
                    ${point.zone}
                </div>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.3rem; font-size:0.8rem;">
                    <span style="color:#7c8fad;">Status</span><span style="font-weight:600;">${cls}</span>
                    <span style="color:#7c8fad;">VCI</span><span>${typeof point.vci === 'number' ? point.vci.toFixed(1) : point.vci}</span>
                    <span style="color:#7c8fad;">NDVI</span><span>${typeof point.ndvi === 'number' ? point.ndvi.toFixed(3) : point.ndvi}</span>
                    <span style="color:#7c8fad;">Rainfall</span><span>${typeof point.rainfall_mm === 'number' ? point.rainfall_mm.toFixed(0) + ' mm' : '--'}</span>
                </div>
            </div>
        `, {
            className: 'dark-popup',
            maxWidth: 220
        });

        // Hover pulse effect
        circle.on('mouseover', function() {
            this.setStyle({ radius: this.options.radius + 2, fillOpacity: 1 });
        });
        circle.on('mouseout', function() {
            this.setStyle({ radius: this.options.radius - 2, fillOpacity: 0.85 });
        });

        if (!layersByZone[point.zone]) layersByZone[point.zone] = [];
        layersByZone[point.zone].push(circle);
        allLayers.push({ layer: circle, zone: point.zone });
    });

    // Update stats
    const totalEl = document.getElementById('total-cells');
    const riskEl = document.getElementById('risk-cells');
    if (totalEl) animateCounter(totalEl, data.length);
    if (riskEl) animateCounter(riskEl, severeCount + warningCount);

    // Update breakdown
    ['severe-count', 'warning-count', 'watch-count', 'normal-count'].forEach((id, i) => {
        const el = document.getElementById(id);
        if (el) animateCounter(el, [severeCount, warningCount, watchCount, normalCount][i]);
    });

    // Zone filter
    const zoneSelect = document.getElementById('zone-filter');
    if (zoneSelect) {
        Array.from(zones).sort().forEach(zone => {
            const opt = document.createElement('option');
            opt.value = zone;
            opt.textContent = zone;
            zoneSelect.appendChild(opt);
        });

        zoneSelect.addEventListener('change', (e) => {
            const selectedZone = e.target.value;
            allLayers.forEach(({ layer, zone }) => {
                if (selectedZone === 'all' || zone === selectedZone) {
                    layer.addTo(mapInstance);
                } else {
                    mapInstance.removeLayer(layer);
                }
            });

            // Pan to zone
            if (selectedZone !== 'all' && layersByZone[selectedZone]) {
                const latlngs = layersByZone[selectedZone].map(l => l.getLatLng());
                if (latlngs.length) mapInstance.flyToBounds(L.latLngBounds(latlngs), { padding: [50, 50], duration: 1 });
            } else {
                mapInstance.flyTo([7.8, 39.8], 7, { duration: 1 });
            }
        });
    }
}

// =================================================================
// TIMESERIES ANALYSIS
// =================================================================
async function loadTimeseriesData(zone) {
    if (!document.getElementById('chart-veg')) return;

    const data = await fetchAPI(`/api/timeseries/${zone}`);
    if (!data) return;

    // NDVI + VCI
    Plotly.newPlot('chart-veg', [
        {
            x: data.dates, y: data.ndvi, name: 'NDVI',
            type: 'scatter', mode: 'lines',
            line: { color: '#34d399', width: 2 },
            fill: 'tozeroy', fillcolor: 'rgba(52,211,153,0.08)'
        },
        {
            x: data.dates, y: data.vci, name: 'VCI',
            type: 'scatter', mode: 'lines', yaxis: 'y2',
            line: { color: '#4f8ef7', width: 2 }
        }
    ], {
        ...plotlyBase,
        yaxis: { ...plotlyBase.yaxis, title: 'NDVI', range: [0, 1] },
        yaxis2: { title: 'VCI', overlaying: 'y', side: 'right', range: [0, 100], gridcolor: 'transparent', tickfont: { size: 11 } }
    }, plotlyConfig);

    // Rainfall
    Plotly.newPlot('chart-rain', [
        {
            x: data.dates, y: data.rainfall, name: 'Rainfall (mm)',
            type: 'bar', marker: { color: '#4f8ef7', opacity: 0.75 }
        }
    ], { ...plotlyBase, yaxis: { ...plotlyBase.yaxis, title: 'mm / month' } }, plotlyConfig);

    // Soil Moisture
    Plotly.newPlot('chart-sm', [
        {
            x: data.dates, y: data.soil_moisture, name: 'Soil Moisture',
            type: 'scatter', mode: 'lines',
            line: { color: '#a78bfa', width: 2 },
            fill: 'tozeroy', fillcolor: 'rgba(167,139,250,0.1)'
        }
    ], { ...plotlyBase, yaxis: { ...plotlyBase.yaxis, title: 'm³/m³' } }, plotlyConfig);

    // Risk Score with thresholds
    Plotly.newPlot('chart-risk', [
        {
            x: data.dates, y: data.risk_score, name: 'Drought Risk Score',
            type: 'scatter', mode: 'lines',
            line: { color: '#ef4444', width: 2.5 },
            fill: 'tozeroy', fillcolor: 'rgba(239,68,68,0.1)'
        }
    ], {
        ...plotlyBase,
        yaxis: { ...plotlyBase.yaxis, range: [0, 1], title: 'Risk Score' },
        shapes: [
            { type: 'line', x0: data.dates[0], x1: data.dates[data.dates.length-1], y0: 0.65, y1: 0.65, line: { color: '#ef4444', dash: 'dot', width: 1.5 } },
            { type: 'line', x0: data.dates[0], x1: data.dates[data.dates.length-1], y0: 0.45, y1: 0.45, line: { color: '#f97316', dash: 'dot', width: 1.5 } }
        ],
        annotations: [
            { x: data.dates[Math.floor(data.dates.length * 0.95)], y: 0.67, text: 'Severe', showarrow: false, font: { color: '#ef4444', size: 10 } },
            { x: data.dates[Math.floor(data.dates.length * 0.95)], y: 0.47, text: 'Warning', showarrow: false, font: { color: '#f97316', size: 10 } }
        ]
    }, plotlyConfig);
}

// =================================================================
// MODELS PAGE
// =================================================================
async function loadModelResults() {
    if (!document.getElementById('models-table')) return;

    const data = await fetchAPI('/api/model-results');
    if (!data) return;

    const tbody = document.querySelector('#models-table tbody');
    const cardsContainer = document.getElementById('models-summary-cards');

    if (cardsContainer && data.comparison) {
        cardsContainer.innerHTML = '';
        cardsContainer.style.cssText = 'display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:1.25rem; margin-bottom:1.5rem;';

        // Sort by F1 to rank
        const sorted = [...data.comparison].sort((a, b) => b.F1 - a.F1);

        data.comparison.forEach(model => {
            const rank = sorted.findIndex(m => m.model === model.model) + 1;
            const card = document.createElement('div');
            card.className = 'glass-card model-score-card animate-in';
            card.innerHTML = `
                <div class="model-rank-badge">#${rank}</div>
                <div class="model-name">${model.model}</div>
                <div class="f1-score" id="f1-${model.model.replace(/\s+/g,'-')}">${model.F1.toFixed(2)}</div>
                <div class="score-label">Macro F1 Score</div>
                <div class="progress-bar-wrap" style="margin-top:0.75rem;">
                    <div class="progress-bar-fill" data-target="${model.F1 * 100}" style="width:0%"></div>
                </div>
            `;
            cardsContainer.appendChild(card);
        });

        // Animate progress bars
        setTimeout(() => {
            document.querySelectorAll('.progress-bar-fill').forEach(bar => {
                bar.style.width = bar.dataset.target + '%';
            });
        }, 300);
    }

    if (tbody && data.comparison) {
        tbody.innerHTML = '';
        data.comparison.forEach(model => {
            const f1Class = model.F1 > 0.7 ? 'good' : model.F1 > 0.5 ? 'medium' : 'poor';
            tbody.innerHTML += `
                <tr>
                    <td><strong>${model.model}</strong></td>
                    <td><span class="metric-badge ${f1Class}">${model.F1.toFixed(3)}</span></td>
                    <td>${model.FAR.toFixed(3)}</td>
                    <td>${model.POD.toFixed(3)}</td>
                    <td>${model.RMSE.toFixed(2)}</td>
                    <td>${model.Lead10.toFixed(3)}</td>
                    <td>${model.Lead30.toFixed(3)}</td>
                </tr>
            `;
        });
    }

    if (data.lead_time_skill) {
        const colors = ['#4f8ef7', '#34d399', '#a78bfa', '#f59e0b'];
        const traces = Object.keys(data.lead_time_skill)
            .filter(k => k !== 'leads')
            .map((key, i) => ({
                x: data.lead_time_skill.leads,
                y: data.lead_time_skill[key],
                name: key.charAt(0).toUpperCase() + key.slice(1),
                type: 'scatter', mode: 'lines+markers',
                line: { color: colors[i % colors.length], width: 2.5 },
                marker: { size: 6 }
            }));

        Plotly.newPlot('chart-lead-time', traces, {
            ...plotlyBase,
            xaxis: { ...plotlyBase.xaxis, title: 'Lead Time (Days)' },
            yaxis: { ...plotlyBase.yaxis, title: 'F1 Score', range: [0, 1] }
        }, plotlyConfig);
    }

    if (data.feature_importance) {
        Plotly.newPlot('chart-importance', [{
            x: data.feature_importance.importance,
            y: data.feature_importance.features,
            type: 'bar', orientation: 'h',
            marker: {
                color: data.feature_importance.importance.map(v =>
                    `rgba(79, 142, 247, ${0.4 + v * 0.6})`
                )
            }
        }], {
            ...plotlyBase,
            margin: { ...plotlyBase.margin, l: 130 },
            yaxis: { ...plotlyBase.yaxis, autorange: 'reversed' },
            xaxis: { ...plotlyBase.xaxis, title: 'Importance Score' }
        }, plotlyConfig);
    }
}

// =================================================================
// ALERTS PAGE
// =================================================================
async function loadAlerts() {
    const container = document.getElementById('alerts-container');
    if (!container) return;

    const alerts = await fetchAPI('/api/alerts');
    container.innerHTML = '';

    if (!alerts) {
        container.innerHTML = '<div class="loading-spinner"><i class="fa-solid fa-exclamation-triangle"></i> Could not load alerts.</div>';
        return;
    }

    let counts = { 'Severe': 0, 'Warning': 0, 'Watch': 0 };

    alerts.forEach((alert, i) => {
        if (counts[alert.level] !== undefined) counts[alert.level]++;

        const icon = alert.level === 'Severe' ? 'triangle-exclamation' :
                     alert.level === 'Warning' ? 'circle-exclamation' : 'eye';

        const card = document.createElement('div');
        card.className = `glass-card alert-card ${alert.level.toLowerCase()} animate-in`;
        card.style.animationDelay = `${i * 0.05}s`;
        card.innerHTML = `
            <div style="flex:1;">
                <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.75rem; flex-wrap:wrap;">
                    <span class="alert-badge bg-${alert.level.toLowerCase()}">
                        <i class="fa-solid fa-${icon}" style="margin-right:0.3rem;"></i>${alert.level}
                    </span>
                    <strong style="font-size:0.95rem;">${alert.zone}</strong>
                    <span class="text-muted text-sm">${alert.date}</span>
                </div>
                <p style="line-height:1.6; color:#9ca3af; margin-bottom:0.75rem; font-size:0.875rem;">${alert.message}</p>
                <div style="display:flex; gap:1.5rem; font-size:0.78rem; color:#7c8fad;">
                    <span><i class="fa-solid fa-bullseye" style="margin-right:0.3rem; color:#4f8ef7;"></i>Confidence: <strong style="color:#e2e8f0;">${alert.confidence}%</strong></span>
                    <span><i class="fa-solid fa-clock" style="margin-right:0.3rem; color:#4f8ef7;"></i>Lead Time: <strong style="color:#e2e8f0;">${alert.lead_time_days} days</strong></span>
                </div>
            </div>
            <div style="margin-left:1rem; flex-shrink:0;">
                <button class="btn btn-secondary text-sm" onclick="window.location='/map'">
                    <i class="fa-solid fa-map-location-dot"></i> View Map
                </button>
            </div>
        `;
        container.appendChild(card);
    });

    // Animate counters
    ['severe', 'warning', 'watch'].forEach(level => {
        const el = document.getElementById(`count-${level}`);
        if (el) animateCounter(el, counts[level.charAt(0).toUpperCase() + level.slice(1)]);
    });
}

// =================================================================
// INIT ON DOM READY
// =================================================================
document.addEventListener('DOMContentLoaded', () => {
    // Global initializations
    initHeroParticles();

    // Animate stat counters on home page
    document.querySelectorAll('[data-count]').forEach(el => {
        animateCounter(el, parseFloat(el.dataset.count));
    });

    // Intersection Observer for animate-in
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, { threshold: 0.1 });

    document.querySelectorAll('.glass-card, .stat-card, .pipeline-step').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(16px)';
        el.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
        observer.observe(el);
    });
});
