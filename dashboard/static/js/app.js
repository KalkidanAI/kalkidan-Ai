// Helper to fetch API
async function fetchAPI(endpoint) {
    try {
        const response = await fetch(endpoint);
        if (!response.ok) throw new Error('Network response was not ok');
        return await response.json();
    } catch (error) {
        console.error(`Error fetching ${endpoint}:`, error);
        return null;
    }
}

// Map Initialization
let map;
async function initMap() {
    if (!document.getElementById('drought-map')) return;
    
    // Centered on Oromia
    map = L.map('drought-map').setView([8.5, 39.5], 7);
    
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; OpenStreetMap &copy; CARTO',
        subdomains: 'abcd',
        maxZoom: 19
    }).addTo(map);
    
    const data = await fetchAPI('/api/drought-data');
    if (!data) return;
    
    let severeCount = 0;
    let warningCount = 0;
    const zones = new Set();
    
    data.forEach(point => {
        zones.add(point.zone);
        let color = '#2ecc71'; // normal
        if (point.risk_class === 'Severe') { color = '#e74c3c'; severeCount++; }
        else if (point.risk_class === 'Warning') { color = '#e67e22'; warningCount++; }
        else if (point.risk_class === 'Watch') { color = '#f1c40f'; }
        
        const marker = L.circleMarker([point.lat, point.lon], {
            radius: 6,
            fillColor: color,
            color: '#000',
            weight: 1,
            opacity: 1,
            fillOpacity: 0.8
        }).addTo(map);
        
        marker.bindPopup(`
            <strong>Zone:</strong> ${point.zone}<br>
            <strong>Class:</strong> ${point.risk_class}<br>
            <strong>VCI:</strong> ${point.vci}<br>
            <strong>NDVI:</strong> ${point.ndvi}
        `);
    });
    
    document.getElementById('total-cells').innerText = data.length;
    document.getElementById('risk-cells').innerText = severeCount + warningCount;
    
    const select = document.getElementById('zone-filter');
    Array.from(zones).sort().forEach(zone => {
        const opt = document.createElement('option');
        opt.value = zone;
        opt.innerText = zone;
        select.appendChild(opt);
    });
}

// Timeseries Analysis
async function loadTimeseriesData(zone) {
    if (!document.getElementById('chart-veg')) return;
    
    const data = await fetchAPI(`/api/timeseries/${zone}`);
    if (!data) return;
    
    const layoutConfig = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#e2e8f0' },
        margin: { t: 30, l: 50, r: 50, b: 50 },
        showlegend: true,
        legend: { orientation: "h", y: -0.2 }
    };
    
    // Veg Chart (NDVI & VCI)
    Plotly.newPlot('chart-veg', [
        { x: data.dates, y: data.ndvi, name: 'NDVI', type: 'scatter', line: {color: '#2ecc71'} },
        { x: data.dates, y: data.vci, name: 'VCI', type: 'scatter', yaxis: 'y2', line: {color: '#3b82f6'} }
    ], {
        ...layoutConfig,
        yaxis: { title: 'NDVI', range: [0, 1] },
        yaxis2: { title: 'VCI', overlaying: 'y', side: 'right', range: [0, 100] }
    });
    
    // Rain Chart
    Plotly.newPlot('chart-rain', [
        { x: data.dates, y: data.rainfall, name: 'Rainfall', type: 'bar', marker: {color: '#3498db'} }
    ], layoutConfig);
    
    // SM Chart
    Plotly.newPlot('chart-sm', [
        { x: data.dates, y: data.soil_moisture, name: 'Soil Moisture', type: 'scatter', fill: 'tozeroy', line: {color: '#9b59b6'} }
    ], layoutConfig);
    
    // Risk Chart
    Plotly.newPlot('chart-risk', [
        { x: data.dates, y: data.risk_score, name: 'Risk Score', type: 'scatter', line: {color: '#e74c3c'} }
    ], {
        ...layoutConfig,
        yaxis: { range: [0, 1] },
        shapes: [
            { type: 'rect', xref: 'x', yref: 'y', x0: '2021-01', x1: '2022-12', y0: 0, y1: 1, fillcolor: 'rgba(231, 76, 60, 0.2)', line: {width: 0} }
        ]
    });
}

// Models Page
async function loadModelResults() {
    if (!document.getElementById('models-table')) return;
    
    const data = await fetchAPI('/api/model-results');
    if (!data) return;
    
    const tbody = document.querySelector('#models-table tbody');
    const cardsContainer = document.getElementById('models-summary-cards');
    
    cardsContainer.style.display = 'grid';
    cardsContainer.style.gridTemplateColumns = 'repeat(auto-fit, minmax(200px, 1fr))';
    cardsContainer.style.gap = '1.5rem';
    
    data.comparison.forEach(model => {
        // Table row
        tbody.innerHTML += `
            <tr>
                <td><strong>${model.model}</strong></td>
                <td>${model.F1.toFixed(2)}</td>
                <td>${model.FAR.toFixed(2)}</td>
                <td>${model.POD.toFixed(2)}</td>
                <td>${model.RMSE.toFixed(1)}</td>
                <td>${model.Lead10.toFixed(2)}</td>
                <td>${model.Lead30.toFixed(2)}</td>
            </tr>
        `;
        
        // Summary Card
        cardsContainer.innerHTML += `
            <div class="glass-card p-4 text-center">
                <h4 class="mb-2">${model.model}</h4>
                <div style="font-size: 2rem; font-weight: bold; color: #3b82f6;">${model.F1.toFixed(2)}</div>
                <div class="text-sm text-muted">F1 Score</div>
            </div>
        `;
    });
    
    const layoutConfig = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#e2e8f0' }
    };
    
    // Lead Time chart
    Plotly.newPlot('chart-lead-time', [
        { x: data.lead_time_skill.leads, y: data.lead_time_skill.xgboost, name: 'XGBoost', type: 'scatter', mode: 'lines+markers' },
        { x: data.lead_time_skill.leads, y: data.lead_time_skill.lstm, name: 'LSTM', type: 'scatter', mode: 'lines+markers' }
    ], { ...layoutConfig, xaxis: { title: 'Lead Time (Days)' }, yaxis: { title: 'F1 Score' } });
    
    // Importance Chart
    Plotly.newPlot('chart-importance', [
        { x: data.feature_importance.importance, y: data.feature_importance.features, type: 'bar', orientation: 'h', marker: {color: '#2ecc71'} }
    ], { ...layoutConfig, margin: {l: 100}, yaxis: {autorange: 'reversed'} });
}

// Alerts Page
async function loadAlerts() {
    if (!document.getElementById('alerts-container')) return;
    
    const alerts = await fetchAPI('/api/alerts');
    const container = document.getElementById('alerts-container');
    container.innerHTML = ''; // clear loading
    
    if (!alerts) {
        container.innerHTML = '<p>Error loading alerts.</p>';
        return;
    }
    
    let counts = { 'Severe': 0, 'Warning': 0, 'Watch': 0 };
    
    alerts.forEach(alert => {
        if (counts[alert.level] !== undefined) counts[alert.level]++;
        
        let badgeClass = `bg-${alert.level.toLowerCase()}`;
        
        container.innerHTML += `
            <div class="glass-card alert-card ${alert.level.toLowerCase()}">
                <div>
                    <div class="mb-2 flex-between" style="justify-content: flex-start; gap: 1rem;">
                        <span class="alert-badge ${badgeClass}">${alert.level}</span>
                        <strong>${alert.zone}</strong>
                        <span class="text-muted text-sm">${alert.date}</span>
                    </div>
                    <p class="mb-2">${alert.message}</p>
                    <div class="text-sm text-muted">
                        <i class="fa-solid fa-bullseye"></i> Confidence: ${alert.confidence}% | 
                        <i class="fa-solid fa-clock"></i> Lead Time: ${alert.lead_time_days} days
                    </div>
                </div>
                <div>
                    <button class="btn btn-secondary text-sm">View Map</button>
                </div>
            </div>
        `;
    });
    
    document.getElementById('count-severe').innerText = counts['Severe'];
    document.getElementById('count-warning').innerText = counts['Warning'];
    document.getElementById('count-watch').innerText = counts['Watch'];
}
