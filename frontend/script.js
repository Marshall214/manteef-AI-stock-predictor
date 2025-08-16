// --- Configuration ---
const API_BASE_URL = window.location.hostname === 'localhost'
    ? 'http://localhost:5000'
    : 'https://manteef-ai-stock-predictor-predictor.up.railway.app';

// --- Elements ---
const elements = {
    tickerInput: document.getElementById('tickerInput'),
    tickerSuggestions: document.getElementById('tickerSuggestions'),
    resultsContent: document.getElementById('resultsContent'),
    loadingState: document.getElementById('loadingState'),
    emptyState: document.getElementById('emptyState'),
    errorState: document.getElementById('errorState'),
    errorMessage: document.getElementById('errorMessage'),
    lastUpdate: document.getElementById('lastUpdate'),
    apiStatus: document.querySelector('.api-status'),
    signalText: document.getElementById('signalText'),
    confidenceText: document.getElementById('confidenceText'),
    strengthText: document.getElementById('strengthText'),
    dateRangeText: document.getElementById('dateRangeText'),
    indicatorsTable: document.getElementById('indicatorsTable'),
    modelName: document.getElementById('modelName'),
    miniChartContainer: null
};

// --- Feature list (fetched from API) ---
let FEATURE_NAMES = [];
let miniChart = null;

// --- Initialize ---
async function init() {
    await fetchFeatures();
    checkAPIStatus();
    elements.emptyState.style.display = 'block';
}

// --- Fetch feature names ---
async function fetchFeatures() {
    try {
        const res = await fetch(`${API_BASE_URL}/features`);
        const data = await res.json();
        FEATURE_NAMES = data.features || [];
        elements.modelName.textContent = "XGBRegressor"; // Hardcoded from backend
    } catch (err) {
        console.error('❌ Error fetching features:', err);
    }
}

// --- Ticker Suggestions ---
elements.tickerInput.addEventListener('input', handleTickerInput);

async function handleTickerInput(event) {
    const query = event.target.value.toUpperCase().replace(/[^A-Z]/g, '');
    elements.tickerInput.value = query;

    if (!query) return hideSuggestions();

    try {
        const res = await fetch(`${API_BASE_URL}/tickers?q=${query}`);
        const data = await res.json();
        showTickerSuggestions(data.tickers || []);
    } catch (err) {
        console.error('❌ Error fetching tickers:', err);
        hideSuggestions();
    }
}

function showTickerSuggestions(tickers) {
    if (!tickers || tickers.length === 0) return hideSuggestions();

    elements.tickerSuggestions.innerHTML = tickers.map(t => `
        <div class="suggestion-item" data-ticker="${t}">${t}</div>
    `).join('');

    elements.tickerSuggestions.querySelectorAll('.suggestion-item')
        .forEach(item => item.addEventListener('click', function() {
            elements.tickerInput.value = this.dataset.ticker;
            hideSuggestions();
            elements.tickerInput.focus();
        }));

    elements.tickerSuggestions.classList.remove('hidden');
}

function hideSuggestions() {
    elements.tickerSuggestions.classList.add('hidden');
}

// --- Predict ---
async function predictTicker(ticker) {
    if (!ticker) return showError('Please enter a ticker');

    showLoadingState();

    try {
        const res = await fetch(`${API_BASE_URL}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ticker })
        });

        const data = await res.json();
        if (res.ok) {
            displayResults(data);
        } else {
            showError(data.error || 'Prediction failed');
        }
    } catch (err) {
        console.error('❌ Prediction error:', err);
        showError('Prediction failed due to network error');
    }
}

// --- Display Results ---
function displayResults(data) {
    showResultsState();

    // Update signals
    elements.signalText.textContent = data.signal;
    elements.confidenceText.textContent = `Confidence: ${data.confidence}`;
    elements.strengthText.textContent = data.signal_strength;

    // Update date range
    elements.dateRangeText.textContent = `Data Range: ${data.date_range.start} → ${data.date_range.end}`;

    // Update indicators table
    if (elements.indicatorsTable && data.features) {
        const rows = Object.entries(data.features)
            .map(([key, value]) => `<tr><td>${key}</td><td>${Number(value).toFixed(4)}</td></tr>`)
            .join('');
        elements.indicatorsTable.innerHTML = rows;
    }

    // Render mini price trend chart
    renderMiniTrend(data.features);

    updateLastUpdate();
}

// --- Mini Trend Chart ---
function renderMiniTrend(features) {
    const labels = Object.keys(features).slice(-30); // Last 30 entries
    const values = Object.values(features).slice(-30);

    if (!elements.miniChartContainer) {
        elements.miniChartContainer = document.createElement('canvas');
        elements.resultsContent.appendChild(elements.miniChartContainer);
    }

    if (miniChart) {
        miniChart.destroy();
    }

    miniChart = new Chart(elements.miniChartContainer, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Price Trend',
                data: values,
                borderColor: '#00ff90',
                backgroundColor: 'rgba(0, 255, 144, 0.2)',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.2
            }]
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
                x: { display: false },
                y: { ticks: { color: '#f0f0f0' }, grid: { color: '#333' } }
            }
        }
    });
}

// --- States ---
function showLoadingState() {
    elements.loadingState.style.display = 'flex';
    elements.emptyState.style.display = 'none';
    elements.resultsContent.style.display = 'none';
    elements.errorState.style.display = 'none';
}

function showResultsState() {
    elements.loadingState.style.display = 'none';
    elements.emptyState.style.display = 'none';
    elements.resultsContent.style.display = 'block';
    elements.errorState.style.display = 'none';
}

function showError(message) {
    elements.loadingState.style.display = 'none';
    elements.emptyState.style.display = 'none';
    elements.resultsContent.style.display = 'none';
    elements.errorState.style.display = 'flex';
    elements.errorMessage.textContent = message;
}

// --- API Status ---
async function checkAPIStatus() {
    try {
        const res = await fetch(`${API_BASE_URL}/`);
        const data = await res.json();
        elements.apiStatus.querySelector('.status-dot').classList.add('connected');
        elements.apiStatus.querySelector('.status-text').textContent =
            `Connected • v${data.version}`;
    } catch (err) {
        elements.apiStatus.querySelector('.status-dot').classList.remove('connected');
        elements.apiStatus.querySelector('.status-text').textContent = 'Offline';
    }
}

// --- Last update ---
function updateLastUpdate() {
    const now = new Date();
    elements.lastUpdate.textContent = `Last updated: ${now.toLocaleTimeString()}`;
}

// --- Global button handlers ---
window.handlePredictClick = () => {
    const ticker = elements.tickerInput.value.toUpperCase();
    predictTicker(ticker);
};

window.handleResetClick = () => {
    clearPrediction();
};

// --- Clear prediction ---
function clearPrediction() {
    elements.tickerInput.value = '';
    hideSuggestions();
    elements.loadingState.style.display = 'none';
    elements.resultsContent.style.display = 'none';
    elements.errorState.style.display = 'none';
    elements.emptyState.style.display = 'block';
    elements.lastUpdate.textContent = '';
    elements.indicatorsTable.innerHTML = '';
    elements.signalText.textContent = '';
    elements.confidenceText.textContent = '';
    elements.strengthText.textContent = '';
    elements.dateRangeText.textContent = '';
    if (miniChart) miniChart.destroy();
}

// --- Initialize app ---
init();
