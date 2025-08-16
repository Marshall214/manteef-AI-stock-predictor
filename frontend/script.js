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
    signalStrengthText: document.getElementById('signalStrengthText'),
    indicatorsTable: document.getElementById('indicatorsTable'),
    dateRangeText: document.getElementById('dateRangeText'),
};

// --- Feature list ---
let FEATURE_NAMES = [];

// --- Initialize ---
async function init() {
    await fetchFeatures();
    checkAPIStatus();
}

// --- Fetch feature names ---
async function fetchFeatures() {
    try {
        const res = await fetch(`${API_BASE_URL}/features`);
        const data = await res.json();
        FEATURE_NAMES = data.features || [];
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

    // Prediction signal
    elements.signalText.textContent = data.signal;
    elements.confidenceText.textContent = `Confidence: ${data.confidence}`;
    elements.signalStrengthText.textContent = `Strength: ${data.signal_strength}`;

    // Date range
    elements.dateRangeText.textContent = `Data Range: ${data.date_range.start} → ${data.date_range.end}`;

    // Technical indicators table
    if (data.features) {
        let tableRows = Object.entries(data.features)
            .map(([key, val]) => `<tr><td>${key}</td><td>${val.toFixed(3)}</td></tr>`)
            .join('');
        elements.indicatorsTable.innerHTML = tableRows;
    }

    updateLastUpdate();
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

// --- Init app ---
init();

// --- Expose for HTML onclick ---
window.predictTicker = () => predictTicker(elements.tickerInput.value.toUpperCase());
