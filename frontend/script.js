const API_BASE_URL = window.location.hostname === 'localhost' 
    ? 'http://localhost:5000' 
    : 'https://manteef-ai-stock-predictor-predictor.up.railway.app';

let probabilityChart = null;
let featuresChart = null;
let currentMode = 'ticker';

const FEATURE_NAMES = [
    'pct_change', 'ma_7', 'ma_21', 'volatility_7', 'volume',
    'RSI_14', 'momentum_7', 'momentum_21', 'ma_diff', 'vol_ratio_20'
];

const SAMPLE_DATA = {
    bullish: {
        pct_change: 0.025, ma_7: 46.50, ma_21: 45.20, volatility_7: 0.12, volume: 3200000,
        RSI_14: 72.3, momentum_7: 1.5, momentum_21: 2.1, ma_diff: 1.30, vol_ratio_20: 1.45
    },
    bearish: {
        pct_change: -0.018, ma_7: 44.20, ma_21: 45.60, volatility_7: 0.22, volume: 4100000,
        RSI_14: 28.7, momentum_7: -0.9, momentum_21: -1.4, ma_diff: -1.40, vol_ratio_20: 1.85
    },
    neutral: {
        pct_change: 0.001, ma_7: 45.08, ma_21: 45.05, volatility_7: 0.08, volume: 1800000,
        RSI_14: 51.2, momentum_7: 0.2, momentum_21: 0.1, ma_diff: 0.03, vol_ratio_20: 0.95
    }
};

// Popular tickers for suggestions
const POPULAR_TICKERS = [
    { ticker: 'AAPL', name: 'Apple Inc.' },
    { ticker: 'MSFT', name: 'Microsoft Corp.' },
    { ticker: 'GOOGL', name: 'Alphabet Inc.' },
    { ticker: 'TSLA', name: 'Tesla Inc.' },
    { ticker: 'NVDA', name: 'NVIDIA Corp.' },
    { ticker: 'AMZN', name: 'Amazon.com Inc.' },
    { ticker: 'META', name: 'Meta Platforms' },
    { ticker: 'AMD', name: 'Advanced Micro Devices' },
    { ticker: 'NFLX', name: 'Netflix Inc.' },
    { ticker: 'BABA', name: 'Alibaba Group' },
    { ticker: 'DIS', name: 'The Walt Disney Co.' },
    { ticker: 'PYPL', name: 'PayPal Holdings' },
    { ticker: 'INTC', name: 'Intel Corp.' },
    { ticker: 'CRM', name: 'Salesforce Inc.' },
    { ticker: 'UBER', name: 'Uber Technologies' }
];

const elements = {
    form: document.getElementById('predictionForm'),
    tickerInput: document.getElementById('tickerInput'),
    tickerSuggestions: document.getElementById('tickerSuggestions'),
    tickerInfoBtn: document.getElementById('tickerInfoBtn'),
    predictTickerBtn: document.getElementById('predictTickerBtn'),
    clearTickerBtn: document.getElementById('clearTickerBtn'),
    predictBtn: document.getElementById('predictBtn'),
    clearBtn: document.getElementById('clearBtn'),
    sampleBtn: document.getElementById('sampleBtn'),
    tickerInputSection: document.getElementById('tickerInputSection'),
    manualInputSection: document.getElementById('manualInputSection'),
    tickerMode: document.getElementById('tickerMode'),
    manualMode: document.getElementById('manualMode'),
    apiStatus: document.getElementById('apiStatus'),
    loadingState: document.getElementById('loadingState'),
    emptyState: document.getElementById('emptyState'),
    resultsContent: document.getElementById('resultsContent'),
    errorState: document.getElementById('errorState'),
    errorMessage: document.getElementById('errorMessage'),
    errorSuggestions: document.getElementById('errorSuggestions'),
    suggestedTickers: document.getElementById('suggestedTickers'),
    tickerInfo: document.getElementById('tickerInfo'),
    tickerSymbol: document.getElementById('tickerSymbol'),
    currentPrice: document.getElementById('currentPrice'),
    previousClose: document.getElementById('previousClose'),
    dataPoints: document.getElementById('dataPoints'),
    dateRange: document.getElementById('dateRange'),
    dataSource: document.getElementById('dataSource'),
    marketStatus: document.getElementById('marketStatus'),
    predictionCard: document.getElementById('predictionCard'),
    signalText: document.getElementById('signalText'),
    signalIcon: document.getElementById('signalIcon'),
    confidenceFill: document.getElementById('confidenceFill'),
    confidenceText: document.getElementById('confidenceText'),
    strengthBadge: document.getElementById('strengthBadge'),
    upProbability: document.getElementById('upProbability'),
    downProbability: document.getElementById('downProbability'),
    recommendation: document.getElementById('recommendation'),
    lastUpdate: document.getElementById('lastUpdate')
};

document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    attachEventListeners();
    checkAPIStatus();
});

function initializeApp() {
    console.log('🚀 Enhanced Manteef Stock Predictor v3.0 Initialized');
    console.log('🔗 API URL:', API_BASE_URL);
    showEmptyState();
    initializeCharts();
}

function attachEventListeners() {
    elements.form.addEventListener('submit', handleFormSubmission);
    elements.tickerInfoBtn.addEventListener('click', getTickerInfo);
    elements.clearTickerBtn.addEventListener('click', clearTickerForm);
    elements.clearBtn.addEventListener('click', clearManualForm);
    elements.sampleBtn.addEventListener('click', fillSampleData);
    elements.tickerInput.addEventListener('input', handleTickerInput);
    elements.tickerInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            e.preventDefault();
            handleFormSubmission(e);
        }
        if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
            e.preventDefault();
            navigateSuggestions(e.key === 'ArrowDown' ? 1 : -1);
        }
        if (e.key === 'Escape') {
            hideSuggestions();
        }
    });
    
    // Click outside to hide suggestions
    document.addEventListener('click', function(e) {
        if (!elements.tickerInput.contains(e.target) && !elements.tickerSuggestions.contains(e.target)) {
            hideSuggestions();
        }
    });
    
    FEATURE_NAMES.forEach(feature => {
        const input = document.getElementById(feature);
        if (input) {
            input.addEventListener('input', debounce(validateInput, 300));
            input.addEventListener('blur', validateInput);
        }
    });
}

function handleTickerInput(event) {
    const input = event.target;
    const value = input.value.toUpperCase().replace(/[^A-Z]/g, '');
    input.value = value;
    
    if (value.length > 0) {
        showTickerSuggestions(value);
    } else {
        hideSuggestions();
    }
}

function showTickerSuggestions(query) {
    const suggestions = POPULAR_TICKERS.filter(item => 
        item.ticker.startsWith(query) || item.ticker.includes(query)
    ).slice(0, 8);
    
    if (suggestions.length === 0) {
        hideSuggestions();
        return;
    }
    
    elements.tickerSuggestions.innerHTML = suggestions.map(item => `
        <div class="suggestion-item" data-ticker="${item.ticker}">
            <span class="suggestion-ticker">${item.ticker}</span>
            <span class="suggestion-name">${item.name}</span>
        </div>
    `).join('');
    
    // Add click event listeners to suggestions
    elements.tickerSuggestions.querySelectorAll('.suggestion-item').forEach(item => {
        item.addEventListener('click', function() {
            const ticker = this.dataset.ticker;
            elements.tickerInput.value = ticker;
            hideSuggestions();
            elements.tickerInput.focus();
        });
    });
    
    elements.tickerSuggestions.classList.remove('hidden');
}

function hideSuggestions() {
    elements.tickerSuggestions.classList.add('hidden');
}

function navigateSuggestions(direction) {
    const suggestions = elements.tickerSuggestions.querySelectorAll('.suggestion-item');
    if (suggestions.length === 0) return;
    
    const current = elements.tickerSuggestions.querySelector('.suggestion-item.highlighted');
    let index = -1;
    
    if (current) {
        index = Array.from(suggestions).indexOf(current);
        current.classList.remove('highlighted');
    }
    
    index += direction;
    if (index < 0) index = suggestions.length - 1;
    if (index >= suggestions.length) index = 0;
    
    suggestions[index].classList.add('highlighted');
    elements.tickerInput.value = suggestions[index].dataset.ticker;
}

function switchMode(mode) {
    currentMode = mode;
    elements.tickerMode.classList.toggle('active', mode === 'ticker');
    elements.manualMode.classList.toggle('active', mode === 'manual');
    elements.tickerInputSection.classList.toggle('hidden', mode !== 'ticker');
    elements.manualInputSection.classList.toggle('hidden', mode !== 'manual');
    clearTickerForm();
    clearManualForm();
    showEmptyState();
    hideSuggestions();
    console.log(`Switched to ${mode} mode`);
}

async function handleFormSubmission(event) {
    event.preventDefault();
    hideSuggestions();
    if (currentMode === 'ticker') {
        await handleTickerPrediction();
    } else {
        await handleManualPrediction();
    }
}

async function handleTickerPrediction() {
    const ticker = elements.tickerInput.value.trim().toUpperCase();
    if (!ticker) {
        showError('Please enter a stock ticker symbol.', []);
        return;
    }
    if (!validateTicker(ticker)) {
        showError('Invalid ticker format. Use 1-5 letter stock symbols (e.g., AAPL, TSLA).', ['AAPL', 'MSFT', 'GOOGL']);
        return;
    }
    
    showLoadingState();
    setButtonLoading(elements.predictTickerBtn, true);
    
    try {
        const response = await fetch(`${API_BASE_URL}/predict-ticker`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({ ticker: ticker })
        });
        
        const result = await response.json();
        
        if (!response.ok || result.error) {
            throw new Error(result.error || `HTTP error! status: ${response.status}`);
        }
        
        displayTickerResults(result);
        console.log('✅ Ticker prediction successful:', result);
        
    } catch (error) {
        console.error('Ticker prediction error:', error);
        
        let errorMessage = 'Ticker prediction failed. Please try again.';
        let suggestions = [];
        
        if (error.message.includes('Insufficient data') || error.message.includes('Not enough')) {
            errorMessage = `Unable to fetch sufficient data for ${ticker}. This might be a newly listed or delisted stock.`;
            suggestions = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'];
        } else if (error.message.includes('Invalid ticker') || error.message.includes('No data available')) {
            errorMessage = `Invalid ticker symbol: ${ticker}. Please verify the symbol and try again.`;
            suggestions = getTickerSuggestions(ticker);
        } else if (error.message.includes('failed')) {
            // Try to extract suggestions from server response
            try {
                const response = await fetch(`${API_BASE_URL}/predict-ticker`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ ticker: ticker })
                });
                const errorData = await response.json();
                if (errorData.suggestions) {
                    suggestions = errorData.suggestions;
                }
            } catch (e) {
                // Fallback suggestions
                suggestions = getTickerSuggestions(ticker);
            }
            errorMessage = error.message;
        } else {
            errorMessage = error.message || 'An unexpected error occurred. Please try again.';
            suggestions = ['AAPL', 'MSFT', 'GOOGL'];
        }
        
        showError(errorMessage, suggestions);
        
    } finally {
        setButtonLoading(elements.predictTickerBtn, false);
    }
}

async function handleManualPrediction() {
    if (!validateManualForm()) {
        showError('Please fill in all required fields with valid numerical values.', []);
        return;
    }
    const formData = getManualFormData();
    showLoadingState();
    setButtonLoading(elements.predictBtn, true);
    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify(formData)
        });
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
        }
        const result = await response.json();
        if (result.error) {
            throw new Error(result.error);
        }
        displayManualResults(result, formData);
        console.log('✅ Manual prediction successful:', result);
    } catch (error) {
        console.error('Manual prediction error:', error);
        showError(error.message || 'Manual prediction failed. Please try again.', []);
    } finally {
        setButtonLoading(elements.predictBtn, false);
    }
}

async function getTickerInfo() {
    const ticker = elements.tickerInput.value.trim().toUpperCase();
    if (!ticker || !validateTicker(ticker)) {
        showError('Please enter a valid ticker symbol first.', ['AAPL', 'MSFT', 'GOOGL']);
        return;
    }
    setButtonLoading(elements.tickerInfoBtn, true);
    try {
        const response = await fetch(`${API_BASE_URL}/ticker-info`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({ ticker: ticker })
        });
        const data = await response.json();
        if (!response.ok || data.error) {
            throw new Error(data.error || `Failed to fetch data. Please try again later.`);
        }
        displayTickerInfo(data);
        console.log('✅ Ticker info retrieved:', data);
    } catch (error) {
        console.error('Ticker info error:', error);
        const suggestions = error.message.includes('Invalid') ? getTickerSuggestions(ticker) : [];
        showError(error.message || 'Failed to get ticker information.', suggestions);
    } finally {
        setButtonLoading(elements.tickerInfoBtn, false);
    }
}

function getTickerSuggestions(ticker) {
    if (!ticker) return ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'];
    
    const firstLetter = ticker[0].toUpperCase();
    const suggestions = POPULAR_TICKERS
        .filter(item => item.ticker.startsWith(firstLetter))
        .slice(0, 3)
        .map(item => item.ticker);
    
    return suggestions.length > 0 ? suggestions : ['AAPL', 'MSFT', 'GOOGL'];
}

function displayTickerResults(result) {
    displayTickerInfo(result);
    displayPredictionResults(result);
    updateCharts(result, result.technical_indicators);
    elements.tickerInfo.classList.remove('hidden');
    showResultsState();
}

function displayTickerInfo(result) {
    const tickerInfo = result.ticker_info;
    elements.tickerSymbol.textContent = tickerInfo.ticker;
    elements.currentPrice.textContent = `${tickerInfo.current_price.toFixed(2)}`;
    elements.previousClose.textContent = `${tickerInfo.previous_close.toFixed(2)}`;
    elements.dataPoints.textContent = tickerInfo.data_points.toString();
    
    // Display data source
    const dataSource = tickerInfo.data_source || 'Unknown';
    elements.dataSource.textContent = dataSource.charAt(0).toUpperCase() + dataSource.slice(1);
    
    // Display market status
    if (tickerInfo.market_status) {
        const status = tickerInfo.market_status;
        const marketText = status.is_open ? 'Open' : (status.is_weekend ? 'Closed (Weekend)' : 'Closed');
        elements.marketStatus.textContent = marketText;
        elements.marketStatus.style.color = status.is_open ? '#10b981' : '#ef4444';
    } else {
        elements.marketStatus.textContent = 'Unknown';
    }
    
    // Format date range
    const startDate = new Date(tickerInfo.date_range.start);
    const endDate = new Date(tickerInfo.date_range.end);
    const daysDiff = Math.ceil((endDate - startDate) / (1000 * 60 * 60 * 24));
    elements.dateRange.textContent = `${daysDiff} days`;
}

function displayManualResults(result, inputData) {
    elements.tickerInfo.classList.add('hidden');
    displayPredictionResults(result);
    updateCharts(result, inputData);
    showResultsState();
}

function displayPredictionResults(result) {
    updateLastUpdate();
    const isPositive = result.prediction === 1;
    const signal = result.signal || (isPositive ? 'BUY' : 'SELL');
    const confidence = result.confidence || 0.5;
    const strength = result.signal_strength || 'UNKNOWN';
    
    elements.predictionCard.className = `prediction-card ${isPositive ? 'buy' : 'sell'}`;
    elements.signalText.textContent = signal;
    elements.signalIcon.innerHTML = `<i class="fas fa-arrow-${isPositive ? 'up' : 'down'}"></i>`;
    
    const confidencePercent = Math.round(confidence * 100);
    elements.confidenceFill.style.width = `${confidencePercent}%`;
    elements.confidenceText.textContent = `${confidencePercent}%`;
    elements.strengthBadge.textContent = strength;
    
    const probabilities = result.probabilities || { up: confidence, down: 1 - confidence };
    elements.upProbability.textContent = `${Math.round(probabilities.up * 100)}%`;
    elements.downProbability.textContent = `${Math.round(probabilities.down * 100)}%`;
    elements.recommendation.textContent = result.recommendation || `${strength}_${signal}`;
}

function validateTicker(ticker) {
    return ticker && ticker.length >= 1 && ticker.length <= 5 && /^[A-Z]+$/.test(ticker);
}

function validateManualForm() {
    let isValid = true;
    FEATURE_NAMES.forEach(feature => {
        const input = document.getElementById(feature);
        if (!input) return;
        const group = input.closest('.input-group');
        const value = input.value.trim();
        if (!value || isNaN(parseFloat(value))) {
            if (group) {
                group.classList.add('error');
                group.classList.remove('success');
            }
            isValid = false;
        } else {
            if (group) {
                group.classList.remove('error');
                group.classList.add('success');
            }
        }
    });
    return isValid;
}

function validateInput(event) {
    const input = event.target;
    const group = input.closest('.input-group');
    const value = input.value.trim();
    if (!value || isNaN(parseFloat(value))) {
        group.classList.add('error');
        group.classList.remove('success');
    } else {
        group.classList.remove('error');
        group.classList.add('success');
    }
}

function getManualFormData() {
    const formData = {};
    FEATURE_NAMES.forEach(feature => {
        const input = document.getElementById(feature);
        if (input && input.value.trim()) {
            const value = parseFloat(input.value);
            formData[feature] = isNaN(value) ? 0 : value;
        } else {
            formData[feature] = 0;
        }
    });
    return formData;
}

function clearTickerForm() {
    elements.tickerInput.value = '';
    hideSuggestions();
}

function clearManualForm() {
    FEATURE_NAMES.forEach(feature => {
        const input = document.getElementById(feature);
        if (input) {
            input.value = '';
            const group = input.closest('.input-group');
            if (group) {
                group.classList.remove('error', 'success');
            }
        }
    });
}

function fillSampleData() {
    const samples = Object.keys(SAMPLE_DATA);
    const randomSample = SAMPLE_DATA[samples[Math.floor(Math.random() * samples.length)]];
    FEATURE_NAMES.forEach(feature => {
        const input = document.getElementById(feature);
        if (input && randomSample[feature] !== undefined) {
            input.value = randomSample[feature];
            const group = input.closest('.input-group');
            if (group) {
                group.classList.remove('error');
                group.classList.add('success');
            }
        }
    });
}

function showLoadingState() {
    setElementDisplay('loadingState', 'flex');
    setElementDisplay('emptyState', 'none');
    setElementDisplay('resultsContent', 'none');
    setElementDisplay('errorState', 'none');
    hideSuggestions();
}

function showEmptyState() {
    setElementDisplay('loadingState', 'none');
    setElementDisplay('emptyState', 'flex');
    setElementDisplay('resultsContent', 'none');
    setElementDisplay('errorState', 'none');
}

function showResultsState() {
    setElementDisplay('loadingState', 'none');
    setElementDisplay('emptyState', 'none');
    setElementDisplay('errorState', 'none');
    setElementDisplay('resultsContent', 'block');
    elements.resultsContent.classList.add('fade-in');
}

function showError(message, suggestions = []) {
    setElementDisplay('loadingState', 'none');
    setElementDisplay('emptyState', 'none');
    setElementDisplay('resultsContent', 'none');
    setElementDisplay('errorState', 'flex');
    elements.errorMessage.textContent = message;
    
    if (suggestions && suggestions.length > 0) {
        elements.errorSuggestions.classList.remove('hidden');
        elements.suggestedTickers.innerHTML = suggestions.map(ticker => 
            `<span class="suggested-ticker" onclick="selectSuggestedTicker('${ticker}')">${ticker}</span>`
        ).join('');
    } else {
        elements.errorSuggestions.classList.add('hidden');
    }
    
    hideSuggestions();
}

function selectSuggestedTicker(ticker) {
    elements.tickerInput.value = ticker;
    resetResults();
    elements.tickerInput.focus();
}

function setElementDisplay(elementKey, display) {
    if (elements[elementKey]) {
        elements[elementKey].style.display = display;
    }
}

function setButtonLoading(button, loading) {
    if (!button) return;
    const span = button.querySelector('span');
    const loader = button.querySelector('.btn-loader');
    if (loading) {
        button.classList.add('loading');
        button.disabled = true;
        if (span) span.style.opacity = '0';
        if (loader) loader.style.display = 'block';
    } else {
        button.classList.remove('loading');
        button.disabled = false;
        if (span) span.style.opacity = '1';
        if (loader) loader.style.display = 'none';
    }
}

function updateLastUpdate() {
    const now = new Date();
    const timeString = now.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
    elements.lastUpdate.textContent = `Last updated: ${timeString}`;
}

function resetResults() {
    showEmptyState();
    if (currentMode === 'ticker') {
        clearTickerForm();
    } else {
        clearManualForm();
    }
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

async function checkAPIStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/`, { timeout: 10000 });
        if (response.ok) {
            const data = await response.json();
            if (data.status === 'healthy') {
                elements.apiStatus.querySelector('.status-dot').classList.add('connected');
                elements.apiStatus.querySelector('.status-text').textContent = 
                    `Connected • v${data.version || '3.0'} • ${data.data_sources ? 'Enhanced' : 'Standard'}`;
                
                // Show data source info if available
                if (data.data_sources && data.data_sources.length > 0) {
                    console.log('📊 Data Sources Available:', data.data_sources);
                }
            } else {
                throw new Error('API not healthy');
            }
        } else {
            throw new Error('API connection failed');
        }
    } catch (error) {
        console.error('API status check failed:', error);
        elements.apiStatus.querySelector('.status-text').textContent = 'API Disconnected';
        elements.apiStatus.querySelector('.status-dot').classList.remove('connected');
    }
}

function initializeCharts() {
    const probabilityCtx = document.getElementById('probabilityChart').getContext('2d');
    const featuresCtx = document.getElementById('featuresChart').getContext('2d');

    probabilityChart = new Chart(probabilityCtx, {
        type: 'bar',
        data: {
            labels: ['Up', 'Down'],
            datasets: [{
                label: 'Prediction Probabilities',
                data: [0, 0],
                backgroundColor: ['#10b981', '#ef4444'],
                borderColor: ['#10b981', '#ef4444'],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    title: {
                        display: true,
                        text: 'Probability',
                        color: '#f8fafc'
                    },
                    ticks: {
                        color: '#94a3b8'
                    }
                },
                x: {
                    ticks: {
                        color: '#94a3b8'
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Prediction Probabilities',
                    color: '#f8fafc'
                }
            }
        }
    });

    featuresChart = new Chart(featuresCtx, {
        type: 'bar',
        data: {
            labels: FEATURE_NAMES,
            datasets: [{
                label: 'Feature Values',
                data: Array(FEATURE_NAMES.length).fill(0),
                backgroundColor: '#06b6d4',
                borderColor: '#06b6d4',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Normalized Value',
                        color: '#f8fafc'
                    },
                    ticks: {
                        color: '#94a3b8'
                    }
                },
                x: {
                    ticks: {
                        color: '#94a3b8',
                        maxRotation: 45,
                        minRotation: 45
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Technical Indicators',
                    color: '#f8fafc'
                }
            }
        }
    });
}

function updateCharts(result, features) {
    const probabilities = result.probabilities || { up: 0.5, down: 0.5 };
    probabilityChart.data.datasets[0].data = [probabilities.up, probabilities.down];
    probabilityChart.update();

    const featureValues = FEATURE_NAMES.map(feature => {
        const value = features[feature] || 0;
        if (feature === 'volume') {
            return value / 1000000; // Normalize volume
        } else if (feature === 'RSI_14') {
            return value / 100; // Normalize RSI
        } else if (['ma_7', 'ma_21'].includes(feature)) {
            return value / 100; // Normalize MAs
        }
        return Math.abs(value); // Use absolute value for others
    });
    featuresChart.data.datasets[0].data = featureValues;
    featuresChart.update();
}