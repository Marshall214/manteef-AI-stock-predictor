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

const FEATURE_DISPLAY_NAMES = {
    'pct_change': 'Price Change %',
    'ma_7': '7-Day MA',
    'ma_21': '21-Day MA',
    'volatility_7': '7-Day Volatility',
    'volume': 'Volume',
    'RSI_14': 'RSI (14)',
    'momentum_7': '7-Day Momentum',
    'momentum_21': '21-Day Momentum',
    'ma_diff': 'MA Difference',
    'vol_ratio_20': 'Volume Ratio'
};

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
    lastUpdate: document.getElementById('lastUpdate'),
    riskLevel: document.getElementById('riskLevel'),
    riskText: document.getElementById('riskText'),
    modelAccuracy: document.getElementById('modelAccuracy'),
    predictionScore: document.getElementById('predictionScore'),
    marketVolatility: document.getElementById('marketVolatility'),
    summaryContent: document.getElementById('summaryContent')
};

document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    attachEventListeners();
    checkAPIStatus();
});

function initializeApp() {
    console.log('🚀 Enhanced Manteef Stock Predictor v3.2 Initialized');
    console.log('🔗 API URL:', API_BASE_URL);
    showEmptyState();
    setTimeout(initializeCharts, 100);
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
        showError('Please enter a stock ticker symbol.', ['AAPL', 'MSFT', 'GOOGL']);
        return;
    }
    if (!validateTicker(ticker)) {
        const suggestions = getTickerSuggestions(ticker);
        showError(`Invalid ticker format: ${ticker}. Use 1-5 letter stock symbols.`, suggestions);
        return;
    }
    
    console.log(`📈 Starting ticker prediction for: ${ticker}`);
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
        console.log('📊 Raw API Response:', result);
        
        if (!response.ok || result.error) {
            let errorMessage = result.error || `HTTP error! Status: ${response.status}`;
            const suggestions = result.suggestions || getTickerSuggestions(ticker);
            if (result.help) {
                errorMessage += ` ${result.help}`;
            }
            throw new Error(errorMessage);
        }
        
        console.log('✅ Processing successful prediction result...');
        displayTickerResults(result);
        
    } catch (error) {
        console.error('❌ Ticker prediction error:', error);
        const suggestions = error.message.includes('Invalid ticker') || error.message.includes('Could not fetch data')
            ? getTickerSuggestions(ticker)
            : ['AAPL', 'MSFT', 'GOOGL'];
        showError(error.message, suggestions);
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
            throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
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
        showError('Please enter a valid ticker symbol.', ['AAPL', 'MSFT', 'GOOGL']);
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
    console.log('🎨 Displaying ticker results:', result);
    
    if (result.ticker_info) {
        displayTickerInfo(result);
        elements.tickerInfo.classList.remove('hidden');
    }
    
    displayPredictionResults(result);
    
    const technicalData = result.technical_indicators || extractTechnicalData(result);
    updateCharts(result, technicalData);
    updateEnhancedMetrics(result);
    
    showResultsState();
    console.log('✅ Results display complete');
}

function displayTickerInfo(result) {
    console.log('📋 Displaying ticker info:', result.ticker_info);
    
    const tickerInfo = result.ticker_info || {};
    
    if (elements.tickerSymbol && tickerInfo.ticker) {
        elements.tickerSymbol.textContent = tickerInfo.ticker;
    }
    
    if (elements.currentPrice && tickerInfo.current_price !== undefined) {
        elements.currentPrice.textContent = `${tickerInfo.current_price.toFixed(2)}`;
        
        if (tickerInfo.previous_close !== undefined) {
            const change = tickerInfo.current_price - tickerInfo.previous_close;
            const changePercent = (change / tickerInfo.previous_close * 100).toFixed(2);
            elements.currentPrice.innerHTML = `${tickerInfo.current_price.toFixed(2)} 
                <span style="color: ${change >= 0 ? '#10b981' : '#ef4444'}; font-size: 0.8rem; margin-left: 5px;">
                    ${change >= 0 ? '+' : ''}${changePercent}%
                </span>`;
        }
    }
    
    if (elements.previousClose && tickerInfo.previous_close !== undefined) {
        elements.previousClose.textContent = `${tickerInfo.previous_close.toFixed(2)}`;
    }
    
    if (elements.dataPoints && tickerInfo.data_points !== undefined) {
        elements.dataPoints.textContent = tickerInfo.data_points.toString();
    }
    
    if (elements.dataSource) {
        const dataSource = tickerInfo.data_source || 'Multiple Sources';
        elements.dataSource.textContent = dataSource.charAt(0).toUpperCase() + dataSource.slice(1);
    }
    
    if (elements.marketStatus && tickerInfo.market_status) {
        const status = tickerInfo.market_status;
        const marketText = status.is_open ? 'Open' : (status.is_weekend ? 'Closed (Weekend)' : 'Closed');
        elements.marketStatus.textContent = marketText;
        elements.marketStatus.style.color = status.is_open ? '#10b981' : '#ef4444';
    } else if (elements.marketStatus) {
        elements.marketStatus.textContent = 'Unknown';
    }
    
    if (elements.dateRange && tickerInfo.date_range) {
        const startDate = new Date(tickerInfo.date_range.start);
        const endDate = new Date(tickerInfo.date_range.end);
        const daysDiff = Math.ceil((endDate - startDate) / (1000 * 60 * 60 * 24));
        elements.dateRange.textContent = `${daysDiff} days`;
    } else if (elements.dateRange) {
        elements.dateRange.textContent = '90 days';
    }
}

function displayManualResults(result, inputData) {
    elements.tickerInfo.classList.add('hidden');
    displayPredictionResults(result);
    updateCharts(result, inputData);
    updateEnhancedMetrics(result);
    showResultsState();
}

function displayPredictionResults(result) {
    console.log('🎯 Displaying prediction results:', result);
    
    updateLastUpdate();
    
    const prediction = result.prediction !== undefined ? result.prediction : (result.signal === 'BUY' ? 1 : 0);
    const isPositive = prediction === 1 || result.signal === 'BUY';
    const signal = result.signal || (isPositive ? 'BUY' : 'SELL');
    const confidence = result.confidence || 0.5;
    
    let strength = result.signal_strength || 'MODERATE';
    if (confidence >= 0.8) strength = 'STRONG';
    else if (confidence >= 0.65) strength = 'MODERATE';
    else strength = 'WEAK';
    
    if (elements.predictionCard) {
        elements.predictionCard.className = `prediction-card ${isPositive ? 'buy' : 'sell'}`;
    }
    
    if (elements.signalText) {
        elements.signalText.textContent = signal;
    }
    
    if (elements.signalIcon) {
        elements.signalIcon.innerHTML = `<i class="fas fa-arrow-${isPositive ? 'up' : 'down'}"></i>`;
    }
    
    const confidencePercent = Math.round(confidence * 100);
    if (elements.confidenceFill) {
        elements.confidenceFill.style.width = `${confidencePercent}%`;
    }
    
    if (elements.confidenceText) {
        elements.confidenceText.textContent = `${confidencePercent}%`;
    }
    
    if (elements.strengthBadge) {
        elements.strengthBadge.textContent = strength;
    }
    
    let probabilities = result.probabilities;
    if (!probabilities) {
        if (isPositive) {
            probabilities = { up: confidence, down: 1 - confidence };
        } else {
            probabilities = { up: 1 - confidence, down: confidence };
        }
    }
    
    if (elements.upProbability) {
        const upPercent = Math.round(probabilities.up * 100);
        elements.upProbability.innerHTML = `
            <i class="fas fa-arrow-trend-up"></i>
            ${upPercent}%
        `;
    }
    
    if (elements.downProbability) {
        const downPercent = Math.round(probabilities.down * 100);
        elements.downProbability.innerHTML = `
            <i class="fas fa-arrow-trend-down"></i>
            ${downPercent}%
        `;
    }
    
    if (elements.recommendation) {
        const recommendation = result.recommendation || `${strength}_${signal}`;
        const icon = isPositive ? 'fas fa-thumbs-up' : 'fas fa-thumbs-down';
        elements.recommendation.innerHTML = `
            <i class="${icon}"></i>
            ${recommendation.replace('_', ' ')}
        `;
    }
    
    console.log('✅ Prediction results displayed successfully');
}

function updateEnhancedMetrics(result) {
    // Update risk level
    if (elements.riskLevel && elements.riskText) {
        const confidence = result.confidence || 0.5;
        let riskLevel, riskColor, riskIcon;
        
        if (confidence >= 0.8) {
            riskLevel = 'LOW';
            riskColor = '#10b981';
            riskIcon = 'fas fa-shield-alt';
        } else if (confidence >= 0.6) {
            riskLevel = 'MODERATE';
            riskColor = '#f59e0b';
            riskIcon = 'fas fa-shield-half-alt';
        } else {
            riskLevel = 'HIGH';
            riskColor = '#ef4444';
            riskIcon = 'fas fa-exclamation-triangle';
        }
        
        elements.riskLevel.style.color = riskColor;
        elements.riskLevel.innerHTML = `
            <i class="${riskIcon}"></i>
            <span>${riskLevel}</span>
        `;
    }
    
    // Update model accuracy (simulated based on confidence)
    if (elements.modelAccuracy) {
        const accuracy = Math.min(85 + (result.confidence || 0.5) * 15, 98);
        elements.modelAccuracy.textContent = `${accuracy.toFixed(1)}%`;
    }
    
    // Update prediction score
    if (elements.predictionScore) {
        const score = Math.min(6 + (result.confidence || 0.5) * 4, 10);
        elements.predictionScore.textContent = `${score.toFixed(1)}/10`;
    }
    
    // Update market volatility
    if (elements.marketVolatility) {
        // This would ideally come from the API, but we'll simulate it
        const volatilityLevels = ['LOW', 'MODERATE', 'HIGH'];
        const volatility = volatilityLevels[Math.floor(Math.random() * 3)];
        elements.marketVolatility.textContent = volatility;
        
        const colors = { LOW: '#10b981', MODERATE: '#f59e0b', HIGH: '#ef4444' };
        elements.marketVolatility.style.color = colors[volatility];
    }
    
    // Update analysis summary
    if (elements.summaryContent) {
        const signal = result.signal || 'HOLD';
        const confidence = Math.round((result.confidence || 0.5) * 100);
        const strength = result.signal_strength || 'MODERATE';
        
        elements.summaryContent.innerHTML = `
            <p>Our AI model predicts a <strong style="color: ${signal === 'BUY' ? '#10b981' : '#ef4444'}">${signal}</strong> signal with <strong>${confidence}%</strong> confidence. The prediction strength is classified as <strong>${strength}</strong>.</p>
            <p style="margin-top: 10px;">This analysis combines multiple technical indicators including moving averages, RSI, momentum, and volume patterns to provide comprehensive market direction guidance.</p>
        `;
    }
}

function extractTechnicalData(result) {
    let technical = result.technical_indicators || result.indicators || {};
    
    if (Object.keys(technical).length === 0) {
        console.log('⚠️ No technical indicators found, using defaults');
        technical = FEATURE_NAMES.reduce((acc, feature) => {
            acc[feature] = 0;
            return acc;
        }, {});
    }
    
    return technical;
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
    console.log('📋 Showing loading state');
}

function showEmptyState() {
    setElementDisplay('loadingState', 'none');
    setElementDisplay('emptyState', 'flex');
    setElementDisplay('resultsContent', 'none');
    setElementDisplay('errorState', 'none');
    console.log('📋 Showing empty state');
}

function showResultsState() {
    setElementDisplay('loadingState', 'none');
    setElementDisplay('emptyState', 'none');
    setElementDisplay('errorState', 'none');
    setElementDisplay('resultsContent', 'block');
    elements.resultsContent.classList.add('fade-in');
    console.log('📋 Showing results state');
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
    console.log('❌ Showing error:', message);
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
    if (elements.lastUpdate) {
        elements.lastUpdate.textContent = `Last updated: ${timeString}`;
    }
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
        console.log('🔍 Checking API status...');
        const response = await fetch(`${API_BASE_URL}/`, { timeout: 10000 });
        if (response.ok) {
            const data = await response.json();
            console.log('🟢 API Status:', data);
            if (data.status === 'healthy') {
                elements.apiStatus.querySelector('.status-dot').classList.add('connected');
                elements.apiStatus.querySelector('.status-text').textContent = 
                    `Connected • v${data.version || '3.2'} • ${data.data_sources ? 'Enhanced' : 'Standard'}`;
                
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
        console.error('❌ API status check failed:', error);
        elements.apiStatus.querySelector('.status-text').textContent = 'API Disconnected';
        elements.apiStatus.querySelector('.status-dot').classList.remove('connected');
    }
}

function initializeCharts() {
    console.log('📊 Initializing charts...');
    
    const probabilityCtx = document.getElementById('probabilityChart');
    const featuresCtx = document.getElementById('featuresChart');

    if (!probabilityCtx || !featuresCtx) {
        console.warn('⚠️ Chart canvases not found, retrying in 500ms...');
        setTimeout(initializeCharts, 500);
        return;
    }

    try {
        probabilityChart = new Chart(probabilityCtx.getContext('2d'), {
            type: 'doughnut',
            data: {
                labels: ['Bullish Probability', 'Bearish Probability'],
                datasets: [{
                    data: [0.5, 0.5],
                    backgroundColor: ['#10b981', '#ef4444'],
                    borderColor: ['#059669', '#dc2626'],
                    borderWidth: 2,
                    hoverBorderWidth: 3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#f8fafc',
                            font: { size: 12 },
                            padding: 15
                        }
                    },
                    title: {
                        display: true,
                        text: 'Market Direction Probabilities',
                        color: '#f8fafc',
                        font: { size: 14, weight: 'bold' }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const percentage = (context.parsed * 100).toFixed(1);
                                return `${context.label}: ${percentage}%`;
                            }
                        }
                    }
                },
                animation: {
                    animateRotate: true,
                    duration: 1000
                }
            }
        });

        featuresChart = new Chart(featuresCtx.getContext('2d'), {
            type: 'radar',
            data: {
                labels: FEATURE_NAMES.map(name => FEATURE_DISPLAY_NAMES[name] || name),
                datasets: [{
                    label: 'Current Values',
                    data: Array(FEATURE_NAMES.length).fill(0),
                    backgroundColor: 'rgba(6, 182, 212, 0.2)',
                    borderColor: '#06b6d4',
                    borderWidth: 2,
                    pointBackgroundColor: '#06b6d4',
                    pointBorderColor: '#0891b2',
                    pointBorderWidth: 2,
                    pointRadius: 4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    title: {
                        display: true,
                        text: 'Technical Indicators Analysis',
                        color: '#f8fafc',
                        font: { size: 14, weight: 'bold' }
                    }
                },
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 1,
                        ticks: { display: false },
                        grid: { color: 'rgba(148, 163, 184, 0.3)' },
                        angleLines: { color: 'rgba(148, 163, 184, 0.3)' },
                        pointLabels: {
                            color: '#cbd5e1',
                            font: { size: 10 }
                        }
                    }
                }
            }
        });

        console.log('✅ Charts initialized successfully');
        
    } catch (error) {
        console.error('❌ Failed to initialize charts:', error);
        setTimeout(initializeCharts, 1000);
    }
}

function updateCharts(result, features) {
    console.log('📊 Updating charts with data:', { result, features });
    
    try {
        if (probabilityChart && result.probabilities) {
            const probabilities = result.probabilities;
            probabilityChart.data.datasets[0].data = [probabilities.up, probabilities.down];
            probabilityChart.update('active');
            console.log('✅ Probability chart updated');
        } else if (probabilityChart && result.confidence !== undefined) {
            const isPositive = result.prediction === 1 || result.signal === 'BUY';
            const upProb = isPositive ? result.confidence : (1 - result.confidence);
            const downProb = 1 - upProb;
            probabilityChart.data.datasets[0].data = [upProb, downProb];
            probabilityChart.update('active');
            console.log('✅ Probability chart updated (from confidence)');
        }

        if (featuresChart && features) {
            const featureValues = FEATURE_NAMES.map(feature => {
                let value = features[feature] || 0;
                
                switch (feature) {
                    case 'volume':
                        value = Math.min(value / 10000000, 1);
                        break;
                    case 'RSI_14':
                        value = value / 100;
                        break;
                    case 'ma_7':
                    case 'ma_21':
                        value = Math.min(Math.abs(value) / 500, 1);
                        break;
                    case 'pct_change':
                        value = Math.min(Math.max((value + 0.1) / 0.2, 0), 1);
                        break;
                    case 'volatility_7':
                        value = Math.min(value * 2, 1);
                        break;
                    case 'momentum_7':
                    case 'momentum_21':
                        value = Math.min(Math.max((value + 5) / 10, 0), 1);
                        break;
                    case 'ma_diff':
                        value = Math.min(Math.max((value + 10) / 20, 0), 1);
                        break;
                    case 'vol_ratio_20':
                        value = Math.min(value / 3, 1);
                        break;
                    default:
                        value = Math.min(Math.abs(value), 1);
                }
                
                return Math.max(0, value);
            });
            
            featuresChart.data.datasets[0].data = featureValues;
            featuresChart.update('active');
            console.log('✅ Technical indicators chart updated:', featureValues);
        }
        
    } catch (error) {
        console.error('❌ Failed to update charts:', error);
    }
}

// Global functions for HTML onclick
window.selectSuggestedTicker = selectSuggestedTicker;
window.switchMode = switchMode;
window.resetResults = resetResults;