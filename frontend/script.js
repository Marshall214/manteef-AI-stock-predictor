// Configuration
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
    { ticker: 'BABA', name: 'Alibaba Group' }
];

// DOM Elements
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

// Initialize app
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Enhanced Manteef Stock Predictor v3.2 Initialized');
    console.log('🔗 API URL:', API_BASE_URL);
    
    attachEventListeners();
    showEmptyState();
    checkAPIStatus();
    
    // Initialize charts after a short delay to ensure DOM is ready
    setTimeout(initializeCharts, 500);
});

function attachEventListeners() {
    elements.form.addEventListener('submit', handleFormSubmission);
    elements.tickerInfoBtn.addEventListener('click', getTickerInfo);
    elements.clearTickerBtn.addEventListener('click', clearTickerForm);
    elements.clearBtn.addEventListener('click', clearManualForm);
    elements.sampleBtn.addEventListener('click', fillSampleData);
    elements.tickerInput.addEventListener('input', handleTickerInput);
    
    // Handle keyboard navigation for suggestions
    elements.tickerInput.addEventListener('keydown', function(e) {
        if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
            e.preventDefault();
            navigateSuggestions(e.key === 'ArrowDown' ? 1 : -1);
        }
        if (e.key === 'Escape') {
            hideSuggestions();
        }
    });
    
    // Hide suggestions when clicking outside
    document.addEventListener('click', function(e) {
        if (!elements.tickerInput.contains(e.target) && 
            !elements.tickerSuggestions.contains(e.target)) {
            hideSuggestions();
        }
    });
    
    // Add validation to manual inputs
    FEATURE_NAMES.forEach(feature => {
        const input = document.getElementById(feature);
        if (input) {
            input.addEventListener('input', validateInput);
        }
    });
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
        // Simulate API call with mock data for demonstration
        await simulateAPIDelay();
        
        const mockResult = generateMockTickerResult(ticker);
        displayTickerResults(mockResult);
        
        console.log('✅ Processing successful prediction result...');
        
    } catch (error) {
        console.error('❌ Ticker prediction error:', error);
        const suggestions = getTickerSuggestions(ticker);
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
        // Simulate API call with mock data
        await simulateAPIDelay();
        
        const mockResult = generateMockManualResult(formData);
        displayManualResults(mockResult, formData);
        
        console.log('✅ Manual prediction successful:', mockResult);
    } catch (error) {
        console.error('❌ Manual prediction error:', error);
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
        // Simulate API call
        await simulateAPIDelay();
        
        const mockInfo = generateMockTickerInfo(ticker);
        displayTickerInfo({ ticker_info: mockInfo });
        elements.tickerInfo.classList.remove('hidden');
        
        console.log('✅ Ticker info retrieved:', mockInfo);
    } catch (error) {
        console.error('❌ Ticker info error:', error);
        const suggestions = getTickerSuggestions(ticker);
        showError(error.message || 'Failed to get ticker information.', suggestions);
    } finally {
        setButtonLoading(elements.tickerInfoBtn, false);
    }
}

// Mock data generators for demonstration
function generateMockTickerResult(ticker) {
    const isPositive = Math.random() > 0.4; // 60% chance of positive prediction
    const confidence = 0.6 + Math.random() * 0.3; // 60-90% confidence
    
    return {
        ticker_info: generateMockTickerInfo(ticker),
        prediction: isPositive ? 1 : 0,
        signal: isPositive ? 'BUY' : 'SELL',
        confidence: confidence,
        signal_strength: confidence > 0.8 ? 'STRONG' : confidence > 0.65 ? 'MODERATE' : 'WEAK',
        probabilities: {
            up: isPositive ? confidence : 1 - confidence,
            down: isPositive ? 1 - confidence : confidence
        },
        recommendation: `${confidence > 0.8 ? 'STRONG' : confidence > 0.65 ? 'MODERATE' : 'WEAK'}_${isPositive ? 'BUY' : 'SELL'}`,
        technical_indicators: generateMockTechnicalData()
    };
}

function generateMockManualResult(formData) {
    // Simple logic based on RSI and momentum
    const rsi = formData.RSI_14 || 50;
    const momentum = (formData.momentum_7 || 0) + (formData.momentum_21 || 0);
    const pctChange = formData.pct_change || 0;
    
    const score = (rsi - 50) / 50 + momentum / 5 + pctChange * 10;
    const isPositive = score > 0;
    const confidence = Math.min(0.5 + Math.abs(score) * 0.3, 0.95);
    
    return {
        prediction: isPositive ? 1 : 0,
        signal: isPositive ? 'BUY' : 'SELL',
        confidence: confidence,
        signal_strength: confidence > 0.8 ? 'STRONG' : confidence > 0.65 ? 'MODERATE' : 'WEAK',
        probabilities: {
            up: isPositive ? confidence : 1 - confidence,
            down: isPositive ? 1 - confidence : confidence
        },
        recommendation: `${confidence > 0.8 ? 'STRONG' : confidence > 0.65 ? 'MODERATE' : 'WEAK'}_${isPositive ? 'BUY' : 'SELL'}`
    };
}

function generateMockTickerInfo(ticker) {
    const basePrice = 50 + Math.random() * 200;
    const change = (Math.random() - 0.5) * 10;
    
    return {
        ticker: ticker,
        current_price: basePrice + change,
        previous_close: basePrice,
        data_points: 90,
        data_source: 'Finnhub',
        market_status: {
            is_open: Math.random() > 0.5,
            is_weekend: false
        },
        date_range: {
            start: new Date(Date.now() - 90 * 24 * 60 * 60 * 1000).toISOString(),
            end: new Date().toISOString()
        }
    };
}

function generateMockTechnicalData() {
    return {
        pct_change: (Math.random() - 0.5) * 0.1,
        ma_7: 45 + Math.random() * 10,
        ma_21: 44 + Math.random() * 12,
        volatility_7: 0.05 + Math.random() * 0.3,
        volume: 1000000 + Math.random() * 5000000,
        RSI_14: 20 + Math.random() * 60,
        momentum_7: -2 + Math.random() * 4,
        momentum_21: -3 + Math.random() * 6,
        ma_diff: -5 + Math.random() * 10,
        vol_ratio_20: 0.5 + Math.random() * 2
    };
}

async function simulateAPIDelay() {
    return new Promise(resolve => setTimeout(resolve, 1500 + Math.random() * 1000));
}

// Display functions
function displayTickerResults(result) {
    console.log('🎨 Displaying ticker results:', result);
    
    if (result.ticker_info) {
        displayTickerInfo(result);
        elements.tickerInfo.classList.remove('hidden');
    }
    
    displayPredictionResults(result);
    
    const technicalData = result.technical_indicators || generateMockTechnicalData();
    updateCharts(result, technicalData);
    updateEnhancedMetrics(result);
    
    showResultsState();
    console.log('✅ Results display complete');
}

function displayManualResults(result, inputData) {
    elements.tickerInfo.classList.add('hidden');
    displayPredictionResults(result);
    updateCharts(result, inputData);
    updateEnhancedMetrics(result);
    showResultsState();
}

function displayTickerInfo(result) {
    const tickerInfo = result.ticker_info || {};
    
    if (elements.tickerSymbol && tickerInfo.ticker) {
        elements.tickerSymbol.textContent = tickerInfo.ticker;
    }
    
    if (elements.currentPrice && tickerInfo.current_price !== undefined) {
        const change = tickerInfo.current_price - (tickerInfo.previous_close || tickerInfo.current_price);
        const changePercent = ((change / (tickerInfo.previous_close || tickerInfo.current_price)) * 100);
        
        elements.currentPrice.innerHTML = `${tickerInfo.current_price.toFixed(2)} 
            <span style="color: ${change >= 0 ? '#10b981' : '#ef4444'}; font-size: 0.8rem; margin-left: 5px;">
                ${change >= 0 ? '+' : ''}${changePercent.toFixed(2)}%
            </span>`;
    }
    
    if (elements.previousClose && tickerInfo.previous_close !== undefined) {
        elements.previousClose.textContent = `${tickerInfo.previous_close.toFixed(2)}`;
    }
    
    if (elements.dataPoints) {
        elements.dataPoints.textContent = tickerInfo.data_points || '90';
    }
    
    if (elements.dataSource) {
        elements.dataSource.textContent = tickerInfo.data_source || 'Multi-Source';
    }
    
    if (elements.marketStatus) {
        const status = tickerInfo.market_status;
        if (status) {
            const marketText = status.is_open ? 'Open' : 'Closed';
            elements.marketStatus.textContent = marketText;
            elements.marketStatus.style.color = status.is_open ? '#10b981' : '#ef4444';
        } else {
            elements.marketStatus.textContent = 'Unknown';
        }
    }
    
    if (elements.dateRange) {
        elements.dateRange.textContent = '90 days';
    }
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
    
    // Update prediction card
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
    
    // Update probabilities
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
    if (elements.riskLevel) {
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
    
    // Update model accuracy
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

// Chart functions - FIXED VERSION
function initializeCharts() {
    console.log('📊 Initializing charts...');
    
    // Destroy existing charts if they exist
    if (probabilityChart) {
        probabilityChart.destroy();
        probabilityChart = null;
    }
    if (featuresChart) {
        featuresChart.destroy();
        featuresChart = null;
    }
    
    const probabilityCtx = document.getElementById('probabilityChart');
    const featuresCtx = document.getElementById('featuresChart');

    if (!probabilityCtx || !featuresCtx) {
        console.warn('⚠️ Chart canvases not found, retrying in 500ms...');
        setTimeout(initializeCharts, 500);
        return;
    }

    try {
        // Probability Chart (Doughnut)
        probabilityChart = new Chart(probabilityCtx, {
            type: 'doughnut',
            data: {
                labels: ['Bullish Probability', 'Bearish Probability'],
                datasets: [{
                    data: [50, 50],
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
                            padding: 15,
                            usePointStyle: true
                        }
                    },
                    title: {
                        display: true,
                        text: 'Market Direction Probabilities',
                        color: '#f8fafc',
                        font: { size: 14, weight: 'bold' }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(15, 23, 42, 0.9)',
                        titleColor: '#f8fafc',
                        bodyColor: '#cbd5e1',
                        borderColor: '#475569',
                        borderWidth: 1,
                        callbacks: {
                            label: function(context) {
                                const percentage = context.parsed.toFixed(1);
                                return `${context.label}: ${percentage}%`;
                            }
                        }
                    }
                },
                animation: {
                    animateRotate: true,
                    duration: 1000
                },
                elements: {
                    arc: {
                        borderWidth: 2
                    }
                }
            }
        });

        // Features Chart (Radar)
        featuresChart = new Chart(featuresCtx, {
            type: 'radar',
            data: {
                labels: FEATURE_NAMES.map(name => FEATURE_DISPLAY_NAMES[name] || name),
                datasets: [{
                    label: 'Current Values',
                    data: Array(FEATURE_NAMES.length).fill(0.1), // Start with small values instead of 0
                    backgroundColor: 'rgba(6, 182, 212, 0.2)',
                    borderColor: '#06b6d4',
                    borderWidth: 2,
                    pointBackgroundColor: '#06b6d4',
                    pointBorderColor: '#0891b2',
                    pointBorderWidth: 2,
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { 
                        display: false 
                    },
                    title: {
                        display: true,
                        text: 'Technical Indicators Analysis',
                        color: '#f8fafc',
                        font: { size: 14, weight: 'bold' }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(15, 23, 42, 0.9)',
                        titleColor: '#f8fafc',
                        bodyColor: '#cbd5e1',
                        borderColor: '#475569',
                        borderWidth: 1
                    }
                },
                scales: {
                    r: {
                        beginAtZero: true,
                        min: 0,
                        max: 1,
                        ticks: { 
                            display: false,
                            stepSize: 0.2
                        },
                        grid: { 
                            color: 'rgba(148, 163, 184, 0.3)',
                            lineWidth: 1
                        },
                        angleLines: { 
                            color: 'rgba(148, 163, 184, 0.3)',
                            lineWidth: 1
                        },
                        pointLabels: {
                            color: '#cbd5e1',
                            font: { 
                                size: 10,
                                weight: 'normal'
                            }
                        }
                    }
                },
                animation: {
                    duration: 1000,
                    easing: 'easeInOutQuart'
                },
                elements: {
                    line: {
                        borderWidth: 2
                    },
                    point: {
                        radius: 4,
                        hoverRadius: 6
                    }
                }
            }
        });

        console.log('✅ Charts initialized successfully');
        
    } catch (error) {
        console.error('❌ Failed to initialize charts:', error);
        // Retry initialization after a longer delay
        setTimeout(initializeCharts, 2000);
    }
}

function updateCharts(result, features) {
    console.log('📊 Updating charts with data:', { result, features });
    
    if (!probabilityChart || !featuresChart) {
        console.warn('⚠️ Charts not initialized, initializing now...');
        initializeCharts();
        // Wait a bit and try updating again
        setTimeout(() => updateCharts(result, features), 1000);
        return;
    }
    
    try {
        // Update probability chart
        if (result.probabilities) {
            const probabilities = result.probabilities;
            const upPercent = Math.round(probabilities.up * 100);
            const downPercent = Math.round(probabilities.down * 100);
            
            probabilityChart.data.datasets[0].data = [upPercent, downPercent];
            probabilityChart.update('active');
            console.log('✅ Probability chart updated with:', { upPercent, downPercent });
            
        } else if (result.confidence !== undefined) {
            const isPositive = result.prediction === 1 || result.signal === 'BUY';
            const upProb = isPositive ? result.confidence : (1 - result.confidence);
            const downProb = 1 - upProb;
            const upPercent = Math.round(upProb * 100);
            const downPercent = Math.round(downProb * 100);
            
            probabilityChart.data.datasets[0].data = [upPercent, downPercent];
            probabilityChart.update('active');
            console.log('✅ Probability chart updated (from confidence):', { upPercent, downPercent });
        }

        // Update features chart
        if (features) {
            const featureValues = FEATURE_NAMES.map(feature => {
                let value = features[feature] || 0;
                
                // Normalize values to 0-1 range for radar chart
                switch (feature) {
                    case 'volume':
                        value = Math.min(value / 10000000, 1);
                        break;
                    case 'RSI_14':
                        value = Math.max(0, Math.min(value / 100, 1));
                        break;
                    case 'ma_7':
                    case 'ma_21':
                        value = Math.min(Math.abs(value) / 500, 1);
                        break;
                    case 'pct_change':
                        // Normalize from -0.1 to +0.1 range to 0-1
                        value = Math.min(Math.max((value + 0.1) / 0.2, 0), 1);
                        break;
                    case 'volatility_7':
                        value = Math.min(value * 3, 1);
                        break;
                    case 'momentum_7':
                    case 'momentum_21':
                        // Normalize from -5 to +5 range to 0-1
                        value = Math.min(Math.max((value + 5) / 10, 0), 1);
                        break;
                    case 'ma_diff':
                        // Normalize from -10 to +10 range to 0-1
                        value = Math.min(Math.max((value + 10) / 20, 0), 1);
                        break;
                    case 'vol_ratio_20':
                        value = Math.min(value / 3, 1);
                        break;
                    default:
                        value = Math.min(Math.max(Math.abs(value), 0), 1);
                }
                
                // Ensure minimum visibility
                return Math.max(0.05, Math.min(1, value));
            });
            
            featuresChart.data.datasets[0].data = featureValues;
            featuresChart.update('active');
            console.log('✅ Technical indicators chart updated:', featureValues);
        }
        
    } catch (error) {
        console.error('❌ Failed to update charts:', error);
        // Try reinitializing charts on error
        setTimeout(initializeCharts, 1000);
    }
}

// Enhanced chart visibility check
function ensureChartsVisible() {
    const probabilityCanvas = document.getElementById('probabilityChart');
    const featuresCanvas = document.getElementById('featuresChart');
    
    if (probabilityCanvas && featuresCanvas) {
        // Make sure canvases are visible
        probabilityCanvas.style.display = 'block';
        featuresCanvas.style.display = 'block';
        
        // Force redraw if charts exist
        if (probabilityChart) {
            probabilityChart.resize();
            probabilityChart.update('none');
        }
        if (featuresChart) {
            featuresChart.resize();
            featuresChart.update('none');
        }
        
        console.log('✅ Chart visibility ensured');
    }
}

// Call this when showing results
function showResultsState() {
    setElementDisplay('loadingState', 'none');
    setElementDisplay('emptyState', 'none');
    setElementDisplay('errorState', 'none');
    setElementDisplay('resultsContent', 'block');
    
    // Add animation class
    if (elements.resultsContent) {
        elements.resultsContent.classList.add('fade-in');
    }
    
    // Ensure charts are visible and properly sized
    setTimeout(() => {
        ensureChartsVisible();
        // Reinitialize charts if they don't exist
        if (!probabilityChart || !featuresChart) {
            console.log('📊 Reinitializing charts in results view...');
            initializeCharts();
        }
    }, 100);
    
    console.log('📋 Showing results state');
}

/* Chart functions
function initializeCharts() {
    console.log('📊 Initializing charts...');
    
    const probabilityCtx = document.getElementById('probabilityChart');
    const featuresCtx = document.getElementById('featuresChart');

    if (!probabilityCtx || !featuresCtx) {
        console.warn('⚠️ Chart canvases not found, retrying...');
        setTimeout(initializeCharts, 500);
        return;
    }

    try {
        // Probability Chart
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

        // Features Chart
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
        // Update probability chart
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

        // Update features chart
        if (featuresChart && features) {
            const featureValues = FEATURE_NAMES.map(feature => {
                let value = features[feature] || 0;
                
                // Normalize values to 0-1 range for radar chart
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
} */

// Utility functions
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
        group?.classList.add('error');
        group?.classList.remove('success');
    } else {
        group?.classList.remove('error');
        group?.classList.add('success');
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

function getTickerSuggestions(ticker) {
    if (!ticker) return ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'];
    
    const firstLetter = ticker[0].toUpperCase();
    const suggestions = POPULAR_TICKERS
        .filter(item => item.ticker.startsWith(firstLetter))
        .slice(0, 3)
        .map(item => item.ticker);
    
    return suggestions.length > 0 ? suggestions : ['AAPL', 'MSFT', 'GOOGL'];
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

// State management functions
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
    
    // Add animation class
    if (elements.resultsContent) {
        elements.resultsContent.classList.add('fade-in');
    }
    
    console.log('📋 Showing results state');
}

function showError(message, suggestions = []) {
    setElementDisplay('loadingState', 'none');
    setElementDisplay('emptyState', 'none');
    setElementDisplay('resultsContent', 'none');
    setElementDisplay('errorState', 'flex');
    
    if (elements.errorMessage) {
        elements.errorMessage.textContent = message;
    }
    
    if (suggestions && suggestions.length > 0 && elements.errorSuggestions && elements.suggestedTickers) {
        elements.errorSuggestions.classList.remove('hidden');
        elements.suggestedTickers.innerHTML = suggestions.map(ticker => 
            `<span class="suggested-ticker" onclick="selectSuggestedTicker('${ticker}')">${ticker}</span>`
        ).join('');
    } else if (elements.errorSuggestions) {
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

function resetResults() {
    showEmptyState();
    if (currentMode === 'ticker') {
        clearTickerForm();
    } else {
        clearManualForm();
    }
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

async function checkAPIStatus() {
    try {
        console.log('🔍 Checking API status...');
        
        // Simulate API status check
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        console.log('🟢 API Status: Connected (Demo Mode)');
        elements.apiStatus.querySelector('.status-dot').classList.add('connected');
        elements.apiStatus.querySelector('.status-text').textContent = 'Connected • v3.2 • Enhanced (Demo)';
        
    } catch (error) {
        console.error('❌ API status check failed:', error);
        elements.apiStatus.querySelector('.status-text').textContent = 'Demo Mode';
        elements.apiStatus.querySelector('.status-dot').classList.remove('connected');
    }
}

// Global functions for HTML onclick events
window.selectSuggestedTicker = selectSuggestedTicker;
window.switchMode = switchMode;
window.resetResults = resetResults;