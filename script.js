// Global variables
const API_BASE_URL = window.location.hostname === 'localhost' 
    ? 'http://localhost:5000' 
    : 'https://your-railway-app-url.up.railway.app'; // Replace with your Railway URL

let probabilityChart = null;
let featuresChart = null;

// Feature names as provided by the backend
const FEATURE_NAMES = [
    'pct_change', 'ma_7', 'ma_21', 'volatility_7', 'volume',
    'RSI_14', 'momentum_7', 'momentum_21', 'ma_diff', 'vol_ratio_20'
];

// Sample data for testing
const SAMPLE_DATA = {
    bullish: {
        pct_change: 0.025,
        ma_7: 46.50,
        ma_21: 45.20,
        volatility_7: 0.12,
        volume: 3200000,
        RSI_14: 72.3,
        momentum_7: 1.5,
        momentum_21: 2.1,
        ma_diff: 1.30,
        vol_ratio_20: 1.45
    },
    bearish: {
        pct_change: -0.018,
        ma_7: 44.20,
        ma_21: 45.60,
        volatility_7: 0.22,
        volume: 4100000,
        RSI_14: 28.7,
        momentum_7: -0.9,
        momentum_21: -1.4,
        ma_diff: -1.40,
        vol_ratio_20: 1.85
    },
    neutral: {
        pct_change: 0.001,
        ma_7: 45.08,
        ma_21: 45.05,
        volatility_7: 0.08,
        volume: 1800000,
        RSI_14: 51.2,
        momentum_7: 0.2,
        momentum_21: 0.1,
        ma_diff: 0.03,
        vol_ratio_20: 0.95
    }
};

// DOM Elements
const elements = {
    form: document.getElementById('predictionForm'),
    predictBtn: document.getElementById('predictBtn'),
    clearBtn: document.getElementById('clearBtn'),
    sampleBtn: document.getElementById('sampleBtn'),
    apiStatus: document.getElementById('apiStatus'),
    loadingState: document.getElementById('loadingState'),
    emptyState: document.getElementById('emptyState'),
    resultsContent: document.getElementById('resultsContent'),
    errorState: document.getElementById('errorState'),
    errorMessage: document.getElementById('errorMessage'),
    lastUpdate: document.getElementById('lastUpdate'),
    predictionCard: document.getElementById('predictionCard'),
    signalText: document.getElementById('signalText'),
    signalIcon: document.getElementById('signalIcon'),
    confidenceFill: document.getElementById('confidenceFill'),
    confidenceText: document.getElementById('confidenceText'),
    strengthBadge: document.getElementById('strengthBadge'),
    upProbability: document.getElementById('upProbability'),
    downProbability: document.getElementById('downProbability'),
    recommendation: document.getElementById('recommendation')
};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    attachEventListeners();
    checkAPIStatus();
});

// Initialize application
function initializeApp() {
    console.log('🚀 Manteef Stock Predictor Initialized');
    console.log('🔗 API URL:', API_BASE_URL);
    showEmptyState();
    initializeCharts();
}

// Attach event listeners
function attachEventListeners() {
    elements.form.addEventListener('submit', handlePrediction);
    elements.clearBtn.addEventListener('click', clearForm);
    elements.sampleBtn.addEventListener('click', fillSampleData);
    
    // Add input validation listeners
    FEATURE_NAMES.forEach(feature => {
        const input = document.getElementById(feature);
        if (input) {
            input.addEventListener('input', validateInput);
            input.addEventListener('blur', validateInput);
        }
    });
}

// Check API status with better error handling for production
async function checkAPIStatus() {
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout
        
        const response = await fetch(`${API_BASE_URL}/`, {
            signal: controller.signal,
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            }
        });
        
        clearTimeout(timeoutId);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.status === 'healthy' && data.model_loaded) {
            updateAPIStatus('connected', 'API Connected');
            console.log('✅ API Connected:', data);
        } else {
            updateAPIStatus('error', 'Model Not Loaded');
        }
    } catch (error) {
        console.error('API Status Check Failed:', error);
        if (error.name === 'AbortError') {
            updateAPIStatus('error', 'API Timeout');
        } else {
            updateAPIStatus('error', 'API Offline');
        }
    }
}

// Update API status indicator
function updateAPIStatus(status, message) {
    const statusDot = elements.apiStatus.querySelector('.status-dot');
    const statusText = elements.apiStatus.querySelector('.status-text');
    
    statusDot.className = `status-dot ${status === 'connected' ? 'connected' : ''}`;
    statusText.textContent = message;
}

// Handle form submission with better error handling
async function handlePrediction(event) {
    event.preventDefault();
    
    if (!validateForm()) {
        showError('Please fill in all required fields with valid values.');
        return;
    }
    
    const formData = getFormData();
    
    showLoadingState();
    setButtonLoading(true);
    
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout
        
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify(formData),
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.error) {
            throw new Error(result.error);
        }
        
        displayResults(result, formData);
        console.log('✅ Prediction successful:', result);
        
    } catch (error) {
        console.error('Prediction Error:', error);
        
        let errorMessage = 'Prediction failed. Please try again.';
        if (error.name === 'AbortError') {
            errorMessage = 'Request timed out. Please check your connection and try again.';
        } else if (error.message) {
            errorMessage = `Prediction failed: ${error.message}`;
        }
        
        showError(errorMessage);
    } finally {
        setButtonLoading(false);
    }
}

// Get form data
function getFormData() {
    const formData = {};
    
    FEATURE_NAMES.forEach(feature => {
        const input = document.getElementById(feature);
        if (input) {
            formData[feature] = parseFloat(input.value) || 0;
        }
    });
    
    return formData;
}

// Validate form
function validateForm() {
    let isValid = true;
    
    FEATURE_NAMES.forEach(feature => {
        const input = document.getElementById(feature);
        const group = input.closest('.input-group');
        
        if (!input.value.trim() || isNaN(parseFloat(input.value))) {
            group.classList.add('error');
            group.classList.remove('success');
            isValid = false;
        } else {
            group.classList.remove('error');
            group.classList.add('success');
        }
    });
    
    return isValid;
}

// Validate individual input
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

// Display prediction results
function displayResults(result, inputData) {
    updateLastUpdate();
    
    // Update prediction card
    const isPositive = result.prediction === 1;
    const signal = result.signal || (isPositive ? 'BUY' : 'SELL');
    const confidence = result.confidence || 0;
    const strength = result.signal_strength || 'UNKNOWN';
    
    // Update card styling
    elements.predictionCard.className = `prediction-card ${isPositive ? 'buy' : 'sell'}`;
    elements.signalText.textContent = signal;
    elements.signalIcon.innerHTML = `<i class="fas fa-arrow-${isPositive ? 'up' : 'down'}"></i>`;
    
    // Update confidence
    const confidencePercent = Math.round(confidence * 100);
    elements.confidenceFill.style.width = `${confidencePercent}%`;
    elements.confidenceText.textContent = `${confidencePercent}%`;
    elements.strengthBadge.textContent = strength;
    
    // Update metrics
    const probabilities = result.probabilities || { up: confidence, down: 1 - confidence };
    elements.upProbability.textContent = `${Math.round(probabilities.up * 100)}%`;
    elements.downProbability.textContent = `${Math.round(probabilities.down * 100)}%`;
    elements.recommendation.textContent = result.recommendation || `${strength}_${signal}`;
    
    // Update charts
    updateCharts(result, inputData);
    
    // Show results with animation
    showResultsState();
}

// Update charts
function updateCharts(result, inputData) {
    updateProbabilityChart(result);
    updateFeaturesChart(inputData);
}

// Initialize charts
function initializeCharts() {
    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                labels: {
                    color: '#CBD5E1',
                    font: {
                        family: 'Inter'
                    }
                }
            }
        },
        scales: {
            x: {
                ticks: { color: '#94A3B8' },
                grid: { color: '#334155' }
            },
            y: {
                ticks: { color: '#94A3B8' },
                grid: { color: '#334155' }
            }
        }
    };
    
    // Initialize probability chart
    const probCtx = document.getElementById('probabilityChart').getContext('2d');
    probabilityChart = new Chart(probCtx, {
        type: 'doughnut',
        data: {
            labels: ['Down Probability', 'Up Probability'],
            datasets: [{
                data: [50, 50],
                backgroundColor: ['#EF4444', '#10B981'],
                borderColor: ['#DC2626', '#059669'],
                borderWidth: 2
            }]
        },
        options: {
            ...chartOptions,
            plugins: {
                ...chartOptions.plugins,
                title: {
                    display: true,
                    text: 'Prediction Probabilities',
                    color: '#F8FAFC',
                    font: { family: 'Inter', size: 14, weight: '600' }
                }
            }
        }
    });
    
    // Initialize features chart
    const featCtx = document.getElementById('featuresChart').getContext('2d');
    featuresChart = new Chart(featCtx, {
        type: 'radar',
        data: {
            labels: [],
            datasets: [{
                label: 'Feature Values',
                data: [],
                borderColor: '#06B6D4',
                backgroundColor: 'rgba(6, 182, 212, 0.1)',
                pointBackgroundColor: '#06B6D4',
                pointBorderColor: '#0891B2',
                pointRadius: 4,
                borderWidth: 2
            }]
        },
        options: {
            ...chartOptions,
            plugins: {
                ...chartOptions.plugins,
                title: {
                    display: true,
                    text: 'Technical Indicators',
                    color: '#F8FAFC',
                    font: { family: 'Inter', size: 14, weight: '600' }
                }
            },
            scales: {
                r: {
                    ticks: { color: '#64748B', backdropColor: 'transparent' },
                    grid: { color: '#334155' },
                    pointLabels: { color: '#94A3B8', font: { size: 10 } }
                }
            }
        }
    });
}

// Update probability chart
function updateProbabilityChart(result) {
    if (!probabilityChart) return;
    
    const probabilities = result.probabilities || { up: result.confidence, down: 1 - result.confidence };
    
    probabilityChart.data.datasets[0].data = [
        Math.round(probabilities.down * 100),
        Math.round(probabilities.up * 100)
    ];
    
    probabilityChart.update('active');
}

// Update features chart
function updateFeaturesChart(inputData) {
    if (!featuresChart) return;
    
    // Normalize values for better visualization
    const normalizedData = normalizeFeatures(inputData);
    
    featuresChart.data.labels = FEATURE_NAMES.map(name => 
        name.replace(/_/g, ' ').toUpperCase()
    );
    featuresChart.data.datasets[0].data = normalizedData;
    
    featuresChart.update('active');
}

// Normalize feature values for radar chart
function normalizeFeatures(data) {
    const normalized = [];
    
    FEATURE_NAMES.forEach(feature => {
        let value = data[feature] || 0;
        
        // Apply feature-specific normalization
        switch(feature) {
            case 'pct_change':
                value = Math.abs(value) * 1000; // Scale percentage
                break;
            case 'ma_7':
            case 'ma_21':
                value = value / 100; // Scale price
                break;
            case 'volatility_7':
                value = value * 100; // Scale volatility
                break;
            case 'volume':
                value = value / 1000000; // Scale to millions
                break;
            case 'RSI_14':
                value = value; // Already 0-100
                break;
            case 'momentum_7':
            case 'momentum_21':
                value = Math.abs(value) * 10; // Scale momentum
                break;
            case 'ma_diff':
                value = Math.abs(value) * 10; // Scale difference
                break;
            case 'vol_ratio_20':
                value = value * 10; // Scale ratio
                break;
        }
        
        normalized.push(Math.min(Math.max(value, 0), 100)); // Clamp to 0-100
    });
    
    return normalized;
}

// Clear form
function clearForm() {
    elements.form.reset();
    
    // Remove validation classes
    document.querySelectorAll('.input-group').forEach(group => {
        group.classList.remove('error', 'success');
    });
    
    showEmptyState();
}

// Fill sample data
function fillSampleData() {
    const samples = Object.keys(SAMPLE_DATA);
    const randomSample = SAMPLE_DATA[samples[Math.floor(Math.random() * samples.length)]];
    
    FEATURE_NAMES.forEach(feature => {
        const input = document.getElementById(feature);
        if (input && randomSample[feature] !== undefined) {
            input.value = randomSample[feature];
            
            // Trigger validation
            const group = input.closest('.input-group');
            group.classList.remove('error');
            group.classList.add('success');
        }
    });
}

// State management
function showLoadingState() {
    elements.emptyState.style.display = 'none';
    elements.resultsContent.style.display = 'none';
    elements.errorState.style.display = 'none';
    elements.loadingState.style.display = 'block';
}

function showEmptyState() {
    elements.loadingState.style.display = 'none';
    elements.resultsContent.style.display = 'none';
    elements.errorState.style.display = 'none';
    elements.emptyState.style.display = 'block';
}

function showResultsState() {
    elements.loadingState.style.display = 'none';
    elements.emptyState.style.display = 'none';
    elements.errorState.style.display = 'none';
    elements.resultsContent.style.display = 'block';
    elements.resultsContent.classList.add('fade-in');
}

function showError(message) {
    elements.loadingState.style.display = 'none';
    elements.emptyState.style.display = 'none';
    elements.resultsContent.style.display = 'none';
    elements.errorState.style.display = 'block';
    elements.errorMessage.textContent = message;
}

// Button loading state
function setButtonLoading(loading) {
    const btn = elements.predictBtn;
    const span = btn.querySelector('span');
    const loader = btn.querySelector('.btn-loader');
    
    if (loading) {
        btn.classList.add('loading');
        btn.disabled = true;
        span.style.opacity = '0';
        loader.style.display = 'block';
    } else {
        btn.classList.remove('loading');
        btn.disabled = false;
        span.style.opacity = '1';
        loader.style.display = 'none';
    }
}

// Update last update timestamp
function updateLastUpdate() {
    const now = new Date();
    const timeString = now.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
    elements.lastUpdate.textContent = `Last updated: ${timeString}`;
}

// Reset results (called from HTML)
function resetResults() {
    showEmptyState();
    clearForm();
}

// Utility functions
function formatNumber(num, decimals = 2) {
    return Number(num).toFixed(decimals);
}

function formatPercent(num) {
    return `${Math.round(num * 100)}%`;
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

// Error handling
window.addEventListener('error', function(event) {
    console.error('JavaScript Error:', event.error);
    showError('An unexpected error occurred. Please refresh the page and try again.');
});

// API health check interval - less frequent in production
setInterval(checkAPIStatus, 60000); // Check every 60 seconds

console.log('Manteef Stock Predictor - Ready for predictions!');
console.log('🌐 Environment:', window.location.hostname === 'localhost' ? 'Development' : 'Production');