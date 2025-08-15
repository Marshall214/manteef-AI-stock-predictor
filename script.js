// Global variables
const API_BASE_URL = window.location.hostname === 'localhost' 
    ? 'http://localhost:5000' 
    : 'https://manteef-ai-stock-predictor-predictor.up.railway.app';

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
    if (elements.form) {
        elements.form.addEventListener('submit', handlePrediction);
    }
    
    if (elements.clearBtn) {
        elements.clearBtn.addEventListener('click', clearForm);
    }
    
    if (elements.sampleBtn) {
        elements.sampleBtn.addEventListener('click', fillSampleData);
    }
    
    // Add input validation listeners
    FEATURE_NAMES.forEach(feature => {
        const input = document.getElementById(feature);
        if (input) {
            input.addEventListener('input', debounce(validateInput, 300));
            input.addEventListener('blur', validateInput);
        }
    });
}

// Check API status with better error handling
async function checkAPIStatus() {
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 10000);
        
        const response = await fetch(`${API_BASE_URL}/`, {
            method: 'GET',
            signal: controller.signal,
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            }
        });
        
        clearTimeout(timeoutId);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        if (data.status === 'healthy' && data.model_loaded) {
            updateAPIStatus('connected', 'API Connected');
            console.log('API Connected:', data);
        } else {
            updateAPIStatus('error', data.model_loaded === false ? 'Model Not Loaded' : 'API Issues');
            console.warn('API Issues:', data);
        }
    } catch (error) {
        console.error('API Status Check Failed:', error);
        let errorMessage = 'API Offline';
        
        if (error.name === 'AbortError') {
            errorMessage = 'API Timeout';
        } else if (error.message.includes('Failed to fetch')) {
            errorMessage = 'Connection Failed';
        } else if (error.message.includes('CORS')) {
            errorMessage = 'CORS Error';
        }
        
        updateAPIStatus('error', errorMessage);
    }
}

// Update API status indicator
function updateAPIStatus(status, message) {
    if (!elements.apiStatus) return;
    
    const statusDot = elements.apiStatus.querySelector('.status-dot');
    const statusText = elements.apiStatus.querySelector('.status-text');
    
    if (statusDot) {
        statusDot.className = `status-dot ${status === 'connected' ? 'connected' : ''}`;
    }
    
    if (statusText) {
        statusText.textContent = message;
    }
}

// Handle form submission with better error handling
async function handlePrediction(event) {
    event.preventDefault();
    
    if (!validateForm()) {
        showError('Please fill in all required fields with valid numerical values.');
        return;
    }
    
    const formData = getFormData();
    console.log('📤 Sending prediction request:', formData);
    
    showLoadingState();
    setButtonLoading(true);
    
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000);
        
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
        
        console.log('📡 Response status:', response.status, response.statusText);
        
        if (!response.ok) {
            let errorData;
            try {
                errorData = await response.json();
            } catch {
                errorData = { error: `HTTP ${response.status}: ${response.statusText}` };
            }
            throw new Error(errorData.error || `Server error: ${response.status}`);
        }
        
        const result = await response.json();
        console.log('Prediction result:', result);
        
        if (result.error) {
            throw new Error(result.error);
        }
        
        displayResults(result, formData);
        
    } catch (error) {
        console.error('Prediction Error:', error);
        
        let errorMessage = 'Prediction failed. Please try again.';
        
        if (error.name === 'AbortError') {
            errorMessage = 'Request timed out. Please check your connection and try again.';
        } else if (error.message.includes('Failed to fetch')) {
            errorMessage = 'Cannot connect to prediction service. Please check your internet connection.';
        } else if (error.message.includes('CORS')) {
            errorMessage = 'Cross-origin request blocked. Please contact support.';
        } else if (error.message) {
            errorMessage = error.message.startsWith('Prediction failed:') 
                ? error.message 
                : `Prediction failed: ${error.message}`;
        }
        
        showError(errorMessage);
    } finally {
        setButtonLoading(false);
    }
}

// Get form data with validation
function getFormData() {
    const formData = {};
    
    FEATURE_NAMES.forEach(feature => {
        const input = document.getElementById(feature);
        if (input && input.value.trim()) {
            const value = parseFloat(input.value);
            if (!isNaN(value)) {
                formData[feature] = value;
            } else {
                console.warn(`Invalid value for ${feature}: ${input.value}`);
                formData[feature] = 0; // Default fallback
            }
        } else {
            console.warn(`Missing value for ${feature}`);
            formData[feature] = 0; // Default fallback
        }
    });
    
    return formData;
}

// Enhanced form validation
function validateForm() {
    let isValid = true;
    const errors = [];
    
    FEATURE_NAMES.forEach(feature => {
        const input = document.getElementById(feature);
        if (!input) {
            console.error(`Input element not found for feature: ${feature}`);
            return;
        }
        
        const group = input.closest('.input-group');
        const value = input.value.trim();
        
        if (!value) {
            if (group) {
                group.classList.add('error');
                group.classList.remove('success');
            }
            errors.push(`${feature} is required`);
            isValid = false;
        } else if (isNaN(parseFloat(value))) {
            if (group) {
                group.classList.add('error');
                group.classList.remove('success');
            }
            errors.push(`${feature} must be a valid number`);
            isValid = false;
        } else {
            if (group) {
                group.classList.remove('error');
                group.classList.add('success');
            }
        }
    });
    
    if (!isValid) {
        console.warn('Form validation errors:', errors);
    }
    
    return isValid;
}

// Validate individual input
function validateInput(event) {
    const input = event.target;
    if (!input) return;
    
    const group = input.closest('.input-group');
    if (!group) return;
    
    const value = input.value.trim();
    
    if (!value || isNaN(parseFloat(value))) {
        group.classList.add('error');
        group.classList.remove('success');
    } else {
        group.classList.remove('error');
        group.classList.add('success');
    }
}

// Display prediction results with better error handling
function displayResults(result, inputData) {
    try {
        updateLastUpdate();
        
        // Validate result structure
        if (!result || typeof result !== 'object') {
            throw new Error('Invalid result format received');
        }
        
        // Update prediction card
        const prediction = result.prediction ?? (result.signal === 'BUY' ? 1 : 0);
        const isPositive = prediction === 1;
        const signal = result.signal || (isPositive ? 'BUY' : 'SELL');
        const confidence = result.confidence ?? 0.5;
        const strength = result.signal_strength || 'UNKNOWN';
        
        // Update card styling
        if (elements.predictionCard) {
            elements.predictionCard.className = `prediction-card ${isPositive ? 'buy' : 'sell'}`;
        }
        
        if (elements.signalText) {
            elements.signalText.textContent = signal;
        }
        
        if (elements.signalIcon) {
            elements.signalIcon.innerHTML = `<i class="fas fa-arrow-${isPositive ? 'up' : 'down'}"></i>`;
        }
        
        // Update confidence
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
        
        // Update metrics
        const probabilities = result.probabilities || { up: confidence, down: 1 - confidence };
        if (elements.upProbability) {
            elements.upProbability.textContent = `${Math.round((probabilities.up ?? confidence) * 100)}%`;
        }
        
        if (elements.downProbability) {
            elements.downProbability.textContent = `${Math.round((probabilities.down ?? (1 - confidence)) * 100)}%`;
        }
        
        if (elements.recommendation) {
            elements.recommendation.textContent = result.recommendation || `${strength}_${signal}`;
        }
        
        // Update charts
        updateCharts(result, inputData);
        
        // Show results with animation
        showResultsState();
        
    } catch (error) {
        console.error('Error displaying results:', error);
        showError('Failed to display results. Please try again.');
    }
}

// Update charts with error handling
function updateCharts(result, inputData) {
    try {
        updateProbabilityChart(result);
        updateFeaturesChart(inputData);
    } catch (error) {
        console.error('Error updating charts:', error);
        // Don't show error to user for chart updates, just log it
    }
}

// Initialize charts with better error handling
function initializeCharts() {
    try {
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
        const probCanvas = document.getElementById('probabilityChart');
        if (probCanvas) {
            const probCtx = probCanvas.getContext('2d');
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
        }
        
        // Initialize features chart
        const featCanvas = document.getElementById('featuresChart');
        if (featCanvas) {
            const featCtx = featCanvas.getContext('2d');
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
    } catch (error) {
        console.error('Error initializing charts:', error);
    }
}

// Update probability chart
function updateProbabilityChart(result) {
    if (!probabilityChart) return;
    
    try {
        const probabilities = result.probabilities || { up: result.confidence || 0.5, down: 1 - (result.confidence || 0.5) };
        
        probabilityChart.data.datasets[0].data = [
            Math.round((probabilities.down ?? 0.5) * 100),
            Math.round((probabilities.up ?? 0.5) * 100)
        ];
        
        probabilityChart.update('active');
    } catch (error) {
        console.error('Error updating probability chart:', error);
    }
}

// Update features chart
function updateFeaturesChart(inputData) {
    if (!featuresChart) return;
    
    try {
        // Normalize values for better visualization
        const normalizedData = normalizeFeatures(inputData);
        
        featuresChart.data.labels = FEATURE_NAMES.map(name => 
            name.replace(/_/g, ' ').toUpperCase()
        );
        featuresChart.data.datasets[0].data = normalizedData;
        
        featuresChart.update('active');
    } catch (error) {
        console.error('Error updating features chart:', error);
    }
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
    if (elements.form) {
        elements.form.reset();
    }
    
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
            if (group) {
                group.classList.remove('error');
                group.classList.add('success');
            }
        }
    });
}

// State management functions
function showLoadingState() {
    setElementDisplay('loadingState', 'block');
    setElementDisplay('emptyState', 'none');
    setElementDisplay('resultsContent', 'none');
    setElementDisplay('errorState', 'none');
}

function showEmptyState() {
    setElementDisplay('loadingState', 'none');
    setElementDisplay('emptyState', 'block');
    setElementDisplay('resultsContent', 'none');
    setElementDisplay('errorState', 'none');
}

function showResultsState() {
    setElementDisplay('loadingState', 'none');
    setElementDisplay('emptyState', 'none');
    setElementDisplay('errorState', 'none');
    setElementDisplay('resultsContent', 'block');
    
    if (elements.resultsContent) {
        elements.resultsContent.classList.add('fade-in');
    }
}

function showError(message) {
    setElementDisplay('loadingState', 'none');
    setElementDisplay('emptyState', 'none');
    setElementDisplay('resultsContent', 'none');
    setElementDisplay('errorState', 'block');
    
    if (elements.errorMessage) {
        elements.errorMessage.textContent = message;
    }
}

// Helper function for setting element display
function setElementDisplay(elementKey, display) {
    if (elements[elementKey]) {
        elements[elementKey].style.display = display;
    }
}

// Button loading state
function setButtonLoading(loading) {
    if (!elements.predictBtn) return;
    
    const btn = elements.predictBtn;
    const span = btn.querySelector('span');
    const loader = btn.querySelector('.btn-loader');
    
    if (loading) {
        btn.classList.add('loading');
        btn.disabled = true;
        if (span) span.style.opacity = '0';
        if (loader) loader.style.display = 'block';
    } else {
        btn.classList.remove('loading');
        btn.disabled = false;
        if (span) span.style.opacity = '1';
        if (loader) loader.style.display = 'none';
    }
}

// Update last update timestamp
function updateLastUpdate() {
    if (!elements.lastUpdate) return;
    
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

// Enhanced error handling
window.addEventListener('error', function(event) {
    console.error('JavaScript Error:', event.error);
    showError('An unexpected error occurred. Please refresh the page and try again.');
});

window.addEventListener('unhandledrejection', function(event) {
    console.error('Unhandled Promise Rejection:', event.reason);
    showError('A network error occurred. Please check your connection and try again.');
});

// API health check interval - less frequent in production
const healthCheckInterval = setInterval(checkAPIStatus, 60000); // Check every 60 seconds

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    if (healthCheckInterval) {
        clearInterval(healthCheckInterval);
    }
});

console.log('Manteef Stock Predictor - Ready for predictions!');
console.log('🌐 Environment:', window.location.hostname === 'localhost' ? 'Development' : 'Production');