/**
 * Smart Spam Classifier - Frontend Controller
 * Connects the web UI to the Flask /predict backend endpoint.
 */

document.addEventListener('DOMContentLoaded', () => {
  const messageInput = document.getElementById('messageInput');
  const charCount = document.getElementById('charCount');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const clearBtn = document.getElementById('clearBtn');
  const btnText = document.getElementById('btnText');
  const btnSpinner = document.getElementById('btnSpinner');
  const btnIcon = document.getElementById('btnIcon');
  const errorBanner = document.getElementById('errorBanner');
  const errorMessage = document.getElementById('errorMessage');
  const loadingState = document.getElementById('loadingState');
  const resultSection = document.getElementById('resultSection');

  // Result display elements
  const resultBadge = document.getElementById('resultBadge');
  const metricPrediction = document.getElementById('metricPrediction');
  const metricVerdict = document.getElementById('metricVerdict');
  const metricProbability = document.getElementById('metricProbability');
  const probabilityFill = document.getElementById('probabilityFill');
  const metricConfidence = document.getElementById('metricConfidence');
  const confidenceFill = document.getElementById('confidenceFill');
  const previewText = document.getElementById('previewText');

  // Sample chip buttons
  const chipButtons = document.querySelectorAll('.chip-btn');

  // Update character count on input
  messageInput.addEventListener('input', () => {
    const len = messageInput.value.length;
    charCount.textContent = `${len} character${len === 1 ? '' : 's'}`;
    hideError();
  });

  // Example chip handlers
  chipButtons.forEach((chip) => {
    chip.addEventListener('click', () => {
      const sampleText = chip.getAttribute('data-sample');
      messageInput.value = sampleText;
      messageInput.dispatchEvent(new Event('input'));
      hideError();
      messageInput.focus();
    });
  });

  // Clear button handler
  clearBtn.addEventListener('click', () => {
    messageInput.value = '';
    charCount.textContent = '0 characters';
    hideError();
    resultSection.classList.add('hidden');
    loadingState.classList.add('hidden');
    messageInput.focus();
  });

  // Analyze button handler
  analyzeBtn.addEventListener('click', handleAnalyze);

  // Allow Ctrl+Enter or Cmd+Enter to submit
  messageInput.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      handleAnalyze();
    }
  });

  async function handleAnalyze() {
    const message = messageInput.value.trim();

    // Input Validation
    if (!message) {
      showError('Please enter a message to analyze before submitting.');
      messageInput.focus();
      return;
    }

    hideError();
    setLoading(true);

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({ message: message })
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.error || 'Failed to analyze the message. Please try again.');
      }

      displayResult(data);
    } catch (err) {
      showError(err.message || 'An unexpected error occurred while connecting to the server.');
      resultSection.classList.add('hidden');
    } finally {
      setLoading(false);
    }
  }

  function displayResult(data) {
    const isSpam = data.prediction.toLowerCase() === 'spam';
    const spamProbPercent = (data.spam_probability * 100).toFixed(1);
    const confidencePercent = (data.confidence * 100).toFixed(1);

    // Update Badge
    resultBadge.className = 'result-badge';
    if (isSpam) {
      resultBadge.classList.add('badge-spam');
      resultBadge.textContent = '🚨 SPAM';
      metricVerdict.textContent = 'High probability of unwanted/fraudulent content';
      probabilityFill.className = 'progress-bar-fill';
    } else {
      resultBadge.classList.add('badge-ham');
      resultBadge.textContent = '✅ HAM (LEGITIMATE)';
      metricVerdict.textContent = 'Likely legitimate personal/transactional communication';
      probabilityFill.className = 'progress-bar-fill fill-ham';
    }

    // Update Metric Cards
    metricPrediction.textContent = isSpam ? 'Spam' : 'Ham';
    metricProbability.textContent = `${spamProbPercent}%`;
    probabilityFill.style.width = `${Math.min(100, Math.max(0, data.spam_probability * 100))}%`;

    metricConfidence.textContent = `${confidencePercent}%`;
    confidenceFill.style.width = `${Math.min(100, Math.max(0, data.confidence * 100))}%`;

    // Preview
    previewText.textContent = data.message;

    // Show Result with smooth scroll
    resultSection.classList.remove('hidden');
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function setLoading(isLoading) {
    if (isLoading) {
      analyzeBtn.disabled = true;
      btnSpinner.classList.remove('hidden');
      btnIcon.classList.add('hidden');
      btnText.textContent = 'Analyzing...';
      loadingState.classList.remove('hidden');
      resultSection.classList.add('hidden');
    } else {
      analyzeBtn.disabled = false;
      btnSpinner.classList.add('hidden');
      btnIcon.classList.remove('hidden');
      btnText.textContent = 'Analyze Message';
      loadingState.classList.add('hidden');
    }
  }

  function showError(msg) {
    errorMessage.textContent = msg;
    errorBanner.classList.remove('hidden');
  }

  function hideError() {
    errorBanner.classList.add('hidden');
  }
});
