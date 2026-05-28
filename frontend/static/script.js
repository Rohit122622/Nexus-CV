/* ================================================================
   NEXUS CV — Interaction Engine
   Score gauges, count-up, toasts, loading, accordion, pipeline
   ================================================================ */

// ── TOAST NOTIFICATION SYSTEM ──
const NexusToast = {
    container: null,
    init() {
        if (!this.container) {
            this.container = document.createElement('div');
            this.container.className = 'toast-container';
            document.body.appendChild(this.container);
        }
    },
    show(message, type = 'info', duration = 4000) {
        this.init();
        const icons = { success: '✅', error: '❌', info: 'ℹ️' };
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `
            <span class="toast-icon">${icons[type] || icons.info}</span>
            <span class="toast-msg">${message}</span>
            <button class="toast-close" onclick="this.parentElement.remove()">×</button>
        `;
        this.container.appendChild(toast);
        setTimeout(() => {
            toast.classList.add('removing');
            setTimeout(() => toast.remove(), 300);
        }, duration);
    },
    success(msg) { this.show(msg, 'success'); },
    error(msg)   { this.show(msg, 'error'); },
    info(msg)    { this.show(msg, 'info'); }
};

// ── COUNT-UP ANIMATION ──
function animateCountUp(element, target, duration = 1200, suffix = '') {
    if (!element) return;
    const start = performance.now();
    const initial = 0;
    function update(now) {
        const elapsed = now - start;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // easeOutCubic
        const current = Math.round(initial + (target - initial) * eased);
        element.textContent = current + suffix;
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

// ── SVG SCORE GAUGE ──
function initScoreGauge(container, score) {
    if (!container) return;
    const radius = 80;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (score / 100) * circumference;
    
    // Create SVG
    container.innerHTML = `
        <svg viewBox="0 0 180 180" width="180" height="180">
            <defs>
                <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#14b8a6"/>
                    <stop offset="100%" style="stop-color:#6366f1"/>
                </linearGradient>
            </defs>
            <circle class="gauge-bg" cx="90" cy="90" r="${radius}" 
                    fill="none" stroke-width="10"/>
            <circle class="gauge-fill" cx="90" cy="90" r="${radius}" 
                    fill="none" stroke="url(#gaugeGradient)" stroke-width="10" 
                    stroke-linecap="round"
                    stroke-dasharray="${circumference}" 
                    stroke-dashoffset="${circumference}"
                    transform="rotate(-90 90 90)"/>
        </svg>
        <div class="gauge-value">
            <span class="gauge-number" data-target="${score}">0%</span>
            <span class="gauge-label">${score >= 80 ? 'Excellent' : score >= 60 ? 'Good' : 'Needs Work'}</span>
        </div>
    `;
    
    // Animate after brief delay
    setTimeout(() => {
        const fill = container.querySelector('.gauge-fill');
        if (fill) fill.style.strokeDashoffset = offset;
        const numberEl = container.querySelector('.gauge-number');
        animateCountUp(numberEl, score, 1500, '%');
    }, 200);
}

// ── BUTTON LOADING STATE ──
function setButtonLoading(btn, isLoading, loadingText) {
    if (!btn) return;
    if (isLoading) {
        btn.classList.add('btn--loading');
        btn.disabled = true;
        btn.dataset.originalText = btn.innerHTML;
        btn.innerHTML = `<span class="btn-spinner"></span><span class="btn-text">${loadingText || btn.textContent}</span>`;
    } else {
        btn.classList.remove('btn--loading');
        btn.disabled = false;
        if (btn.dataset.originalText) {
            btn.innerHTML = btn.dataset.originalText;
        }
    }
}

// ── ACCORDION TOGGLE ──
function toggleAccordion(header) {
    const accordion = header.closest('.accordion');
    if (!accordion) return;
    accordion.classList.toggle('open');
}

// ── PIPELINE STEPPER ──
function initPipelineStepper(containerId, steps) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    container.innerHTML = steps.map((step, i) => `
        <div class="pipeline-step ${i === 0 ? 'active' : ''}" data-step="${step.id}">
            <div class="step-dot">${step.icon}</div>
            <div class="step-label">${step.label}</div>
            ${i < steps.length - 1 ? '<div class="step-line"></div>' : ''}
        </div>
    `).join('');
}

function updatePipelineStep(containerId, stepId) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    const steps = container.querySelectorAll('.pipeline-step');
    let found = false;
    
    steps.forEach(step => {
        if (step.dataset.step === stepId) {
            step.classList.remove('done');
            step.classList.add('active');
            found = true;
        } else if (!found) {
            step.classList.remove('active');
            step.classList.add('done');
        } else {
            step.classList.remove('active', 'done');
        }
    });
}

// ── DEBOUNCED INPUT ──
function debounce(fn, delay = 300) {
    let timer;
    return function(...args) {
        clearTimeout(timer);
        timer = setTimeout(() => fn.apply(this, args), delay);
    };
}

// ── INTERSECTION OBSERVER FOR LAZY ANIMATIONS ──
function initLazyAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1 });
    
    document.querySelectorAll('.lazy-animate').forEach(el => observer.observe(el));
}

// ── FORM SUBMISSION WITH LOADING ──
function initFormLoading() {
    document.querySelectorAll('form[data-loading]').forEach(form => {
        form.addEventListener('submit', function(e) {
            const btn = form.querySelector('button[type="submit"], .btn[type="submit"]');
            if (btn && !btn.disabled) {
                setButtonLoading(btn, true, form.dataset.loading || 'Processing...');
            }
        });
    });
}

// ── PROGRESS BAR ANIMATION ──
function initProgressBars() {
    document.querySelectorAll('.progress-bar[data-score]').forEach(bar => {
        setTimeout(() => {
            bar.style.width = bar.dataset.score + '%';
        }, 300);
    });
}

// ── INIT ON DOM READY ──
document.addEventListener('DOMContentLoaded', function() {
    initLazyAnimations();
    initFormLoading();
    initProgressBars();
    
    // Initialize score gauges
    document.querySelectorAll('.score-gauge[data-score]').forEach(gauge => {
        initScoreGauge(gauge, parseInt(gauge.dataset.score));
    });
    
    // Initialize count-up elements
    document.querySelectorAll('[data-count-up]').forEach(el => {
        animateCountUp(el, parseInt(el.dataset.countUp), 1200, el.dataset.suffix || '');
    });
    
    // Card stagger animation
    document.querySelectorAll('.card').forEach((card, i) => {
        card.style.animationDelay = `${i * 0.06}s`;
    });
});
