/* ==========================================================================
   Employee Attrition Prediction System — client-side interactivity
   - Mobile nav toggle
   - Range slider live value display
   - Client-side form validation (mirrors server-side rules in app.py)
   ========================================================================== */

document.addEventListener('DOMContentLoaded', function () {

  // ---- Mobile nav toggle ----
  var navToggle = document.getElementById('navToggle');
  var navLinks = document.getElementById('navLinks');
  if (navToggle && navLinks) {
    navToggle.addEventListener('click', function () {
      var isOpen = navLinks.classList.toggle('open');
      navToggle.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
    });
    // Close menu on link click (mobile)
    navLinks.querySelectorAll('a').forEach(function (link) {
      link.addEventListener('click', function () {
        navLinks.classList.remove('open');
        navToggle.setAttribute('aria-expanded', 'false');
      });
    });
  }

  // ---- Range slider live value labels ----
  document.querySelectorAll('input[type="range"]').forEach(function (slider) {
    var out = document.querySelector('[data-value-for="' + slider.id + '"]');
    if (!out) return;
    out.textContent = slider.value;
    slider.addEventListener('input', function () {
      out.textContent = slider.value;
    });
  });

  // ---- Prediction form validation ----
  var form = document.getElementById('predictForm');
  if (!form) return;

  var VALIDATORS = {
    Age: { min: 18, max: 65, type: 'number', label: 'Age' },
    MonthlyIncome: { min: 10000, max: 300000, type: 'number', label: 'Monthly Income' },
    YearsAtCompany: { min: 0, max: 40, type: 'number', label: 'Years at Company' },
    DistanceFromHome: { min: 0, max: 100, type: 'number', label: 'Distance From Home' },
    TrainingTimesLastYear: { min: 0, max: 6, type: 'number', label: 'Training Times Last Year' },
    JobSatisfaction: { min: 1, max: 4, type: 'number', label: 'Job Satisfaction' },
    WorkLifeBalance: { min: 1, max: 4, type: 'number', label: 'Work Life Balance' },
    PerformanceRating: { min: 1, max: 4, type: 'number', label: 'Performance Rating' },
    Gender: { type: 'select', label: 'Gender' },
    Department: { type: 'select', label: 'Department' },
    JobRole: { type: 'select', label: 'Job Role' },
    OverTime: { type: 'select', label: 'Overtime' },
  };

  function showError(fieldName, message) {
    var el = form.querySelector('[name="' + fieldName + '"]');
    var errorEl = form.querySelector('[data-error-for="' + fieldName + '"]');
    if (el) el.closest('.field').classList.toggle('error', Boolean(message));
    if (errorEl) errorEl.textContent = message || '';
  }

  function validateField(name) {
    var rule = VALIDATORS[name];
    var el = form.querySelector('[name="' + name + '"]');
    if (!el || !rule) return true;

    var value = el.value.trim();

    if (value === '') {
      showError(name, rule.label + ' is required.');
      return false;
    }

    if (rule.type === 'number') {
      var num = Number(value);
      if (Number.isNaN(num)) {
        showError(name, rule.label + ' must be a number.');
        return false;
      }
      if (num < rule.min || num > rule.max) {
        showError(name, rule.label + ' must be between ' + rule.min + ' and ' + rule.max + '.');
        return false;
      }
    }

    showError(name, '');
    return true;
  }

  // Validate on blur for immediate feedback
  Object.keys(VALIDATORS).forEach(function (name) {
    var el = form.querySelector('[name="' + name + '"]');
    if (!el) return;
    el.addEventListener('blur', function () { validateField(name); });
    if (el.tagName === 'SELECT') {
      el.addEventListener('change', function () { validateField(name); });
    }
  });

  form.addEventListener('submit', function (e) {
    var allValid = true;
    Object.keys(VALIDATORS).forEach(function (name) {
      if (!validateField(name)) allValid = false;
    });

    if (!allValid) {
      e.preventDefault();
      var firstError = form.querySelector('.field.error');
      if (firstError) {
        firstError.scrollIntoView({ behavior: 'smooth', block: 'center' });
        var input = firstError.querySelector('input, select');
        if (input) input.focus();
      }
    }
  });
});
