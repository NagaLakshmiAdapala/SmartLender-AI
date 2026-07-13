const darkModeToggle = document.getElementById('darkModeToggle');
const body = document.body;

if (darkModeToggle) {
  darkModeToggle.addEventListener('click', () => {
    body.classList.toggle('dark-mode');
    const icon = darkModeToggle.querySelector('i');
    if (body.classList.contains('dark-mode')) {
      icon.classList.replace('fa-moon', 'fa-sun');
      darkModeToggle.classList.add('active');
    } else {
      icon.classList.replace('fa-sun', 'fa-moon');
      darkModeToggle.classList.remove('active');
    }
  });
}



const updateSummaryView = () => {
  const summaryName = document.getElementById('summaryName');
  const summaryArea = document.getElementById('summaryArea');
  const summaryEducation = document.getElementById('summaryEducation');
  const summaryCredit = document.getElementById('summaryCredit');
  if (summaryName) summaryName.textContent = document.getElementById('applicant_name')?.value || '—';
  if (summaryArea) summaryArea.textContent = document.getElementById('property_area')?.value || '—';
  if (summaryEducation) summaryEducation.textContent = document.getElementById('education')?.value || '—';
  if (summaryCredit) {
    const credit = document.getElementById('credit_history')?.value;
    summaryCredit.textContent = credit === '1' ? 'Good' : credit === '0' ? 'Bad' : '—';
  }
};



const maybeUpdateSummary = () => {
  if (document.getElementById('summaryName')) {
    updateSummaryView();
  }
};

const formFields = [
  'applicant_name', 'gender', 'married', 'dependents', 'education',
  'self_employed', 'property_area', 'applicant_income', 'coapplicant_income', 'loan_amount', 'loan_term', 'credit_history',
];

formFields.forEach((name) => {
  const element = document.querySelector(`[name="${name}"]`);
  if (element) {
    element.addEventListener('change', maybeUpdateSummary);
    element.addEventListener('input', maybeUpdateSummary);
  }
});

const updateSummaryOnLoad = () => {
  updateSummaryView();
};

window.addEventListener('DOMContentLoaded', updateSummaryOnLoad);

const exportHistoryButton = document.getElementById('exportHistory');
if (exportHistoryButton) {
  exportHistoryButton.addEventListener('click', () => {
    if (window.historyExportUrl) {
      window.location.href = window.historyExportUrl;
    } else {
      alert('No history export is available yet.');
    }
  });
}

const downloadBatchButton = document.getElementById('downloadBatchResults');
if (downloadBatchButton) {
  downloadBatchButton.addEventListener('click', () => {
    if (window.batchDownloadUrl) {
      window.location.href = window.batchDownloadUrl;
    } else {
      alert('No batch results are available yet.');
    }
  });
}

const initCharts = () => {
  const trendCanvas = document.getElementById('trendChart');
  const incomeCanvas = document.getElementById('incomeChart');
  const loanCanvas = document.getElementById('loanChart');
  const propertyCanvas = document.getElementById('propertyChart');
  const educationCanvas = document.getElementById('educationChart');

  if (trendCanvas) {
    new Chart(trendCanvas, {
      type: 'line',
      data: {
        labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
        datasets: [{
          label: 'Approval Trend',
          data: [62, 68, 70, 74, 72, 78],
          borderColor: '#ff8a3d',
          backgroundColor: 'rgba(255,138,61,0.18)',
          fill: true,
          tension: 0.4,
        }],
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false },
        },
      },
    });
  }

  if (incomeCanvas) {
    new Chart(incomeCanvas, {
      type: 'bar',
      data: {
        labels: ['< 50k', '50k-100k', '100k-200k', '200k+'],
        datasets: [{
          label: 'Applicants',
          data: [22, 41, 34, 18],
          backgroundColor: ['#ff8a3d', '#ffa65a', '#f55a4e', '#f6e7d8'],
        }],
      },
      options: { responsive: true, plugins: { legend: { display: false } } },
    });
  }

  if (loanCanvas) {
    new Chart(loanCanvas, {
      type: 'bar',
      data: {
        labels: ['< 100k', '100k-200k', '200k-300k', '300k+'],
        datasets: [{
          label: 'Loan Amount',
          data: [18, 30, 28, 24],
          backgroundColor: ['#ff8a3d', '#ffa65a', '#f55a4e', '#f6e7d8'],
        }],
      },
      options: { responsive: true, plugins: { legend: { display: false } } },
    });
  }

  if (propertyCanvas) {
    new Chart(propertyCanvas, {
      type: 'pie',
      data: {
        labels: ['Urban', 'Rural', 'Semiurban'],
        datasets: [{
          data: [45, 22, 33],
          backgroundColor: ['#ff8a3d', '#ffa65a', '#f55a4e'],
        }],
      },
      options: { responsive: true },
    });
  }

  if (educationCanvas) {
    new Chart(educationCanvas, {
      type: 'doughnut',
      data: {
        labels: ['Graduate', 'Not Graduate', 'Good Credit', 'Bad Credit'],
        datasets: [{
          data: [56, 44, 67, 33],
          backgroundColor: ['#ff8a3d', '#ffa65a', '#f55a4e', '#f6e7d8'],
        }],
      },
      options: { responsive: true },
    });
  }
};

window.addEventListener('DOMContentLoaded', initCharts);
// --- AI Assistant + Prediction Renderer ---
(function() {
  function formatCurrency(v) {
    try { return '₹' + Number(v).toLocaleString(); } catch(e) { return v; }
  }

  function generateReasons(pred, form) {
    const reasons = [];
    const income = Number(form.applicant_income || form.applicant_income === 0 ? form.applicant_income : 0);
    const coIncome = Number(form.coapplicant_income || 0);
    const loan = Number(form.loan_amount || 0);
    const term = Number(form.loan_term || 0);
    const credit = String(form.credit_history || '0');
    const totalIncome = income + coIncome;

    if (pred === 'Eligible') {
      if (credit === '1') reasons.push({ok:true,title:'Good Credit History',explanation:'Applicant has a positive credit history that supports approval.'});
      if (totalIncome >= 5000 && loan <= totalIncome * 10) reasons.push({ok:true,title:'Stable Monthly Income',explanation:'Income level and requested loan are within a healthy ratio.'});
      if (loan <= totalIncome * 12) reasons.push({ok:true,title:'Suitable Loan Amount',explanation:'Requested loan amount is appropriate for the applicant\'s income.'});
      if (term >= 120 && term <= 360) reasons.push({ok:true,title:'Appropriate Loan Term',explanation:'Loan term is within typical acceptable ranges.'});
      if (form.property_area && ['Urban','Semiurban'].includes(form.property_area)) reasons.push({ok:true,title:'Property Area Influence',explanation:'Property area indicates lower risk for lending.'});
      if (reasons.length === 0) reasons.push({ok:true,title:'Balanced Profile',explanation:'Overall financial profile satisfies eligibility requirements.'});
    } else {
      if (credit !== '1') reasons.push({ok:false,title:'Poor Credit History',explanation:'Credit history indicates past defaults or issues.'});
      if (totalIncome < 3000) reasons.push({ok:false,title:'Low Monthly Income',explanation:'Income level may be insufficient for the requested loan.'});
      if (loan > totalIncome * 15) reasons.push({ok:false,title:'Loan Amount Exceeds Affordability',explanation:'Requested loan is very high compared to combined income.'});
      if (term < 60) reasons.push({ok:false,title:'Short Loan Term',explanation:'Short term increases monthly repayment burden.'});
      if (reasons.length === 0) reasons.push({ok:false,title:'Weak Applicant Profile',explanation:'Some risk factors in the submitted profile caused rejection.'});
    }
    return reasons;
  }

  function generateSuggestions(pred, form) {
    const suggestions = [];
    suggestions.push('Increase monthly income');
    suggestions.push('Maintain good credit history');
    suggestions.push('Reduce requested loan amount');
    suggestions.push('Increase loan repayment term');
    suggestions.push('Add a co-applicant');
    suggestions.push('Reduce existing liabilities');
    return suggestions;
  }

  function renderPredictionCard(current) {
    try {
      const panel = document.querySelector('.result-panel');
      if (!panel) return;
      const pred = current.prediction ?? '';
      const prob = Math.round((current.probability || 0) * 100);
      const form = current.formData || {};

      const reasons = generateReasons(pred, form);
      const riskLevel = prob >= 75 ? 'Low' : prob >= 45 ? 'Medium' : 'High';

      const html = `
        <div class="prediction-card">
          <div class="prediction-card-header">
            <div class="header-label">AI Loan Decision Snapshot</div>
            <div class="header-copy">A premium underwriting view with outcome, confidence and risk explanation.</div>
          </div>
          <div class="result-top-card">
            <div>
              <div class="top-label">Decision</div>
              <div class="top-value ${pred === 'Eligible' ? 'risk-pill low' : 'risk-pill high'}">${pred === 'Eligible' ? 'Eligible' : 'Not Eligible'}</div>
            </div>
            <div>
              <div class="top-label">Confidence</div>
              <div class="top-progress">
                <span class="percent-value">${prob}%</span>
                <div class="single-progress"><div class="single-progress-fill" style="width:${prob}%"></div></div>
              </div>
            </div>
            <div>
              <div class="top-label">Risk Level</div>
              <div class="top-value risk-pill ${riskLevel === 'Low' ? 'low' : riskLevel === 'Medium' ? 'medium' : 'high'}">${riskLevel}</div>
            </div>
          </div>

          <div class="result-body-grid">
            <div class="reasons-card">
              <h4>Key decision drivers</h4>
              <div class="reason-chip-group">
                ${reasons.map(r => `
                  <div class="reason-chip ${r.ok ? 'ok' : 'bad'}">
                    <span class="chip-icon">${r.ok ? '<i class="fa-solid fa-circle-check"></i>' : '<i class="fa-solid fa-triangle-exclamation"></i>'}</span>
                    <span>${r.title}</span>
                  </div>
                `).join('')}
              </div>
            </div>

            <div class="summary-grid">
              <div class="summary-card small-card">
                <div class="summary-title">Applicant</div>
                <div class="summary-value">${form.applicant_name || '—'}</div>
              </div>
              <div class="summary-card small-card">
                <div class="summary-title">Monthly Income</div>
                <div class="summary-value">${formatCurrency(form.applicant_income || 0)}</div>
              </div>
              <div class="summary-card small-card">
                <div class="summary-title">Loan Amount</div>
                <div class="summary-value">${formatCurrency(form.loan_amount || 0)}</div>
              </div>
              <div class="summary-card small-card">
                <div class="summary-title">Loan Term</div>
                <div class="summary-value">${form.loan_term ? `${form.loan_term} months` : '—'}</div>
              </div>
              <div class="summary-card small-card">
                <div class="summary-title">Property Area</div>
                <div class="summary-value">${form.property_area || '—'}</div>
              </div>
              <div class="summary-card small-card">
                <div class="summary-title">Credit Status</div>
                <div class="summary-value">${form.credit_history === '1' ? 'Good' : 'Bad'}</div>
              </div>
            </div>
          </div>

          <div class="action-row">
            <button class="btn-primary" id="downloadReportBtn2"><i class="fa-solid fa-file-arrow-down"></i> Download PDF</button>
            <button class="btn-secondary" id="printReportBtn2"><i class="fa-solid fa-print"></i> Print Report</button>
          </div>
        </div>
      `;

      panel.innerHTML = html;

      setTimeout(() => {
        const dr = document.getElementById('downloadReportBtn2');
        const pr = document.getElementById('printReportBtn2');
        if (dr) dr.addEventListener('click', () => downloadReport(current));
        if (pr) pr.addEventListener('click', () => printReport(current));
      }, 50);
    } catch (e) {
      console.error('renderPredictionCard error', e);
    }
  }

  function downloadReport(current) {
    const html = buildReportHtml(current);
    const w = window.open('', '_blank');
    if (!w) { alert('Pop-up blocked. Please allow pop-ups to download report.'); return; }
    w.document.write(html);
    w.document.close();
  }

  function printReport(current) {
    const html = buildReportHtml(current);
    const w = window.open('', '_blank');
    if (!w) { alert('Pop-up blocked. Please allow pop-ups to print report.'); return; }
    w.document.write(html);
    w.document.close();
    w.focus();
    w.print();
  }

  function buildReportHtml(current) {
    const now = new Date().toLocaleString();
    const pred = current.prediction ?? '';
    const prob = Math.round((current.probability || 0) * 100);
    const form = current.formData || {};
    const reasons = generateReasons(pred, form);
    const suggestions = generateSuggestions(pred, form);
    return `<!doctype html><html><head><meta charset="utf-8"><title>Prediction Report</title><style>body{font-family:Inter,Arial;padding:24px;color:#222}h1{color:#ff8a3d}</style></head><body><h1>AI Loan Prediction Report</h1><p>${now}</p><h2>Applicant</h2><p>${form.applicant_name||''}</p><h2>Prediction</h2><p>${pred} (${prob}%)</p><h3>Reasons</h3><ul>${reasons.map(r=>`<li>${r.title}: ${r.explanation}</li>`).join('')}</ul><h3>Suggestions</h3><ul>${suggestions.map(s=>`<li>${s}</li>`).join('')}</ul></body></html>`;
  }

  function initAssistant() {
    try {
      // render assistant actions
      const downloadBtn = document.getElementById('downloadReportBtn');
      const printBtn = document.getElementById('printReportBtn');
      const newBtn = document.getElementById('newPredictionBtn');
      if (downloadBtn) downloadBtn.addEventListener('click', () => { if (window.currentPrediction) downloadReport(window.currentPrediction); else alert('No prediction available'); });
      if (printBtn) printBtn.addEventListener('click', () => { if (window.currentPrediction) printReport(window.currentPrediction); else alert('No prediction available'); });
      if (newBtn) newBtn.addEventListener('click', () => window.location.href = window.location.pathname);

      // If prediction present from server, render redesigned card
      if (window.currentPrediction) {
        renderPredictionCard(window.currentPrediction);
      }
    } catch (e) { console.error('initAssistant error', e); }
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initAssistant); else initAssistant();
})();
