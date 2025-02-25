// Fonction pour récupérer et afficher les métriques de performance
async function loadMetrics() {
  try {
    const response = await fetch('/api/metrics');
    if (!response.ok) throw new Error("Erreur lors du chargement des métriques");
    const data = await response.json();
    document.getElementById('balance').innerText = `BTC: ${data.balance ? data.balance.BTC : 'N/A'} | USDT: ${data.balance ? data.balance.USDT : 'N/A'}`;
    document.getElementById('pnl_total').innerText = data.pnl_total !== undefined ? data.pnl_total : 'N/A';
    document.getElementById('trade_stats').innerText = data.winning_trades !== undefined ? `${data.winning_trades} / ${data.losing_trades}` : 'N/A';
    document.getElementById('win_rate').innerText = data.win_rate !== undefined ? `${data.win_rate}%` : 'N/A';
  } catch (error) {
    console.error(error);
  }
}

// Fonction pour récupérer et afficher l'historique des trades
async function loadTrades() {
  try {
    const response = await fetch('/api/trades');
    if (!response.ok) throw new Error("Erreur lors du chargement de l'historique des trades");
    const trades = await response.json();
    const tbody = document.getElementById('tradesTable');
    tbody.innerHTML = '';
    trades.forEach(trade => {
      const row = document.createElement('tr');
      row.innerHTML = `
        <td>${trade.id || ''}</td>
        <td>${trade.timestamp || ''}</td>
        <td>${trade.asset || ''}</td>
        <td>${trade.type || ''}</td>
        <td>${trade.quantity || ''}</td>
        <td>${trade.entry_price || ''}</td>
        <td>${trade.exit_price || ''}</td>
        <td>${trade.stop_loss || ''} / ${trade.take_profit || ''}</td>
        <td>${trade.pnl_amount || ''} / ${trade.pnl_percent || ''}%</td>
        <td>${trade.fees || ''}</td>
        <td>${trade.status || ''}</td>
        <td>${trade.strategy || ''}</td>
        <td>${trade.indicators || ''}</td>
        <td>${trade.confidence_score || ''}</td>
      `;
      tbody.appendChild(row);
    });
  } catch (error) {
    console.error(error);
  }
}

// Fonction pour charger et afficher les données du graphique de performance
async function loadPerformanceChart() {
  try {
    const response = await fetch('/api/performance');
    if (!response.ok) throw new Error("Erreur lors du chargement des données de performance");
    const data = await response.json();
    const labels = data.map(point => point.time);
    const pnlValues = data.map(point => point.pnl);

    const ctx = document.getElementById('pnlChart').getContext('2d');
    // Si un graphique existe déjà, le détruire avant de recréer
    if (window.pnlChart) {
      window.pnlChart.destroy();
    }
    window.pnlChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [{
          label: 'PnL (USD)',
          data: pnlValues,
          borderColor: 'rgba(75, 192, 192, 1)',
          backgroundColor: 'rgba(75, 192, 192, 0.2)',
          fill: true,
          tension: 0.3
        }]
      },
      options: {
        scales: {
          y: {
            beginAtZero: false,
            ticks: { color: '#f5f5f5' }
          },
          x: {
            ticks: { color: '#f5f5f5' }
          }
        },
        plugins: {
          legend: { labels: { color: '#f5f5f5' } }
        }
      }
    });
  } catch (error) {
    console.error(error);
  }
}

// Fonction pour charger les données de risque
async function loadRiskData() {
  try {
    const response = await fetch('/api/risk');
    if (!response.ok) throw new Error("Erreur lors du chargement des données de risque");
    const data = await response.json();
    document.getElementById('drawdown').innerText = data.max_drawdown !== undefined ? data.max_drawdown : 'N/A';
    document.getElementById('maxExposure').innerText = data.max_exposure !== undefined ? data.max_exposure : 'N/A';
    document.getElementById('riskReward').innerText = data.risk_reward_ratio !== undefined ? data.risk_reward_ratio : 'N/A';
  } catch (error) {
    console.error(error);
  }
}

// Fonction d'exportation des données en CSV
async function exportData() {
  try {
    const response = await fetch('/api/export');
    if (!response.ok) throw new Error("Erreur lors de l'exportation des données");
    const trades = await response.json();
    if (!trades || trades.length === 0) return;
    const headers = Object.keys(trades[0]).join(',');
    const rows = trades.map(trade => Object.values(trade).join(','));
    const csvContent = headers + "\n" + rows.join("\n");

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'trades_export.csv';
    a.click();
    URL.revokeObjectURL(url);
  } catch (error) {
    console.error(error);
  }
}

// Fonction pour rafraîchir l'ensemble des données du dashboard
function refreshDashboard() {
  loadMetrics();
  loadTrades();
  loadPerformanceChart();
  loadRiskData();
}

// Initialisation et rafraîchissement périodique (toutes les 60 secondes)
function initDashboard() {
  refreshDashboard();
  document.getElementById('exportBtn').addEventListener('click', exportData);
  document.getElementById('refreshBtn').addEventListener('click', refreshDashboard);
  setInterval(refreshDashboard, 60000);
}

// Exécution après chargement de la page
window.onload = initDashboard;
