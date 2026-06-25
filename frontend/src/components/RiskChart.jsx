import { useMemo } from 'react'
import {
    Chart as ChartJS,
    RadialLinearScale,
    PointElement,
    LineElement,
    Filler,
    Tooltip,
    Legend,
    CategoryScale,
    LinearScale,
    BarElement
} from 'chart.js'
import { Radar, Bar } from 'react-chartjs-2'
import './RiskChart.css'

ChartJS.register(
    RadialLinearScale,
    PointElement,
    LineElement,
    Filler,
    Tooltip,
    Legend,
    CategoryScale,
    LinearScale,
    BarElement
)

function RiskChart({ predictions }) {
    const hasData = predictions && Object.keys(predictions).length > 0

    const sortedPredictions = useMemo(() => {
        if (!hasData) return []
        // Sort by probability descending
        return Object.entries(predictions)
            .sort(([, a], [, b]) => b - a)
    }, [predictions, hasData])

    const chartData = useMemo(() => {
        if (!hasData || sortedPredictions.length === 0) return null

        const labels = sortedPredictions.map(([d]) =>
            d.charAt(0).toUpperCase() + d.slice(1).replace('_', ' ')
        )
        const values = sortedPredictions.map(([, v]) => v * 100)

        const colors = {
            diabetes: 'rgba(255, 99, 132, 0.7)',
            obesity: 'rgba(54, 162, 235, 0.7)',
            hypertension: 'rgba(255, 206, 86, 0.7)',
            cancer: 'rgba(75, 192, 192, 0.7)',
            'cardiovascular disease': 'rgba(153, 102, 255, 0.7)',
            cardiovascular: 'rgba(153, 102, 255, 0.7)'
        }

        const backgroundColors = sortedPredictions.map(([d]) => colors[d] || 'rgba(13, 148, 136, 0.7)')

        return {
            labels,
            datasets: [{
                label: 'Risk Probability (%)',
                data: values,
                backgroundColor: backgroundColors,
                borderColor: backgroundColors.map(c => c.replace('0.7', '1')),
                borderWidth: 2
            }]
        }
    }, [sortedPredictions, hasData])

    const barOptions = {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: 'y',
        plugins: {
            legend: { display: false },
            tooltip: {
                callbacks: {
                    label: (context) => `${context.parsed.x.toFixed(1)}% risk`
                }
            }
        },
        scales: {
            x: {
                beginAtZero: true,
                max: 100,
                title: { display: true, text: 'Risk Probability (%)' }
            }
        }
    }

    const radarOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { display: false }
        },
        scales: {
            r: {
                beginAtZero: true,
                max: 100,
                ticks: { stepSize: 20 }
            }
        }
    }

    if (!hasData) {
        return (
            <div className="risk-chart empty">
                <p>No risk predictions available</p>
            </div>
        )
    }

    return (
        <div className="risk-chart">
            <div className="chart-container bar-chart">
                <Bar data={chartData} options={barOptions} />
            </div>

            <div className="risk-summary">
                {sortedPredictions.map(([disease, prob]) => {
                    const level = prob > 0.6 ? 'high' : prob > 0.3 ? 'moderate' : 'low'
                    return (
                        <div key={disease} className={`risk-item ${level}`}>
                            <span className="disease-name">
                                {disease.charAt(0).toUpperCase() + disease.slice(1).replace('_', ' ')}
                            </span>
                            <div className="risk-bar-container">
                                <div
                                    className={`risk-bar ${level}`}
                                    style={{ width: `${prob * 100}%` }}
                                />
                            </div>
                            <span className="risk-value">{(prob * 100).toFixed(1)}%</span>
                            <span className={`risk-badge ${level}`}>{level}</span>
                        </div>
                    )
                })}
            </div>
        </div>
    )
}

export default RiskChart
