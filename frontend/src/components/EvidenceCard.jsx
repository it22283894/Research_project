import './EvidenceCard.css'

function EvidenceCard({ disease, papers }) {
    if (!papers || papers.length === 0) return null

    const diseaseColors = {
        diabetes: '#FF6384',
        obesity: '#36A2EB',
        hypertension: '#FFCE56',
        cancer: '#4BC0C0',
        'cardiovascular disease': '#9966FF',
        cardiovascular: '#9966FF'
    }

    const color = diseaseColors[disease] || '#667eea'

    return (
        <div className="evidence-card" style={{ borderTopColor: color }}>
            <div className="evidence-header" style={{ color }}>
                <h4>{disease.charAt(0).toUpperCase() + disease.slice(1).replace('_', ' ')}</h4>
                <span className="paper-count">{papers.length} papers</span>
            </div>

            <ul className="evidence-list">
                {papers.map((paper, index) => (
                    <li key={index} className="evidence-item">
                        <div className="paper-title">{paper.title || 'Untitled study'}</div>
                        <div className="paper-meta">
                            <span className="journal">{paper.journal || 'Unknown Journal'}</span>
                            {paper.year && <span className="year">({paper.year})</span>}
                        </div>
                        {paper.pmid && (
                            <a
                                href={`https://pubmed.ncbi.nlm.nih.gov/${paper.pmid}/`}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="pubmed-link"
                            >
                                PMID: {paper.pmid} →
                            </a>
                        )}
                    </li>
                ))}
            </ul>
        </div>
    )
}

export default EvidenceCard
