import './Suggestions.css'

function Suggestions({ suggestions }) {
    if (!suggestions || suggestions.length === 0) {
        return null
    }

    return (
        <div className="suggestions">
            <ul className="suggestions-list">
                {suggestions.map((suggestion, index) => (
                    <li key={index} className="suggestion-item">
                        {suggestion}
                    </li>
                ))}
            </ul>
        </div>
    )
}

export default Suggestions
