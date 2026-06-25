import { useState } from 'react'
import './SearchBar.css'

function SearchBar({ onSearch, loading }) {
    const [ingredients, setIngredients] = useState([{ name: '', amount: '', unit: 'g' }])
    const [mode, setMode] = useState('simple') // 'simple' or 'quantity'

    const handleSimpleSubmit = (e) => {
        e.preventDefault()
        const query = ingredients[0].name.trim()
        if (query && !loading) {
            onSearch({ query, mode: 'simple' })
        }
    }

    const handleQuantitySubmit = (e) => {
        e.preventDefault()
        const validIngredients = ingredients.filter(ing => ing.name.trim())
        if (validIngredients.length > 0 && !loading) {
            onSearch({
                ingredients: validIngredients.map(ing => ({
                    name: ing.name.trim(),
                    amount: parseFloat(ing.amount) || 100,
                    unit: ing.unit
                })),
                mode: 'quantity'
            })
        }
    }

    const handleKeyPress = (e) => {
        if (e.key === 'Enter') {
            mode === 'simple' ? handleSimpleSubmit(e) : handleQuantitySubmit(e)
        }
    }

    const addIngredient = () => {
        setIngredients([...ingredients, { name: '', amount: '', unit: 'g' }])
    }

    const removeIngredient = (index) => {
        if (ingredients.length > 1) {
            setIngredients(ingredients.filter((_, i) => i !== index))
        }
    }

    const updateIngredient = (index, field, value) => {
        const updated = [...ingredients]
        updated[index][field] = value
        setIngredients(updated)
    }

    const quickSearches = ['protein', 'sugar', 'saturated fat', 'sodium', 'fiber']
    const units = [
        { value: 'g', label: 'grams' },
        { value: 'mg', label: 'mg' },
        { value: '%', label: '% DV' },
        { value: 'ml', label: 'ml' },
        { value: 'tsp', label: 'tsp' },
        { value: 'tbsp', label: 'tbsp' }
    ]

    return (
        <div className="search-bar">
            {/* Mode Toggle */}
            <div className="mode-toggle">
                <button
                    className={`mode-btn ${mode === 'simple' ? 'active' : ''}`}
                    onClick={() => setMode('simple')}
                    type="button"
                >
                    🔍 Simple Search
                </button>
                <button
                    className={`mode-btn ${mode === 'quantity' ? 'active' : ''}`}
                    onClick={() => setMode('quantity')}
                    type="button"
                >
                    ⚖️ With Quantities
                </button>
            </div>

            {mode === 'simple' ? (
                /* Simple Mode - Original Search */
                <form onSubmit={handleSimpleSubmit} className="search-form">
                    <div className="search-input-container">
                        <span className="search-icon">🔍</span>
                        <input
                            type="text"
                            value={ingredients[0].name}
                            onChange={(e) => updateIngredient(0, 'name', e.target.value)}
                            onKeyPress={handleKeyPress}
                            placeholder="Enter ingredient (e.g., sugar, sodium, fiber...)"
                            className="search-input"
                            disabled={loading}
                        />
                        <button
                            type="submit"
                            className="search-button"
                            disabled={loading || !ingredients[0].name.trim()}
                        >
                            {loading ? 'Analyzing...' : 'Analyze'}
                        </button>
                    </div>
                </form>
            ) : (
                /* Quantity Mode - With Amount and Unit */
                <form onSubmit={handleQuantitySubmit} className="search-form quantity-form">
                    <div className="ingredients-list">
                        {ingredients.map((ing, index) => (
                            <div key={index} className="ingredient-row">
                                <input
                                    type="text"
                                    value={ing.name}
                                    onChange={(e) => updateIngredient(index, 'name', e.target.value)}
                                    placeholder="Ingredient name"
                                    className="ingredient-name-input"
                                    disabled={loading}
                                />
                                <input
                                    type="number"
                                    value={ing.amount}
                                    onChange={(e) => updateIngredient(index, 'amount', e.target.value)}
                                    placeholder="Amount"
                                    className="ingredient-amount-input"
                                    disabled={loading}
                                    min="0"
                                    step="any"
                                />
                                <select
                                    value={ing.unit}
                                    onChange={(e) => updateIngredient(index, 'unit', e.target.value)}
                                    className="ingredient-unit-select"
                                    disabled={loading}
                                >
                                    {units.map(u => (
                                        <option key={u.value} value={u.value}>{u.label}</option>
                                    ))}
                                </select>
                                {ingredients.length > 1 && (
                                    <button
                                        type="button"
                                        className="remove-ingredient-btn"
                                        onClick={() => removeIngredient(index)}
                                        disabled={loading}
                                    >
                                        ✕
                                    </button>
                                )}
                            </div>
                        ))}
                    </div>
                    <div className="quantity-actions">
                        <button
                            type="button"
                            className="add-ingredient-btn"
                            onClick={addIngredient}
                            disabled={loading}
                        >
                            + Add Ingredient
                        </button>
                        <button
                            type="submit"
                            className="search-button"
                            disabled={loading || !ingredients.some(ing => ing.name.trim())}
                        >
                            {loading ? 'Analyzing...' : 'Analyze with Quantities'}
                        </button>
                    </div>
                </form>
            )}

            <div className="quick-search">
                <span className="quick-label">Quick search:</span>
                <div className="quick-buttons">
                    {quickSearches.map((term) => (
                        <button
                            key={term}
                            onClick={() => onSearch({ query: term, mode: 'simple' })}
                            className="quick-button"
                            disabled={loading}
                        >
                            {term}
                        </button>
                    ))}
                </div>
            </div>
        </div>
    )
}

export default SearchBar
