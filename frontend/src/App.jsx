import { useState, useCallback } from "react";
import VoiceInput from "./components/VoiceInput";
import SearchBar from "./components/SearchBar";
import RiskChart from "./components/RiskChart";
import EvidenceCard from "./components/EvidenceCard";
import Suggestions from "./components/Suggestions";
import "./App.css";

const API_BASE = "/api";

function App() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const analyzeIngredient = useCallback(async (searchData) => {
    // Handle both simple string (from voice) and object (from SearchBar)
    const isSimple =
      typeof searchData === "string" || searchData.mode === "simple";
    const queryStr =
      typeof searchData === "string"
        ? searchData
        : searchData.query ||
          searchData.ingredients?.map((i) => i.name).join(", ");

    if (!queryStr?.trim()) return;

    setLoading(true);
    setError(null);
    setQuery(queryStr);

    try {
      let response;

      if (isSimple || typeof searchData === "string") {
        // Simple mode - use original /api/analyze endpoint
        response = await fetch(`${API_BASE}/analyze`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query:
              typeof searchData === "string" ? searchData : searchData.query,
            include_explanation: true,
          }),
        });
      } else {
        // Quantity mode - use /api/analyze/v2 endpoint
        response = await fetch(`${API_BASE}/analyze/v2`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            ingredients: searchData.ingredients,
            include_explanation: true,
          }),
        });
      }

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`);
      }

      const data = await response.json();

      // Normalize response format for compatibility
      const normalizedResult = {
        ingredient: data.ingredient || data.ingredients?.join(", "),
        risk_predictions: data.risk_predictions,
        evidence: data.evidence,
        explanation: data.explanation,
        suggestions: data.suggestions,
        // V2 specific fields
        base_predictions: data.base_predictions,
        dose_response_info: data.dose_response_info,
        quantity_summary: data.quantity_summary,
      };

      setResult(normalizedResult);
    } catch (err) {
      setError(err.message);
      console.error("Analysis error:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleVoiceResult = useCallback(
    (transcript) => {
      if (transcript) {
        analyzeIngredient(transcript);
      }
    },
    [analyzeIngredient],
  );

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <h1>Food LensAI: Analysis starts here 👇</h1>
          <p className="tagline">
            AI-powered nutritional insights with scientific evidence
          </p>
        </div>
      </header>

      <main className="app-main">
        <section className="search-section">
          <div className="search-container">
            <SearchBar onSearch={analyzeIngredient} loading={loading} />
            <div className="voice-divider">
              <span>or</span>
            </div>
            <VoiceInput onResult={handleVoiceResult} />
          </div>
        </section>

        {loading && (
          <div className="loading-container">
            <div className="loading-spinner"></div>
            <p>Analyzing "{query}"...</p>
          </div>
        )}

        {error && (
          <div className="error-container">
            <p>⚠️ {error}</p>
          </div>
        )}

        {result && !loading && (
          <div className="results-container">
            <h2 className="results-title">
              Analysis for:{" "}
              <span className="highlight">{result.ingredient}</span>
            </h2>

            {/* Quantity Summary Banner (V2 only) */}
            {result.quantity_summary && (
              <div className="quantity-summary-banner">
                <span className="quantity-icon">⚖️</span>
                <div className="quantity-details">
                  <strong>Quantity Analysis:</strong> {result.quantity_summary}
                </div>
              </div>
            )}

            {/* Dose Response Info (V2 only) */}
            {result.dose_response_info &&
              result.dose_response_info.length > 0 && (
                <div className="dose-response-section">
                  <h3>📊 Dose-Response Analysis</h3>
                  <div className="dose-cards">
                    {result.dose_response_info.map((info, idx) => (
                      <div key={idx} className={`dose-card ${info.risk_level}`}>
                        <div className="dose-ingredient">{info.ingredient}</div>
                        <div className="dose-amount">
                          {info.amount}
                          {info.unit}
                        </div>
                        <div className="dose-rda">
                          {info.is_known_nutrient ? (
                            <>
                              <span className="rda-percent">
                                {info.pct_rda.toFixed(0)}%
                              </span>
                              <span className="rda-label">
                                of daily{" "}
                                {info.rda_unit === "g" ? "limit" : "value"}
                              </span>
                            </>
                          ) : (
                            <span className="rda-unknown">
                              RDA not available
                            </span>
                          )}
                        </div>
                        <div className={`dose-badge ${info.risk_level}`}>
                          {info.risk_level.replace("_", " ")}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

            <div className="results-grid">
              <div className="chart-section">
                <h3>🎯 Health Risk Probabilities</h3>
                {result.base_predictions && (
                  <p className="risk-note">
                    <em>
                      Adjusted from base predictions based on quantity consumed
                    </em>
                  </p>
                )}
                <RiskChart predictions={result.risk_predictions} />
              </div>

              <div className="suggestions-section">
                <h3>💡 Dietary Suggestions</h3>
                <Suggestions suggestions={result.suggestions} />
              </div>
            </div>

            <div className="explanation-section">
              <h3>📋 AI Analysis</h3>
              <div className="explanation-content">{result.explanation}</div>
            </div>

            <div className="evidence-section">
              <h3>📚 Scientific Evidence</h3>
              <div className="evidence-grid">
                {Object.entries(result.evidence).map(
                  ([disease, papers]) =>
                    papers.length > 0 && (
                      <EvidenceCard
                        key={disease}
                        disease={disease}
                        papers={papers}
                      />
                    ),
                )}
              </div>
            </div>
          </div>
        )}

        {!result && !loading && !error && (
          <div className="welcome-container">
            <div className="welcome-card">
              <h2>🔬 How It Works</h2>
              <div className="steps">
                <div className="step">
                  <span className="step-icon">1️⃣</span>
                  <p>Enter an ingredient or use voice input</p>
                </div>
                <div className="step">
                  <span className="step-icon">2️⃣</span>
                  <p>
                    Optionally specify quantities for dose-response analysis
                  </p>
                </div>
                <div className="step">
                  <span className="step-icon">3️⃣</span>
                  <p>Our AI analyzes health risks using GNN predictions</p>
                </div>
                <div className="step">
                  <span className="step-icon">4️⃣</span>
                  <p>Get evidence from 30,000+ PubMed studies</p>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      <footer className="app-footer">
        <p>
          Powered by Knowledge Graph + GNN + RAG Pipeline + Dose-Response
          Analysis
        </p>
      </footer>
    </div>
  );
}

export default App;
