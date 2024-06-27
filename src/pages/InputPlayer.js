import React, { useState } from 'react';
import axios from 'axios';

function InputPlayer() {
  const [playerData, setPlayerData] = useState({
    // Initialize with default values
    PassesLeadingToGoals: '',
    Age: '',
    NumberOfTimesPassTarget: '',
    NumberOfTimesReceivedPass: '',
    TouchesInAttackingPenaltyBox: '',
    TotalCarriesInForwardDirection: '',
    ContractYearsLeft: '',
    SuccessfulPressurePercent: '',
    CarriesIntoAttackingPenaltyBox: '',
    GoalCreatingActionsPer90: '',
    GlsAstScoredPenaltiesPer90: '',
    GoalCreatingActions: '',
    GA90: '',
    xG: '',
    League: '',
    Position: '',
    Player: '',
    Value: ''
  });

  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false); // State to manage loading indicator
  const [error, setError] = useState(null); // State to track errors

  const handleChange = (e) => {
    const { name, value } = e.target;
    setPlayerData({
      ...playerData,
      [name]: value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    try {
      const response = await axios.post('http://localhost:8000/predict', playerData);
      setPrediction(response.data);
    } catch (error) {
      console.error('Error making prediction:', error);
      setError('Error making prediction: ' + error.message); // Update error state with detailed message
    }
    setIsLoading(false);
  };

  return (
    <div>
      <h2>Input Player Data</h2>
      <form onSubmit={handleSubmit}>
        {/* Create input fields for each player attribute */}
        {Object.keys(playerData).map((key) => (
          <div key={key}>
            <label>
              {/* Insert spaces between capitalized words */}
              {key.replace(/([A-Z])/g, ' $1').trim()}:
              <input type="text" name={key} value={playerData[key]} onChange={handleChange} required />
            </label>
          </div>
        ))}
        <button type="submit">Predict Value</button>
      </form>
      {isLoading && <p>Loading...</p>}
      {error && <p>{error}</p>}
      {prediction && !isLoading && (
        <div>
          <h3>Prediction Result</h3>
          <p>Actual Value: {playerData.Value}</p>
          <p>Predicted Value: {prediction.predictedValue}</p>
          <p>{prediction.isOvervalued ? 'Overvalued' : 'Undervalued'}</p>
        </div>
      )}
    </div>
  );
}

export default InputPlayer;
