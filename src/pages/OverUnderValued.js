import React, { useEffect, useState } from 'react';
import axios from 'axios';

function OverUnderValued() {
  const [players, setPlayers] = useState({ overvalued: [], undervalued: [] });
  const [loading, setLoading] = useState(true); // State to track loading state
  const [error, setError] = useState(null); // State to track error

  useEffect(() => {
    const fetchPlayers = async () => {
      try {
        const response = await axios.get('http://localhost:8000/over-under-valued');
        setPlayers(response.data);
        setLoading(false); // Set loading to false after data is fetched
      } catch (error) {
        console.error('Error fetching players:', error);
        setError('Error fetching data: ' + error.message); // Set error state with detailed message
        setLoading(false); // Ensure loading is set to false on error
      }
    };
    fetchPlayers();
  }, []);

  if (loading) {
    return <p>Loading...</p>; // Display a loading indicator while fetching data
  }

  if (error) {
    return <p>{error}</p>; // Display an error message with detailed error description
  }

  // Display the fetched data
  return (
    <div>
      <h2>Most Overvalued/Undervalued Players</h2>
      <div>
        <h3>Overvalued Players</h3>
        <ul>
          {players.overvalued.map((player, index) => (
            <li key={index}>
              <strong>{player.Player}</strong> - Actual: {player.Actual}, Predicted: {Math.trunc(player.Predicted)}, Overvaluation: {Math.trunc(player.Undervaluation) * -1}
            </li>
          ))}
        </ul>
      </div>
      <div>
        <h3>Undervalued Players</h3>
        <ul>
          {players.undervalued.map((player, index) => (
            <li key={index}>
              <strong>{player.Player}</strong> - Actual: {player.Actual}, Predicted: {Math.trunc(player.Predicted)}, Undervaluation: {Math.trunc(player.Undervaluation)}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export default OverUnderValued;
