import React from 'react';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import InputPlayer from './pages/InputPlayer';
import OverUnderValued from './pages/OverUnderValued';
import './App.css'; // Import your CSS file for styling

import MessiImage from './assets/messi.jpeg';
import RonaldoImage from './assets/ronaldo.jpeg';

function App() {
  return (
    <Router>
      <div className="App">
        <img src={MessiImage} alt="Messi" className="side-image left-image" />
        <div className="App-main">
          <header className="App-header">
            <h1>Football Player Valuation Predictor</h1>
            <nav className="App-nav">
              <Link to="/input-player" className="App-link">
                Input a Player
              </Link>
              <Link to="/over-undervalued" className="App-link">
                Most Overvalued/Undervalued Players
              </Link>
            </nav>
          </header>
          <main className="App-content">
            <Routes>
              <Route path="/input-player" element={<InputPlayer />} />
              <Route path="/over-undervalued" element={<OverUnderValued />} />
            </Routes>
          </main>
          <footer className="App-footer">
            <p>© 2024 Football Valuation Predictor. All rights reserved.</p>
          </footer>
        </div>
        <img src={RonaldoImage} alt="Ronaldo" className="side-image right-image" />
      </div>
    </Router>
  );
}

export default App;
