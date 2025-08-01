import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import SearchBar from './SearchBar'; // Import your search bar component
import "./App.css";

// This component will be our main settings page content
function MainPage() {
  return (
    <main className="container">
      <h1>Multi-Search Settings</h1>
      <p>This is where connectors and other settings will go.</p>
    </main>
  );
}

// The App component is now the router for the whole application
function App() {
  return (
    <Router>
      <Routes>
        {/* Route for the launcher window */}
        <Route path="/launcher" element={<SearchBar />} />
        {/* Route for the main settings window */}
        <Route path="/" element={<MainPage />} />
      </Routes>
    </Router>
  );
}

export default App;