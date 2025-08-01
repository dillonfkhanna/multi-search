import { useState, useRef, useEffect } from "react";
import "./SearchBar.css";

const SearchBar = () => {
  const [searchQuery, setSearchQuery] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [isFading, setIsFading] = useState(false);
  const [isFadingIn, setIsFadingIn] = useState(false);
  const typingTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const fadingTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const fadeInTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setSearchQuery(value);
    
    // Clear any existing timeouts
    if (typingTimeoutRef.current) {
      clearTimeout(typingTimeoutRef.current);
    }
    if (fadingTimeoutRef.current) {
      clearTimeout(fadingTimeoutRef.current);
    }
    if (fadeInTimeoutRef.current) {
      clearTimeout(fadeInTimeoutRef.current);
    }
    
    // If not already typing, start with fade-in
    if (!isTyping) {
      setIsFadingIn(true);
      setIsTyping(true);
      setIsFading(false);
      
      // Remove fade-in state after fade-in completes
      fadeInTimeoutRef.current = setTimeout(() => {
        setIsFadingIn(false);
      }, 600); // Match CSS fade-in duration
    } else {
      // Already typing, just continue
      setIsFading(false);
      setIsFadingIn(false);
    }
    
    // Set a timeout to start fading after user stops typing
    typingTimeoutRef.current = setTimeout(() => {
      setIsFading(true);
      
      // Then stop typing completely after fade duration
      fadingTimeoutRef.current = setTimeout(() => {
        setIsTyping(false);
        setIsFading(false);
      }, 1200); // Match CSS transition duration
    }, 1000); // Start fading 1 second after user stops typing
    
    // Call the search function as user types
    handleSearch(value);
  };

  const handleSearch = (query: string) => {
    // TODO: Implement search functionality
    // This method will be used for search as the user types
    console.log("Searching for:", query);
  };

  // Cleanup timeouts on unmount
  useEffect(() => {
    return () => {
      if (typingTimeoutRef.current) {
        clearTimeout(typingTimeoutRef.current);
      }
      if (fadingTimeoutRef.current) {
        clearTimeout(fadingTimeoutRef.current);
      }
      if (fadeInTimeoutRef.current) {
        clearTimeout(fadeInTimeoutRef.current);
      }
    };
  }, []);

  return (
    <div className={`search-container ${isTyping ? 'typing' : ''} ${isFading ? 'fading' : ''} ${isFadingIn ? 'fading-in' : ''}`}>
      <div className="search-content">
        {/* Search Icon (from Feather Icons) */}
        <svg 
          xmlns="http://www.w3.org/2000/svg" 
          viewBox="0 0 24 24" 
          fill="none" 
          stroke="currentColor" 
          strokeWidth="2" 
          strokeLinecap="round" 
          strokeLinejoin="round" 
          className="search-icon"
        >
          <circle cx="11" cy="11" r="8"></circle>
          <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
        </svg>

        {/* The Actual Input Field */}
        <input 
          type="text" 
          className="search-input" 
          placeholder="Search anything..."
          value={searchQuery}
          onChange={handleInputChange}
          autoFocus
        />
      </div>
    </div>
  );
};

export default SearchBar;
