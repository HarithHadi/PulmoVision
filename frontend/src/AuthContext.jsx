// src/AuthContext.jsx
import { createContext, useContext, useState } from "react";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [token, setToken] = useState(null);
  const [radiologistName, setRadiologistName] = useState(null);

  const login = (accessToken, radId) => {
    setToken(accessToken);
    setRadiologistName(`Dr. ${radId}`);
  };

  const logout = () => {
    setToken(null);
    setRadiologistName(null);
  };

  return (
    <AuthContext.Provider value={{ token, radiologistName, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}