import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from './assets/vite.svg'
import heroImg from './assets/hero.png'
import './App.css'
import { BrowserRouter, Routes, Route } from "react-router-dom"
import HeroPage from "./pages/HeroPage"
import TBDiagnosis from './pages/Tbdiagnosis'
import Navbar from './components/Navbar'
import PatientRecords from './Pages/PatientRecords'


export default function App() {
  return (
    <BrowserRouter>
    <Navbar />
      <Routes>
        <Route path="/" element={<HeroPage />} />
        <Route path="/diagnose" element={<TBDiagnosis />} />
        <Route path="/patients" element={<PatientRecords />} />
      </Routes>
    </BrowserRouter>
  )
}
