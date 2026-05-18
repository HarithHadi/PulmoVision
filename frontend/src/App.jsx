import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from './assets/vite.svg'
import heroImg from './assets/hero.png'
import './App.css'

function App() {
  const uploadImage = async (event) => {
    const file = event.target.files[0];
    const formData = new FormData();
    formData.append('file',file);

    const response = await fetch('http://127.0.0.1:8000/diagnose', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();
    console.log("Model Response:", data);
  }
  

  return (
    <>
      <h1 class="text-3xl font-bold">
        Pulmovision
      </h1>
    </>
  )
}

export default App
