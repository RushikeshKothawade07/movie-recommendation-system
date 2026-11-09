/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        cinema: {
          950: '#0B0F19',
          900: '#111827',
          800: '#1F2937',
          accent: '#E50914',
          gold: '#F5C518'
        }
      }
    },
  },
  plugins: [],
}
