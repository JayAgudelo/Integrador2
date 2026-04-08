/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        app: "#0a0e14",
        "app-surface": "#111720",
        "app-surface-bright": "#18202b",
        "app-text": "#f1f3fc",
        "app-muted": "#a8abb3",
        "app-ink": "#042d30",
        "app-primary": "#a1faff",
        "app-primary-strong": "#00f4fe",
        "app-secondary": "#d873ff",
        "app-success": "#a7ffb3",
      },
      fontFamily: {
        display: ["Space Grotesk", "sans-serif"],
        body: ["Inter", "sans-serif"],
      },
    },
  },
  plugins: [],
};
