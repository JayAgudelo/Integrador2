export const BACKEND_URL =
  process.env.REACT_APP_BACKEND_URL ||
  (process.env.NODE_ENV === "production"
    ? "https://integrador2-37j4.onrender.com"
    : "http://localhost:8000");
