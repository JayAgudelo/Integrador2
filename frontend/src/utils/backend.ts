export const BACKEND_URL =
  process.env.REACT_APP_BACKEND_URL ||
  (process.env.NODE_ENV === "production"
    ? "https://integrador2-production-72e9.up.railway.app"
    : "http://localhost:8000");
