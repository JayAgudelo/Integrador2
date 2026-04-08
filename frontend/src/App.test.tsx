import React from "react";
import { render, screen } from "@testing-library/react";
import App from "./App";
import { I18nProvider } from "./i18n";

test("renders the Sonic Architect home shell", () => {
  render(
    <I18nProvider>
      <App />
    </I18nProvider>
  );
  expect(screen.getByText(/Sonic Architect/i)).toBeInTheDocument();
  expect(screen.getByRole("heading", { name: /streaming potential/i })).toBeInTheDocument();
});
