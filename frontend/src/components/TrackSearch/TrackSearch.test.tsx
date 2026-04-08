import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import TrackSearch from "./TrackSearch";
import { I18nProvider } from "../../i18n";

jest.mock("canvas-confetti", () => jest.fn());

function renderTrackSearch() {
  return render(
    <I18nProvider>
      <TrackSearch onAnalysisComplete={jest.fn()} onTryAlternateRoute={jest.fn()} />
    </I18nProvider>
  );
}

test("requires spotify source and genre before spotify analysis can start", () => {
  renderTrackSearch();

  const analyzeButton = screen.getByRole("button", { name: /analyze spotify track/i });
  expect(analyzeButton).toBeDisabled();

  fireEvent.change(screen.getByLabelText(/spotify track source/i), { target: { value: "spotify:track:12345" } });
  expect(analyzeButton).toBeDisabled();

  fireEvent.change(screen.getByLabelText(/genre hint/i), { target: { value: "rock" } });
  expect(analyzeButton).toBeEnabled();
});
