import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import AudioUploader from "./AudioUploader";
import { I18nProvider } from "../../i18n";

jest.mock("canvas-confetti", () => jest.fn());

function renderUploader() {
  return render(
    <I18nProvider>
      <AudioUploader onAnalysisComplete={jest.fn()} onTryAlternateRoute={jest.fn()} />
    </I18nProvider>
  );
}

test("requires both audio file and genre before upload analysis can start", () => {
  renderUploader();

  const analyzeButton = screen.getByRole("button", { name: /analyze my track/i });
  expect(analyzeButton).toBeDisabled();

  const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
  fireEvent.change(fileInput, { target: { files: [new File(["demo"], "demo.mp3", { type: "audio/mp3" })] } });
  expect(analyzeButton).toBeDisabled();

  fireEvent.change(screen.getByLabelText(/primary genre/i), { target: { value: "pop" } });
  expect(analyzeButton).toBeEnabled();
});
