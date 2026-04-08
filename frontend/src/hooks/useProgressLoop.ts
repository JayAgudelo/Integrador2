import { useEffect, useState } from "react";

type ProgressStatus = "idle" | "loading" | "success" | "error";

export default function useProgressLoop(status: ProgressStatus): number {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (status !== "loading") {
      return undefined;
    }

    setProgress(12);

    const intervalId = window.setInterval(() => {
      setProgress((current) => {
        const next = current + 7 + Math.random() * 12;
        return next >= 92 ? 18 : Math.min(next, 92);
      });
    }, 220);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [status]);

  useEffect(() => {
    if (status === "success") {
      setProgress(100);
      return;
    }

    if (status === "error" || status === "idle") {
      setProgress(0);
    }
  }, [status]);

  return progress;
}
