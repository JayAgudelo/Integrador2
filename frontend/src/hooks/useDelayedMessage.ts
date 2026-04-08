import { useEffect, useState } from "react";

export default function useDelayedMessage(
  active: boolean,
  delayMs: number = 8000,
): boolean {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    if (!active) {
      setVisible(false);
      return undefined;
    }

    const timeoutId = window.setTimeout(() => setVisible(true), delayMs);

    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [active, delayMs]);

  return visible;
}
