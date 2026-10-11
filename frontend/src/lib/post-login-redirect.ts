const PENDING_KEY = "signbridge:postLoginRedirect";

export function storePostLoginRedirect(path: string | undefined) {
  try {
    if (path) sessionStorage.setItem(PENDING_KEY, path);
  } catch {
    /* storage unavailable */
  }
}

export function consumePostLoginRedirect(): string | undefined {
  try {
    const v = sessionStorage.getItem(PENDING_KEY);
    sessionStorage.removeItem(PENDING_KEY);
    return v ?? undefined;
  } catch {
    return undefined;
  }
}
