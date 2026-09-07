import { apiUrl, fetchApi } from "./apiBase";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

export type AuthUser = {
  email: string;
  name?: string;
  picture?: string;
  credits?: number;
};

type AuthConfig = {
  auth_required: boolean;
  google_client_id: string | null;
  initial_credits?: number;
  credits_per_document?: number;
  credits_per_page?: number;
};

type AuthContextValue = {
  user: AuthUser | null;
  token: string | null;
  authRequired: boolean;
  googleClientId: string | null;
  creditsPerDocument: number;
  creditsPerPage: number;
  initialCredits: number;
  loading: boolean;
  signInWithGoogle: (credential: string) => Promise<void>;
  signInWithEmail: (email: string, password: string) => Promise<void>;
  registerWithEmail: (email: string, password: string) => Promise<void>;
  setUserCredits: (credits: number) => void;
  signOut: () => void;
};

const STORAGE_KEY = "docparse_session";

const AuthContext = createContext<AuthContextValue | null>(null);

async function parseAuthError(res: Response): Promise<string> {
  const err = await res.json().catch(() => ({}));
  const detail = err.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail[0]?.msg) return String(detail[0].msg);
  return "Sign-in failed";
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [authRequired, setAuthRequired] = useState(true);
  const [googleClientId, setGoogleClientId] = useState<string | null>(null);
  const [creditsPerDocument, setCreditsPerDocument] = useState(2);
  const [creditsPerPage, setCreditsPerPage] = useState(2);
  const [initialCredits, setInitialCredits] = useState(20);
  const [loading, setLoading] = useState(true);

  const persist = useCallback((nextToken: string, nextUser: AuthUser) => {
    setToken(nextToken);
    setUser(nextUser);
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ token: nextToken, user: nextUser }));
  }, []);

  useEffect(() => {
    const raw = localStorage.getItem(STORAGE_KEY);
    let savedToken: string | null = null;
    if (raw) {
      try {
        const saved = JSON.parse(raw) as { token: string; user: AuthUser };
        savedToken = saved.token;
        setToken(saved.token);
        setUser(saved.user);
      } catch {
        localStorage.removeItem(STORAGE_KEY);
      }
    }

    fetchApi("/api/auth/config")
      .then((r) => r.json())
      .then((cfg: AuthConfig) => {
        setAuthRequired(cfg.auth_required);
        const envGoogle = import.meta.env.VITE_GOOGLE_CLIENT_ID?.trim() || null;
        setGoogleClientId(cfg.google_client_id || envGoogle);
        if (typeof cfg.credits_per_page === "number") setCreditsPerPage(cfg.credits_per_page);
        else if (typeof cfg.credits_per_document === "number") setCreditsPerPage(cfg.credits_per_document);
        if (typeof cfg.credits_per_document === "number") setCreditsPerDocument(cfg.credits_per_document);
        if (typeof cfg.initial_credits === "number") setInitialCredits(cfg.initial_credits);
      })
      .catch(() => setAuthRequired(true))
      .finally(() => setLoading(false));

    if (savedToken) {
      fetch(apiUrl("/api/auth/me"), {
        headers: { Authorization: `Bearer ${savedToken}` },
      })
        .then((r) => (r.ok ? r.json() : null))
        .then((me) => {
          if (!me?.email) return;
          setUser((prev) => {
            const next: AuthUser = {
              email: me.email,
              name: me.name ?? prev?.name,
              picture: me.picture ?? prev?.picture,
              credits: me.credits,
            };
            const t = savedToken;
            if (t) {
              localStorage.setItem(STORAGE_KEY, JSON.stringify({ token: t, user: next }));
            }
            return next;
          });
        })
        .catch(() => undefined);
    }
  }, []);

  const applyAuthResponse = useCallback(
    (data: { token: string; user: AuthUser }) => {
      persist(data.token, data.user);
    },
    [persist],
  );

  const signInWithGoogle = useCallback(
    async (credential: string) => {
      const res = await fetch(apiUrl("/api/auth/google"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id_token: credential }),
      });
      if (!res.ok) throw new Error(await parseAuthError(res));
      const data = await res.json();
      applyAuthResponse(data);
    },
    [applyAuthResponse],
  );

  const signInWithEmail = useCallback(
    async (email: string, password: string) => {
      const res = await fetch(apiUrl("/api/auth/login"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      if (!res.ok) throw new Error(await parseAuthError(res));
      const data = await res.json();
      applyAuthResponse(data);
    },
    [applyAuthResponse],
  );

  const registerWithEmail = useCallback(
    async (email: string, password: string) => {
      const res = await fetch(apiUrl("/api/auth/register"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      if (!res.ok) throw new Error(await parseAuthError(res));
      const data = await res.json();
      applyAuthResponse(data);
    },
    [applyAuthResponse],
  );

  const setUserCredits = useCallback((credits: number) => {
    setUser((prev) => {
      if (!prev) return prev;
      const next = { ...prev, credits };
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {
        try {
          const saved = JSON.parse(raw) as { token: string; user: AuthUser };
          localStorage.setItem(STORAGE_KEY, JSON.stringify({ token: saved.token, user: next }));
        } catch {
          /* ignore */
        }
      }
      return next;
    });
  }, []);

  const signOut = useCallback(() => {
    setToken(null);
    setUser(null);
    localStorage.removeItem(STORAGE_KEY);
  }, []);

  const value = useMemo(
    () => ({
      user,
      token,
      authRequired,
      googleClientId,
      creditsPerDocument,
      creditsPerPage,
      initialCredits,
      loading,
      signInWithGoogle,
      signInWithEmail,
      registerWithEmail,
      setUserCredits,
      signOut,
    }),
    [
      user,
      token,
      authRequired,
      googleClientId,
      creditsPerDocument,
      creditsPerPage,
      initialCredits,
      loading,
      signInWithGoogle,
      signInWithEmail,
      registerWithEmail,
      setUserCredits,
      signOut,
    ],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
