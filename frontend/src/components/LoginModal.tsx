import { FormEvent, useEffect, useRef, useState } from "react";
import { useAuth } from "../auth";
import { signupCreditsMessage } from "../creditsCopy";

type Props = {
  open: boolean;
  onClose: () => void;
  onSuccess?: () => void;
};

type Mode = "signup" | "signin";

declare global {
  interface Window {
    google?: {
      accounts: {
        id: {
          initialize: (cfg: {
            client_id: string;
            callback: (res: { credential: string }) => void;
          }) => void;
          renderButton: (
            el: HTMLElement,
            cfg: { theme?: string; size?: string; width?: number; text?: string },
          ) => void;
        };
      };
    };
  }
}

export default function LoginModal({ open, onClose, onSuccess }: Props) {
  const {
    googleClientId,
    signInWithGoogle,
    signInWithEmail,
    registerWithEmail,
    creditsPerPage,
    initialCredits,
  } = useAuth();
  const btnRef = useRef<HTMLDivElement>(null);
  const [mode, setMode] = useState<Mode>("signup");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!open) return;
    setError(null);
  }, [open, mode]);

  useEffect(() => {
    if (!open || !googleClientId || !btnRef.current || !window.google?.accounts?.id) return;

    const handleCredential = async (res: { credential: string }) => {
      setBusy(true);
      setError(null);
      try {
        await signInWithGoogle(res.credential);
        onSuccess?.();
        onClose();
      } catch (e) {
        setError(e instanceof Error ? e.message : "Sign-in failed");
      } finally {
        setBusy(false);
      }
    };

    window.google.accounts.id.initialize({
      client_id: googleClientId,
      callback: handleCredential,
    });
    btnRef.current.innerHTML = "";
    window.google.accounts.id.renderButton(btnRef.current, {
      theme: "outline",
      size: "large",
      width: 320,
      text: "continue_with",
    });
  }, [open, googleClientId, onClose, onSuccess, signInWithGoogle]);

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      if (mode === "signup") {
        await registerWithEmail(email.trim(), password);
      } else {
        await signInWithEmail(email.trim(), password);
      }
      onSuccess?.();
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setBusy(false);
    }
  };

  if (!open) return null;

  return (
    <div className="login-modal-backdrop" onClick={onClose} role="presentation">
      <div className="login-modal" onClick={(e) => e.stopPropagation()} role="dialog" aria-modal="true">
        <button type="button" className="login-modal-close" onClick={onClose} aria-label="Close">
          ×
        </button>
        <div className="login-modal-brand">
          <div className="login-modal-logo">E</div>
          <h2>{mode === "signup" ? "Create your account" : "Sign in"}</h2>
          <p>
            {mode === "signup"
              ? signupCreditsMessage(creditsPerPage, initialCredits)
              : "Sign in with your email and password to continue extracting."}
          </p>
        </div>

        <div className="login-modal-tabs">
          <button
            type="button"
            className={mode === "signup" ? "login-tab active" : "login-tab"}
            onClick={() => setMode("signup")}
          >
            Sign up
          </button>
          <button
            type="button"
            className={mode === "signin" ? "login-tab active" : "login-tab"}
            onClick={() => setMode("signin")}
          >
            Sign in
          </button>
        </div>

        <form className="login-email-form" onSubmit={(e) => void onSubmit(e)}>
          <label className="login-field">
            <span>Email</span>
            <input
              type="email"
              autoComplete="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@company.com"
            />
          </label>
          <label className="login-field">
            <span>Password</span>
            <input
              type="password"
              autoComplete={mode === "signup" ? "new-password" : "current-password"}
              required
              minLength={8}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="At least 8 characters"
            />
          </label>
          <button type="submit" className="login-email-submit" disabled={busy}>
            {busy ? "Please wait…" : mode === "signup" ? "Create account" : "Sign in"}
          </button>
          {error && <p className="login-modal-error">{error}</p>}
        </form>

        {googleClientId && (
          <>
            <p className="login-modal-divider">or</p>
            <div className="login-modal-actions">
              <div ref={btnRef} className="google-btn-host" />
            </div>
          </>
        )}

        <p className="login-modal-foot">
          By continuing you agree to our terms. Extraction usage is stored with your account.
        </p>
      </div>
    </div>
  );
}
