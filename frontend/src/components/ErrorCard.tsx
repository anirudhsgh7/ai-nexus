interface ErrorLike {
  type: string;
  message: string;
  hint?: string;
}

interface ErrorCardProps {
  error: ErrorLike;
  title?: string;
}

export function ErrorCard({ error, title = "ERROR" }: ErrorCardProps) {
  return (
    <div className="error-card" role="alert">
      <p className="error-title">
        {title} · {error.type}
      </p>
      <p className="error-message">{error.message}</p>
      {error.hint !== undefined && error.hint !== "" ? (
        <p className="error-hint">{error.hint}</p>
      ) : null}
    </div>
  );
}
