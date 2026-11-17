// Loading, empty and error states. Every screen uses all three.
import type { ReactNode } from "react";

export function Loading({ what }: { what: string }) {
  return (
    <div className="status" role="status" aria-live="polite">
      <span className="spinner" aria-hidden="true" />
      Loading {what}…
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="status status-empty">{children}</div>;
}

export function ErrorState({ error, retry }: { error: unknown; retry?: () => void }) {
  const message = error instanceof Error ? error.message : "Something went wrong.";
  return (
    <div className="status status-error" role="alert">
      <strong>Could not load this.</strong> <span>{message}</span>
      {retry ? (
        <button type="button" className="button button-quiet" onClick={retry}>
          Try again
        </button>
      ) : null}
    </div>
  );
}

type QueryLike<T> = {
  isPending: boolean;
  isError: boolean;
  error: unknown;
  data: T | undefined;
  refetch: () => unknown;
};

/** Render ``children`` with the data once loaded, or the loading or error state. */
export function Loaded<T>({
  query,
  what,
  children,
}: {
  query: QueryLike<T>;
  what: string;
  children: (data: T) => ReactNode;
}) {
  if (query.isPending) return <Loading what={what} />;
  if (query.isError || query.data === undefined) {
    return <ErrorState error={query.error} retry={() => void query.refetch()} />;
  }
  return <>{children(query.data)}</>;
}
