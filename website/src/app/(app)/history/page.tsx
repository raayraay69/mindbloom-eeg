import { SessionHistory } from "./components/session-history";

export default function HistoryPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl md:text-3xl font-bold tracking-tight font-headline">
          Session History
        </h1>
        <p className="text-muted-foreground mt-1">
          Review your past sessions and observe trends over time.
        </p>
      </div>
      <SessionHistory />
    </div>
  );
}
