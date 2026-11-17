// The console shell: the simulated-data banner, navigation and the five screens.
// The queue loads with the shell; the chart-heavy screens load when first opened.
import { QueryClientProvider, type QueryClient } from "@tanstack/react-query";
import { lazy, Suspense } from "react";
import { NavLink, Route, Routes } from "react-router";

import { makeQueryClient } from "./api/queryClient";
import { Loading } from "./components/Status";
import { AlertQueue } from "./pages/AlertQueue";

const CaseView = lazy(() => import("./pages/CaseView").then((m) => ({ default: m.CaseView })));
const ThresholdLab = lazy(() =>
  import("./pages/ThresholdLab").then((m) => ({ default: m.ThresholdLab })),
);
const ModelPage = lazy(() => import("./pages/ModelPage").then((m) => ({ default: m.ModelPage })));
const Experiments = lazy(() =>
  import("./pages/Experiments").then((m) => ({ default: m.Experiments })),
);

export function App({ client = makeQueryClient() }: { client?: QueryClient }) {
  return (
    <QueryClientProvider client={client}>
      <div className="banner" role="note">
        Simulated data. Every transaction here is generated; no real cardholder is involved.
      </div>
      <nav className="nav" aria-label="Main">
        <span className="brand">FraudLens</span>
        <NavLink to="/" end>
          Alerts
        </NavLink>
        <NavLink to="/thresholds">Threshold lab</NavLink>
        <NavLink to="/model">Model and monitoring</NavLink>
        <NavLink to="/experiments">Experiments</NavLink>
      </nav>
      <main className="main">
        <Suspense fallback={<Loading what="the page" />}>
          <Routes>
            <Route path="/" element={<AlertQueue />} />
            <Route path="/cases/:transactionId" element={<CaseView />} />
            <Route path="/thresholds" element={<ThresholdLab />} />
            <Route path="/model" element={<ModelPage />} />
            <Route path="/experiments" element={<Experiments />} />
            <Route path="*" element={<p>No such page.</p>} />
          </Routes>
        </Suspense>
      </main>
    </QueryClientProvider>
  );
}
