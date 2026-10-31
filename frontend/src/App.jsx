import { useState } from "react";
import Header from "./components/Header.jsx";
import SearchPage from "./pages/SearchPage.jsx";
import EvaluationPage from "./pages/EvaluationPage.jsx";

export default function App() {
  const [tab, setTab] = useState("search");
  const header = <Header tab={tab} onTab={setTab} />;

  return (
    <>
      {/* Both pages stay mounted so results survive a visit to the other tab. */}
      <div hidden={tab !== "search"}><SearchPage header={header} /></div>
      <div hidden={tab !== "eval"}><EvaluationPage header={header} active={tab === "eval"} /></div>
      <footer className="wrap">
        Internship project. Detection with YOLOv8, search with CLIP and Qdrant, explanations with Gemini.
      </footer>
    </>
  );
}
