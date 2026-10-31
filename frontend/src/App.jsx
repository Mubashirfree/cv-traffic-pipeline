import { useCallback, useState } from "react";
import Header from "./components/Header.jsx";
import SearchPage from "./pages/SearchPage.jsx";
import LibraryPage from "./pages/LibraryPage.jsx";
import EvaluationPage from "./pages/EvaluationPage.jsx";

export default function App() {
  const [tab, setTab] = useState("search");
  const [libraryVersion, setLibraryVersion] = useState(0);   // bumps when videos become ready or are removed
  const [focusCamera, setFocusCamera] = useState("");        // video chosen with "Search this video"
  const header = <Header tab={tab} onTab={setTab} />;

  const onChanged = useCallback(() => setLibraryVersion((v) => v + 1), []);
  const searchVideo = (id) => { setFocusCamera(id); setTab("search"); };

  return (
    <>
      {/* All pages stay mounted so results and uploads survive a visit to another tab. */}
      <div hidden={tab !== "search"}>
        <SearchPage header={header} libraryVersion={libraryVersion} focusCamera={focusCamera} />
      </div>
      <div hidden={tab !== "library"}>
        <LibraryPage header={header} active={tab === "library"} onChanged={onChanged} onSearchVideo={searchVideo} />
      </div>
      <div hidden={tab !== "eval"}><EvaluationPage header={header} active={tab === "eval"} /></div>
      <footer className="wrap">
        Internship project. Detection with YOLOv8, search with CLIP and Qdrant, explanations with Gemini.
      </footer>
    </>
  );
}
