import { useState } from "react";
import CameraPanel from "./components/CameraPanel";

import {
  LayoutDashboard,
  BookOpen,
  History,
  Info,
  Settings,
  Video,
  Activity,
  BrainCircuit,
  Bell,
  Hand,
  Zap,
  ChevronRight,
  CircleHelp,
  Languages,
} from "lucide-react";

const navigation = [
  { name: "Dashboard", icon: LayoutDashboard },
  { name: "Vocabulary", icon: BookOpen },
  { name: "History", icon: History },
  { name: "About Project", icon: Info },
];

function App() {
  const [activePage, setActivePage] = useState("Dashboard");
  const [prediction, setPrediction] = useState(null);
  const [history, setHistory] = useState([]);
  const [inferenceConfig, setInferenceConfig] = useState(null);

  const handlePrediction = (result) => {
    setPrediction(result);

    setHistory((previous) => [
      {
        id: `${Date.now()}-${Math.random()}`,
        sign: result.predicted_sign,
        confidence: result.confidence,
        time: new Date().toLocaleTimeString(),
      },
      ...previous,
    ].slice(0, 10));
  };

  return (
    <div className="min-h-screen bg-slate-50 flex">

      {/* SIDEBAR */}
      <aside className="w-64 bg-slate-950 text-white flex flex-col min-h-screen">

        <div className="px-6 py-7 border-b border-slate-800">
          <div className="flex items-center gap-3">

            <div className="w-11 h-11 rounded-xl bg-blue-600 flex items-center justify-center">
              <Hand size={25} />
            </div>

            <div>
              <h1 className="text-xl font-bold tracking-wide">
                SIGNBRIDGE
              </h1>

              <p className="text-xs text-slate-400">
                AI Sign Recognition
              </p>
            </div>

          </div>
        </div>

        <div className="px-4 py-7 flex-1">

          <p className="text-xs text-slate-500 font-semibold tracking-widest px-3 mb-4">
            WORKSPACE
          </p>

          <nav className="space-y-2">

            {navigation.map((item) => {
              const Icon = item.icon;
              const active = activePage === item.name;

              return (
                <button
                  key={item.name}
                  onClick={() => setActivePage(item.name)}
                  className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm transition ${
                    active
                      ? "bg-blue-600 text-white shadow-lg shadow-blue-950/40"
                      : "text-slate-400 hover:bg-slate-900 hover:text-white"
                  }`}
                >
                  <Icon size={19} />

                  {item.name}

                  {active && (
                    <ChevronRight size={16} className="ml-auto" />
                  )}
                </button>
              );
            })}

          </nav>

          <div className="mt-12 p-4 rounded-2xl bg-slate-900 border border-slate-800">

            <div className="flex items-center gap-2 text-blue-400 mb-3">
              <BrainCircuit size={20} />

              <span className="text-sm font-semibold">
                AI Engine
              </span>
            </div>

            <p className="text-xs text-slate-400 leading-relaxed">
              MobileNetV3-based Indian Sign Language recognition.
            </p>

            <div className="mt-4 flex items-center gap-2">
              <div className="w-2 h-2 bg-emerald-400 rounded-full" />

              <span className="text-xs text-emerald-400">
                Model configured
              </span>
            </div>

          </div>
        </div>

        <div className="p-5 border-t border-slate-800">
          <button className="flex items-center gap-3 text-slate-400 hover:text-white text-sm">
            <Settings size={18} />
            Settings
          </button>
        </div>

      </aside>

      {/* MAIN CONTENT */}
      <main className="flex-1 min-w-0">

        {/* HEADER */}
        <header className="h-20 bg-white border-b border-slate-200 px-8 flex items-center justify-between">

          <div>
            <p className="text-sm text-slate-400">
              Workspace / {activePage}
            </p>

            <h2 className="text-xl font-bold text-slate-900">
              {activePage === "Dashboard"
                ? "Sign Recognition Dashboard"
                : activePage}
            </h2>
          </div>

          <div className="flex items-center gap-5">

            <button className="text-slate-400 hover:text-slate-700">
              <Bell size={20} />
            </button>

            <div className="flex items-center gap-3">

              <div className="w-10 h-10 rounded-full bg-blue-100 text-blue-700 flex items-center justify-center font-bold">
                AS
              </div>

              <div className="hidden sm:block">
                <p className="text-sm font-semibold">User</p>

                <p className="text-xs text-slate-400">
                  ISL Explorer
                </p>
              </div>

            </div>
          </div>
        </header>

        {/* DASHBOARD BODY */}
        <div className="p-8 max-w-7xl mx-auto">

          {/* WELCOME */}
          <div className="flex flex-wrap justify-between items-center gap-4 mb-8">

            <div>
              <h1 className="text-2xl font-bold text-slate-900">
                Welcome to SIGNBRIDGE 👋
              </h1>

              <p className="text-slate-500 mt-1">
                Bridge communication through Indian Sign Language.
              </p>
            </div>

            <div className="flex items-center gap-2 px-4 py-2 bg-emerald-50 text-emerald-700 rounded-full text-sm">
              <Activity size={17} />
              System Ready
            </div>

          </div>

          {/* STAT CARDS */}
          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-5 mb-8">

            <StatCard
              title="Supported Signs"
              value="50"
              description="INCLUDE-50 vocabulary"
              icon={Hand}
              color="blue"
            />

            <StatCard
              title="Recognition Model"
              value="MobileNetV3"
              description="CNN-based classifier"
              icon={BrainCircuit}
              color="purple"
            />

            <StatCard
              title="Input Frames"
              value={inferenceConfig?.frame_count ?? "--"}
              description={inferenceConfig ? "Checkpoint-configured frames" : "Loading model configuration"}
              icon={Video}
              color="emerald"
            />

            <StatCard
              title="Backend"
              value="FastAPI"
              description="Inference API"
              icon={Zap}
              color="orange"
            />

          </div>

          {/* CAMERA + PREDICTION */}
          <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">

            {/* CAMERA PANEL */}
            <div className="xl:col-span-2 bg-white border border-slate-200 rounded-2xl overflow-hidden shadow-sm">

              <div className="p-6 border-b border-slate-100 flex items-center justify-between">

                <div>
                  <h3 className="font-bold text-lg">
                    Live Sign Recognition
                  </h3>

                  <p className="text-sm text-slate-400 mt-1">
                    Position your hands inside the camera frame.
                  </p>
                </div>

                <div className="flex items-center gap-2 text-xs text-slate-500">
                  <div className="w-2 h-2 bg-emerald-400 rounded-full" />
                  Webcam Module
                </div>

              </div>

              <div className="p-6">
                <CameraPanel onPrediction={handlePrediction} onConfig={setInferenceConfig} />
              </div>

            </div>

            {/* PREDICTION PANEL */}
            <div className="bg-white border border-slate-200 rounded-2xl shadow-sm">

              <div className="p-6 border-b border-slate-100">

                <h3 className="font-bold text-lg">
                  Recognition Result
                </h3>

                <p className="text-sm text-slate-400 mt-1">
                  Latest model prediction
                </p>

              </div>

              <div className="p-6">

                <div className="bg-blue-50 rounded-2xl p-6 text-center">

                  <div className="w-16 h-16 bg-blue-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
                    <Languages size={30} className="text-blue-600" />
                  </div>

                  <p className="text-xs text-blue-500 uppercase tracking-widest font-semibold">
                    Predicted Sign
                  </p>

                  <h2 className="text-2xl font-bold text-slate-800 mt-2 break-words">
                    {prediction?.predicted_sign || "Awaiting Input"}
                  </h2>

                  <p className="text-sm text-slate-500 mt-2">
                    {prediction
                      ? `Confidence: ${prediction.confidence}%`
                      : "Start recognition to see predictions."}
                  </p>

                </div>

                <div className="mt-7">

                  <div className="flex items-center justify-between mb-3">
                    <h4 className="font-semibold text-sm">
                      Top Predictions
                    </h4>

                    <span className="text-xs text-slate-400">
                      Top 5
                    </span>
                  </div>

                  <div className="space-y-4">

                    {prediction?.top_predictions?.map((item, index) => (
                      <div key={`${item.sign}-${index}`}>

                        <div className="flex items-center justify-between text-sm mb-1">
                          <div className="flex items-center gap-2">
                            <span className="w-6 h-6 rounded-lg bg-slate-100 flex items-center justify-center text-xs text-slate-500">
                              {index + 1}
                            </span>

                            <span className="text-slate-700">
                              {item.sign}
                            </span>
                          </div>

                          <span className="font-semibold text-slate-700">
                            {item.confidence}%
                          </span>
                        </div>

                        <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-blue-500 rounded-full"
                            style={{ width: `${item.confidence}%` }}
                          />
                        </div>

                      </div>
                    ))}

                    {!prediction && (
                      <p className="text-sm text-slate-400">
                        No predictions available.
                      </p>
                    )}

                  </div>

                </div>

              </div>
            </div>

          </div>

          {/* BOTTOM SECTION */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-8">

            {/* HISTORY */}
            <div className="bg-white border border-slate-200 rounded-2xl p-6">

              <div className="flex justify-between items-center mb-5">
                <h3 className="font-bold">
                  Recent Recognition History
                </h3>

                <History size={19} className="text-slate-400" />
              </div>

              {history.length === 0 ? (
                <div className="text-center py-10">
                  <History
                    size={35}
                    className="mx-auto text-slate-300 mb-3"
                  />

                  <p className="text-sm text-slate-400">
                    No recognition history yet.
                  </p>
                </div>
              ) : (
                <div className="space-y-4">

                  {history.map((item) => (
                    <div
                      key={item.id}
                      className="flex items-center justify-between border-b border-slate-100 pb-3"
                    >

                      <div>
                        <p className="font-semibold text-sm text-slate-800">
                          {item.sign}
                        </p>

                        <p className="text-xs text-slate-400 mt-1">
                          {item.time}
                        </p>
                      </div>

                      <span className="text-sm font-semibold text-blue-600">
                        {item.confidence}%
                      </span>

                    </div>
                  ))}

                </div>
              )}

            </div>

            {/* ABOUT */}
            <div className="bg-gradient-to-br from-blue-600 to-indigo-700 rounded-2xl p-6 text-white">

              <div className="flex items-center gap-3 mb-5">
                <CircleHelp size={25} />

                <h3 className="font-bold text-lg">
                  About SIGNBRIDGE
                </h3>
              </div>

              <p className="text-blue-100 text-sm leading-relaxed">
                SIGNBRIDGE is a deep learning-based Indian Sign Language
                recognition system. It uses MobileNetV3 to classify
                isolated signs from sequences of video frames.
              </p>

              <div className="mt-6 flex items-center gap-2 text-sm text-blue-100">
                <BrainCircuit size={18} />
                Spatial Feature Extraction
              </div>

              <div className="mt-3 flex items-center gap-2 text-sm text-blue-100">
                <Video size={18} />
                Temporal Frame Sampling
              </div>

              <div className="mt-3 flex items-center gap-2 text-sm text-blue-100">
                <Languages size={18} />
                50 Isolated Sign Classes
              </div>

            </div>

          </div>

          <footer className="text-center text-xs text-slate-400 mt-10 pb-4">
            SIGNBRIDGE © 2026 · Deep Learning-Based ISL Recognition
          </footer>

        </div>
      </main>
    </div>
  );
}

function StatCard({
  title,
  value,
  description,
  icon: Icon,
  color,
}) {
  const colors = {
    blue: "bg-blue-50 text-blue-600",
    purple: "bg-purple-50 text-purple-600",
    emerald: "bg-emerald-50 text-emerald-600",
    orange: "bg-orange-50 text-orange-600",
  };

  return (
    <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm">

      <div className="flex items-center justify-between mb-5">
        <p className="text-sm text-slate-500">
          {title}
        </p>

        <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${colors[color]}`}>
          <Icon size={20} />
        </div>
      </div>

      <h3 className="text-xl font-bold text-slate-900">
        {value}
      </h3>

      <p className="text-xs text-slate-400 mt-2">
        {description}
      </p>

    </div>
  );
}

export default App;
