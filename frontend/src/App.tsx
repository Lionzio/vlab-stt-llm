import { useState, useRef } from 'react';
import { UploadCloud, FileAudio, Activity, AlertTriangle, CheckCircle2, Loader2 } from 'lucide-react';

// O contrato esperado da sua API
interface Extraction {
  intent: string;
  parameter: string | null;
  value: number | null;
  unit: string | null;
  status: string;
  notes: string | null;
}

interface PipelineResponse {
  transcription: string;
  extraction: Extraction;
}

// Configuração Cloud-Ready: Lê a URL da API do ambiente.
// Em produção (Vercel), usará a URL real.
// Localmente, se não houver .env, faz fallback para o localhost:8000.
const VITE_API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [response, setResponse] = useState<PipelineResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setFileName(file.name);
    setError(null);
    setResponse(null);
    await sendAudioToAPI(file);
    
    // Limpa o input para permitir enviar o mesmo arquivo de novo se quiser
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const sendAudioToAPI = async (file: File) => {
    setIsProcessing(true);
    const formData = new FormData();
    // O FastAPI espera um arquivo chamado "audio_file"
    formData.append("audio_file", file);

    try {
      // AJUSTE CIRÚRGICO: Usando a constante dinâmica com Template Literals
      const res = await fetch(`${VITE_API_URL}/api/v1/extract-from-audio`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        // Tenta capturar o detalhe do erro vindo do FastAPI
        const errorData = await res.json().catch(() => null);
        throw new Error(errorData?.detail || `Erro na API: ${res.statusText}`);
      }

      const data: PipelineResponse = await res.json();
      setResponse(data);
    } catch (err: any) {
      setError(err.message || "Falha de comunicação com a API.");
    } finally {
      setIsProcessing(false);
    }
  };

  // Lógica visual: Se a API voltar OUT_OF_BOUNDS, a tela deve avisar.
  const isDanger = response?.extraction.status === "OUT_OF_BOUNDS";
  const isMissing = response?.extraction.status === "MISSING_VALUE" || response?.extraction.status === "REQUIRES_CLARIFICATION";

  return (
    <div className={`min-h-screen p-8 transition-colors duration-500 ${isDanger ? 'bg-red-50' : 'bg-slate-50'}`}>
      <div className="max-w-3xl mx-auto space-y-8">
        
        {/* Header */}
        <header className="text-center space-y-2">
          <div className="flex justify-center items-center gap-3">
            <Activity className={`w-8 h-8 ${isDanger ? 'text-red-500' : 'text-blue-500'}`} />
            <h1 className="text-3xl font-bold text-slate-800">VLab Command Center</h1>
          </div>
          <p className="text-slate-500">Interface de Avaliação de Comandos (Mock Data)</p>
          {/* Opcional: Mostra a API conectada para debug em portfólio */}
          <p className="text-xs text-slate-400 font-mono">Connected to: {VITE_API_URL}</p>
        </header>

        {/* Área de Upload */}
        <div className="flex flex-col items-center justify-center py-8">
          <input 
            type="file" 
            ref={fileInputRef}
            onChange={handleFileChange}
            accept="audio/*"
            className="hidden"
          />
          
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isProcessing}
            className={`
              relative flex flex-col items-center justify-center w-full max-w-md p-10 rounded-2xl border-2 border-dashed transition-all
              ${isProcessing 
                ? 'bg-slate-100 border-slate-300 cursor-not-allowed' 
                : 'bg-white border-blue-300 hover:border-blue-500 hover:bg-blue-50 hover:shadow-lg'}
            `}
          >
            {isProcessing ? (
              <Loader2 className="w-12 h-12 text-blue-500 animate-spin mb-4" />
            ) : (
              <UploadCloud className="w-12 h-12 text-blue-500 mb-4" />
            )}
            
            <p className="text-lg font-medium text-slate-700">
              {isProcessing ? 'Processando IA...' : 'Clique para anexar um áudio'}
            </p>
            <p className="text-sm text-slate-500 mt-2">
              Arquivos suportados: .mp3, .wav, .m4a
            </p>

            {fileName && !isProcessing && (
              <div className="mt-4 flex items-center gap-2 text-sm text-blue-600 bg-blue-100 px-3 py-1 rounded-full">
                <FileAudio className="w-4 h-4" />
                {fileName}
              </div>
            )}
          </button>
        </div>

        {/* Tratamento de Erros de Rede */}
        {error && (
          <div className="p-4 bg-red-100 border-l-4 border-red-500 rounded shadow-sm flex items-start gap-3 animate-in fade-in slide-in-from-bottom-4">
            <AlertTriangle className="w-6 h-6 text-red-600 shrink-0" />
            <p className="text-red-800 font-medium">{error}</p>
          </div>
        )}

        {/* Dashboard de Resultados */}
        {response && (
          <div className={`
            p-6 rounded-xl shadow-lg border-2 transition-all animate-in fade-in slide-in-from-bottom-8
            ${isDanger ? 'bg-white border-red-500 shadow-red-200' : 
              isMissing ? 'bg-white border-amber-400 shadow-amber-100' : 
              'bg-white border-green-500 shadow-green-100'}
          `}>
            
            {/* Status Visual */}
            <div className="flex items-center gap-3 mb-6 pb-4 border-b border-slate-100">
              {isDanger ? <AlertTriangle className="w-8 h-8 text-red-500" /> : 
               isMissing ? <AlertTriangle className="w-8 h-8 text-amber-500" /> :
               <CheckCircle2 className="w-8 h-8 text-green-500" />}
               
              <div>
                <h2 className={`text-xl font-bold uppercase tracking-wide
                  ${isDanger ? 'text-red-600' : isMissing ? 'text-amber-600' : 'text-green-600'}`}>
                  {response.extraction.status}
                </h2>
                <p className="text-slate-500 italic text-sm">"{response.transcription}"</p>
              </div>
            </div>

            {/* Dados Estruturados */}
            <div className="grid grid-cols-2 gap-4 mb-6">
              <div className="bg-slate-50 p-4 rounded-lg">
                <p className="text-xs text-slate-400 uppercase font-bold mb-1">Intenção</p>
                <p className="text-lg font-mono text-slate-700">{response.extraction.intent}</p>
              </div>
              <div className="bg-slate-50 p-4 rounded-lg">
                <p className="text-xs text-slate-400 uppercase font-bold mb-1">Parâmetro</p>
                <p className="text-lg font-mono text-slate-700">{response.extraction.parameter || 'N/A'}</p>
              </div>
              <div className="bg-slate-50 p-4 rounded-lg col-span-2 flex items-center justify-between">
                <div>
                  <p className="text-xs text-slate-400 uppercase font-bold mb-1">Valor Final</p>
                  <div className="flex items-baseline gap-2">
                    <span className={`text-4xl font-black ${isDanger ? 'text-red-600' : 'text-slate-800'}`}>
                      {response.extraction.value !== null ? response.extraction.value : '--'}
                    </span>
                    <span className="text-lg font-bold text-slate-400">
                      {response.extraction.unit || ''}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Notes / Pydantic Rationale */}
            {response.extraction.notes && (
              <div className={`p-4 rounded-lg text-sm font-medium border
                ${isDanger ? 'bg-red-50 border-red-200 text-red-800' : 'bg-blue-50 border-blue-200 text-blue-800'}`}>
                <strong className="block mb-1 text-xs uppercase tracking-wider opacity-70">Justificativa do Sistema:</strong>
                {response.extraction.notes}
              </div>
            )}

          </div>
        )}

      </div>
    </div>
  );
}

export default App;