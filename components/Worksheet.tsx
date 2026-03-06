import React, { useMemo, useRef, useEffect } from 'react';
import { AcademicLevel, Theme, PaperType, BrandSettings } from '../types';

interface WorksheetProps {
  content: string;
  onContentChange: (val: string) => void;
  isGenerating: boolean;
  level: AcademicLevel;
  module: string;
  topic: string;
  paperType: PaperType;
  theme: Theme;
  brandSettings: BrandSettings;
}

const Worksheet: React.FC<WorksheetProps> = ({
  content, onContentChange, isGenerating, brandSettings
}) => {
  const editorRef = useRef<HTMLDivElement>(null);

  const placeholderHtml = useMemo(() => {
    return `
      <div class="flex flex-col items-center h-full min-h-[600px] text-center px-4 py-20 relative overflow-hidden bg-white text-black">
        <div class="border-[12px] border-black p-12 max-w-2xl w-full">
            <div class="space-y-6 text-center">
                <h1 class="text-2xl md:text-3xl font-black uppercase tracking-widest">${brandSettings.schoolName}</h1>
                <hr class="border-black border-2 w-full" />
                <p class="text-sm md:text-lg font-bold italic tracking-[0.3em] uppercase">${brandSettings.schoolAddress}</p>
            </div>
        </div>
        <div class="mt-20 w-full max-w-2xl text-left font-bold text-[10px] space-y-4">
            <div class="flex justify-between border-b border-slate-200 pb-2 text-slate-400">
                <span>NAME : _________________________________</span>
                <span>DATE : ____ / ____ / ____</span>
            </div>
        </div>
      </div>
    `;
  }, [brandSettings.schoolName, brandSettings.schoolAddress]);

  useEffect(() => {
    if (editorRef.current && content !== editorRef.current.innerHTML) {
      if (!isGenerating) {
        editorRef.current.innerHTML = content || placeholderHtml;
      }
    }
  }, [content, isGenerating, placeholderHtml]);

  return (
    <div className="flex-1 overflow-y-auto p-4 md:p-12 no-scrollbar bg-slate-200/50">
      <div className="max-w-[210mm] mx-auto pb-64 shadow-2xl worksheet-paper">
        <style>{`
          .prose { 
            font-family: 'Times New Roman', Times, serif !important; 
            font-size: ${brandSettings.fontSize || 12}pt !important; 
          }
          .prose table { border-collapse: collapse !important; width: 100% !important; border: 1.5pt solid black !important; table-layout: fixed; }
          .prose th, .prose td { border: 1pt solid black !important; padding: 6pt !important; vertical-align: top !important; }
          @media print {
            .no-print { display: none !important; }
            .bg-white { background-color: white !important; }
          }
        `}</style>
        
        <div className="worksheet-page bg-white min-h-[297mm] p-[0.7in] relative rounded-sm border border-slate-300"
             contentEditable={!isGenerating}
             onInput={(e) => onContentChange(e.currentTarget.innerHTML)}
             dangerouslySetInnerHTML={{ __html: content || placeholderHtml }}
        />
      </div>
    </div>
  );
};

export default Worksheet;
