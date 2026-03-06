
import React from 'react';
import { OutlineItem } from '../types';

interface OutlinePanelProps {
  isOpen: boolean;
  onToggle: () => void;
  outline: OutlineItem[];
  isGenerating: boolean;
  onGenerate: () => void;
  onToggleExpand: (id: string) => void;
}

const OutlineNode: React.FC<{ item: OutlineItem; onToggle: (id: string) => void; depth: number }> = ({ item, onToggle, depth }) => {
  const hasChildren = item.children && item.children.length > 0;
  
  return (
    <div className="select-none">
      <div 
        className={`flex items-center gap-2 py-2.5 px-3 rounded-xl hover:bg-slate-50 transition-colors cursor-pointer group`}
        style={{ paddingLeft: `${depth * 1.5 + 0.75}rem` }}
        onClick={() => hasChildren && onToggle(item.id)}
      >
        {hasChildren ? (
          <i className={`fa-solid fa-chevron-right text-[9px] text-slate-300 transition-transform ${item.expanded ? 'rotate-90 text-orange-500' : ''}`}></i>
        ) : (
          <div className="w-[10px] h-[10px] flex items-center justify-center">
            <div className="w-1 h-1 rounded-full bg-slate-200 group-hover:bg-orange-400"></div>
          </div>
        )}
        <span className={`text-[11px] ${depth === 0 ? 'text-slate-900 font-bold uppercase tracking-wider' : 'text-slate-600 font-medium'}`}>{item.title}</span>
      </div>
      {hasChildren && item.expanded && (
        <div className="outline-children">
          {item.children?.map(child => (
            <OutlineNode key={child.id} item={child} onToggle={onToggle} depth={depth + 1} />
          ))}
        </div>
      )}
    </div>
  );
};

const OutlinePanel: React.FC<OutlinePanelProps> = ({ isOpen, onToggle, outline, isGenerating, onGenerate, onToggleExpand }) => {
  return (
    <>
      <div className={`w-[320px] bg-white border-l border-slate-100 h-full fixed right-0 top-0 flex flex-col z-[80] shadow-2xl transition-transform duration-300 ease-in-out ${isOpen ? 'translate-x-0' : 'translate-x-full'}`}>
        <div className="p-7 border-b border-slate-50 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <i className="fa-solid fa-list-check text-orange-500"></i>
            <h2 className="text-[11px] font-black uppercase tracking-[0.15em] text-slate-900">Neural Outline</h2>
          </div>
          <button onClick={onToggle} className="h-9 w-9 rounded-full bg-slate-50 flex items-center justify-center text-slate-400 hover:text-slate-900 transition-all">
            <i className="fa-solid fa-xmark text-xs"></i>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-5 no-scrollbar">
          {outline.length > 0 ? (
            <div className="space-y-1">
              {outline.map(item => (
                <OutlineNode key={item.id} item={item} onToggle={onToggleExpand} depth={0} />
              ))}
            </div>
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-center p-8 space-y-5">
              <div className="h-14 w-14 bg-slate-50 rounded-2xl flex items-center justify-center text-slate-200">
                <i className="fa-solid fa-route text-xl"></i>
              </div>
              <p className="text-[10px] font-black text-slate-300 uppercase tracking-widest leading-loose">No structural roadmap synthesized</p>
            </div>
          )}
        </div>

        <div className="p-6 border-t border-slate-50 bg-slate-50/30">
          <button
            onClick={onGenerate}
            disabled={isGenerating}
            className="w-full py-4 rounded-xl bg-slate-900 text-white text-[10px] font-black uppercase tracking-widest hover:bg-black transition-all disabled:opacity-50 flex items-center justify-center gap-3 shadow-xl"
          >
            {isGenerating ? <i className="fa-solid fa-circle-notch animate-spin"></i> : <i className="fa-solid fa-sparkles"></i>}
            <span>Synthesis Outline</span>
          </button>
        </div>
      </div>

      {!isOpen && (
        <button 
          onClick={onToggle}
          className="fixed right-6 top-6 h-11 w-11 bg-white rounded-xl shadow-xl flex items-center justify-center text-slate-900 border border-slate-100 hover:scale-105 active:scale-95 transition-all z-[60] no-print"
        >
          <i className="fa-solid fa-list-ul"></i>
        </button>
      )}
    </>
  );
};

export default OutlinePanel;
