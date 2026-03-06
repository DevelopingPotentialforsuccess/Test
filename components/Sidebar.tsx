import React, { useState } from 'react';
import { AcademicLevel, HistoryItem, BrandSettings } from '../types';

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
  curriculum: string[];
  activeModule: string;
  onModuleChange: (m: string) => void;
  activeLevel: AcademicLevel;
  onLevelChange: (l: AcademicLevel) => void;
  topic: string;
  onTopicChange: (t: string) => void;
  onClearCanvas: () => void;
  onToggleSettings: () => void;
  history: HistoryItem[];
  onLoadHistory: (item: HistoryItem) => void;
  onDeleteHistory: (id: string) => void;
  onRenameHistory?: (id: string, newTitle: string) => void;
  brandSettings: BrandSettings;
}

const Sidebar: React.FC<SidebarProps> = ({
  isOpen, onClose,
  onClearCanvas, onToggleSettings,
  history, onLoadHistory, onDeleteHistory, onRenameHistory,
  brandSettings
}) => {
  const [editingHistId, setEditingHistId] = useState<string | null>(null);
  const [tempTitle, setTempTitle] = useState('');

  const startRename = (e: React.MouseEvent, item: HistoryItem) => {
    e.stopPropagation();
    setEditingHistId(item.id);
    setTempTitle(item.title);
  };

  const submitRename = (id: string) => {
    if (onRenameHistory && tempTitle.trim()) {
      onRenameHistory(id, tempTitle);
    }
    setEditingHistId(null);
  };

  return (
    <aside className={`w-[280px] bg-white h-full fixed left-0 top-0 text-slate-900 flex flex-col z-[110] shadow-[0_0_40px_rgba(0,0,0,0.05)] border-r border-slate-100 transition-transform duration-300 ease-in-out ${isOpen ? 'translate-x-0' : '-translate-x-full'}`}>
      <div className="p-8 relative">
        <button 
          onClick={onClose}
          className="absolute top-8 right-6 text-slate-300 hover:text-slate-900 transition-colors h-10 w-10 flex items-center justify-center bg-slate-50 rounded-xl group"
          title="Collapse Sidebar"
        >
          <i className="fa-solid fa-angles-left text-sm group-hover:-translate-x-1 transition-transform"></i>
        </button>

        <div className="flex flex-col gap-3 mb-10 px-2 mt-4 text-center">
          {brandSettings.logoData && (
              <img src={brandSettings.logoData} alt="School Logo" className="mb-4 rounded-lg mx-auto" style={{ width: '100%', maxWidth: '200px' }} />
          )}
          <div className="flex flex-col items-center">
            {!brandSettings.logoData && (
                <div className="h-12 w-12 bg-orange-600 rounded-xl flex items-center justify-center shadow-lg shadow-orange-600/30 mb-3">
                  <i className="fa-solid fa-bolt text-white text-xl"></i>
                </div>
            )}
            <h1 
              style={{ 
                fontSize: `14px`, 
                fontWeight: '900',
                letterSpacing: '0.05em',
                textTransform: 'uppercase'
              }}
              className="text-slate-900 leading-tight transition-all duration-300"
            >
              DPSS Test Builder
            </h1>
            <div className="text-[7px] font-black text-orange-500 tracking-[0.3em] uppercase mt-1">Assessment Architecture Node</div>
          </div>
          
          <div className="mt-4 border-t border-slate-100 pt-4 space-y-1">
             <div className="text-[9px] font-bold khmer-font text-orange-600">សាលាអភិវឌ្ឍន៍សក្ដានុពលដើម្បីជោគជ័យ</div>
             <div className="text-[7px] font-black text-slate-400 uppercase tracking-widest">Simple Vocab • Complex Logic</div>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-2 no-scrollbar border-t border-slate-50">
        <div className="flex items-center justify-between mb-4 mt-4 px-2">
            <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Logic Registry</label>
            <button onClick={onClearCanvas} className="text-[9px] font-black text-slate-300 uppercase hover:text-orange-600 transition-colors">Wipe</button>
        </div>
        <div className="space-y-2 pb-6">
          {history.map(item => (
            <div key={item.id} className="group relative">
                {editingHistId === item.id ? (
                  <div className="p-3 bg-slate-50 rounded-2xl border border-orange-500 flex flex-col gap-2">
                    <input 
                      autoFocus
                      className="bg-transparent text-[11px] font-bold text-slate-900 outline-none w-full"
                      value={tempTitle}
                      onChange={(e) => setTempTitle(e.target.value)}
                      onKeyDown={(e) => e.key === 'Enter' && submitRename(item.id)}
                    />
                    <div className="flex justify-end gap-2">
                      <button onClick={() => submitRename(item.id)} className="text-[8px] font-black text-emerald-600 uppercase">Save</button>
                      <button onClick={() => setEditingHistId(null)} className="text-[8px] font-black text-slate-400 uppercase">Cancel</button>
                    </div>
                  </div>
                ) : (
                  <>
                    <button
                        onClick={() => onLoadHistory(item)}
                        className="w-full text-left p-4 rounded-2xl bg-slate-50 border border-transparent hover:border-orange-100 hover:bg-white transition-all pr-12 group/btn shadow-sm hover:shadow-md"
                    >
                        <div className="text-[9px] font-black text-slate-300 mb-1">{new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</div>
                        <div className="text-[11px] font-bold text-slate-600 line-clamp-1 group-hover/btn:text-slate-900">{item.title}</div>
                    </button>
                    <div className="absolute right-3 top-1/2 -translate-y-1/2 flex flex-col gap-1 opacity-0 lg:group-hover:opacity-100 transition-all">
                        <button 
                            onClick={(e) => startRename(e, item)}
                            className="h-6 w-6 bg-orange-500/10 text-orange-500 hover:bg-orange-500 hover:text-white rounded-lg flex items-center justify-center transition-all"
                        >
                            <i className="fa-solid fa-pen text-[8px]"></i>
                        </button>
                        <button 
                            onClick={(e) => { e.stopPropagation(); onDeleteHistory(item.id); }}
                            className="h-6 w-6 bg-red-500/10 text-red-500 hover:bg-red-500 hover:text-white rounded-lg flex items-center justify-center transition-all"
                        >
                            <i className="fa-solid fa-trash-can text-[8px]"></i>
                        </button>
                    </div>
                  </>
                )}
            </div>
          ))}
          {history.length === 0 && <div className="text-center py-12 text-[10px] font-black text-slate-200 uppercase tracking-widest">Archive Empty</div>}
        </div>
      </div>

      <div className="p-6 mt-auto">
        <button
          onClick={onToggleSettings}
          className="w-full flex items-center justify-between p-5 rounded-2xl bg-orange-600 text-white hover:bg-orange-700 transition-all group shadow-2xl shadow-orange-600/20"
        >
          <div className="flex items-center gap-4">
            <i className="fa-solid fa-gear text-sm group-hover:rotate-90 transition-transform duration-500"></i>
            <span className="text-[11px] font-black uppercase tracking-widest">Workspace</span>
          </div>
          <i className="fa-solid fa-chevron-right text-[10px] opacity-70 group-hover:translate-x-1 transition-transform"></i>
        </button>
      </div>
    </aside>
  );
};

export default Sidebar;