import React, { useState, useRef, useEffect } from 'react';
import { QuickSource, ChatMessage } from '../types';

interface NeuralChatAssistantProps {
  messages: ChatMessage[];
  input: string;
  onInputChange: (val: string) => void;
  onSendMessage: (msg: string, file?: QuickSource) => void;
  isGenerating: boolean;
  quickSource: QuickSource | null;
  inline?: boolean;
}

const NeuralChatAssistant: React.FC<NeuralChatAssistantProps> = ({ 
  messages, input, onInputChange, onSendMessage, isGenerating, inline = false
}) => {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages]);

  const handleSubmit = (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim()) return;
    onSendMessage(input);
  };

  if (inline) {
    return (
      <div className="flex flex-col gap-4 w-full">
        <div className="bg-slate-900 rounded-2xl p-4 max-h-[180px] overflow-y-auto no-scrollbar space-y-3" ref={scrollRef}>
           {messages.map((msg) => (
             <div key={msg.id}>
                 <div className="text-[7px] font-black uppercase tracking-widest text-slate-500">{msg.role}</div>
                 <p className="text-[10px] text-slate-300">{msg.text}</p>
             </div>
           ))}
        </div>
        <div className="flex items-center gap-2 bg-white p-2 rounded-full border">
           <input 
             type="text" 
             value={input}
             onChange={(e) => onInputChange(e.target.value)}
             onKeyDown={(e) => e.key === 'Enter' && handleSubmit()}
             placeholder="Instruct architect..."
             className="flex-1 px-4 text-[11px] outline-none"
           />
           <button onClick={handleSubmit} className="h-8 w-8 bg-orange-600 text-white rounded-full flex items-center justify-center">
             <i className="fa-solid fa-bolt"></i>
           </button>
        </div>
      </div>
    );
  }
  return null;
};

export default NeuralChatAssistant;
