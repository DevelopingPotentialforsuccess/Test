import {
  NeuralEngine,
  AcademicLevel,
  HistoryItem,
  QuickSource,
  StrictRule,
  SettingsTab,
  UserSession,
  BrandSettings,
  InstructionTemplate,
  Priority,
  RuleCategory,
  ExternalKeys,
  ChatMessage,
  AnswerStrategy
} from './types';
import {
  INITIAL_MODULES,
  LANGUAGES,
  ACADEMIC_LEVELS,
  GLOBAL_STRICT_COMMAND,
  DEFAULT_STRICT_RULES,
  DEFAULT_MASTER_PROTOCOLS,
  INITIAL_TEMPLATES,
  THEMES
} from './constants';

import { callNeuralEngine } from './services/neuralService';
import { exportToWord } from './services/wordExportService';
import React, { useState, useEffect, useRef } from 'react';
import Worksheet from './components/Worksheet';
import NeuralChatAssistant from './components/NeuralChatAssistant'
import { OnboardingTutorial } from './components/OnboardingTutorial';
const DEFAULT_BRAND_SETTINGS: BrandSettings = {
  fontSize: 12,
  fontWeight: '800',
  letterSpacing: 0,
  textTransform: 'none',
  schoolName: 'DPSS ULTIMATE TEST BUILDER',
  schoolAddress: 'Developing Potential for Success School',
  logos: Array(30).fill(undefined),
  logoWidth: 300,
  logoData: undefined
};

const MASTER_PROTOCOLS_KEY = 'dp_master_v46';
const STRICT_RULES_KEY = 'dp_rules_v46';
const TEMPLATES_KEY = 'dp_templates_v46';
const HISTORY_KEY = 'dp_history_v46';
const BRAND_SETTINGS_KEY = 'dp_brand_v46';
const USER_SESSION_KEY = 'dp_session_v46';
const ENGINE_CONFIG_KEY = 'dp_engine_config_v46';
const ONBOARDING_KEY = 'dp_onboarding_v1';

function App() {
  const [session, setSession] = useState<UserSession | null>(() => {
    try {
      const saved = localStorage.getItem(USER_SESSION_KEY);
      return saved ? JSON.parse(saved) : null;
    } catch { return null; }
  });

  const [loginName, setLoginName] = useState('');
  const [loginCode, setLoginCode] = useState('');
  const [loginError, setLoginError] = useState('');

  const [viewMode, setViewMode] = useState<'generator' | 'preview' | 'book_creation' | 'ielts_master' | 'dpss_studio' | 'grammar_iframe'>('grammar_iframe');
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isAssistantVisible, setIsAssistantVisible] = useState(false);
  const [activeModule, setActiveModule] = useState<string>('Grammar');
  const [activeLanguage, setActiveLanguage] = useState<string>('English');
  const [activeLevel, setActiveLevel] = useState<AcademicLevel>('Level 1');
  const [answerStrategy, setAnswerStrategy] = useState<AnswerStrategy>('GENERAL_MIXED');
  const [topic, setTopic] = useState<string>('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [worksheetContent, setWorksheetContent] = useState<string>('');
  const [showSettings, setShowSettings] = useState(false);
  const [settingsTab, setSettingsTab] = useState<SettingsTab>('COMMAND');
  
  const [activeLogicCategory, setActiveLogicCategory] = useState<RuleCategory>('General');
  const [expandedRuleId, setExpandedRuleId] = useState<string | null>(null);
  const [activeProtocolCategory, setActiveProtocolCategory] = useState<RuleCategory>('General');
  const [expandedProtocolId, setExpandedProtocolId] = useState<string | null>(null);
  const [activeTemplateCategory, setActiveTemplateCategory] = useState<string>('GRAMMAR');
  const [expandedTemplateId, setExpandedTemplateId] = useState<string | null>(null);

  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState('');

  const [activeThemeId, setActiveThemeId] = useState<string>(() => {
    try {
      const saved = localStorage.getItem('dp_theme_v30');
      return saved || 'default';
    } catch { return 'default'; }
  });

  const [activeEngine, setActiveEngine] = useState<NeuralEngine>(() => {
    try {
      const saved = localStorage.getItem(ENGINE_CONFIG_KEY);
      return saved ? JSON.parse(saved).active : NeuralEngine.GEMINI_3_FLASH;
    } catch { return NeuralEngine.GEMINI_3_FLASH; }
  });

  const [externalKeys, setExternalKeys] = useState<ExternalKeys>(() => {
    try {
      const saved = localStorage.getItem(ENGINE_CONFIG_KEY);
      return saved ? JSON.parse(saved).keys : {};
    } catch { return {}; }
  });
  
  const [brandSettings, setBrandSettings] = useState<BrandSettings>(() => {
    try {
      const saved = localStorage.getItem(BRAND_SETTINGS_KEY);
      return saved ? JSON.parse(saved) : DEFAULT_BRAND_SETTINGS;
    } catch { return DEFAULT_BRAND_SETTINGS; }
  });
  
  const [history, setHistory] = useState<HistoryItem[]>(() => {
    try {
      const saved = localStorage.getItem(HISTORY_KEY);
      return saved ? JSON.parse(saved) : [];
    } catch { return []; }
  });

  const [masterProtocols, setMasterProtocols] = useState<StrictRule[]>(() => {
    try {
      const saved = localStorage.getItem(MASTER_PROTOCOLS_KEY);
      let parsed = saved ? JSON.parse(saved) : DEFAULT_MASTER_PROTOCOLS;
      if (!Array.isArray(parsed)) parsed = DEFAULT_MASTER_PROTOCOLS;
      // Auto-merge missing defaults
      const existingIds = new Set(parsed.map((p: any) => p.id));
      const missing = DEFAULT_MASTER_PROTOCOLS.filter(p => !existingIds.has(p.id));
      return [...parsed, ...missing];
    } catch { return DEFAULT_MASTER_PROTOCOLS; }
  });
  const [strictRules, setStrictRules] = useState<StrictRule[]>(() => {
    try {
      const saved = localStorage.getItem(STRICT_RULES_KEY);
      let parsed = saved ? JSON.parse(saved) : DEFAULT_STRICT_RULES;
      if (!Array.isArray(parsed)) parsed = DEFAULT_STRICT_RULES;
      // Auto-merge missing defaults
      const existingIds = new Set(parsed.map((r: any) => r.id));
      const missing = DEFAULT_STRICT_RULES.filter(r => !existingIds.has(r.id));
      return [...parsed, ...missing];
    } catch { return DEFAULT_STRICT_RULES; }
  });
  const [instructionTemplates, setInstructionTemplates] = useState<InstructionTemplate[]>(() => {
    try {
      const saved = localStorage.getItem(TEMPLATES_KEY);
      let parsed = saved ? JSON.parse(saved) : INITIAL_TEMPLATES;
      if (!Array.isArray(parsed)) parsed = INITIAL_TEMPLATES;
      // Auto-merge missing defaults
      const existingIds = new Set(parsed.map((t: any) => t.id));
      const missing = INITIAL_TEMPLATES.filter(t => !existingIds.has(t.id));
      return [...parsed, ...missing];
    } catch { return INITIAL_TEMPLATES; }
  });

  const [selectedInstructionIds, setSelectedInstructionIds] = useState<string[]>([]);
  const [columnOverrides, setColumnOverrides] = useState<Record<string, number>>({});
  const [itemCountOverrides, setItemCountOverrides] = useState<Record<string, number>>({});
  
  const [sourceMaterial, setSourceMaterial] = useState<QuickSource | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const logoUploadRef = useRef<HTMLInputElement>(null);

  const [showOnboarding, setShowOnboarding] = useState(() => {
    try {
      const saved = localStorage.getItem(ONBOARDING_KEY);
      return saved !== 'completed';
    } catch { return true; }
  });

  useEffect(() => { 
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(history)); 
    } catch (e) {
      console.warn("History storage limit reached. Oldest items may be lost.", e);
      // Optional: Try to save a smaller subset if full save fails
      try {
        const smallerHistory = history.slice(0, 10);
        localStorage.setItem(HISTORY_KEY, JSON.stringify(smallerHistory));
      } catch (innerE) {
        console.error("Critical storage failure for history", innerE);
      }
    }
  }, [history]);

  useEffect(() => { 
    try {
      localStorage.setItem(TEMPLATES_KEY, JSON.stringify(instructionTemplates)); 
    } catch (e) { console.warn("Templates storage limit reached", e); }
  }, [instructionTemplates]);

  useEffect(() => { 
    try {
      localStorage.setItem(STRICT_RULES_KEY, JSON.stringify(strictRules)); 
    } catch (e) { console.warn("Rules storage limit reached", e); }
  }, [strictRules]);

  useEffect(() => { 
    try {
      localStorage.setItem(MASTER_PROTOCOLS_KEY, JSON.stringify(masterProtocols)); 
    } catch (e) { console.warn("Protocols storage limit reached", e); }
  }, [masterProtocols]);
  useEffect(() => { 
    try {
      localStorage.setItem(BRAND_SETTINGS_KEY, JSON.stringify(brandSettings)); 
    } catch (e) {
      console.warn("Storage quota exceeded. Some branding settings might not persist locally.", e);
    }
  }, [brandSettings]);
  useEffect(() => { 
    localStorage.setItem('dp_theme_v30', activeThemeId); 
    const theme = THEMES.find(t => t.id === activeThemeId) || THEMES[0];
    
    // Core Colors
    document.documentElement.style.setProperty('--primary-orange', theme.color);
    document.documentElement.style.setProperty('--accent-orange-light', theme.accent);
    document.documentElement.style.setProperty('--accent-orange-dark', theme.color);
    
    // Body Background
    const body = document.body;
    if (theme.bg.startsWith('linear-gradient') || theme.bg.startsWith('radial-gradient')) {
      body.style.backgroundImage = theme.bg;
      body.style.backgroundColor = 'transparent';
    } else {
      body.style.backgroundColor = theme.bg;
      body.style.backgroundImage = 'none';
    }

    // Handle text contrast (simple heuristic)
    const isDark = theme.id === 'midnight' || theme.id === 'nebula';
    body.style.color = isDark ? '#f8fafc' : '#1e293b';
    
    // Update sidebar/main backgrounds if they are too dark
    const main = document.querySelector('main');
    const aside = document.querySelector('aside');
    if (main) {
      main.style.backgroundColor = isDark ? '#0b1221' : 'rgba(255, 255, 255, 0.4)';
      main.style.backdropFilter = 'blur(20px)';
    }
    if (aside) {
      aside.style.backgroundColor = isDark ? '#0b1221' : 'rgba(255, 255, 255, 0.6)';
      aside.style.backdropFilter = 'blur(20px)';
    }

  }, [activeThemeId]);
  useEffect(() => { 
    localStorage.setItem(ENGINE_CONFIG_KEY, JSON.stringify({ active: activeEngine, keys: externalKeys }));
  }, [activeEngine, externalKeys]);

  useEffect(() => {
    const interval = setInterval(() => {
      const randomTheme = THEMES[Math.floor(Math.random() * THEMES.length)];
      setActiveThemeId(randomTheme.id);
    }, 30000);
    return () => clearInterval(interval);
  }, []);

  const cyclePriority = (current: Priority): Priority => {
    const priorities: Priority[] = ['Low', 'Average', 'Medium', 'High'];
    const currentIndex = priorities.indexOf(current);
    return priorities[(currentIndex + 1) % priorities.length];
  };

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    
    const code = loginCode.toLowerCase().trim();
    if (!loginName.trim()) { setLoginError('Full name is required.'); return; }
    
    // Master Passcodes for Access
    const validCodes = ['virtues', 'gratitude', 'dpss'];
    
    if (validCodes.includes(code)) {
      const newSession: UserSession = { 
        name: loginName, 
        code: code, 
        loginTime: Date.now(),
        email: 'local@user.com'
      };
      setSession(newSession);
      localStorage.setItem(USER_SESSION_KEY, JSON.stringify(newSession));
    } else { 
      setLoginError('Invalid Access Code.'); 
    }
  };

  const handleLogout = () => { 
    setSession(null); 
    localStorage.removeItem(USER_SESSION_KEY); 
  };
  
  const handleOnboardingComplete = () => {
    setShowOnboarding(false);
    localStorage.setItem(ONBOARDING_KEY, 'completed');
  };

  const toggleInstruction = (id: string) => setSelectedInstructionIds(prev => prev.includes(id) ? prev.filter(i => i !== id) : [...prev, id]);
  const setItemCount = (id: string, count: number) => setItemCountOverrides(prev => ({ ...prev, [id]: count }));
  const adjustColumns = (id: string, delta: number) => {
    setColumnOverrides(prev => ({ ...prev, [id]: Math.max(0, Math.min(6, (prev[id] || 0) + delta)) }));
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => setSourceMaterial({ data: (event.target?.result as string).split(',')[1], mimeType: file.type, name: file.name });
      reader.readAsDataURL(file);
    }
  };

  const handleLogoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      // 1. Initial size check
      if (file.size > 10 * 1024 * 1024) {
        alert("Image is too large. Please use a file smaller than 10MB.");
        if (logoUploadRef.current) logoUploadRef.current.value = '';
        return;
      }

      const reader = new FileReader();
      reader.onload = (event) => {
        const img = new Image();
        img.onload = () => {
          // 2. Resize & Compress Logic
          const canvas = document.createElement('canvas');
          let width = img.width;
          let height = img.height;
          
          // Max dimension 600px is sufficient for A4 header logos
          // This keeps file size very low (~50-100KB)
          const MAX_DIM = 600; 
          
          if (width > height) {
            if (width > MAX_DIM) {
              height *= MAX_DIM / width;
              width = MAX_DIM;
            }
          } else {
            if (height > MAX_DIM) {
              width *= MAX_DIM / height;
              height = MAX_DIM;
            }
          }
          
          canvas.width = width;
          canvas.height = height;
          const ctx = canvas.getContext('2d');
          if (ctx) {
             ctx.drawImage(img, 0, 0, width, height);
             // Compress to JPEG 0.7 quality
             const dataUrl = canvas.toDataURL('image/jpeg', 0.7);
             
             setBrandSettings(prev => {
               const newLogos = [...prev.logos];
               const firstEmpty = newLogos.findIndex(l => !l);
               if (firstEmpty !== -1) {
                 newLogos[firstEmpty] = dataUrl;
               } else {
                 newLogos.push(dataUrl);
               }
               return { ...prev, logos: newLogos, logoData: dataUrl };
             });
          }
          
          if (logoUploadRef.current) logoUploadRef.current.value = '';
        };
        img.src = event.target?.result as string;
      };
      reader.readAsDataURL(file);
    }
  };

  const removeLogo = (index: number) => {
    setBrandSettings(prev => {
      const newLogos = [...prev.logos];
      newLogos[index] = undefined;
      return { ...prev, logos: newLogos };
    });
  };

  const generateNeuralBlueprint = (count: number) => {
    const keys = ['A', 'B', 'C', 'D'];
    const blueprint: string[] = [];
    const counts: Record<string, number> = { A: 0, B: 0, C: 0, D: 0 };
    
    // Max 35% per letter
    const maxPerLetter = Math.max(1, Math.floor(count * 0.35));
    // Min 2 per letter if count is large enough
    const minPerLetter = count >= 8 ? 2 : 1;

    // 1. Fill with minimums
    for (const key of keys) {
      for (let i = 0; i < minPerLetter; i++) {
        if (blueprint.length < count) {
          blueprint.push(key);
          counts[key]++;
        }
      }
    }

    // 2. Fill the rest randomly
    while (blueprint.length < count) {
      const availableKeys = keys.filter(k => counts[k] < maxPerLetter);
      if (availableKeys.length === 0) {
        blueprint.push(keys[Math.floor(Math.random() * keys.length)]);
      } else {
        const randomKey = availableKeys[Math.floor(Math.random() * availableKeys.length)];
        blueprint.push(randomKey);
        counts[randomKey]++;
      }
    }

    // 3. Shuffle
    for (let i = blueprint.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [blueprint[i], blueprint[j]] = [blueprint[j], blueprint[i]];
    }

    return blueprint;
  };

  const handleGenerate = async () => {
    if (selectedInstructionIds.length === 0) { alert("Please select at least one component."); return; }
    setIsGenerating(true);
    
    const selectedTemps = instructionTemplates.filter(t => selectedInstructionIds.includes(t.id));

    // Filter Master Protocols and Strict Rules by category
    const filterByCategory = (rules: StrictRule[]) => 
      rules.filter(r => r.active && (r.category === 'General' || r.category.toLowerCase() === activeModule.toLowerCase()));

    const filteredProtocols = filterByCategory(masterProtocols);
    const filteredRules = filterByCategory(strictRules);

    // Enforce Shift + Underscore style
    const selectedBlankStyle = '____________________';

    const protocolsPrompt = filteredProtocols.map(p => `[PROTOCOL - ${p.priority}]: ${p.promptInjection.replace(/{{BLANK}}/g, selectedBlankStyle)}`).join('\n');
    const rulesPrompt = filteredRules.map(r => `[STRICT RULE - ${r.priority}]: ${r.promptInjection.replace(/{{BLANK}}/g, selectedBlankStyle)}`).join('\n');
    
    const strategyInstruction = answerStrategy === 'GENERAL_MIXED' 
      ? `[STRATEGY]: GENERAL-MIXED (Horizontal Logic). The context is {{TOPIC}}, but distractors should test high-frequency "general" errors (Gerunds, Prepositions, Agreement).`
      : `[STRATEGY]: TOPIC-FOCUSED (Vertical Logic). Every item and distractor must focus strictly on the rules of {{TOPIC}}.`;

    const componentLogic = selectedTemps.map((t, idx) => {
      const overrideCol = columnOverrides[t.id] || 0;
      const overrideItems = itemCountOverrides[t.id] || 10;
      
      // Generate unique blueprint for this part
      const blueprint = generateNeuralBlueprint(overrideItems);
      const blueprintStr = blueprint.map((key, i) => `${i + 1}:${key}`).join(', ');

      // ONLY add table instruction if columns are explicitly requested (> 0)
      const formatInstruction = overrideCol > 0 
        ? `(FORMAT: HTML <table> with ${overrideCol} columns. Ensure items are distributed evenly.)` 
        : `(FORMAT: Standard numbered list. DO NOT use tables or columns.)`;
        
      return `PART ${String.fromCharCode(65 + idx)} [MANDATORY INSTRUCTION HEADER: ${t.label}]: ${t.prompt.replace(/{{BLANK}}/g, selectedBlankStyle)} (GENERATE EXACTLY ${overrideItems} ITEMS) (USE THIS ANSWER KEY: ${blueprintStr}) ${formatInstruction}`;
    }).join('\n\n');

    const moduleSafetyGuard = activeModule === 'Grammar'
      ? `[MODULE SAFETY GUARD - CRITICAL]: You are generating a GRAMMAR assessment. You are strictly FORBIDDEN from including reading passages or vocabulary-only definitions. Focus 100% on grammar rules, situational logic, and positional word order. Ensure NO LEAKAGE from Reading or Vocabulary modules.`
      : activeModule === 'Vocabulary'
      ? `[MODULE SAFETY GUARD - CRITICAL]: You are generating a VOCABULARY assessment. You are strictly FORBIDDEN from testing grammar rules, injecting grammar errors, or including reading passages. 
         - NO READING LOGIC: Do NOT include "Not Mentioned" or "Unknown" options. 
         - NO GRAMMAR LOGIC: Protocol 21 (Cross-Topic Injection) and Rule 1 (No-Free-Verb) are DISABLED. 
         - NO GRAMMAR TOPICS: Avoid using sentences that test "Must/Have to", "Should", or other modal verbs. Focus on the meaning of the word itself.
         - PURE SEMANTICS: Focus 100% on word meanings. All distractors must be grammatically identical to the correct answer.`
      : activeModule === 'Reading'
      ? `[MODULE SAFETY GUARD - CRITICAL]: You are generating a READING assessment. You are strictly FORBIDDEN from testing grammar rules or injecting grammar errors. Focus 100% on comprehension and inference logic.`
      : '';

    const mandatorySequence = activeModule === 'Grammar' 
      ? `1. PRE-ASSIGN balanced answer keys (A-D).\n2. GENERATE ALL REQUESTED PARTS. ADAPT TITLES TO MATCH "${topic}".\n3. ENFORCE "NO FREE VERB" & "SITUATIONAL EVIDENCE" rules for all grammar stems.`
      : activeModule === 'Reading'
      ? `1. GENERATE A PASSAGE (~300-500 words) about "${topic}".\n2. APPLY [NATURAL PARAPHRASE] logic to all questions (No keyword matching).\n3. ENFORCE [READING LOGIC FIREWALL] (Strictly forbidden from testing grammar).\n4. ENSURE all distractors are grammatically identical to the correct answer.`
      : `1. PRE-ASSIGN balanced answer keys (A-D).\n2. GENERATE ALL REQUESTED PARTS. ADAPT TITLES TO MATCH "${topic}".\n3. ENFORCE [VOCABULARY FIREWALL] (No grammar clues).`;

    const finalLogic = `
${moduleSafetyGuard}
${GLOBAL_STRICT_COMMAND.replace(/{{TOPIC}}/g, topic || "General English").replace(/{{BLANK}}/g, selectedBlankStyle)}
${protocolsPrompt}
${strategyInstruction.replace(/{{TOPIC}}/g, topic || "General English")}
${rulesPrompt}

[SYSTEM OBJECTIVE]: Generate a COMPLETE assessment based on the requested components.
[TARGET TOPIC]: "${topic || "General English"}"
[TARGET LEVEL]: ${activeLevel}
[LANGUAGE]: ${activeLanguage}

### MANDATORY SEQUENCE ###
${mandatorySequence}

${componentLogic}
    `;
    
    try {
      // Randomize logo from available logos
      const availableLogos = brandSettings.logos.filter(l => !!l);
      if (availableLogos.length > 0) {
        const randomLogo = availableLogos[Math.floor(Math.random() * availableLogos.length)];
        setBrandSettings(prev => ({ ...prev, logoData: randomLogo }));
      }

      // FIREBASE CLOUD SAVE IMPLEMENTATION
      // ==================================================
      // 1. Call the AI Brain
      const result = await callNeuralEngine(activeEngine, finalLogic, protocolsPrompt, sourceMaterial, externalKeys);
      setWorksheetContent(result.text);
      setIsGenerating(false);
      setViewMode('preview');

      // 2. Create the data package
      const newTestItem = {
        id: `hist-${Date.now()}`,
        title: `${activeLanguage} ${activeModule}: ${activeLevel} - ${topic || "Synthesis"}`,
        content: result.text,
        timestamp: Date.now(),
        promptId: 'manual',
        logicSnapshot: finalLogic,
        module: activeModule,
        level: activeLevel,
        topic: topic,
        // Add who created it
        authorName: session?.name || 'Anonymous',
        authorCode: session?.code || 'N/A',
        authorEmail: session?.email || 'N/A'
      };

      // 3. Update Local History (so you see it on screen)
      setHistory(prev => [newTestItem, ...prev].slice(0, 30));
    } catch (error: any) {
      console.error("Generation failed:", error);
      alert("Neural synthesis failed. Please check your connection or API keys.");
      setIsGenerating(false);
    }
  };

  const handleAssistantMessage = async (msg: string, file?: QuickSource) => {
    const userMsg: ChatMessage = { id: `msg-${Date.now()}`, role: 'user', text: msg, timestamp: Date.now() };
    setChatMessages(prev => [...prev, userMsg]);
    setIsGenerating(true);
    const context = `Assistant Mode. Worksheet: ${worksheetContent.slice(0, 1000)}. Edit based on: ${msg}`;
    const result = await callNeuralEngine(activeEngine, msg, context, file || sourceMaterial, externalKeys);
    setChatMessages(prev => [...prev, { id: `msg-bot-${Date.now()}`, role: 'architect', text: "Synthesis updated.", timestamp: Date.now() }]);
    setWorksheetContent(result.text);
    setIsGenerating(false);
  };

  const handleExportWord = () => {
    if (!worksheetContent) return;
    const logoHtml = brandSettings.logoData ? `<table style="width: 100%; border: none; margin-bottom: 8pt;"><tr><td style="border: none; text-align: center;"><img src="${brandSettings.logoData}" width="621" style="width: 16.43cm;" /></td></tr></table>` : '';
    const header = `${logoHtml}<table style="width: 100%; border-bottom: 2pt solid black; margin-bottom: 15pt; font-family: 'Times New Roman', serif;"><tr><td style="border: none; width: 50%;"><b>Teacher:</b> ${session?.name || '________________'}</td><td style="border: none; width: 50%; text-align: right;"><b>${activeModule}: ${topic || 'Assessment'}</b><br/>${activeLevel} | ${activeLanguage}</td></tr></table>`;
    exportToWord(worksheetContent, `DPSS_Test_${activeLanguage}_${activeLevel}`, header, '0.6cm');
  };

  const updateRule = (id: string, updates: Partial<StrictRule>) => setStrictRules(prev => prev.map(r => r.id === id ? { ...r, ...updates } : r));
  const updateProtocol = (id: string, updates: Partial<StrictRule>) => setMasterProtocols(prev => prev.map(p => p.id === id ? { ...p, ...updates } : p));
  const updateTemplate = (id: string, updates: Partial<InstructionTemplate>) => setInstructionTemplates(prev => prev.map(t => t.id === id ? { ...t, ...updates } : t));
  const deleteTemplate = (id: string) => setInstructionTemplates(prev => prev.filter(t => t.id !== id));
  const deleteRule = (id: string) => setStrictRules(prev => prev.filter(r => r.id !== id));
  const deleteProtocol = (id: string) => setMasterProtocols(prev => prev.filter(p => p.id !== id));
  
  const syncWithDefaults = () => {
    setMasterProtocols(prev => {
      const existingIds = new Set(prev.map(p => p.id));
      const newItems = DEFAULT_MASTER_PROTOCOLS.filter(p => !existingIds.has(p.id));
      return [...prev, ...newItems];
    });
    setStrictRules(prev => {
      const existingIds = new Set(prev.map(r => r.id));
      const newItems = DEFAULT_STRICT_RULES.filter(r => !existingIds.has(r.id));
      return [...prev, ...newItems];
    });
    setInstructionTemplates(prev => {
      const existingIds = new Set(prev.map(t => t.id));
      const newItems = INITIAL_TEMPLATES.filter(t => !existingIds.has(t.id));
      return [...prev, ...newItems];
    });
    alert("Neural protocols and templates synchronized with latest definitions.");
  };

  const hardReset = () => {
    if (confirm("WARNING: This will delete all custom rules, protocols, and templates. Are you sure?")) {
      localStorage.removeItem(MASTER_PROTOCOLS_KEY);
      localStorage.removeItem(STRICT_RULES_KEY);
      localStorage.removeItem(TEMPLATES_KEY);
      window.location.reload();
    }
  };

  const addRule = () => {
    const newRule: StrictRule = { id: `rule-${Date.now()}`, label: 'NEW LOGIC NODE', description: '', promptInjection: '', active: true, priority: 'Medium', category: activeLogicCategory };
    setStrictRules([...strictRules, newRule]); setExpandedRuleId(newRule.id);
  };
  const addProtocol = () => {
    const newProtocol: StrictRule = { id: `mp-${Date.now()}`, label: 'NEW PROTOCOL', description: '', promptInjection: '', active: true, priority: 'Medium', category: activeProtocolCategory };
    setMasterProtocols([...masterProtocols, newProtocol]); setExpandedProtocolId(newProtocol.id);
  };
  const addTemplate = () => {
    const newId = `temp-${Date.now()}`;
    setInstructionTemplates(prev => [...prev, { id: newId, label: `NEW PART`, prompt: `Detail logic for {{TOPIC}}...`, category: activeTemplateCategory as any, columnCount: 0 }]);
    setExpandedTemplateId(newId);
  };

  if (!session) {
    return (
      <div className="h-screen w-screen bg-[#0b1221] flex items-center justify-center p-6 text-white overflow-hidden">
        <div className="w-full max-w-xl bg-white/5 backdrop-blur-xl border border-white/10 rounded-[64px] p-12 text-center shadow-2xl">
           <div className="h-20 w-20 bg-orange-600 rounded-3xl flex items-center justify-center shadow-2xl mx-auto mb-12"><i className="fa-solid fa-bolt text-white text-3xl"></i></div>
           <h1 className="text-2xl font-[900] uppercase tracking-wider mb-8">DPSS Ultimate Test Builder</h1>
           
           <form onSubmit={handleLogin} className="space-y-6 text-left animate-in fade-in slide-in-from-bottom-6 duration-500">
                <input type="text" value={loginName} onChange={(e) => setLoginName(e.target.value)} placeholder="Full Name" className="w-full bg-white/5 border border-white/10 rounded-3xl px-8 py-5 outline-none focus:border-orange-500 font-bold" />
                <input type="password" value={loginCode} onChange={(e) => setLoginCode(e.target.value)} placeholder="Access Code / Passcode" className="w-full bg-white/5 border border-white/10 rounded-3xl px-8 py-5 outline-none focus:border-orange-500 font-bold" />
                {loginError && <p className="text-rose-500 text-xs font-black uppercase text-center">{loginError}</p>}
                <button type="submit" className="w-full bg-orange-600 text-white py-6 rounded-3xl text-sm font-black uppercase tracking-widest hover:brightness-110 shadow-xl transition-all">
                  Initialize Architecture
                </button>
             </form>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen overflow-hidden text-slate-300 relative transition-all duration-500">
      {showOnboarding && <OnboardingTutorial onComplete={handleOnboardingComplete} />}
      {viewMode === 'generator' && (
        <>
          <aside className={`fixed inset-y-0 left-0 z-50 w-72 border-r border-[#1f2937] flex flex-col shrink-0 lg:relative lg:translate-x-0 lg:w-80 ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'} transition-all duration-500`}>
            <div className="p-6 border-b border-[#1f2937] flex justify-between items-center"><h2 className="text-[10px] font-black uppercase tracking-widest text-slate-500">History Log</h2><button onClick={() => setIsSidebarOpen(false)} className="lg:hidden h-8 w-8 rounded-full bg-white/5 flex items-center justify-center text-slate-400"><i className="fa-solid fa-xmark"></i></button></div>
            <div className="flex-1 overflow-y-auto no-scrollbar p-6 space-y-4">
               {history.map(item => (<button key={item.id} onClick={() => { setWorksheetContent(item.content); setViewMode('preview'); }} className="w-full text-left p-4 rounded-2xl bg-[#111827] border border-[#1f2937] hover:border-orange-500/30 transition-all"><div className="text-[8px] text-slate-600 font-black uppercase mb-1">{new Date(item.timestamp).toLocaleDateString()}</div><div className="text-[11px] font-bold text-slate-400 line-clamp-1">{item.title}</div></button>))}
            </div>
              <div className="p-6 border-t border-[#1f2937] space-y-2">
              <button onClick={() => setShowOnboarding(true)} className="w-full flex items-center justify-between p-4 rounded-2xl bg-white/5 border border-white/10 text-slate-400 text-[9px] font-black uppercase hover:bg-white/10 transition-all"><span>Restart Tutorial</span><i className="fa-solid fa-circle-question"></i></button>
              <button onClick={() => setShowSettings(true)} className="w-full flex items-center justify-between p-5 rounded-2xl bg-gradient-to-r from-accent-orange-dark to-accent-orange-light text-white shadow-lg uppercase text-[11px] font-black hover:brightness-110 transition-all"><span>Architect Settings</span><i className="fa-solid fa-gear"></i></button>
              <button onClick={handleLogout} className="w-full p-4 rounded-2xl bg-white/5 border border-white/10 text-slate-500 text-[9px] font-black uppercase hover:bg-white/10 transition-all">Logout Session</button>
            </div>
          </aside>

          <main className="flex-1 border-r border-[#1f2937] flex flex-col overflow-y-auto no-scrollbar transition-all duration-500">
            <div className="p-6 lg:p-10 max-w-5xl mx-auto w-full space-y-8 lg:space-y-12">
              <div className="flex items-center justify-between gap-4">
                <div className="flex items-center gap-4">
                  <button onClick={() => setIsSidebarOpen(!isSidebarOpen)} className="h-12 w-12 rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center text-white lg:hidden"><i className={`fa-solid ${isSidebarOpen ? 'fa-angles-left' : 'fa-bars'}`}></i></button>
                  <div><h1 className="text-xl lg:text-3xl font-[900] uppercase tracking-wider text-white">DPSS Ultimate Test Builder</h1><span className="text-orange-500 text-[9px] font-black uppercase tracking-widest mt-1 block">Architect: {session.name}</span></div>
                </div>
                <div className="h-12 w-12 bg-orange-600 rounded-xl flex items-center justify-center shadow-lg"><i className="fa-solid fa-bolt text-white text-xl"></i></div>
              </div>
              
              <div className="space-y-4">
                 <h3 className="text-[10px] font-black uppercase text-slate-500 tracking-widest">Assessment Matrix</h3>
                  <div className="flex bg-[#111827] p-1.5 rounded-2xl border border-[#1f2937] gap-1 overflow-x-auto no-scrollbar">
                    {INITIAL_MODULES.map(mod => (
                      <button 
                        key={mod} 
                        onClick={() => { 
                          setActiveModule(mod); 
                          setSelectedInstructionIds([]); 
                          if (mod === 'Grammar') {
                            setViewMode('grammar_iframe');
                          } else {
                            setViewMode('generator');
                          }
                        }} 
                        className={`flex-1 min-w-[100px] py-4 px-6 rounded-xl transition-all ${(activeModule === mod && (viewMode === 'generator' || viewMode === 'grammar_iframe')) ? 'bg-accent-blue text-white shadow-lg' : 'text-slate-600 hover:text-slate-400'}`}
                      >
                        <div className="text-[11px] font-black tracking-widest">{mod}</div>
                      </button>
                    ))}
                    <button 
                      onClick={() => setViewMode('book_creation')} 
                      className={`flex-1 min-w-[150px] py-4 px-6 rounded-xl transition-all ${viewMode === 'book_creation' ? 'bg-accent-blue text-white shadow-lg' : 'text-slate-600 hover:text-slate-400'}`}
                    >
                      <div className="text-[11px] font-black tracking-widest">Book Creation</div>
                    </button>
                    <button 
                      onClick={() => setViewMode('ielts_master')} 
                      className={`flex-1 min-w-[150px] py-4 px-6 rounded-xl transition-all ${viewMode === 'ielts_master' ? 'bg-accent-blue text-white shadow-lg' : 'text-slate-600 hover:text-slate-400'}`}
                    >
                      <div className="text-[11px] font-black tracking-widest">IELTS Master</div>
                    </button>
                    <button 
                      onClick={() => setViewMode('dpss_studio')} 
                      className={`flex-1 min-w-[150px] py-4 px-6 rounded-xl transition-all ${viewMode === 'dpss_studio' ? 'bg-accent-blue text-white shadow-lg' : 'text-slate-600 hover:text-slate-400'}`}
                    >
                      <div className="text-[11px] font-black tracking-widest">DPSS Studio</div>
                    </button>
                  </div>
              </div>

              <div className="space-y-4">
                 <h3 className="text-[10px] font-black uppercase text-slate-500 tracking-widest">Answer Strategy Logic</h3>
                 <div className="flex bg-[#111827] p-1.5 rounded-2xl border border-[#1f2937] gap-1">
                    <button onClick={() => setAnswerStrategy('TOPIC_FOCUSED')} className={`flex-1 py-3.5 px-6 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all ${answerStrategy === 'TOPIC_FOCUSED' ? 'bg-accent-orange-dark text-white shadow-lg' : 'text-slate-600 hover:text-slate-400'}`}>Topic-Focused Focus</button>
                    <button onClick={() => setAnswerStrategy('GENERAL_MIXED')} className={`flex-1 py-3.5 px-6 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all ${answerStrategy === 'GENERAL_MIXED' ? 'bg-accent-blue text-white shadow-lg' : 'text-slate-600 hover:text-slate-400'}`}>General-Mixed Strategy</button>
                 </div>
              </div>

              <div className="space-y-4">
                 <h3 className="text-[10px] font-black uppercase text-slate-500 tracking-widest">Target Language</h3>
                 <div className="grid grid-cols-3 md:grid-cols-5 gap-2">
                    {LANGUAGES.map(lang => (<button key={lang} onClick={() => setActiveLanguage(lang)} className={`py-3 rounded-xl text-[10px] font-black uppercase border transition-all ${activeLanguage === lang ? 'bg-accent-orange-dark border-accent-orange-dark text-white' : 'bg-[#111827] border-[#1f2937] text-slate-600 hover:border-slate-500'}`}>{lang}</button>))}
                 </div>
              </div>

              <div className="space-y-4">
                 <h3 className="text-[10px] font-black uppercase text-slate-500 tracking-widest">Academic Level Selection</h3>
                 <div className="flex flex-wrap gap-2">
                    {ACADEMIC_LEVELS.map(level => (<button key={level} onClick={() => setActiveLevel(level)} className={`py-3.5 px-4 rounded-xl text-[10px] font-black uppercase border transition-all ${activeLevel === level ? 'bg-accent-blue text-white border-accent-blue shadow-xl' : 'bg-[#111827] text-slate-600 border-[#1f2937] hover:border-slate-500'}`}>{level.replace('Level ', '')}</button>))}
                 </div>
              </div>

              <div className="space-y-6">
                <h3 className="text-[10px] font-black uppercase text-slate-500 tracking-widest">{activeModule} Structural Parts</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {instructionTemplates.filter(t => t.category?.toUpperCase() === activeModule.toUpperCase()).map(t => {
                    const isSelected = selectedInstructionIds.includes(t.id);
                    const curItems = itemCountOverrides[t.id] || 10;
                    const curCols = columnOverrides[t.id] || 0;
                    return (
                      <div key={t.id} className={`blueprint-card rounded-3xl p-6 border transition-all ${isSelected ? 'active ring-1 ring-orange-500' : ''}`}>
                        <div className="flex flex-col gap-4">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-4">
                               <button onClick={() => toggleInstruction(t.id)} className={`h-10 w-10 rounded-xl border-2 flex items-center justify-center transition-all ${isSelected ? 'bg-orange-600 border-orange-600 shadow-lg' : 'border-[#374151]'}`}>{isSelected && <i className="fa-solid fa-check text-white text-[14px]"></i>}</button>
                               <span className="text-[12px] font-black uppercase tracking-widest text-white">{t.label}</span>
                            </div>
                          </div>
                          {isSelected && (
                            <div className="flex flex-wrap gap-4 items-center animate-in fade-in">
                               <div className="flex bg-black/40 rounded-xl p-1 gap-1 border border-white/5 no-scrollbar max-w-full">{[5, 10, 15, 20, 30, 40, 60].map(num => (<button key={num} onClick={() => setItemCount(t.id, num)} className={`h-7 px-3 rounded-lg text-[9px] font-black uppercase transition-all ${curItems === num ? 'bg-accent-orange-dark text-white shadow-sm' : 'text-slate-500'}`}>{num}</button>))}</div>
                               <div className="flex items-center bg-black/40 rounded-xl p-1 px-3 border border-white/5 gap-3"><span className="text-[9px] font-black text-slate-500 uppercase">Cols</span><div className="flex items-center gap-3"><button onClick={() => adjustColumns(t.id, -1)} className="h-6 w-6 bg-slate-800 rounded flex items-center justify-center text-slate-400">-</button><span className="text-[11px] font-black text-accent-orange-light w-3 text-center">{curCols || 'L'}</span><button onClick={() => adjustColumns(t.id, 1)} className="h-6 w-6 bg-slate-800 rounded flex items-center justify-center text-slate-400">+</button></div></div>
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              <div className="space-y-4">
                 <div className="flex justify-between items-center px-2"><h3 className="text-[10px] font-black uppercase text-slate-500 tracking-widest">Topic Anchor</h3><button onClick={() => fileInputRef.current?.click()} className={`text-[10px] font-black uppercase flex items-center gap-2 transition-all ${sourceMaterial ? 'text-emerald-500' : 'text-slate-400 hover:text-orange-500'}`}><i className="fa-solid fa-paperclip"></i> {sourceMaterial ? sourceMaterial.name : 'Inject Source'}</button></div>
                 <input type="file" ref={fileInputRef} onChange={handleFileUpload} className="hidden" accept=".pdf,.doc,.docx,image/*" />
                 <textarea value={topic} onChange={e => setTopic(e.target.value)} placeholder="Target Topic (e.g., 'Past Simple vs Present Perfect', 'Daily Life')." className="w-full h-40 bg-[#111827] border border-[#1f2937] rounded-[40px] p-6 lg:p-8 text-white outline-none focus:border-orange-500/50 resize-none font-medium text-base lg:text-lg" />
              </div>
              <button onClick={handleGenerate} className="w-full bg-gradient-to-r from-accent-orange-dark to-accent-orange-light text-white py-6 lg:py-8 rounded-[40px] text-lg lg:text-xl font-black uppercase tracking-[0.25em] shadow-2xl hover:brightness-110 transition-all flex items-center justify-center gap-6">{isGenerating ? <i className="fa-solid fa-spinner animate-spin"></i> : <i className="fa-solid fa-sparkles"></i>}Synthesize Full Test</button>
            </div>
          </main>
        </>
      )}

      {viewMode === 'preview' && (
        <section className="flex-1 flex flex-col overflow-hidden animate-in fade-in duration-500">
          <div className="p-4 lg:p-6 border-b border-[#1f2937] flex flex-wrap gap-4 justify-between items-center backdrop-blur-xl z-10 no-print shadow-2xl">
            <button onClick={() => setViewMode('generator')} className="border border-[#1f2937] text-white px-6 lg:px-8 py-3 rounded-full text-[11px] font-black uppercase tracking-widest hover:bg-slate-900/50 flex items-center gap-4 group shadow-xl"><i className="fa-solid fa-arrow-left group-hover:-translate-x-1 transition-transform"></i> ARCHITECT</button>
            <div className="flex gap-2 lg:gap-3 ml-auto"><button onClick={handleExportWord} className="px-6 lg:px-10 py-3 bg-[#ea580c] text-white rounded-full text-[11px] font-black uppercase tracking-widest hover:brightness-110 shadow-2xl">EXPORT DOC</button><button onClick={() => window.print()} className="h-10 w-10 lg:h-12 lg:w-12 bg-[#111827]/50 border border-[#1f2937] rounded-full flex items-center justify-center text-slate-400 hover:text-orange-500 transition-all shadow-xl"><i className="fa-solid fa-print"></i></button></div>
          </div>
          <div className="flex-1 overflow-y-auto no-scrollbar"><Worksheet content={worksheetContent} onContentChange={setWorksheetContent} isGenerating={isGenerating} theme={THEMES.find(t => t.id === activeThemeId) || THEMES[0]} paperType="Plain" brandSettings={brandSettings} level={activeLevel} module={activeModule} topic={topic} /></div>
        </section>
      )}

      {viewMode === 'grammar_iframe' && (
        <section className="flex-1 flex flex-col overflow-hidden animate-in fade-in duration-500">
          <div className="p-4 lg:p-6 border-b border-[#1f2937] flex flex-wrap gap-4 justify-between items-center backdrop-blur-xl z-10 no-print shadow-2xl">
            <button onClick={() => setViewMode('generator')} className="border border-[#1f2937] text-white px-6 lg:px-8 py-3 rounded-full text-[11px] font-black uppercase tracking-widest hover:bg-slate-900/50 flex items-center gap-4 group shadow-xl">
              <i className="fa-solid fa-arrow-left group-hover:-translate-x-1 transition-transform"></i> ARCHITECT
            </button>
            <div className="flex-1 text-center">
              <h2 className="text-white font-black uppercase tracking-widest text-[12px]">Neural Grammar Architect</h2>
            </div>
            <div className="flex gap-2">
              <a 
                href="https://aistudio.google.com/apps/f6448ec0-06de-44f2-93d6-13cd43bceb87?showPreview=true&showAssistant=true" 
                target="_blank" 
                rel="noopener noreferrer"
                className="px-6 py-3 bg-orange-600 text-white rounded-full text-[11px] font-black uppercase tracking-widest hover:bg-orange-500 shadow-xl flex items-center gap-2"
              >
                <i className="fa-solid fa-arrow-up-right-from-square"></i> Launch Tool
              </a>
            </div>
          </div>
          <div className="flex-1 bg-white overflow-hidden relative">
            <div className="absolute inset-0 flex flex-col items-center justify-center p-10 text-center bg-slate-50 -z-10">
              <i className="fa-solid fa-circle-exclamation text-4xl text-slate-300 mb-4"></i>
              <p className="text-slate-500 font-bold text-sm">If the tool refuses to connect, please use the "Launch Tool" button above.</p>
            </div>
            <iframe 
              src="https://aistudio.google.com/apps/f6448ec0-06de-44f2-93d6-13cd43bceb87?showPreview=true&showAssistant=true"
              className="w-full h-full min-h-[800px] border-none relative z-10"
              title="Grammar Tool"
              sandbox="allow-scripts allow-same-origin allow-forms allow-popups"
            />
          </div>
        </section>
      )}

      {viewMode === 'book_creation' && (
        <section className="flex-1 flex flex-col overflow-hidden animate-in fade-in duration-500">
          <div className="p-4 lg:p-6 border-b border-[#1f2937] flex flex-wrap gap-4 justify-between items-center backdrop-blur-xl z-10 no-print shadow-2xl">
            <button onClick={() => setViewMode('generator')} className="border border-[#1f2937] text-white px-6 lg:px-8 py-3 rounded-full text-[11px] font-black uppercase tracking-widest hover:bg-slate-900/50 flex items-center gap-4 group shadow-xl">
              <i className="fa-solid fa-arrow-left group-hover:-translate-x-1 transition-transform"></i> ARCHITECT
            </button>
            <div className="flex-1 text-center">
              <h2 className="text-white font-black uppercase tracking-widest text-[12px]">Neural Book Architect</h2>
            </div>
            <div className="flex gap-2">
              <a 
                href="https://remix-book-creation-4-deploy-370806846570.us-west1.run.app/" 
                target="_blank" 
                rel="noopener noreferrer"
                className="px-6 py-3 bg-orange-600 text-white rounded-full text-[11px] font-black uppercase tracking-widest hover:bg-orange-500 shadow-xl flex items-center gap-2"
              >
                <i className="fa-solid fa-arrow-up-right-from-square"></i> Launch Tool
              </a>
            </div>
          </div>
          <div className="flex-1 bg-white overflow-hidden relative">
            <div className="absolute inset-0 flex flex-col items-center justify-center p-10 text-center bg-slate-50 -z-10">
              <i className="fa-solid fa-circle-exclamation text-4xl text-slate-300 mb-4"></i>
              <p className="text-slate-500 font-bold text-sm">If the tool refuses to connect, please use the "Launch Tool" button above.</p>
            </div>
            <iframe 
              src="https://remix-book-creation-4-deploy-370806846570.us-west1.run.app/"
              className="w-full h-full min-h-[800px] border-none relative z-10"
              title="Book Creation Tool"
            />
          </div>
        </section>
      )}

      {viewMode === 'ielts_master' && (
        <section className="flex-1 flex flex-col overflow-hidden animate-in fade-in duration-500">
          <div className="p-4 lg:p-6 border-b border-[#1f2937] flex flex-wrap gap-4 justify-between items-center backdrop-blur-xl z-10 no-print shadow-2xl">
            <button onClick={() => setViewMode('generator')} className="border border-[#1f2937] text-white px-6 lg:px-8 py-3 rounded-full text-[11px] font-black uppercase tracking-widest hover:bg-slate-900/50 flex items-center gap-4 group shadow-xl">
              <i className="fa-solid fa-arrow-left group-hover:-translate-x-1 transition-transform"></i> ARCHITECT
            </button>
            <div className="flex-1 text-center">
              <h2 className="text-white font-black uppercase tracking-widest text-[12px]">IELTS Mastermind</h2>
            </div>
          </div>
          <div className="flex-1 bg-white overflow-hidden">
            <iframe 
              src="https://ielts-mastermind-dpss-deploy-11-370806846570.us-west1.run.app/"
              className="w-full h-full min-h-[800px] border-none"
              title="IELTS Master Tool"
            />
          </div>
        </section>
      )}

      {viewMode === 'dpss_studio' && (
        <section className="flex-1 flex flex-col overflow-hidden animate-in fade-in duration-500">
          <div className="p-4 lg:p-6 border-b border-[#1f2937] flex flex-wrap gap-4 justify-between items-center backdrop-blur-xl z-10 no-print shadow-2xl">
            <button onClick={() => setViewMode('generator')} className="border border-[#1f2937] text-white px-6 lg:px-8 py-3 rounded-full text-[11px] font-black uppercase tracking-widest hover:bg-slate-900/50 flex items-center gap-4 group shadow-xl">
              <i className="fa-solid fa-arrow-left group-hover:-translate-x-1 transition-transform"></i> ARCHITECT
            </button>
            <div className="flex-1 text-center">
              <h2 className="text-white font-black uppercase tracking-widest text-[12px]">DPSS Studio</h2>
            </div>
          </div>
          <div className="flex-1 bg-white overflow-hidden">
            <iframe 
              src="https://remix-100-book-creation-deploy-370806846570.us-west1.run.app/"
              className="w-full h-full min-h-[800px] border-none"
              title="DPSS Studio Tool"
            />
          </div>
        </section>
      )}
      {!showSettings && isAssistantVisible && (
        <div className="fixed bottom-24 right-6 w-[340px] max-w-[90vw] h-[500px] bg-slate-900/90 backdrop-blur-3xl rounded-[40px] border border-white/10 shadow-2xl overflow-hidden flex flex-col animate-in slide-in-from-bottom-10">
            <div className="p-6 border-b border-white/5 flex items-center justify-between"><div className="flex items-center gap-3"><div className="h-2 w-2 bg-orange-500 rounded-full animate-pulse"></div><span className="text-[10px] font-black uppercase tracking-widest text-white">Live Assistant</span></div><button onClick={() => setIsAssistantVisible(false)} className="h-8 w-8 rounded-full bg-white/5 flex items-center justify-center text-slate-400 hover:text-white"><i className="fa-solid fa-minus"></i></button></div>
            <div className="flex-1 overflow-hidden p-4"><NeuralChatAssistant messages={chatMessages} input={chatInput} onInputChange={setChatInput} onSendMessage={handleAssistantMessage} isGenerating={isGenerating} quickSource={sourceMaterial} inline={true} /></div>
        </div>
      )}
      {!showSettings && (<button onClick={() => setIsAssistantVisible(!isAssistantVisible)} className={`fixed bottom-6 right-6 h-16 w-16 rounded-full flex items-center justify-center text-white shadow-2xl transition-all ${isAssistantVisible ? 'bg-orange-600 rotate-90' : 'bg-slate-800'}`}><i className={`fa-solid ${isAssistantVisible ? 'fa-xmark' : 'fa-wand-magic-sparkles text-xl'}`}></i></button>)}

      {showSettings && (
        <div className="fixed inset-0 z-[250] bg-slate-950/80 backdrop-blur-2xl flex items-center justify-center p-4">
          <div className="bg-[#f8fafc] bg-[radial-gradient(circle_at_top_right,rgba(234,88,12,0.03),transparent_40%),radial-gradient(circle_at_bottom_left,rgba(37,99,235,0.03),transparent_40%)] rounded-[48px] lg:rounded-[64px] w-full max-w-7xl h-full max-h-[95vh] overflow-hidden shadow-2xl flex flex-col border border-white/50">
             <div className="p-8 lg:p-12 pb-4 flex justify-between items-center"><div className="flex items-center gap-4"><div className="h-4 w-4 bg-orange-600 rounded-full animate-pulse"></div><h2 className="text-[12px] font-black uppercase text-slate-900 tracking-widest">Workspace Control Node</h2></div><button onClick={() => setShowSettings(false)} className="h-10 w-10 bg-slate-100 rounded-full flex items-center justify-center text-slate-400 hover:text-slate-900"><i className="fa-solid fa-xmark text-xl"></i></button></div>
             <div className="px-6 lg:px-12 mb-8"><div className="flex bg-slate-100/70 p-2 rounded-[32px] gap-1 overflow-x-auto no-scrollbar shadow-inner">{['ACCOUNT', 'COMMAND', 'ENGINE', 'BACKBONE LOGIC', 'DISPLAY', 'DESIGN'].map(tab => (<button key={tab} onClick={() => setSettingsTab(tab as SettingsTab)} className={`px-6 lg:px-10 py-4 rounded-[28px] text-[10px] font-black uppercase tracking-widest transition-all ${settingsTab === tab ? 'bg-orange-600 text-white shadow-xl' : 'text-slate-400 hover:text-slate-600'}`}>{tab}</button>))}</div></div>
             <div className="flex-1 overflow-y-auto px-6 lg:px-12 pb-12 space-y-12 no-scrollbar">
                {settingsTab === 'DESIGN' && (
                  <div className="space-y-12 animate-in fade-in slide-in-from-bottom-6">
                    <div className="space-y-8">
                      <h3 className="text-[13px] font-black text-slate-900 uppercase tracking-widest">Branding Architecture</h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
                        <div className="space-y-6">
                          <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest">School Identity</label>
                          <input value={brandSettings.schoolName} onChange={e => setBrandSettings({ ...brandSettings, schoolName: e.target.value })} className="w-full bg-slate-100 border border-slate-200 rounded-3xl px-8 py-5 text-[14px] font-black text-slate-900 uppercase focus:border-orange-500 outline-none" placeholder="School Name" />
                          <input value={brandSettings.schoolAddress} onChange={e => setBrandSettings({ ...brandSettings, schoolAddress: e.target.value })} className="w-full bg-slate-100 border border-slate-200 rounded-3xl px-8 py-5 text-[14px] font-black text-slate-900 uppercase focus:border-orange-500 outline-none" placeholder="Address / Motto" />
                        </div>
                        <div className="space-y-6">
                          <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Header Logo (A4 Precision)</label>
                          <div className="border-4 border-dashed border-slate-200 rounded-[48px] p-10 flex flex-col items-center justify-center gap-6 hover:border-orange-500 transition-all cursor-pointer relative" onClick={() => logoUploadRef.current?.click()}>
                            {brandSettings.logoData ? <img src={brandSettings.logoData} className="max-h-24 w-auto rounded-xl" /> : <i className="fa-solid fa-cloud-arrow-up text-4xl text-slate-300"></i>}
                            <span className="text-[10px] font-black text-slate-400 uppercase">Upload Header Graphic</span>
                            <input type="file" ref={logoUploadRef} onChange={handleLogoUpload} className="hidden" accept="image/*" />
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="space-y-8">
                      <div className="flex justify-between items-center px-2">
                        <h3 className="text-[13px] font-black text-slate-900 uppercase tracking-widest">Neural Logo Registry ({brandSettings.logos.filter(l => !!l).length} / {brandSettings.logos.length})</h3>
                        <div className="flex gap-4">
                          <button onClick={() => { if(window.confirm("Clear all logos to free up space?")) setBrandSettings(prev => ({ ...prev, logos: Array(30).fill(undefined) })); }} className="text-[11px] font-black text-rose-500 uppercase border-b-2 border-rose-500">Clear All</button>
                          <button onClick={() => logoUploadRef.current?.click()} className="text-[11px] font-black text-orange-600 uppercase border-b-2 border-orange-600">+ Add Logo</button>
                        </div>
                      </div>
                      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-6">
                        {brandSettings.logos.map((logo, idx) => (
                          <div key={idx} className={`aspect-video rounded-3xl border-2 flex items-center justify-center relative group overflow-hidden transition-all ${logo ? 'border-slate-200 bg-white' : 'border-dashed border-slate-100 bg-slate-50/50'}`}>
                            {logo ? (
                              <>
                                <img src={logo} className="max-h-full max-w-full p-4 object-contain" />
                                <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-3">
                                  <button onClick={() => setBrandSettings(prev => ({ ...prev, logoData: logo }))} className="h-10 w-10 bg-white rounded-full flex items-center justify-center text-slate-900 hover:bg-orange-500 hover:text-white transition-all shadow-lg"><i className="fa-solid fa-eye"></i></button>
                                  <button onClick={() => removeLogo(idx)} className="h-10 w-10 bg-rose-500 rounded-full flex items-center justify-center text-white hover:bg-rose-600 transition-all shadow-lg"><i className="fa-solid fa-trash-can"></i></button>
                                </div>
                              </>
                            ) : (
                              <div 
                                onClick={() => logoUploadRef.current?.click()} 
                                className="w-full h-full flex items-center justify-center cursor-pointer group/slot"
                              >
                                <i className="fa-solid fa-plus text-2xl text-slate-200 group-hover/slot:text-orange-500 transition-colors"></i>
                              </div>
                            )}
                            <div className="absolute bottom-3 left-4 text-[8px] font-black text-slate-300 uppercase tracking-widest">Slot {idx + 1}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
                {settingsTab === 'COMMAND' && (
                   <div className="space-y-10 animate-in fade-in slide-in-from-bottom-6">
                     <div className="flex justify-between items-center px-2"><h3 className="text-[13px] font-black text-slate-900 uppercase tracking-widest">Instruction Templates</h3><button onClick={addTemplate} className="text-[11px] font-black text-orange-600 uppercase border-b-2 border-orange-600">+ New Part</button></div>
                     <div className="flex bg-slate-100/50 p-1.5 rounded-[24px] gap-1 overflow-x-auto no-scrollbar shadow-sm border border-slate-100 self-start">{['GRAMMAR', 'VOCABULARY', 'READING', 'TABLES', 'KIDS'].map(cat => (<button key={cat} onClick={() => setActiveTemplateCategory(cat)} className={`px-6 py-2.5 rounded-[18px] text-[10px] font-black uppercase tracking-widest transition-all ${activeTemplateCategory === cat ? 'bg-orange-600 text-white shadow-md' : 'text-slate-400 hover:text-slate-600'}`}>{cat}</button>))}</div>
                     <div className="space-y-3">
                        {instructionTemplates.filter(t => t.category === activeTemplateCategory).map(t => {
                            const isExpanded = expandedTemplateId === t.id;
                            return (
                              <div key={t.id} className={`bg-white border rounded-[32px] overflow-hidden transition-all duration-300 ${isExpanded ? 'border-orange-200 shadow-xl' : 'border-slate-100 shadow-sm'}`}>
                                 <div className="p-6 lg:p-8 cursor-pointer flex items-center justify-between" onClick={() => setExpandedTemplateId(isExpanded ? null : t.id)}><div className="flex items-center gap-4 flex-1"><div className={`h-8 w-8 rounded-full flex items-center justify-center transition-transform ${isExpanded ? 'rotate-90 bg-orange-600 text-white' : 'bg-slate-50 text-slate-400'}`}><i className="fa-solid fa-chevron-right text-[10px]"></i></div><div className="flex flex-col gap-0.5"><div className={`text-[13px] font-black uppercase tracking-wide transition-colors ${isExpanded ? 'text-orange-600' : 'text-slate-900'}`}>{t.label}</div>{!isExpanded && <div className="text-[9px] font-black text-slate-300 uppercase line-clamp-1">{t.prompt.slice(0, 100)}...</div>}</div></div><div className="flex items-center gap-3"><div className="px-4 py-1.5 rounded-full bg-slate-100 text-slate-400 text-[8px] font-black uppercase">{t.category}</div>{isExpanded && <button onClick={() => deleteTemplate(t.id)} className="h-8 w-8 bg-rose-50 text-rose-500 rounded-full flex items-center justify-center hover:bg-rose-500 hover:text-white transition-all"><i className="fa-solid fa-trash-can text-[10px]"></i></button>}</div></div>
                                 {isExpanded && (<div className="px-8 pb-8 space-y-6 animate-in fade-in slide-in-from-top-4"><div className="h-px bg-slate-100 w-full mb-6"></div><div className="space-y-4"><label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-2">Display Name</label><input value={t.label} onChange={e => updateTemplate(t.id, { label: e.target.value })} className="w-full bg-slate-50 border border-slate-100 rounded-2xl px-6 py-4 outline-none focus:border-orange-500 font-bold text-slate-700" /></div><div className="space-y-4"><label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-2">Neural Prompt Logic</label><textarea value={t.prompt} onChange={e => updateTemplate(t.id, { prompt: e.target.value })} className="w-full h-32 bg-slate-50 border border-slate-100 rounded-3xl p-6 text-[11px] text-slate-600 font-medium italic outline-none resize-none focus:bg-white transition-all" /></div></div>)}
                              </div>
                            );
                        })}
                     </div>
                   </div>
                )}
                {settingsTab === 'ACCOUNT' && (
                  <div className="space-y-10 animate-in fade-in slide-in-from-bottom-6">
                    <h3 className="text-[13px] font-black text-slate-900 uppercase tracking-widest">Session Identity</h3>
                    <div className="bg-white border border-slate-100 rounded-[40px] p-10 flex items-center gap-8 shadow-sm">
                      <div className="h-20 w-20 bg-orange-600 rounded-full flex items-center justify-center text-white text-3xl font-black">{session.name.charAt(0)}</div>
                      <div className="space-y-1">
                        <div className="text-2xl font-black text-slate-900 uppercase">{session.name}</div>
                        <div className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Access Code: {session.code} • Active since {new Date(session.loginTime).toLocaleTimeString()}</div>
                      </div>
                    </div>
                  </div>
                )}
                {settingsTab === 'ENGINE' && (
                  <div className="space-y-10 animate-in fade-in slide-in-from-bottom-6">
                    <h3 className="text-[13px] font-black text-slate-900 uppercase tracking-widest">Neural Core Configuration</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      {[
                        { id: NeuralEngine.GEMINI_3_FLASH, name: 'Gemini 3 Flash', desc: 'High-speed, balanced reasoning.' },
                        { id: NeuralEngine.GEMINI_3_PRO, name: 'Gemini 3 Pro', desc: 'Maximum intelligence for complex tests.' },
                        { id: NeuralEngine.GPT_4O, name: 'GPT-4o', desc: 'Advanced multimodal capabilities.' },
                        { id: NeuralEngine.GROK_3, name: 'Grok 3', desc: 'Real-time knowledge and reasoning.' },
                        { id: NeuralEngine.DEEPSEEK_V3, name: 'DeepSeek V3', desc: 'Efficient large-scale processing.' }
                      ].map(engine => (
                        <div key={engine.id} className={`p-8 rounded-[40px] border-2 transition-all ${activeEngine === engine.id ? 'bg-white border-orange-600 shadow-xl' : 'bg-slate-50 border-slate-100'}`}>
                          <div className="flex justify-between items-start mb-6">
                            <div className="space-y-1">
                              <div className="text-[14px] font-black text-slate-900 uppercase">{engine.name}</div>
                              <div className="text-[10px] font-medium text-slate-400">{engine.desc}</div>
                            </div>
                            {activeEngine === engine.id && <div className="h-6 w-6 bg-orange-600 rounded-full flex items-center justify-center text-white text-[10px]"><i className="fa-solid fa-check"></i></div>}
                          </div>
                          <div className="space-y-4">
                            <input 
                              type="password"
                              value={externalKeys[engine.id as keyof ExternalKeys] || ''} 
                              onChange={e => setExternalKeys({ ...externalKeys, [engine.id]: e.target.value })}
                              placeholder="Custom API Key (Optional)" 
                              className="w-full bg-white border border-slate-200 rounded-2xl px-6 py-3 text-[11px] outline-none focus:border-orange-500"
                            />
                            <button 
                              onClick={() => setActiveEngine(engine.id as NeuralEngine)}
                              className={`w-full py-3 rounded-2xl text-[10px] font-black uppercase tracking-widest transition-all ${activeEngine === engine.id ? 'bg-orange-600 text-white shadow-lg' : 'bg-white border border-slate-200 text-slate-400 hover:text-slate-900'}`}
                            >
                              {activeEngine === engine.id ? 'Currently Active' : 'Switch Engine'}
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {settingsTab === 'DISPLAY' && (
                  <div className="space-y-10 animate-in fade-in slide-in-from-bottom-6">
                    <h3 className="text-[13px] font-black text-slate-900 uppercase tracking-widest">Typography & Layout</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
                      <div className="space-y-6">
                        <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Font Size (px)</label>
                        <div className="flex items-center gap-4">
                          <input type="range" min="8" max="24" value={brandSettings.fontSize} onChange={e => setBrandSettings({ ...brandSettings, fontSize: parseInt(e.target.value) })} className="flex-1 accent-orange-600" />
                          <span className="text-xl font-black text-slate-900 w-12">{brandSettings.fontSize}</span>
                        </div>
                      </div>
                      <div className="space-y-6">
                        <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Font Weight</label>
                        <div className="flex bg-slate-100 p-1.5 rounded-2xl gap-1">
                          {['400', '500', '600', '700', '800', '900'].map(weight => (
                            <button key={weight} onClick={() => setBrandSettings({ ...brandSettings, fontWeight: weight })} className={`flex-1 py-2 rounded-xl text-[10px] font-black transition-all ${brandSettings.fontWeight === weight ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-400'}`}>{weight}</button>
                          ))}
                        </div>
                      </div>
                      <div className="space-y-6">
                        <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Letter Spacing</label>
                        <div className="flex items-center gap-4">
                          <input type="range" min="-2" max="10" step="0.5" value={brandSettings.letterSpacing} onChange={e => setBrandSettings({ ...brandSettings, letterSpacing: parseFloat(e.target.value) })} className="flex-1 accent-orange-600" />
                          <span className="text-xl font-black text-slate-900 w-12">{brandSettings.letterSpacing}</span>
                        </div>
                      </div>
                      <div className="space-y-6">
                        <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Text Transform</label>
                        <div className="flex bg-slate-100 p-1.5 rounded-2xl gap-1">
                          {['none', 'uppercase', 'capitalize'].map(transform => (
                            <button key={transform} onClick={() => setBrandSettings({ ...brandSettings, textTransform: transform as any })} className={`flex-1 py-2 rounded-xl text-[10px] font-black uppercase transition-all ${brandSettings.textTransform === transform ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-400'}`}>{transform}</button>
                          ))}
                        </div>
                      </div>
                    </div>
                    <div className="space-y-6">
                      <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Worksheet Theme</label>
                      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
                        {THEMES.map(theme => (
                          <button key={theme.id} onClick={() => setActiveThemeId(theme.id)} className={`p-4 rounded-2xl border-2 transition-all text-left space-y-2 ${activeThemeId === theme.id ? 'border-orange-600 bg-white shadow-lg' : 'border-slate-100 bg-slate-50 hover:border-slate-200'}`}>
                            <div className="w-full h-2 rounded-full" style={{ backgroundColor: theme.color }}></div>
                            <div className="text-[10px] font-black uppercase text-slate-900 truncate">{theme.name}</div>
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
                {settingsTab === 'BACKBONE LOGIC' && (
                  <div className="space-y-12 animate-in fade-in slide-in-from-bottom-6 duration-700">
                    <div className="space-y-8">
                       <div className="flex justify-between items-center px-2">
                         <h3 className="text-[13px] font-black text-master-green uppercase tracking-widest">Master Protocols</h3>
                         {!(session?.code === 'dpss' || session?.code === 'gratitude') && (
                           <div className="flex items-center gap-2 text-rose-500 animate-pulse">
                             <i className="fa-solid fa-lock text-[10px]"></i>
                             <span className="text-[10px] font-black uppercase tracking-widest">Restricted Access</span>
                           </div>
                         )}
                         {(session?.code === 'dpss' || session?.code === 'gratitude') && (
                           <button onClick={addProtocol} className="text-[11px] font-black text-master-green uppercase border-b-2 border-master-green">+ New Protocol</button>
                         )}
                       </div>
                       {(session?.code === 'dpss' || session?.code === 'gratitude') ? (
                         <>
                           <div className="flex bg-slate-100/50 p-1.5 rounded-[24px] gap-1 overflow-x-auto no-scrollbar shadow-sm border border-slate-100 self-start">
                             {['General', 'Grammar', 'Vocabulary', 'Reading'].map(cat => (
                               <button key={cat} onClick={() => setActiveProtocolCategory(cat as RuleCategory)} className={`px-6 py-2.5 rounded-[18px] text-[10px] font-black uppercase tracking-widest transition-all ${activeProtocolCategory === cat ? 'bg-master-green text-white shadow-md' : 'text-slate-400 hover:text-slate-600'}`}>{cat}</button>
                             ))}
                           </div>
                           <div className="space-y-3">
                             {masterProtocols.filter(p => p.category === activeProtocolCategory).map(p => {
                               const isExpanded = expandedProtocolId === p.id;
                               return (
                                 <div key={p.id} className={`bg-white border rounded-[32px] overflow-hidden transition-all duration-300 ${isExpanded ? 'border-master-green/30 shadow-xl' : 'border-slate-100 shadow-sm'}`}>
                                   <div className="p-6 lg:p-8 cursor-pointer flex items-center justify-between" onClick={() => setExpandedProtocolId(isExpanded ? null : p.id)}>
                                     <div className="flex items-center gap-4 flex-1">
                                       <div className={`h-8 w-8 rounded-full flex items-center justify-center transition-transform ${isExpanded ? 'rotate-90 bg-master-green text-white' : 'bg-slate-50 text-slate-400'}`}>
                                         <i className="fa-solid fa-chevron-right text-[10px]"></i>
                                       </div>
                                       <div className="flex flex-col gap-0.5">
                                         <div className={`text-[13px] font-black uppercase tracking-wide transition-colors ${isExpanded ? 'text-master-green' : 'text-slate-900'}`}>{p.label}</div>
                                         {!isExpanded && <div className="text-[9px] font-black text-slate-300 uppercase line-clamp-1">{p.promptInjection.slice(0, 100)}...</div>}
                                       </div>
                                     </div>
                                     <div className="flex items-center gap-3">
                                       <button 
                                         onClick={(e) => { e.stopPropagation(); updateProtocol(p.id, { priority: cyclePriority(p.priority) }); }}
                                         className={`px-3 py-1 rounded-full text-[8px] font-black uppercase transition-all hover:scale-105 ${p.priority === 'High' ? 'bg-rose-100 text-rose-600' : p.priority === 'Medium' ? 'bg-orange-100 text-orange-600' : p.priority === 'Average' ? 'bg-blue-100 text-blue-600' : 'bg-slate-100 text-slate-400'}`}
                                       >
                                         {p.priority}
                                       </button>
                                       <button 
                                         onClick={(e) => { e.stopPropagation(); updateProtocol(p.id, { active: !p.active }); }} 
                                         className={`px-4 py-1.5 rounded-full text-[8px] font-black uppercase transition-all ${p.active ? 'bg-master-green/10 text-master-green' : 'bg-slate-100 text-slate-400'}`}
                                       >
                                         {p.active ? 'Active' : 'Disabled'}
                                       </button>
                                       {isExpanded && <button onClick={(e) => { e.stopPropagation(); deleteProtocol(p.id); }} className="h-8 w-8 bg-rose-50 text-rose-500 rounded-full flex items-center justify-center hover:bg-rose-500 hover:text-white transition-all"><i className="fa-solid fa-trash-can text-[10px]"></i></button>}
                                     </div>
                                   </div>
                                   {isExpanded && (
                                     <div className="px-8 pb-8 space-y-6 animate-in fade-in slide-in-from-top-4">
                                       <div className="h-px bg-slate-100 w-full mb-6"></div>
                                       <div className="grid grid-cols-2 gap-4">
                                          <div className="space-y-4">
                                            <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-2">Protocol Name</label>
                                            <input value={p.label} onChange={e => updateProtocol(p.id, { label: e.target.value })} className="w-full bg-slate-50 border border-slate-100 rounded-2xl px-6 py-4 outline-none focus:border-master-green font-bold text-slate-700" />
                                          </div>
                                          <div className="space-y-4">
                                            <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-2">Priority Level</label>
                                            <button 
                                              onClick={() => updateProtocol(p.id, { priority: cyclePriority(p.priority) })}
                                              className="w-full bg-slate-50 border border-slate-100 rounded-2xl px-6 py-4 outline-none hover:border-master-green font-bold text-slate-700 uppercase text-left flex justify-between items-center"
                                            >
                                              <span>{p.priority}</span>
                                              <i className="fa-solid fa-rotate text-[10px] text-slate-300"></i>
                                            </button>
                                          </div>
                                       </div>
                                       <div className="space-y-4">
                                         <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-2">Protocol Logic</label>
                                         <textarea value={p.promptInjection} onChange={e => updateProtocol(p.id, { promptInjection: e.target.value })} className="w-full h-32 bg-slate-50 border border-slate-100 rounded-3xl p-6 text-[11px] text-slate-600 font-medium italic outline-none resize-none focus:bg-white transition-all" />
                                       </div>
                                     </div>
                                   )}
                                 </div>
                               );
                             })}
                           </div>
                         </>
                       ) : (
                         <div className="p-12 border-2 border-dashed border-slate-100 rounded-[40px] flex flex-col items-center justify-center gap-4 bg-slate-50/50">
                           <i className="fa-solid fa-shield-halved text-slate-200 text-4xl"></i>
                           <div className="text-[11px] font-black text-slate-400 uppercase tracking-widest">Neural Protocols Encrypted</div>
                           <div className="text-[9px] font-medium text-slate-400 text-center max-w-[200px]">Please authenticate with a Master Architect code to modify core protocols.</div>
                         </div>
                       )}
                    </div>
                     <div className="space-y-8">
                        <div className="flex justify-between items-center px-2">
                          <h3 className="text-[13px] font-black text-strict-purple uppercase tracking-widest">Logic Node Registry</h3>
                          <button onClick={addRule} className="text-[11px] font-black text-strict-purple uppercase border-b-2 border-strict-purple">+ New Logic Node</button>
                        </div>
                        <div className="flex bg-slate-100/50 p-1.5 rounded-[24px] gap-1 overflow-x-auto no-scrollbar shadow-sm border border-slate-100 self-start">
                          {['General', 'Grammar', 'Vocabulary', 'Reading'].map(cat => (
                            <button key={cat} onClick={() => setActiveLogicCategory(cat as RuleCategory)} className={`px-6 py-2.5 rounded-[18px] text-[10px] font-black uppercase tracking-widest transition-all ${activeLogicCategory === cat ? 'bg-strict-purple text-white shadow-md' : 'text-slate-400 hover:text-slate-600'}`}>{cat}</button>
                          ))}
                        </div>
                        <div className="space-y-3">
                          {strictRules.filter(rule => rule.category === activeLogicCategory).map(rule => {
                            const isExpanded = expandedRuleId === rule.id;
                            return (
                              <div key={rule.id} className={`bg-white border rounded-[32px] overflow-hidden transition-all duration-300 ${isExpanded ? 'border-strict-purple/30 shadow-xl' : 'border-slate-100 shadow-sm'}`}>
                                <div className="p-6 lg:p-8 cursor-pointer flex items-center justify-between" onClick={() => setExpandedRuleId(isExpanded ? null : rule.id)}>
                                  <div className="flex items-center gap-4 flex-1">
                                    <div className={`h-8 w-8 rounded-full flex items-center justify-center transition-transform ${isExpanded ? 'rotate-90 bg-strict-purple text-white' : 'bg-slate-50 text-slate-400'}`}>
                                      <i className="fa-solid fa-chevron-right text-[10px]"></i>
                                    </div>
                                    <div className="flex flex-col gap-0.5">
                                      <div className={`text-[13px] font-black uppercase tracking-wide transition-colors ${isExpanded ? 'text-strict-purple' : 'text-slate-900'}`}>{rule.label}</div>
                                      {!isExpanded && <div className="text-[9px] font-black text-slate-300 uppercase line-clamp-1">{rule.promptInjection.slice(0, 100)}...</div>}
                                    </div>
                                  </div>
                                  <div className="flex items-center gap-3">
                                    <button 
                                      onClick={(e) => { e.stopPropagation(); updateRule(rule.id, { priority: cyclePriority(rule.priority) }); }}
                                      className={`px-3 py-1 rounded-full text-[8px] font-black uppercase transition-all hover:scale-105 ${rule.priority === 'High' ? 'bg-rose-100 text-rose-600' : rule.priority === 'Medium' ? 'bg-orange-100 text-orange-600' : rule.priority === 'Average' ? 'bg-blue-100 text-blue-600' : 'bg-slate-100 text-slate-400'}`}
                                    >
                                      {rule.priority}
                                    </button>
                                    <button 
                                      onClick={(e) => { e.stopPropagation(); updateRule(rule.id, { active: !rule.active }); }} 
                                      className={`px-4 py-1.5 rounded-full text-[8px] font-black uppercase transition-all ${rule.active ? 'bg-strict-purple/10 text-strict-purple' : 'bg-slate-100 text-slate-400'}`}
                                    >
                                      {rule.active ? 'Active' : 'Disabled'}
                                    </button>
                                    {isExpanded && <button onClick={(e) => { e.stopPropagation(); deleteRule(rule.id); }} className="h-8 w-8 bg-rose-50 text-rose-500 rounded-full flex items-center justify-center hover:bg-rose-500 hover:text-white transition-all"><i className="fa-solid fa-trash-can text-[10px]"></i></button>}
                                  </div>
                                </div>
                                {isExpanded && (
                                  <div className="px-8 pb-8 space-y-6 animate-in fade-in slide-in-from-top-4">
                                    <div className="h-px bg-slate-100 w-full mb-6"></div>
                                    <div className="grid grid-cols-2 gap-4">
                                       <div className="space-y-4">
                                         <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-2">Logic Name</label>
                                         <input value={rule.label} onChange={e => updateRule(rule.id, { label: e.target.value })} className="w-full bg-slate-50 border border-slate-100 rounded-2xl px-6 py-4 outline-none focus:border-strict-purple font-bold text-slate-700" />
                                       </div>
                                       <div className="space-y-4">
                                         <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-2">Priority Level</label>
                                         <button 
                                           onClick={() => updateRule(rule.id, { priority: cyclePriority(rule.priority) })}
                                           className="w-full bg-slate-50 border border-slate-100 rounded-2xl px-6 py-4 outline-none hover:border-strict-purple font-bold text-slate-700 uppercase text-left flex justify-between items-center"
                                         >
                                           <span>{rule.priority}</span>
                                           <i className="fa-solid fa-rotate text-[10px] text-slate-300"></i>
                                         </button>
                                       </div>
                                    </div>
                                    <div className="space-y-4">
                                      <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-2">Prompt Injection</label>
                                      <textarea value={rule.promptInjection} onChange={e => updateRule(rule.id, { promptInjection: e.target.value })} className="w-full h-32 bg-slate-50 border border-slate-100 rounded-3xl p-6 text-[11px] text-slate-600 font-medium italic outline-none resize-none focus:bg-white transition-all" />
                                    </div>
                                  </div>
                                )}
                              </div>
                            );
                          })}
                        </div>
                     </div>
                  </div>
                )}
             </div>
              <div className="p-12 bg-slate-50 border-t border-slate-100 flex justify-end gap-4">
                <button onClick={handleLogout} className="px-16 py-6 bg-white border border-slate-200 text-rose-500 rounded-full text-[12px] font-black uppercase shadow-sm hover:bg-rose-50 transition-all">Terminate Architecture</button>
                <button onClick={hardReset} className="px-16 py-6 bg-rose-600 text-white rounded-full text-[12px] font-black uppercase shadow-xl hover:bg-rose-700 transition-all">Hard Reset</button>
                <button onClick={syncWithDefaults} className="px-16 py-6 bg-slate-900 text-white rounded-full text-[12px] font-black uppercase shadow-xl hover:bg-black transition-all">Sync Settings</button>
                <button onClick={() => setShowSettings(false)} className="px-16 py-6 bg-gradient-to-r from-accent-orange-dark to-accent-orange-light text-white rounded-full text-[12px] font-black uppercase shadow-xl hover:brightness-110 transition-all">Close Panel</button>
              </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
