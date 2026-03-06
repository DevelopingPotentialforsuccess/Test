import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Sparkles, Layout, Type, Zap, ChevronRight, X } from 'lucide-react';

interface OnboardingTutorialProps { onComplete: () => void; }

const steps = [
  { title: "Welcome", description: "Craft perfect assessments in seconds.", icon: <Sparkles className="w-8 h-8 text-orange-500" /> },
  { title: "Modules", description: "Select specialized AI protocols.", icon: <Layout className="w-8 h-8 text-blue-500" /> },
  { title: "Synthesize", description: "Hit the button to build your test.", icon: <Zap className="w-8 h-8 text-yellow-500" /> }
];

export const OnboardingTutorial: React.FC<OnboardingTutorialProps> = ({ onComplete }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const handleNext = () => currentStep < steps.length - 1 ? setCurrentStep(currentStep + 1) : onComplete();

  return (
    <div className="fixed inset-0 z-[1000] flex items-center justify-center bg-slate-950/60 backdrop-blur-sm">
      <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} className="bg-white rounded-[40px] p-12 max-w-lg w-full">
        <div className="flex justify-between mb-8">{steps[currentStep].icon}<button onClick={onComplete}><X /></button></div>
        <h2 className="text-2xl font-black uppercase mb-4">{steps[currentStep].title}</h2>
        <p className="text-slate-500 text-lg mb-12">{steps[currentStep].description}</p>
        <button onClick={handleNext} className="w-full bg-orange-600 text-white py-4 rounded-2xl font-black uppercase">
          {currentStep === steps.length - 1 ? 'Get Started' : 'Next'}
        </button>
      </motion.div>
    </div>
  );
};
