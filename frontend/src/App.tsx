import { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronLeft, ChevronRight, Settings2, Sparkles, Send, Bot, Filter, ArrowDownUp } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { ContextMenu, ContextMenuContent, ContextMenuItem, ContextMenuTrigger } from '@/components/ui/context-menu';

import rawEntries from './entries.json';

// Persona mappings & Chameleon Colors
const PERSONAS: Record<string, { name: string, color: string, glow: string }> = {
  van_gogh: { name: 'Vincent Van Gogh', color: 'text-yellow-400/80', glow: 'shadow-yellow-400/10' },
  mansfield: { name: 'Katherine Mansfield', color: 'text-emerald-400/80', glow: 'shadow-emerald-400/10' },
  pepys: { name: 'Samuel Pepys', color: 'text-blue-400/80', glow: 'shadow-blue-400/10' },
  maclane: { name: 'Mary MacLane', color: 'text-rose-400/80', glow: 'shadow-rose-400/10' }
};

export default function App() {
  const [isLeftOpen, setIsLeftOpen] = useState(true);
  const [isRightOpen, setIsRightOpen] = useState(true);
  
  const [currentIndex, setCurrentIndex] = useState(100); 
  const [activeFilter, setActiveFilter] = useState<string | null>(null);
  const [activeVector, setActiveVector] = useState<string | null>(null);
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('asc');

  const [chatHistory, setChatHistory] = useState<{role: string, content: string}[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);

  const entries = useMemo(() => {
    let filtered = activeFilter ? rawEntries.filter((e: any) => e.persona === activeFilter) : [...rawEntries];
    if (sortOrder === 'desc') {
      filtered.reverse();
    }
    return filtered;
  }, [activeFilter, sortOrder]);

  const currentEntry = entries[currentIndex] || entries[0];
  const personaData = PERSONAS[currentEntry?.persona] || PERSONAS['maclane'];
  const activeColor = personaData?.color || 'text-purple-400/80';
  const activeGlow = personaData?.glow || 'shadow-purple-400/10';

  const chatPersona = activeVector ? PERSONAS[activeVector] : personaData;
  const chatColor = chatPersona?.color || activeColor;
  const chatGlow = chatPersona?.glow || activeGlow;

  const handleNext = () => setCurrentIndex(i => Math.min(i + 1, entries.length - 1));
  const handlePrev = () => setCurrentIndex(i => Math.max(i - 1, 0));

  return (
    <TooltipProvider>
      <div className="h-screen w-full bg-background flex overflow-hidden text-foreground">
        
        {/* Subtle ambient background tied to persona */}
        <div className={`absolute inset-0 pointer-events-none z-0 transition-colors duration-1000 ${activeGlow.replace('shadow', 'bg')}`} style={{
          background: 'radial-gradient(circle at 50% 50%, rgba(255,255,255,0.02) 0%, transparent 60%)'
        }} />

        {/* LEFT SIDEBAR: Timeline & Filters */}
        <AnimatePresence initial={false}>
          {isLeftOpen && (
            <motion.aside
              initial={{ width: 0, opacity: 0 }}
              animate={{ width: 320, opacity: 1 }}
              exit={{ width: 0, opacity: 0 }}
              transition={{ duration: 0.3, ease: 'easeInOut' }}
              className="h-full border-r border-white/5 bg-background/95 backdrop-blur-xl z-20 flex flex-col shrink-0 overflow-hidden relative group"
            >
              <div className="p-6 border-b border-white/5 flex items-center justify-between">
                <h2 className="font-serif text-2xl tracking-wide flex items-center gap-3">
                  <Settings2 className="w-5 h-5 opacity-50" />
                  Chronology
                </h2>
                <Button 
                  variant="ghost" 
                  size="icon" 
                  className="h-8 w-8 rounded-full"
                  onClick={() => { setSortOrder(prev => prev === 'asc' ? 'desc' : 'asc'); setCurrentIndex(0); }}
                >
                  <ArrowDownUp className="w-4 h-4 opacity-50" />
                </Button>
              </div>
              
              <div className="p-4 border-b border-white/5">
                <div className="flex items-center gap-2 mb-3 px-2">
                  <Filter className="w-4 h-4 opacity-50" />
                  <span className="text-sm font-sans uppercase tracking-widest text-muted-foreground">Filter Persona</span>
                </div>
                <div className="flex flex-wrap gap-2 px-2">
                  <Button 
                    variant={activeFilter === null ? 'secondary' : 'ghost'} 
                    size="sm" 
                    onClick={() => { setActiveFilter(null); setCurrentIndex(0); }}
                    className="text-xs h-7 rounded-full"
                  >
                    All
                  </Button>
                  {Object.entries(PERSONAS).map(([key, data]) => (
                    <Button 
                      key={key}
                      variant={activeFilter === key ? 'secondary' : 'ghost'} 
                      size="sm" 
                      onClick={() => { setActiveFilter(key); setCurrentIndex(0); }}
                      className={`text-xs h-7 rounded-full ${activeFilter === key ? data.color : ''}`}
                    >
                      {data.name.split(' ').pop()}
                    </Button>
                  ))}
                </div>
              </div>

              <div className="flex-1 overflow-y-auto p-4 space-y-2 relative no-scrollbar">
                {entries.slice(Math.max(0, currentIndex - 20), currentIndex + 20).map((entry: any, i: number) => {
                  const actualIndex = Math.max(0, currentIndex - 20) + i;
                  const isActive = actualIndex === currentIndex;
                  return (
                    <button
                      key={actualIndex}
                      onClick={() => setCurrentIndex(actualIndex)}
                      className={`w-full text-left p-3 rounded-lg transition-all duration-200 ${
                        isActive 
                          ? 'bg-white/10 border border-white/10' 
                          : 'hover:bg-white/5 border border-transparent'
                      }`}
                    >
                      <div className={`text-xs font-sans tracking-widest uppercase mb-1 ${isActive ? activeColor : 'text-primary/50'}`}>
                        {entry.date || 'Undated'}
                      </div>
                      <div className={`text-sm font-serif line-clamp-2 ${isActive ? 'text-white' : 'text-muted-foreground'}`}>
                        {entry.text}
                      </div>
                    </button>
                  );
                })}
              </div>
              
              <button 
                onClick={() => setIsLeftOpen(false)}
                className="absolute right-0 top-0 bottom-0 w-8 bg-gradient-to-l from-black/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center cursor-ew-resize"
              >
                <div className="w-1 h-8 bg-white/20 rounded-full" />
              </button>
            </motion.aside>
          )}
        </AnimatePresence>

        <AnimatePresence>
          {!isLeftOpen && (
            <motion.button
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setIsLeftOpen(true)}
              className="absolute left-0 top-1/2 -translate-y-1/2 h-24 w-6 bg-white/5 hover:bg-white/10 border border-white/10 rounded-r-xl z-30 flex items-center justify-center backdrop-blur"
            >
              <ChevronRight className="w-4 h-4 opacity-50" />
            </motion.button>
          )}
        </AnimatePresence>

        {/* CENTER PANE: Reading Area */}
        <main className="flex-1 h-full flex flex-col relative z-10">
          
          <div className="flex-1 overflow-y-auto px-12 py-24 flex flex-col items-center">
            
            <div className="w-full max-w-2xl mb-12 flex justify-center">
              <h1 className="text-2xl font-serif tracking-widest text-primary/80 uppercase">Ghostwriter</h1>
            </div>

            <ContextMenu>
              <ContextMenuTrigger className="w-full max-w-2xl">
                <AnimatePresence mode="wait">
                  <motion.article 
                    key={currentIndex}
                    initial={{ opacity: 0, scale: 0.98, filter: 'blur(4px)' }}
                    animate={{ opacity: 1, scale: 1, filter: 'blur(0px)' }}
                    exit={{ opacity: 0, scale: 1.02, filter: 'blur(4px)' }}
                    transition={{ duration: 0.5, ease: "easeOut" }}
                    className={`w-full p-8 rounded-2xl border border-white/5 bg-white/[0.02] ${activeGlow} shadow-2xl transition-all duration-700`}
                  >
                    <header className="flex items-center justify-between mb-8 pb-6 border-b border-white/5">
                      <div className="flex flex-col gap-1">
                        <span className={`text-sm tracking-[0.2em] uppercase font-sans ${activeColor}`}>
                          {currentEntry?.date || 'Undated'}
                        </span>
                        <span className="text-xs text-primary/40 font-serif italic">
                          {personaData?.name || 'Blended Voice'}
                        </span>
                      </div>
                      <div className="flex items-center gap-4">
                        <Popover>
                          <PopoverTrigger asChild>
                            <Button variant="ghost" size="icon" className="h-8 w-8 rounded-full hover:bg-white/5">
                              <Sparkles className={`w-4 h-4 ${activeColor}`} />
                            </Button>
                          </PopoverTrigger>
                          <PopoverContent className="w-64 p-4 bg-card/95 backdrop-blur border-white/10" sideOffset={12}>
                            <h4 className="font-serif text-lg mb-4 text-white/90">Resonance Vector</h4>
                            <div className="space-y-3 text-sm font-sans text-muted-foreground">
                              {Object.entries(PERSONAS).map(([key, data]) => {
                                const isActive = key === (activeVector || currentEntry?.persona);
                                const pct = isActive ? 85 : 5;
                                return (
                                  <div key={key}>
                                    <div className="flex justify-between items-center mb-1">
                                      <span className={isActive ? "text-white" : ""}>{data.name}</span>
                                      <span className={isActive ? data.color : ""}>{pct}%</span>
                                    </div>
                                    <div className="h-1 w-full bg-white/5 rounded-full overflow-hidden">
                                      <motion.div 
                                        initial={{ width: 0 }}
                                        animate={{ width: `${pct}%` }}
                                        className={`h-full ${isActive ? data.color.replace('text-', 'bg-').replace('/80', '') : 'bg-white/10'}`} 
                                      />
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                          </PopoverContent>
                        </Popover>
                      </div>
                    </header>



                    <div className="prose prose-invert prose-p:font-serif prose-p:text-xl prose-p:leading-[1.8] prose-p:text-foreground/90 whitespace-pre-wrap">
                      {currentEntry?.text}
                    </div>
                  </motion.article>
                </AnimatePresence>
              </ContextMenuTrigger>
              <ContextMenuContent className="w-56 bg-card/95 backdrop-blur border-white/10">
                {Object.values(PERSONAS).map((data) => (
                  <ContextMenuItem key={data.name} className="font-serif py-2 cursor-pointer">
                    Regenerate as {data.name.split(' ').pop()}
                  </ContextMenuItem>
                ))}
              </ContextMenuContent>
            </ContextMenu>
            
          </div>

          <footer className="h-24 shrink-0 flex items-center justify-center gap-24 border-t border-white/5 bg-background/50 backdrop-blur">
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="ghost" size="icon" onClick={handlePrev} disabled={currentIndex === 0} className="rounded-full h-12 w-12 hover:bg-white/5 disabled:opacity-20">
                  <ChevronLeft className="w-6 h-6 opacity-60" />
                </Button>
              </TooltipTrigger>
              <TooltipContent side="top">Previous Entry</TooltipContent>
            </Tooltip>

            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="ghost" size="icon" onClick={handleNext} disabled={currentIndex === entries.length - 1} className="rounded-full h-12 w-12 hover:bg-white/5 disabled:opacity-20">
                  <ChevronRight className="w-6 h-6 opacity-60" />
                </Button>
              </TooltipTrigger>
              <TooltipContent side="top">Next Entry</TooltipContent>
            </Tooltip>
          </footer>
        </main>

        <AnimatePresence>
          {!isRightOpen && (
            <motion.button
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setIsRightOpen(true)}
              className="absolute right-0 top-1/2 -translate-y-1/2 h-24 w-6 bg-white/5 hover:bg-white/10 border border-white/10 rounded-l-xl z-30 flex items-center justify-center backdrop-blur"
            >
              <ChevronLeft className="w-4 h-4 opacity-50" />
            </motion.button>
          )}
        </AnimatePresence>

        {/* RIGHT SIDEBAR: Chat / Séance */}
        <AnimatePresence initial={false}>
          {isRightOpen && (
            <motion.aside
              initial={{ width: 0, opacity: 0 }}
              animate={{ width: 380, opacity: 1 }}
              exit={{ width: 0, opacity: 0 }}
              transition={{ duration: 0.3, ease: 'easeInOut' }}
              className="h-full border-l border-white/5 bg-background/95 backdrop-blur-xl z-20 flex flex-col shrink-0 relative group"
            >
              <button 
                onClick={() => setIsRightOpen(false)}
                className="absolute left-0 top-0 bottom-0 w-8 bg-gradient-to-r from-black/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center cursor-ew-resize z-10"
              >
                <div className="w-1 h-8 bg-white/20 rounded-full" />
              </button>

              <div className="p-6 border-b border-white/5 flex flex-col gap-4">
                <h2 className="font-serif text-2xl tracking-wide font-normal flex items-center gap-3">
                  <Bot className={`w-5 h-5 ${chatColor}`} />
                  The Séance
                </h2>
                
                <div className="flex bg-white/5 rounded-full p-1 border border-white/10 self-start">
                  <button 
                    onClick={() => setActiveVector(null)} 
                    className={`px-3 py-1 rounded-full text-[10px] font-sans tracking-widest uppercase transition-colors ${activeVector === null ? 'bg-white/20 text-white' : 'text-white/40 hover:text-white/80'}`}
                  >
                    Auto
                  </button>
                  {Object.entries(PERSONAS).map(([key, data]) => (
                    <button 
                      key={key} 
                      onClick={() => setActiveVector(key)} 
                      className={`px-3 py-1 rounded-full text-[10px] font-sans tracking-widest uppercase transition-colors ${activeVector === key ? data.color.replace('text-', 'bg-').replace('/80', '/20') + ' ' + data.color : 'text-white/40 hover:text-white/80'}`}
                    >
                      {data.name.split(' ').pop()}
                    </button>
                  ))}
                </div>
              </div>
              
              <div className="flex-1 p-6 overflow-y-auto space-y-6">
                <div className={`border p-4 rounded-lg rounded-tl-none bg-white/[0.02] ${chatGlow} border-white/5`}>
                  <p className="font-serif text-sm text-foreground/80 leading-relaxed">
                    I am the resonance of {chatPersona?.name || 'the spirits'}. Ask me of the thoughts penned on this day, or what shadows danced at the edge of my vision...
                  </p>
                </div>
                
                {chatHistory.map((msg, idx) => (
                  <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-[85%] p-4 rounded-lg border ${
                      msg.role === 'user' 
                        ? 'bg-white/10 border-white/10 rounded-tr-none' 
                        : `bg-white/[0.02] ${chatGlow} border-white/5 rounded-tl-none`
                    }`}>
                      <p className={`text-sm leading-relaxed ${msg.role === 'user' ? 'font-sans' : 'font-serif'}`}>
                        {msg.content}
                      </p>
                    </div>
                  </div>
                ))}
                
                {isTyping && (
                  <div className="flex justify-start">
                    <div className={`border p-4 rounded-lg rounded-tl-none bg-white/[0.02] ${chatGlow} border-white/5 flex items-center gap-2`}>
                      <div className="w-2 h-2 rounded-full bg-white/30 animate-pulse" />
                      <div className="w-2 h-2 rounded-full bg-white/30 animate-pulse delay-75" />
                      <div className="w-2 h-2 rounded-full bg-white/30 animate-pulse delay-150" />
                    </div>
                  </div>
                )}
              </div>

              <div className="p-4 border-t border-white/5 bg-background">
                <form 
                  className="relative"
                  onSubmit={async (e) => {
                    e.preventDefault();
                    if (!chatInput.trim() || isTyping) return;
                    
                    const userMsg = chatInput;
                    setChatInput("");
                    setChatHistory(prev => [...prev, { role: 'user', content: userMsg }]);
                    setIsTyping(true);
                    
                    try {
                      // Determine which persona to hit. If activeVector is set, use it.
                      // Otherwise, use the current entry's persona. 
                      // If the entry is "Blended Voice" (no persona), fallback to Van Gogh or someone.
                      const targetPersona = activeVector || currentEntry?.persona || 'van_gogh';
                      
                      const res = await fetch("http://localhost:8000/api/chat", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                          persona: targetPersona,
                          message: userMsg,
                          history: chatHistory
                        })
                      });
                      
                      const data = await res.json();
                      setChatHistory(prev => [...prev, { role: 'assistant', content: data.response }]);
                    } catch (err) {
                      console.error("Error communicating with spirits:", err);
                      setChatHistory(prev => [...prev, { role: 'assistant', content: "The connection to the other side was lost..." }]);
                    } finally {
                      setIsTyping(false);
                    }
                  }}
                >
                  <input 
                    type="text" 
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    placeholder="Whisper into the void..." 
                    className="w-full bg-white/5 border border-white/10 rounded-full py-3 pl-5 pr-12 text-sm font-sans text-white placeholder:text-white/30 focus:outline-none focus:ring-1 focus:ring-white/20"
                  />
                  <Button type="submit" disabled={isTyping || !chatInput.trim()} size="icon" variant="ghost" className="absolute right-1 top-1 h-9 w-9 rounded-full hover:bg-white/10">
                    <Send className={`w-4 h-4 ${chatColor}`} />
                  </Button>
                </form>
              </div>
            </motion.aside>
          )}
        </AnimatePresence>

      </div>
    </TooltipProvider>
  );
}
