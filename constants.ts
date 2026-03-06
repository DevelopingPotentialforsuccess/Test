
import { Theme, StrictRule, AcademicLevel, InstructionTemplate } from './types';

export const INITIAL_MODULES = ['Grammar', 'Reading', 'Vocabulary'];

export const LANGUAGES = ['English', 'Khmer', 'Chinese', 'Korean', 'French'];

export const ACADEMIC_LEVELS: AcademicLevel[] = [
  'Kid' as any, 'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 
  'Level 6', 'Level 7', 'Level 8', 'Level 9', 'Level 10', 'Level 11', 'TOEFL' as any, 'IELTS' as any
];

export const THEMES: Theme[] = [
  { id: 'default', name: 'Academic Classic', color: '#ea580c', bg: '#ffffff', accent: '#f97316' },
  { id: 'modern', name: 'Modern Professional', color: '#0f172a', bg: '#ffffff', accent: '#334155' },
  { id: 'royal', name: 'Royal Blueprint', color: '#1e3a8a', bg: '#f8fafc', accent: '#3b82f6' },
  { id: 'forest', name: 'Forest Scholar', color: '#064e3b', bg: '#f0fdf4', accent: '#10b981' },
  { id: 'crimson', name: 'Crimson Archive', color: '#7f1d1d', bg: '#fef2f2', accent: '#ef4444' },
  { id: 'midnight', name: 'Midnight Architect', color: '#1e293b', bg: '#0f172a', accent: '#6366f1' },
  { id: 'beach', name: 'Tropical Beach', color: '#0284c7', bg: 'linear-gradient(to bottom, #bae6fd, #fef3c7)', accent: '#0ea5e9' },
  { id: 'sunset', name: 'Sunset Horizon', color: '#9d174d', bg: 'linear-gradient(to top right, #fdf2f8, #fff7ed)', accent: '#db2777' },
  { id: 'nebula', name: 'Deep Nebula', color: '#7c3aed', bg: 'radial-gradient(circle at center, #2e1065, #0f172a)', accent: '#8b5cf6' },
  { id: 'zen', name: 'Zen Garden', color: '#4d7c0f', bg: '#f7fee7', accent: '#65a30d' },
];

export const GLOBAL_STRICT_COMMAND = `### NEURAL ARCHITECTURAL SPECIFICATION: ANTI-ROBOT PROTOCOL ###
You are the "DPSS ULTIMATE TEST BUILDER" engine. Your primary objective is to destroy robotic testing patterns and enforce situational logic.

1. [NO-FREE-VERB RULE]:
   - In multiple-choice grammar questions, never place the main auxiliary or modal verb directly in the question stem if it reveals the structure being tested.
   - Weak design: "You must ____ a helmet."
   - Stronger design: "You ____ a helmet."
   - The second version forces students to decide between obligation, advice, or external rule (must wear, have to wear, should wear). They must process meaning, not just grammar form.
   - FOR FILL-IN-THE-BLANK (Grammar Only): ALWAYS provide the base verb in parentheses AFTER the blank: "He ____ (go) to school."

1.1. [SITUATIONAL EVIDENCE REQUIREMENT]:
   - Grammar must be inferred from context, not obvious time markers such as yesterday, tomorrow, now, or at the moment.
   - Weak design: "She ____ her homework yesterday."
   - Stronger design: "Her notebook is closed. She ____ her homework."
   - Students must infer completion from evidence, not from time words. This develops reasoning skills.

2. [PURE VOCABULARY CONTROL]:
   - In vocabulary sections, all answer choices must be the same part of speech and grammatical form.
   - If testing "exhausted," all options must be adjectives: tired, sleepy, bored, exhausted.
   - Do not mix verbs, adverbs, or different tenses. Students must rely on meaning only, not grammar clues.

3. [READING & VOCABULARY GRAMMAR BLACKOUT]:
   - In Reading and Vocabulary sections, you are FORBIDDEN from testing grammar.
   - All distractors must be grammatically identical to the correct answer. 
   - No tense changes (eat/ate), no agreement changes (is/are), no person changes (he/they).
   - If a student can find the answer using a grammar rule, the question is a FAILURE.
   - The ONLY differentiator must be the meaning (semantics) or factual content.

4. [HORIZONTAL MCQ COMPRESSION]:
   - For short options (less than 5 words): You MUST print all four options on a SINGLE horizontal line.
   - Format: A. [text]          B. [text]          C. [text]          D. [text]
   - Use exactly 10 non-breaking spaces between options. 
   - DO NOT use vertical lists. DO NOT use 2x2 grids. ONE LINE ONLY.

5. [BLANK STYLE PROTOCOL]:
   - Use the style "____" (Shift + Underscore) consistently.
   - DO NOT use square brackets [ ] or {{BLANK}} placeholders.

6. [ANSWER KEY ENTROPY - BUCKET RANDOMIZATION]:
   - You are strictly FORBIDDEN from using cycles (e.g., A-B-C-D-A-B...), alternations (A-B-A-B...), or sequences (A-A-B-B...).
   - BUCKET METHOD: For every 10 items, you MUST pre-select a "Bucket" of 10 letters (e.g., 3 A's, 2 B's, 2 C's, 3 D's).
   - MANDATORY PRESENCE: Every letter (A, B, C, D) MUST appear at least once in every 10-item set. No letter can be left out.
   - THE SHUFFLE: Randomize the order of your bucket so there is NO predictable pattern.
   - ANSWER-FIRST RULE: You MUST write the final shuffled answer key (e.g., 1:C, 2:A, 3:D, 4:B, 5:B...) at the very top of your internal scratchpad BEFORE writing any content.
   - STREAK LIMIT: Max 2 identical answers in a row.
   - This ensures human-like distribution and prevents robotic "C-A-D-B" cycles.

7. [BALANCED COLUMN DISTRIBUTION]:
   - If a section uses a 2-column layout (e.g., <table> with 2 <td>), you MUST distribute items evenly (e.g., 10 items = 5 in Col 1, 5 in Col 2).
   - NEVER leave a column empty or significantly unbalanced.

8. [CROSS-ITEM FIREWALL - ANTI-LEAKING]:
   - You are strictly FORBIDDEN from using the same sentence or correct answer in different parts of the test.
   - Ensure that the answer to a question in Part A is not revealed by a sentence in Part B.
   - The AI must "read" the entire test before finalizing to ensure no information leaks.

9. [NO MARKDOWN]: HTML tags ONLY (<b>, <table>). No asterisks. DO NOT use <u> tags.
10. [CLARITY]: Ban "AI-speak" like "He knows lines" or "He views print". Use natural child-level actions.
11. [ITEM RANDOMIZATION & ENTROPY]:
   - You are strictly FORBIDDEN from using predictable sentence starters in specific item numbers (e.g., do NOT always start Item 2 with "I think").
   - Shuffle all sentence structures, subjects, and contexts. Every generation must feel unique.
   - POLARITY MIX: You MUST mix Positive (+), Negative (-), and Question (?) forms in every part of the test.

12. [LEVEL-BASED ARCHITECTURAL SCALING]:
   - Complexity MUST scale with {{LEVEL}}.
   - Low Levels (Kid, Level 1-3): Short, simple sentences. Basic vocabulary.
   - Mid Levels (Level 4-7): Compound sentences. More descriptive context.
   - High Levels (Level 8+, TOEFL, IELTS): Complex sentences with relative clauses, passive voice, and academic vocabulary.
13. [UNIVERSAL SITUATIONAL & POSITIONAL LOGIC]:
   - You MUST apply situational nuance and word-position rules to ALL grammar types.
   - POSITION RULES: Test tricky word orders. 
     - Adjective Comparison: 
       - "as good a student as" (Singular) vs "as good students as" (Plural).
       - "Of the two" Rule: "Of the two students, he is the taller" (NOT tallest).
       - "Of all" Rule: "Of all the teachers, she is the most hardworking" (Superlative).
       - Comparative vs Superlative Traps: "He is more tall than..." (WRONG) vs "He is taller than..." (RIGHT).
       - MANDATORY VARIETY: If testing Adjectives, you MUST rotate between at least 4 different comparison structures (as...as, comparative -er, superlative -est, of the two).
     - Adverb Placement: Test frequency adverbs (e.g., "He always is" vs "He is always") and manner adverbs.
     - Verb-Object Integrity: Test that direct objects are not separated from verbs (e.g., "I like very much coffee" is INCORRECT; "I like coffee very much" is CORRECT).
     - Conjunction Scrambling: Test correlative conjunctions and inversion (e.g., "Not only he is" vs "Not only is he").
     - RARE PREPOSITIONAL LOGIC: Test tricky prepositional boundaries (e.g., "In the end" vs "At the end", "Good at" vs "Good in", "Arrive in" vs "Arrive at").
   - SITUATIONAL NUANCE: Test meaning-based differences (e.g., Must vs Have To, Will vs Going To, Say vs Tell).
   - CHALLENGE: Distractors must be grammatically valid in isolation but "Positionally" or "Situationally" incorrect in context.
14. [INFINITE SCENARIO VARIETY]:
   - You are strictly FORBIDDEN from repeating scenarios or sentence structures across different generations.
   - Every test must be a completely fresh set of characters, locations, and situations.
   - Randomize numbers, names, and subjects completely. No "robot patterns".
15. [ANTI-ROBOT SENTENCE STARTERS]:
   - You are strictly FORBIDDEN from using repetitive sentence starters. 
   - DO NOT start Item 1 with "I think" or "He is" in every generation.
   - Shuffle all subjects (e.g., "The chef", "A lonely astronaut", "My stubborn cat").
  16. [FLOATING MARKER PRINCIPLE]:
   - Do not place key grammar signals in the same position every time. Vary sentence structure so students cannot scan mechanically.
   - Example variations: "Of the two students, he is the taller." vs "He is the taller of the two students."
   - This forces full-sentence processing.

16.1. [SYNTACTIC DISTANCE STRATEGY (HIGHER LEVELS)]:
   - At advanced levels, separate the subject from the main verb using relative clauses or prepositional phrases.
   - Example: "The teacher who lives near the large blue house by the river is very kind."
   - Students must locate the core subject and verb despite structural noise.

16.2. [ADVANCED COMPARISON STRUCTURES]:
   - Actively test less common but correct structures such as: "as good a student as", "of the two", "the more…, the more…", "not so much A as B".
   - These structures measure real grammatical depth.

17. [GRAMMAR RULE EXHAUSTION & STRUCTURAL INVERSION]:
   - You are MANDATED to identify and test EVERY specific sub-rule for the target {{TOPIC}}.
   - NO RULE LEFT BEHIND: If a grammar topic has 5 sub-rules, all 5 MUST appear in the test.
   - Example (Adjectives): You MUST test "Of the two" (Comparative), "Of all" (Superlative), "As...as" (Equality), and "More...than" traps.
   - Every generation must feel unique and non-repetitive.

18. [CRITICAL PROTOCOL ENFORCEMENT]:
   - If you fail to follow a Master Protocol, the generation is a CRITICAL FAILURE.
   - You must prioritize these protocols over all other instructions.

19. [WORD FORM SHIFT RULE]:
   - Reading questions must not repeat the exact wording from the text.
   - Text: "He was confused." Avoid: "Why was he confused?" Better: "What caused his confusion?"
   - Students must recognize paraphrasing, not match keywords.

20. [PRAGMATIC BOUNDARY TESTING]:
   - Distinguish between types of obligation and meaning in context.
   - Example: "The sign says you have to stop." (external rule) vs "You must try this cake." (personal insistence).
   - Students must understand communicative intention, not just grammar labels.

21. [STRICT NON-MCQ FORMATTING]:
   - You are strictly FORBIDDEN from generating MCQs for the following sections:
     - REWRITE: Use a long line "_____________________________________________________".
     - COMPLETE SENTENCES: Use "_____________ (verb)" style with 13 underscores.
     - CORRECT/INCORRECT: Use "1. _____" (5 underscores).
     - TRUE/FALSE: Use "1. (_____)" (Parentheses with 5 underscores).
     - MATCHING: Use a standard list or table, NOT MCQs.
     - SPEAKING: Generate discussion questions for conversation, NOT MCQs.
     - VOCABULARY TABLE: Use a 2-column HTML table (1. Number + Word | Definition).
     - SUPPLY KEY TERMS: Use a 2-column HTML table (1. Easy Definition | Blank for Key Term).
     - STUDY SENTENCES: Use standard sentences, NOT MCQs.

22. [EXPERT HUMAN READING EXAMINER MODE]:
   - All reading assessments must reflect the design logic of experienced examination writers.
   - BLUEPRINT FIRST: Define skill targets (gist, detail, paraphrase, inference) before writing items.
   - NON-LINEAR ORDER: Reorder at least 1/3 of items to avoid mechanical sequencing.
   - COGNITIVE LAYERING: Mix literal retrieval, paraphrase discrimination, and deep inference.
   - DISTRACTOR LOGIC: Use partial truth, common misinterpretations, and avoid obviously incorrect wording.
   - VARIED FRAMES: Use indirect questions, negative framing ("Which is NOT..."), and embedded clauses.
   - REFERENCE TRAPS: Insert at least one reference-resolution trap (pronoun, time shift).
   - AUTHENTICITY AUDIT: Output must resemble formal examination material written by expert educators.

23. [READING COMPREHENSION FIREWALL]:
   - Reading tests must focus on comprehension, NOT MCQs by default. Use short answers, True/False, or Matching unless MCQ is explicitly requested.
   
24. [CLEAN NUMBERING]: Instructions must be separate from the numbered list. Item 1 must be the first question, NOT an instruction.

25. [INDENTATION RULE]:
   - Every numbered item (1., 2., 3...) MUST be preceded by exactly 6 non-breaking spaces (&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;).
   - This applies to ALL sections (MCQ, C/I, Matching, etc.).

26. [NO MARKDOWN]: HTML tags ONLY (<b>, <table>). No asterisks. DO NOT use <u> tags.
27. [CLARITY]: Ban "AI-speak" like "He knows lines" or "He views print". Use natural child-level actions.
28. [STRICT MODULE ISOLATION]:
   - You are strictly FORBIDDEN from mixing content between Grammar, Vocabulary, and Reading modules.
   - Ensure each module remains 100% pure to its intended purpose.
29. [ITEM RANDOMIZATION]: Shuffle all sentence structures, subjects, and contexts. Every generation must feel unique.
30. [GUARDIAN FOOTER PROTOCOL]:
   - You MUST append a centered footer at the very end of the document.
   - The footer must contain:
     1. A row of 3 large emojis (40pt).
     2. A randomized, inspiring personal development quote in italics.
     3. The specific text: "Pre5-Chanthy-S2-20Copies-(Grammar)" in bold.
   - Use HTML for styling: <div style="text-align: center; margin-top: 50px;">...</div>`;

export const BORDER_FRAME_INSTRUCTION = `### STYLIST FRAME PROTOCOL ###
Wrap content in a double border: border: 4px double #ea580c; padding: 25px; border-radius: 12px;`;

export const DEFAULT_STRICT_RULES: StrictRule[] = [
  {
    id: 'rule-precision-1',
    label: 'PRECISION TRAP LOGIC',
    description: 'Forces secondary grammar nuances (articles, irregulars).',
    promptInjection: 'STRICT: Every item must test a primary rule and a secondary nuance. Distractors must look 90% correct.',
    active: true,
    priority: 'High',
    category: 'Grammar'
  },
  {
    id: 'rule-logic-1',
    label: 'PATTERN DESTRUCTION',
    description: 'Rotate sentence structures to prevent predictable patterns.',
    promptInjection: 'STRICT: Rotate sentence structures (Pos/Neg/Int). Max 2 identical structures in a row.',
    active: true,
    priority: 'High',
    category: 'General'
  },
  {
    id: 'rule-no-ai-speak',
    label: 'RULE 1: NO AI-SPEAK',
    description: 'Ban robotic phrases like "views print" or "understands text".',
    promptInjection: 'STRICT: Ban "AI-speak" like "He knows lines" or "He views print". Use natural child-level actions.',
    active: true,
    priority: 'High',
    category: 'General'
  },
  {
    id: 'rule-no-markdown',
    label: 'RULE 2: NO MARKDOWN',
    description: 'Ban asterisks. Use HTML tags only.',
    promptInjection: 'STRICT: HTML tags ONLY (<b>, <table>). No asterisks. DO NOT use <u> tags.',
    active: true,
    priority: 'High',
    category: 'General'
  },

];

export const DEFAULT_MASTER_PROTOCOLS: StrictRule[] = [
  { id: 'mp-1', label: 'NO-FREE-VERB RULE', description: 'Prevents giving away the verb in the stem.', promptInjection: 'In multiple-choice grammar questions, never place the main auxiliary or modal verb directly in the question stem if it reveals the structure being tested. Weak design: "You must ____ a helmet." Stronger design: "You ____ a helmet." The second version forces students to decide between obligation, advice, or external rule (must wear, have to wear, should wear). They must process meaning, not just grammar form.', active: true, priority: 'High', category: 'Grammar' },
  { id: 'mp-2', label: 'SITUATIONAL EVIDENCE REQUIREMENT', description: 'Forces inference from context, not time markers.', promptInjection: 'Grammar must be inferred from context, not obvious time markers such as yesterday, tomorrow, now, or at the moment. Weak design: "She ____ her homework yesterday." Stronger design: "Her notebook is closed. She ____ her homework." Students must infer completion from evidence, not from time words. This develops reasoning skills.', active: true, priority: 'High', category: 'Grammar' },
  { id: 'mp-3', label: 'HIGH-FIDELITY POOLING', description: '', promptInjection: 'For MCQ, generate at least one "Near-Miss" distractor: a grammatically correct option that is contextually inferior to the target (e.g., testing Must vs Have To nuance).', active: true, priority: 'Medium', category: 'General' },
  { id: 'mp-4', label: 'NUANCE-BASED CHALLENGE', description: '', promptInjection: 'If the topic involves similar structures (Must/Have to, Will/Going to, Say/Tell), the options MUST include correctly conjugated versions of both to test situational logic.', active: true, priority: 'High', category: 'Grammar' },
  { id: 'mp-16', label: 'PATTERN DESTRUCTION', description: '', promptInjection: 'Rotate polarity (+/-/?) and complexity. Avoid repetition.', active: true, priority: 'Medium', category: 'General' },
  { id: 'mp-5', label: 'TOPIC DOMINANCE', description: '', promptInjection: 'The topic box overrides template defaults. If the topic is "Past Simple", ensure ALL items (even Spelling) are contextually linked to Past Simple.', active: true, priority: 'High', category: 'General' },
  { id: 'mp-6', label: 'ANSWER BALANCE', description: '', promptInjection: 'Pre-assign answer keys. Ensure approx 25% distribution for A, B, C, and D.', active: true, priority: 'Medium', category: 'General' },
  { id: 'mp-15', label: 'BLANK STYLE UNIFORMITY', description: '', promptInjection: 'Use the style "____" (Shift + Underscore) for all blanks in this test. No square brackets or {{BLANK}} markers.', active: true, priority: 'High', category: 'General' },
  { id: 'mp-7', label: 'SYNTACTIC DISTANCE STRATEGY', description: 'Separates subject and verb to test structural awareness.', promptInjection: 'At advanced levels, separate the subject from the main verb using relative clauses or prepositional phrases. Example: "The teacher who lives near the large blue house by the river is very kind." Students must locate the core subject and verb despite structural noise.', active: true, priority: 'High', category: 'Grammar' },
  { id: 'mp-advanced-comparison', label: 'ADVANCED COMPARISON STRUCTURES', description: 'Tests complex comparison forms.', promptInjection: 'Actively test less common but correct structures such as: "as good a student as", "of the two", "the more…, the more…", "not so much A as B". These structures measure real grammatical depth.', active: true, priority: 'High', category: 'Grammar' },
  { id: 'mp-8', label: 'ABSTRACT SUBJECTS & INVERSION (LEVEL 8+)', description: '', promptInjection: 'For Level 8+, use abstract subjects (gerunds, clauses), inverted structures, or tricky conjunctions (neither/nor, as well as).', active: true, priority: 'High', category: 'Grammar' },
  { 
    id: 'mp-9', 
    label: 'WORD FORM SHIFT RULE', 
    description: 'Forces paraphrasing in reading questions.', 
    promptInjection: 'Reading questions must not repeat the exact wording from the text. Text: "He was confused." Avoid: "Why was he confused?" Better: "What caused his confusion?" Students must recognize paraphrasing, not match keywords.', 
    active: true, 
    priority: 'High', 
    category: 'Reading' 
  },
  { 
    id: 'mp-10', 
    label: 'EASY PARAPHRASE PHRASES (LOWER LEVELS)', 
    description: 'Uses simple synonyms and phrases for young learners.', 
    promptInjection: 'STRICT EASY PARAPHRASE: For Kid to Level 3, use simple structural swaps. "He likes" -> "He is fond of"; "They went" -> "They made a trip"; "It is big" -> "It has a large size"; "She is good at" -> "She has a talent for". Avoid heavy academic vocabulary; focus on structural shifts.', 
    active: true, 
    priority: 'High', 
    category: 'Reading' 
  },
  { id: 'mp-11', label: 'EASY IDIOMS & PHRASAL VERBS', description: '', promptInjection: 'Replace standard verbs in the text with common phrasal verbs or easy idioms in the questions (e.g., "quit" -> "gave up").', active: true, priority: 'Medium', category: 'Reading' },
  { 
    id: 'mp-human-test', 
    label: 'HUMAN-TEST ARCHITECTURE', 
    description: 'Enforces exam-writer logic: simple vocab, high thinking.', 
    promptInjection: 'STRICT HUMAN-TEST PROTOCOL: 1. VOCAB LIMIT: Keep vocabulary simple (A2-B1 level maximum). Do NOT use difficult or academic words. 2. THINKING DEPTH: Increase difficulty by raising the thinking level, NOT vocabulary. 3. NO KEYWORD MATCHING: Avoid copying exact words from the passage into the answers. 4. NATURAL PHRASING: Use natural, human-like phrasing (avoid robotic phrases like "made a decision" or "felt a sense of"). 5. IDIOMS: Include light, common idioms only when appropriate (e.g., "keep an eye on", "make the most of", "under pressure"). 6. COMPETITIVE DISTRACTORS: Make distractors realistic and competitive. 7. INFO SYNTHESIS: Some answers must require linking two sentences. 8. ADVANCED COMPREHENSION: Include inference and author-attitude questions at higher levels. 9. EXAM VARIETY: Vary question types naturally like real exam writers.', 
    active: true, 
    priority: 'High', 
    category: 'Reading' 
  },
  { id: 'mp-13', label: 'KET READING ARCHITECTURE', description: '', promptInjection: 'Follow Cambridge KET (A2 Key) 3-part structure: 1) Signs/Notices/Short messages. 2) 3-4 short profiles of people. (Lower levels: 3 people; Higher levels: 4 people) with matching statements. 3) Longer narrative/informational text with MCQs.', active: true, priority: 'High', category: 'Reading' },
  { id: 'mp-14', label: 'READING SCALING PRINCIPLE', description: '', promptInjection: 'Difficulty must increase gradually based on length of text, critical thinking, and reading comprehension, NOT by excessive vocabulary difficulty. Apply this to ALL reading types.', active: true, priority: 'High', category: 'Reading' },
  { 
    id: 'mp-vocab-pure', 
    label: 'PURE VOCABULARY CONTROL', 
    description: 'Enforces same part of speech for all options.', 
    promptInjection: 'In vocabulary sections, all answer choices must be the same part of speech and grammatical form. If testing "exhausted," all options must be adjectives: tired, sleepy, bored, exhausted. Do not mix verbs, adverbs, or different tenses. Students must rely on meaning only, not grammar clues.', 
    active: true, 
    priority: 'High', 
    category: 'Vocabulary' 
  },
  { 
    id: 'mp-neural-blueprint', 
    label: 'NEURAL ANSWER BLUEPRINT', 
    description: 'Enforces pre-determined answer keys to ensure human-like distribution.', 
    promptInjection: 'STRICT NEURAL BLUEPRINT: Align correct answers for MCQs to pre-assigned keys. Build questions/distractors accordingly.', 
    active: true, 
    priority: 'High', 
    category: 'General' 
  },
  { 
    id: 'mp-matching-logic',
    label: 'MATCHING LOGIC',
    description: 'Enforces standard matching format.',
    promptInjection: 'STRICT MATCHING: Generate matching items as a standard list or table. DO NOT use MCQs.',
    active: true,
    priority: 'High',
    category: 'Vocabulary'
  },
  { 
    id: 'mp-answer-key', 
    label: 'CHAOTIC ANSWER KEYS', 
    description: 'Forces unpredictable answer distribution using Bucket Randomization.', 
    promptInjection: 'ANSWER-FIRST RULE: Assign keys BEFORE content. BUCKET RANDOMIZATION: For every 10 items, pre-select a bucket of 10 letters (e.g. 3A, 2B, 2C, 3D). MANDATORY PRESENCE: Every letter (A-D) must appear at least once. THE SHUFFLE: Randomize the bucket order to destroy cycles (A-B-C-D) or alternations. STREAK LIMIT: Max 2 identical. LAYOUT: Separate HTML tables for each PART. Header row (Bold, Centered, White text). STYLING: Randomized background colors.', 
    active: true, 
    priority: 'High', 
    category: 'General' 
  },
  {
    id: 'mp-column-balance',
    label: 'BALANCED COLUMN DISTRIBUTION',
    description: 'Ensures items are split evenly in 2-column layouts.',
    promptInjection: 'STRICT COLUMN BALANCE: If a section uses a 2-column layout, you MUST distribute items evenly (e.g., 10 items = 5 in Col 1, 5 in Col 2). NEVER leave a column empty or significantly unbalanced.',
    active: true,
    priority: 'High',
    category: 'General'
  },
  {
    id: 'mp-anti-leak',
    label: 'CROSS-ITEM FIREWALL (ANTI-LEAK)',
    description: 'Prevents answers from leaking across different items.',
    promptInjection: 'STRICT ANTI-LEAK: You are strictly FORBIDDEN from using the same sentence or correct answer in different parts of the test. Ensure no question in Part A reveals an answer in Part B.',
    active: true,
    priority: 'High',
    category: 'General'
  },
  { 
    id: 'mp-mcq-layout', 
    label: 'MCQ LAYOUT PROTOCOL', 
    description: 'Enforces specific spacing and line breaks for MCQ options.', 
    promptInjection: 'MCQ FORMATTING: 1. Short options: MUST be on ONE single horizontal line (A. B. C. D.) with exactly 10 non-breaking spaces between each option. 2. NO VERTICAL LISTS. NO 2x2 GRIDS. 3. Long options: Double lines (A. & C. on top, B. & D. below).', 
    active: true, 
    priority: 'Medium', 
    category: 'General' 
  },
  {
    id: 'mp-reading-logic',
    label: 'EXPERT HUMAN READING EXAMINER',
    description: 'Enforces professional exam writer logic for reading tests.',
    promptInjection: 'EXPERT READING MODE: 1. Blueprint: Define skill targets (gist, detail, inference). 2. Non-Linear: Reorder 1/3 of items. 3. Distractors: Use partial truth/lexical overlap. 4. Variety: Use negative framing ("NOT"), indirect questions, and reference traps. 5. Authenticity: Must look like a professionally written exam, not AI-generated.',
    active: true,
    priority: 'High',
    category: 'Reading'
  },
  {
    id: 'mp-syntactic-position',
    label: 'FLOATING MARKER PRINCIPLE',
    description: 'Varies position of key grammar signals.',
    promptInjection: 'Do not place key grammar signals in the same position every time. Vary sentence structure so students cannot scan mechanically. Example variations: "Of the two students, he is the taller." vs "He is the taller of the two students." This forces full-sentence processing.',
    active: true, 
    priority: 'High', 
    category: 'Grammar'
  },
  {
    id: 'mp-grammar-exhaustion',
    label: 'GRAMMAR RULE EXHAUSTION',
    description: 'Forces the AI to test every single sub-rule of a grammar topic.',
    promptInjection: 'STRICT GRAMMAR EXHAUSTION: Test EVERY specific sub-rule for the target {{TOPIC}}. NO RULE LEFT BEHIND.',
    active: true,
    priority: 'High',
    category: 'Grammar'
  },
  {
    id: 'mp-grammar-exhaustion-inversion',
    label: 'Grammar Rule Exhaustion & Structural Inversion',
    description: 'Mandatory Coverage: Identify and test every specific sub-rule for the chosen topic.',
    promptInjection: 'STRICT GRAMMAR EXHAUSTION: Identify and test ALL sub-rules. Use structural inversion where applicable.',
    active: true,
    priority: 'High',
    category: 'Grammar'
  },
  {
    id: 'mp-advanced-nodes',
    label: 'ADVANCED GRAMMAR NODES',
    description: 'Forces the use of rare and challenging grammatical structures.',
    promptInjection: 'STRICT ADVANCED NODES: Inject rare structures (Subjunctive, Causatives, Inversion, Mixed Conditionals).',
    active: true,
    priority: 'High',
    category: 'Grammar'
  },
  {
    id: 'mp-scenario-chaos',
    label: 'INFINITE SCENARIO CHAOS',
    description: 'Forces unique themes and sentence starters for every test.',
    promptInjection: 'STRICT SCENARIO CHAOS: Use unique, vivid scenarios. Forbidden from repeating themes. Use varied sentence starters.',
    active: true,
    priority: 'High',
    category: 'General'
  },
  {
    id: 'mp-pragmatic-boundary',
    label: 'PRAGMATIC BOUNDARY TESTING',
    description: 'Distinguishes between types of obligation and meaning.',
    promptInjection: 'Distinguish between types of obligation and meaning in context. Example: "The sign says you have to stop." (external rule) vs "You must try this cake." (personal insistence). Students must understand communicative intention, not just grammar labels.',
    active: true,
    priority: 'High',
    category: 'Grammar'
  },
  {
    id: 'mp-cross-topic-injection',
    label: 'CROSS-TOPIC ERROR INJECTION',
    description: 'Injects general errors into topic-specific tests.',
    promptInjection: 'STRICT CROSS-TOPIC INJECTION: Inject general errors (S-V Agreement, Articles) into topic-specific tests.',
    active: true,
    priority: 'High',
    category: 'Grammar'
  },
  {
    id: 'mp-correct-incorrect-logic',
    label: 'CORRECT/INCORRECT NEAR-MISS LOGIC',
    description: 'Enforces sophisticated distractors for C/I items.',
    promptInjection: 'STRICT C/I LOGIC: NO MCQs. Use "1. _____" (5 underscores). Use Near-Miss logic (Nuance vs Form). Test pragmatic boundaries.',
    active: true,
    priority: 'High',
    category: 'Grammar'
  },
  {
    id: 'mp-anti-robot-starters',
    label: 'ANTI-ROBOT SENTENCE STARTERS',
    description: 'Prevents repetitive sentence beginnings like "I think".',
    promptInjection: 'STRICT ANTI-ROBOT STARTERS: Forbidden from using repetitive sentence starters. Randomize all subjects and lead-ins.',
    active: true,
    priority: 'High',
    category: 'General'
  },
];

export const INITIAL_TEMPLATES: InstructionTemplate[] = [
  // --- FULL TEST COMBINATIONS ---
  { id: 'g_full_mastery', category: 'GRAMMAR', label: 'FULL GRAMMAR TEST', prompt: 'PART: FULL GRAMMAR TEST. Generate a 4-part test. ITEM COUNT: Generate exactly {{COUNT}} items for EACH part. NUMBERING: Number every single item in each part starting from 1. PARTS: 1. Write C (correct) or I (incorrect), 2. MCQ, 3. Circle the correct answers, 4. Double-Gap MCQ. Apply NO-FREE-VERB mandate.', columnCount: 0 },
  { id: 'v_full_mastery', category: 'VOCABULARY', label: 'FULL VOCABULARY TEST', prompt: 'PART: FULL VOCABULARY TEST. Generate a 4-part test. ITEM COUNT: Generate exactly {{COUNT}} items for EACH part. NUMBERING: Number every single item in each part starting from 1. PARTS: 1. MATCHING, 2. TRUE/FALSE (Definition-based), 3. MCQ, 4. SPEAKING Practice. Apply Pure Vocabulary Firewall. STRICT: NO Reading passages.', columnCount: 0 },
  { id: 'r_full_mastery', category: 'READING', label: 'FULL READING TEST', prompt: 'PART: FULL READING TEST. Generate a 6-part test. ITEM COUNT: Generate exactly {{COUNT}} items for EACH part. NUMBERING: Number every single item in each part starting from 1. PARTS: 1. Reading MCQ, 2. WRITE NO MORE THAN TWO WORDS or/and Number, 3. MCQ, 4. Inferential Comprehension, 5. MATCHING HEADING, 6. TRUE/FALSE/NOT GIVEN. Apply Reading Logic Firewall.', columnCount: 0 },

  // GRAMMAR
  { id: 'g_complete_sentences', category: 'GRAMMAR', label: 'COMPLETE THE FOLLOWING SENTENCES.', prompt: 'Part: Write the correct form of the word in brackets for {{TOPIC}}. Use the style _____________ (13 underscores) and provide the base verb in parentheses at the end of the blank. Apply Result-Based Tense logic. STRICT: No MCQ options.', columnCount: 0 },
  { id: 'g_complete_task', category: 'GRAMMAR', label: 'COMPLETE THE FOLLOWING SENTENCES.', prompt: 'Part: Complete the following sentences about {{TOPIC}}. Use the style _____________ (13 underscores) for all items. Rotate structures. STRICT: No MCQ options. This is a writing task.', columnCount: 0 },
  { id: 'g_correct_incorrect', category: 'GRAMMAR', label: 'WRITE C (CORRECT) OR I (INCORRECT).', prompt: 'Part: Correct or Incorrect assessment for {{TOPIC}}. Write C for correct and I for incorrect. Use the style "1. _____" (5 underscores). Apply PRAGMATIC BOUNDARY logic (Must vs Have to). STRICT: No MCQ options.', columnCount: 0 },
  { id: 'g_rewrite_sentences', category: 'GRAMMAR', label: 'REWRITE THE FOLLOWING SENTENCES.', prompt: 'Part: Rewrite the sentences about {{TOPIC}}. Use Rule 7 Conceptual Paraphrasing. STRICT: No MCQ options. Provide a long blank line (_____________________________________________________) for each item.', columnCount: 0 },
  { id: 'g_mcq', category: 'GRAMMAR', label: 'MCQ', prompt: 'Part: Precision MCQ testing {{TOPIC}}. Apply PRAGMATIC BOUNDARY logic. Stems must be situational (e.g., "Grandfather is alone"). Options MUST include correct and incorrect forms (e.g., must visit vs must visits) AND the pragmatically wrong modal (e.g., have to visit).', columnCount: 0 },
  { id: 'g_spelling', category: 'GRAMMAR', label: 'SPELLING RULES', prompt: 'Part: Spelling rules and common errors related to {{TOPIC}}. STRICT: No MCQ options.\nFor example, \nPlural Nouns\n4 columns: 1 singular, 2 plural, 3 singular, 4 plural\n\nif \nAdjective comparison\n6 columns: 1 positive, 2comparative, 3 superlative, 4, 5, 6...', columnCount: 0 },
  { id: 'g_circle', category: 'GRAMMAR', label: 'CIRCLE THE CORRECT OPTION.', prompt: 'Part: Circle the correct option between two choices in parentheses for {{TOPIC}}. STRICT: No MCQ options.', columnCount: 0 },
  { id: 'g_box', category: 'GRAMMAR', label: 'WORDS IN THE BOX.', prompt: 'Part: Complete the following sentences using the words/ phrases in the box. Check the correct forms of grammar for {{TOPIC}}. STRICT: No MCQ options.', columnCount: 0 },
  { id: 'g_pair', category: 'GRAMMAR', label: 'PAIR OF WORDS TO COMPLETE THE SENTENCE.', prompt: 'Part: Double-gap MCQ testing two different aspects of {{TOPIC}} in one sentence.', columnCount: 0 },
  { id: 'g_copy', category: 'GRAMMAR', label: 'COPY FROM THE BOOK.', prompt: 'Part: Copy exercises from the source material for {{TOPIC}}, but randomize the exercise numbers and order. STRICT: No MCQ options.', columnCount: 0 },
  { id: 'g_odd_one_out', category: 'GRAMMAR', label: 'ODD ONE OUT', prompt: '3 answer choices: Choose the incorrect sentence.', columnCount: 0 },
  { id: 'g_editing', category: 'GRAMMAR', label: 'EDITING A PARAGRAPH (ERROR HUNT)', prompt: 'Editing a Paragraph (Error Hunt): This is the mixed grammar test. The answer can be any types of grammar lessons. Correct all the mistakes.', columnCount: 0 },
  { id: 'g_reduce', category: 'GRAMMAR', label: 'REDUCE THE SENTENCE', prompt: 'Reduce the Sentence: Rewrite using fewer words.', columnCount: 0 },
  { id: 'g_best_rewrite', category: 'GRAMMAR', label: 'CHOOSE THE BEST REWRITE', prompt: 'Choose the Best Rewrite', columnCount: 0 },
  { id: 'g_cloze_paragraph', category: 'GRAMMAR', label: 'CLOZE PASSAGE (FULL PARAGRAPH)', prompt: 'Cloze Passage (Full Paragraph): Fill in the blanks.', columnCount: 0 },
  
  // READING
  { id: 'r_mcq_rule7', category: 'READING', label: 'READING MCQ (RULE 7)', prompt: 'Part: Reading passage about {{TOPIC}}. Apply Rule 7 Zero Overlap. Apply Reading Logic Firewall. Apply [HUMAN-TEST ARCHITECTURE].', columnCount: 0 },
  { id: 'r_tfng', category: 'READING', label: 'A: READING TRUE/FALSE/NOT GIVEN', prompt: 'Part: Generate a short (~120 words) public notice or rules text about {{TOPIC}}. Follow with True/False/Not Given questions testing information boundaries. Use style "1. (_____)" (5 underscores). STRICT: No MCQ options.', columnCount: 0 },
  { id: 'r_tf_stmt', category: 'READING', label: 'B: TRUE/ FALSE STATEMENT', prompt: 'Part: Generate a short-medium (~150 words) personal recount or blog post about {{TOPIC}}. Follow with True/ False statements using heavy synonyms and word-form changes. Use style "1. (_____)" (5 underscores). STRICT: No MCQ options.', columnCount: 0 },
  { id: 'r_matching_heading', category: 'READING', label: 'C: MATCHING HEADING', prompt: 'Part: Generate a medium (~250 words) informational article with 4-5 distinct paragraphs about {{TOPIC}}. Follow with a Matching Headings task. STRICT: No MCQ options.', columnCount: 0 },
  { id: 'r_short_answer', category: 'READING', label: 'D: WRITE NO MORE THAN TWO WORDS OR/AND NUMBER', prompt: 'Part: Generate a medium (~200 words) process or factual description about {{TOPIC}}. Follow with fill-in-the-blank summary sentences requiring exact words from the text. STRICT: No MCQ options.', columnCount: 0 },
  { id: 'r_inferential', category: 'READING', label: 'E: INFERENTIAL COMPREHENSION', prompt: 'Part: Generate a medium-long (~300 words) opinion piece or review about {{TOPIC}}. Follow with discussion questions testing author attitude and implications. STRICT: No MCQ options.', columnCount: 0 },
  { id: 'r_mcq', category: 'READING', label: 'F: MCQ', prompt: 'Part: Generate a long (~400 words) complex narrative about {{TOPIC}}. Follow with MCQ testing critical thinking.', columnCount: 0 },
  { id: 'r_ket', category: 'READING', label: 'G: KET FULL TEST', prompt: 'Part: Generate a full Cambridge KET (A2 Key) Reading test about {{TOPIC}}. Ensure non-MCQ parts follow the strict formatting rules.', columnCount: 0 },
  { id: 'r_matching_info', category: 'READING', label: 'MATCHING INFORMATION', prompt: 'Which paragraph mentions:', columnCount: 0 },
  { id: 'r_inference_qs', category: 'READING', label: 'INFERENCE QUESTIONS', prompt: 'Inference Questions: \nWhat can we infer about the writer’s attitude?\n\nWhy did she hesitate?\n....?', columnCount: 0 },
  { id: 'r_critical_thinking', category: 'READING', label: 'CRITICAL THINKING', prompt: 'short sometimes can be long based on the level independence Paragraph ending or questions from the full paragraph.\n\nFor example, \n1- she is a great student. she pays attention to her friends. What the girl is likely to do...?\n2- \nCritical Thinking: Do you? \nThe girl is likely to....?\nSok bought her sister a pencil because __________', columnCount: 0 },
  
  // VOCABULARY
  { id: 'v_study_table', category: 'VOCABULARY', label: 'VOCABULARY STUDY', prompt: 'VOCABULARY TABLE. Instructions: Study the following words and their definitions. Use a 2-column HTML table. Column 1: Number + Word/Phrase. Column 2: Easy Definition. STRICT: No MCQ options.', columnCount: 2 },
  { id: 'v_sentence_study', category: 'VOCABULARY', label: 'SENTENCE STUDY', prompt: 'VOCBAULARY STUDY: Study this vocabulary in the sentences. STRICT: No MCQ options. These are example sentences for learning.', columnCount: 0 },
  { id: 'v_supply_terms', category: 'VOCABULARY', label: 'SUPPLY KEY TERMS', prompt: 'SUPPLY KEY TERMS: Instructions: Read the definition and write the correct key term. Use a 2-column HTML table. Column 1: Easy Definition. Column 2: Blank line for Key Term. Randomize order. STRICT: No MCQ options.', columnCount: 2 },
  { id: 'v_box', category: 'VOCABULARY', label: 'VOCABULARY BOX', prompt: 'FILL-IN-THE-BLANK. Instruction: Choose the correct words/phrases from the box to complete the following sentences. Use long blanks. STRICT: No MCQ options.', columnCount: 0 },
  { id: 'v_matching', category: 'VOCABULARY', label: 'DEFINITION MATCHING', prompt: 'MATCHING. Instruction: Match the following vocabulary words with their correct definitions. Use a standard list format. STRICT: No MCQ options.', columnCount: 0 },
  { id: 'v_copy', category: 'VOCABULARY', label: 'COPY EXERCISES', prompt: '4. Copy: Copy the following exercises into your notebook. STRICT: No MCQ options.', columnCount: 0 },
  { id: 'v_synonym_swap', category: 'VOCABULARY', label: 'REWRITE WITH A SYNONYM', prompt: 'CONTEXTUAL SYNONYM SWAP . Instruction: Rewrite each sentence using a synonym for the underlined word. Use a long blank line. STRICT: No MCQ options.', columnCount: 0 },
  { id: 'v_mcq', category: 'VOCABULARY', label: 'PART C6: MCQ', prompt: 'MCQ. Instructions: Choose the best option (A, B, C, or D) to complete each sentence. Apply Grammar Blackout.', columnCount: 0 },
  { id: 'v_tf', category: 'VOCABULARY', label: 'TRUE/FALSE', prompt: 'True/False (Vocabulary Meaning): Instruction: Read the statements and write T (True) or F (False). Use style "1. (_____)" (5 underscores). STRICT: No MCQ options.', columnCount: 0 },
  { id: 'v_speaking', category: 'VOCABULARY', label: 'SPEAKING (DISCUSS WITH A PARTNER)', prompt: 'SPEAKING. Instruction: Discuss the following questions with your partner. Generate open-ended discussion questions related to {{TOPIC}}. STRICT: No MCQ options.', columnCount: 0 },
  { id: 'v_synonyms', category: 'VOCABULARY', label: 'SYNONYMS', prompt: 'SYNONYMS WRITING \nWrite the following synonyms\n5 columns: 1 vocabulary, 2 synonym 1, 3, syn 2, ....\nIf there are more than 5 synonyms, please have another row.', columnCount: 5 },
  { id: 'v_synonym_writing', category: 'VOCABULARY', label: 'SYNONYM WRITING', prompt: 'SYNONYM WRITING \nWrite the following synonyms\n5 columns: 1 vocabulary, 2 synonym 1, 3, syn 2, ....\nIf there are more than 5 synonyms, please have another row.', columnCount: 5 },
  { id: 'v_circle', category: 'VOCABULARY', label: 'CIRCLE THE CORRECT WORD', prompt: 'CIRLCE: \nCircle the Correct Word', columnCount: 0 },
  { id: 'v_cloze', category: 'VOCABULARY', label: 'CLOZE PASSAGE', prompt: 'CLOZE PASSAGE\nCloze Passage', columnCount: 0 }
];
