
import { GoogleGenAI, GenerateContentResponse, Type } from "@google/genai";
import { NeuralEngine, QuickSource, OutlineItem, ExternalKeys } from "../types";

export interface NeuralResult {
  text: string;
  thought?: string;
  keyUsed?: string;
}

// ==========================================
//  THE NEURAL ENGINE CONFIG
// ==========================================

const withRetry = async <T>(
  fn: () => Promise<T>,
  retries: number = 2,
  delay: number = 2000
): Promise<T> => {
  try {
    return await fn();
  } catch (error) {
    if (retries <= 0) throw error;
    await new Promise(resolve => setTimeout(resolve, delay));
    return withRetry(fn, retries - 1, delay * 2);
  }
};

export const callNeuralEngine = async (
  engine: NeuralEngine,
  prompt: string,
  systemInstruction: string,
  file?: QuickSource | null,
  userKeys: ExternalKeys = {}
): Promise<NeuralResult> => {
  
  if (engine === NeuralEngine.GEMINI_3_FLASH || engine === NeuralEngine.GEMINI_3_PRO) {
    return withRetry(async () => {
      const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
      const parts: any[] = [{ text: prompt }];
      if (file) {
        parts.push({
          inlineData: { data: file.data, mimeType: file.mimeType }
        });
      }

      const response: GenerateContentResponse = await ai.models.generateContent({
        model: engine,
        contents: { parts },
        config: {
          systemInstruction,
          temperature: 0.7,
          topP: 0.95,
          topK: 64
        },
      });

      return {
        text: response.text || "No content generated.",
        thought: `Neural synthesis complete via ${engine} node.`
      };
    }).catch((error: any) => ({ 
      text: `<div class="p-6 bg-red-50 text-red-600 rounded-xl">Error: ${error.message}</div>` 
    }));
  }

  const userKey = userKeys[engine];
  if (!userKey) return { text: `<div class="p-6 bg-orange-50 text-orange-600">Key required for ${engine}</div>` };

  return withRetry(async () => {
    let endpoint = "";
    if (engine === NeuralEngine.GPT_4O) endpoint = "https://api.openai.com/v1/chat/completions";
    else if (engine === NeuralEngine.GROK_3) endpoint = "https://api.x.ai/v1/chat/completions";
    else if (engine === NeuralEngine.DEEPSEEK_V3) endpoint = "https://api.deepseek.com/chat/completions";

    const response = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${userKey}` },
      body: JSON.stringify({
        model: engine,
        messages: [{ role: "system", content: systemInstruction }, { role: "user", content: prompt }],
        temperature: 0.7
      })
    });

    const data = await response.json();
    return { text: data.choices[0].message.content, thought: `External synthesis via ${engine}.` };
  }).catch((error: any) => ({ text: `<div class="p-6 bg-red-50 text-red-600">Error: ${error.message}</div>` }));
};

// Fixed initialization and property access for generateNeuralOutline.
export const generateNeuralOutline = async (
  prompt: string
): Promise<OutlineItem[]> => {
  try {
    const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
    const response = await ai.models.generateContent({
      model: 'gemini-3-flash-preview',
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              title: { type: Type.STRING },
              children: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    title: { type: Type.STRING },
                    children: { type: Type.ARRAY, items: { type: Type.OBJECT, properties: { title: { type: Type.STRING } } } }
                  }
                }
              }
            },
            required: ["title"]
          }
        }
      }
    });

    // Access .text property directly.
    const jsonStr = response.text || "[]";
    const data = JSON.parse(jsonStr);
    
    const addIds = (items: any[]): OutlineItem[] => {
      return items.map((item, index) => ({
        id: `outline-${Date.now()}-${index}-${Math.random()}`,
        title: item.title,
        expanded: true,
        children: item.children ? addIds(item.children) : []
      }));
    };

    return addIds(data);
  } catch (error: any) {
    console.error("Outline generation failed.", error.message);
    return [];
  }
};
