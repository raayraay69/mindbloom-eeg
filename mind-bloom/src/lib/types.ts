export type TherapeuticContent = {
  id: string;
  category: 'Mobile Apps' | 'VR Experiences' | 'AI-Guided Exercises' | 'Ambient Soundscapes';
  title: string;
  description: string;
  thumbnailUrl: string;
  imageHint: string;
  platform: ('iOS' | 'Android' | 'Web' | 'VR Headset')[];
  duration: string;
  rating: number;
  intensity: 'Low' | 'Moderate' | 'High';
  accessibility: ('Colorblind Mode' | 'Reduced Motion' | 'Screen Reader Friendly')[];
};

export type Session = {
  id: string;
  date: string;
  prediction: 'positive' | 'negative';
  confidence: number;
  processingTime: number;
};

export type AnalysisResult = {
  prediction: 'positive' | 'negative';
  confidence: number;
  processingTime: number;
};
