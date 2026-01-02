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

export type ChannelStatus = {
  name: string;
  found: boolean;
  quality_score: number | null;
  is_zero: boolean;
  is_noisy: boolean;
  snr_db: number | null;
};

export type SignalQuality = {
  overall_score: number;
  channels_found: number;
  channels_expected: number;
  zero_channels: number;
  noisy_channels: number;
  average_snr_db: number | null;
  preprocessing_status: {
    dc_removal: boolean;
    bandpass_filter: boolean;
    notch_filter: boolean;
  };
};

export type ValidationDetails = {
  file_format: string;
  original_sampling_rate: number;
  resampled: boolean;
  duration_seconds: number;
  total_samples: number;
  channels_in_file: string[];
  channel_statuses: ChannelStatus[];
  signal_quality: SignalQuality;
  validation_passed: boolean;
  validation_errors: string[];
  validation_warnings: string[];
};

export type AnalysisResult = {
  prediction: 'positive' | 'negative';
  confidence: number;
  processingTime: number;
  validation?: ValidationDetails;
  channels_matched?: number;
  recording_length_seconds?: number;
};
