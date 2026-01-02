'use server';

import { adaptiveSeverityDisclosure, type AdaptiveSeverityDisclosureInput } from '@/ai/flows/adaptive-severity-disclosure';
import type { AnalysisResult } from '@/lib/types';
import { z } from 'zod';

/**
 * Perform EEG analysis by sending file to the backend API.
 * Returns prediction results and detailed validation information.
 */
export async function performAnalysis(fileData: FormData): Promise<AnalysisResult> {
  console.log('Performing analysis by calling backend API...');

  // Use the backend API endpoint (adjust based on your deployment)
  const apiEndpoint = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';
  const modelEndpoint = `${apiEndpoint}/predict`;

  try {
    const startTime = Date.now();

    const response = await fetch(modelEndpoint, {
      method: 'POST',
      body: fileData,
    });

    const processingTime = Date.now() - startTime;

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(errorData.detail || `API request failed with status ${response.status}`);
    }

    const result = await response.json();

    // Map backend response to frontend AnalysisResult type
    return {
      prediction: result.prediction.includes('Detected') ? 'positive' : 'negative',
      confidence: result.confidence || result.probability,
      processingTime: processingTime,
      validation: result.validation,
      channels_matched: result.channels_matched,
      recording_length_seconds: result.recording_length_seconds,
    };

  } catch (error) {
    console.error('Error performing analysis:', error);
    if (error instanceof Error) {
      throw error;
    }
    throw new Error('Failed to get analysis from the model service.');
  }
}


const disclosureSchema = z.object({
  prediction: z.enum(['positive', 'negative']),
  confidence: z.number(),
  emotionalState: z.string().optional(),
});

export async function getDisclosure(input: z.infer<typeof disclosureSchema>) {
    const validatedInput = disclosureSchema.parse(input);
    return await adaptiveSeverityDisclosure(validatedInput as AdaptiveSeverityDisclosureInput);
}
