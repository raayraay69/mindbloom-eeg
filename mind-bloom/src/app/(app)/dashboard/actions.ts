'use server';

import { adaptiveSeverityDisclosure, type AdaptiveSeverityDisclosureInput } from '@/ai/flows/adaptive-severity-disclosure';
import type { AnalysisResult } from '@/lib/types';
import { z } from 'zod';

// This function now simulates sending the file to a real model endpoint.
// In a real application, you would pass the file data or a URL to the file.
export async function performAnalysis(
  // The file data would be passed here, e.g., as a FormData object
): Promise<AnalysisResult> {
  console.log('Performing analysis by calling external model endpoint...');

  // Replace this with the actual URL of your deployed model
  const modelEndpoint = 'https://your-model-endpoint.run.app/predict';

  try {
    // In a real scenario, you would send the EEG file data in the request body.
    // For this example, we'll simulate the request and a successful response.
    /*
    const response = await fetch(modelEndpoint, {
      method: 'POST',
      body: fileData, // e.g., FormData containing the file
    });

    if (!response.ok) {
      throw new Error(`Model API request failed with status ${response.status}`);
    }

    const result = await response.json();
    return {
        prediction: result.prediction,
        confidence: result.confidence,
        processingTime: result.processingTime,
    };
    */

    // Simulating the API call latency and response for now.
    const processingTime = 15000 + Math.random() * 10000;
    await new Promise(resolve => setTimeout(resolve, 2000));

    const prediction = Math.random() > 0.6 ? 'positive' : 'negative';
    const confidence = prediction === 'positive' 
      ? 0.45 + Math.random() * 0.45 // 0.45 to 0.9
      : 0.8 + Math.random() * 0.19; // 0.8 to 0.99

    // This is the structure your API should return
    const simulatedResult: AnalysisResult = {
      prediction: prediction as 'positive' | 'negative',
      confidence: parseFloat(confidence.toFixed(4)),
      processingTime: Math.round(processingTime),
    };
    
    return simulatedResult;

  } catch (error) {
    console.error('Error performing analysis:', error);
    // It's good practice to throw the error so the client can handle it
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
