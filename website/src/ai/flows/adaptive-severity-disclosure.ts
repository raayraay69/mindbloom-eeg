'use server';

/**
 * @fileOverview A Genkit flow for adaptive severity disclosure in schizophrenia diagnosis.
 *
 * - adaptiveSeverityDisclosure - A function that determines whether to include severity information based on emotional state and confidence score.
 * - AdaptiveSeverityDisclosureInput - The input type for the adaptiveSeverityDisclosure function.
 * - AdaptiveSeverityDisclosureOutput - The return type for the adaptiveSeverityDisclosure function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const AdaptiveSeverityDisclosureInputSchema = z.object({
  prediction: z
    .enum(['positive', 'negative'])
    .describe('The AI prediction of schizophrenia diagnosis (positive or negative).'),
  confidence: z
    .number()
    .min(0)
    .max(1)
    .describe('The confidence score of the AI prediction, ranging from 0 to 1.'),
  emotionalState: z
    .string()
    .optional()
    .describe('The user\'s emotional state, if available (e.g., anxious, calm).'),
});
export type AdaptiveSeverityDisclosureInput = z.infer<typeof AdaptiveSeverityDisclosureInputSchema>;

const AdaptiveSeverityDisclosureOutputSchema = z.object({
  disclosureText: z
    .string()
    .describe(
      'The diagnostic disclosure text, which may or may not include information about severity.'
    ),
});
export type AdaptiveSeverityDisclosureOutput = z.infer<typeof AdaptiveSeverityDisclosureOutputSchema>;

export async function adaptiveSeverityDisclosure(
  input: AdaptiveSeverityDisclosureInput
): Promise<AdaptiveSeverityDisclosureOutput> {
  return adaptiveSeverityDisclosureFlow(input);
}

const adaptiveSeverityDisclosurePrompt = ai.definePrompt({
  name: 'adaptiveSeverityDisclosurePrompt',
  input: {schema: AdaptiveSeverityDisclosureInputSchema},
  output: {schema: AdaptiveSeverityDisclosureOutputSchema},
  prompt: `You are a compassionate AI assistant delivering diagnostic results for schizophrenia.

  Based on the AI prediction ({{prediction}}), the confidence score ({{confidence}}), and the user\'s emotional state ({{emotionalState}}), determine whether to include information about the potential severity of schizophrenia in the disclosure.

  - If the prediction is negative, provide a reassuring message without mentioning severity.
  - If the prediction is positive and the confidence is high (>= 0.8):
    - If the user reports feeling anxious, provide the results using gentle language.
    - If the user reports feeling calm, provide straightforward results.
  - If the prediction is positive and the confidence is moderate (0.5 - 0.8), avoid mentioning severity regardless of emotional state.
  - If the prediction is positive and the confidence is low (< 0.5), focus on the uncertainty and recommend further evaluation.

  Craft a disclosure text that is sensitive, personalized, and psychologically safe.`,
});

const adaptiveSeverityDisclosureFlow = ai.defineFlow(
  {
    name: 'adaptiveSeverityDisclosureFlow',
    inputSchema: AdaptiveSeverityDisclosureInputSchema,
    outputSchema: AdaptiveSeverityDisclosureOutputSchema,
  },
  async input => {
    const {output} = await adaptiveSeverityDisclosurePrompt(input);
    return output!;
  }
);
