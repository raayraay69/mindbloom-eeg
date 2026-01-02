# **App Name**: MindBloom

## Core Features:

- Secure EEG Upload & Processing: Allows secure uploading of EEG recordings. On upload to Firebase Storage, it triggers a Cloud Function that simulates the Random Forest classifier, and returns a JSON output using trauma-informed language. Keeps protected health information secure according to HIPAA rules.
- AI Diagnostic Prediction Tool: Analyzes uploaded EEG data using an AI Random Forest classifier to predict the likelihood of schizophrenia and provide confidence scores. The LLM acts as a tool when deciding when and if to include information about severity of schizophrenia in its outputs.
- Therapeutic Content Hub: Provides a categorized and searchable media gallery with therapeutic content such as mobile apps, VR experiences, AI-guided exercises, and ambient soundscapes, with metadata for platform compatibility and accessibility features.
- Session History Visualization: For returning users, provides a session history with trend visualization to track progress and changes over time.
- Downloadable Report: Generates a PDF report based on the findings with clear next-steps guidance and confidence score visualization, all using gentle and reassuring language.
- Firebase Authentication: Uses Firebase Authentication for secure user management, offering both email/password login and optional anonymous sessions for privacy-conscious users.
- Admin Content Management: Provides an admin panel for managing and updating therapeutic content within the app.

## Style Guidelines:

- Primary color: Soft sage green (#A7C494) to promote a sense of calm and well-being.
- Background color: Warm, muted beige (#F5F5DC) to create a comforting and neutral backdrop.
- Accent color: Gentle blue (#B0E2FF) used sparingly to highlight key interactive elements.
- Body and headline font: 'Poppins', a rounded sans-serif font, for a friendly and approachable feel.
- Use soft, rounded icons with a line style to represent content categories and actions. Avoid sharp edges.
- Ensure sufficient whitespace and padding around elements to reduce visual clutter. Use a grid-based layout for consistency.
- Implement subtle, slow transitions (300-500ms easing) and animations to enhance user experience without causing distress. Respect `prefers-reduced-motion` setting.