import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';

export default function ResourcesPage() {
  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      <div>
        <h1 className="text-2xl md:text-3xl font-bold tracking-tight font-headline">
          Educational Resources
        </h1>
        <p className="text-muted-foreground mt-1">
          Understanding the science behind MindBloom in approachable language.
        </p>
      </div>

      <Accordion type="single" collapsible className="w-full">
        <AccordionItem value="item-1">
          <AccordionTrigger className="text-lg">
            What is an EEG?
          </AccordionTrigger>
          <AccordionContent className="text-base text-muted-foreground leading-relaxed">
            An electroencephalogram (EEG) is a test that detects electrical activity in your brain using small, metal discs (electrodes) attached to your scalp. Your brain cells communicate via electrical impulses and are active all the time, even when you're asleep. This activity shows up as wavy lines on an EEG recording. It's a non-invasive and painless way to get a glimpse into the brain's functional state.
          </AccordionContent>
        </AccordionItem>
        <AccordionItem value="item-2">
          <AccordionTrigger className="text-lg">
            How does MindBloom analyze EEG data?
          </AccordionTrigger>
          <AccordionContent className="text-base text-muted-foreground leading-relaxed">
            MindBloom uses a simulated advanced machine learning model, specifically a Random Forest classifier, to analyze neurophysiological features from the EEG recording. This includes looking at patterns in different brainwave frequencies (like delta, theta, alpha, beta, and gamma waves). The model has been trained on a large dataset of EEG recordings to identify subtle patterns that may be associated with certain neurological conditions. It provides a prediction based on these patterns, along with a confidence score.
          </AccordionContent>
        </AccordionItem>
        <AccordionItem value="item-3">
          <AccordionTrigger className="text-lg">
            What does a "prediction" mean?
          </AccordionTrigger>
          <AccordionContent className="text-base text-muted-foreground leading-relaxed">
            The prediction from our tool is not a medical diagnosis. It is an indication that the patterns in your EEG data show similarities to patterns commonly found in individuals with a specific condition. A "positive" prediction ("Markers Detected") suggests that you should consult with a healthcare professional for a comprehensive evaluation. A "negative" prediction ("No Markers Detected") is reassuring, but does not rule out the possibility of a condition. All results should be discussed with a qualified doctor.
          </AccordionContent>
        </AccordionItem>
        <AccordionItem value="item-4">
          <AccordionTrigger className="text-lg">
            Why is psychological safety important?
          </AccordionTrigger>
          <AccordionContent className="text-base text-muted-foreground leading-relaxed">
            Psychological safety is the belief that you won't be punished or humiliated for speaking up with ideas, questions, concerns, or mistakes. In the context of mental health, it means creating an environment that feels safe, respectful, and non-judgmental. We designed MindBloom with this in mind, using gentle language, calming visuals, and avoiding alarming or stressful elements. Our goal is to provide information in a way that empowers you without causing distress.
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </div>
  );
}
