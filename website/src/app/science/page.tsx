import { Button } from '@/components/ui/button';
import Link from 'next/link';
import { Leaf, ArrowLeft, Brain, Activity, Cpu, CheckCircle2, AlertTriangle, ExternalLink } from 'lucide-react';

export default function SciencePage() {
  return (
    <div className="flex flex-col min-h-screen">
      {/* Header */}
      <header className="px-4 lg:px-6 h-16 flex items-center border-b">
        <Link href="/" className="flex items-center gap-2" prefetch={false}>
          <Leaf className="h-6 w-6 text-primary" />
          <span className="font-headline font-semibold text-lg">MindBloom</span>
        </Link>
        <nav className="ml-auto flex gap-4">
          <Link href="/" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
            Home
          </Link>
          <Link href="/login" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
            Login
          </Link>
        </nav>
      </header>

      <main className="flex-1">
        {/* Hero */}
        <section className="py-16 px-4 sm:px-6 lg:px-8 border-b">
          <div className="max-w-3xl mx-auto">
            <Link href="/" className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground mb-6">
              <ArrowLeft className="h-4 w-4" />
              Back to Home
            </Link>
            <h1 className="text-3xl sm:text-4xl font-bold font-headline mb-4">
              The Science Behind MindBloom
            </h1>
            <p className="text-lg text-muted-foreground">
              A system-level framework for EEG-based schizophrenia assessment with methodological rigor,
              uncertainty quantification, and hardware feasibility.
            </p>
          </div>
        </section>

        {/* Research Overview */}
        <section className="py-12 px-4 sm:px-6 lg:px-8">
          <div className="max-w-3xl mx-auto space-y-8">
            <div>
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Brain className="h-5 w-5 text-primary" />
                Research Overview
              </h2>
              <p className="text-muted-foreground mb-4">
                Our research addresses a critical problem in medical AI: the inflation of accuracy claims through
                methodological shortcuts. Many published studies report 75-99% accuracy by inadvertently allowing
                the same patient&apos;s data in both training and testing sets—a flaw called &quot;identity leakage.&quot;
              </p>
              <p className="text-muted-foreground">
                We prioritize honest metrics over impressive numbers, establishing a reproducible, clinically-viable
                baseline for EEG-based schizophrenia screening.
              </p>
            </div>

            {/* Key Metrics */}
            <div className="bg-muted/50 rounded-lg p-6">
              <h3 className="font-semibold mb-4">Performance Metrics</h3>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                <div>
                  <div className="text-2xl font-bold text-primary">83.7%</div>
                  <div className="text-sm text-muted-foreground">Accuracy</div>
                  <div className="text-xs text-muted-foreground">95% CI: 77.8-89.5%</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-primary">93.4%</div>
                  <div className="text-sm text-muted-foreground">Sensitivity</div>
                  <div className="text-xs text-muted-foreground">71/76 cases detected</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-primary">74.0%</div>
                  <div className="text-sm text-muted-foreground">Specificity</div>
                  <div className="text-xs text-muted-foreground">57/77 controls correct</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-primary">0.869</div>
                  <div className="text-sm text-muted-foreground">ROC-AUC</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-primary">153</div>
                  <div className="text-sm text-muted-foreground">Subjects</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-primary">1,931</div>
                  <div className="text-sm text-muted-foreground">Recordings</div>
                </div>
              </div>
            </div>

            {/* The Inflation Problem */}
            <div>
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <AlertTriangle className="h-5 w-5 text-primary" />
                The Inflation Problem
              </h2>
              <p className="text-muted-foreground mb-4">
                We quantified how identity leakage inflates performance claims:
              </p>
              <div className="bg-muted/50 rounded-lg p-6 space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">Recording-level accuracy (naive)</span>
                  <span className="font-mono font-semibold">90.9%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">Subject-level accuracy (correct)</span>
                  <span className="font-mono font-semibold">83.7%</span>
                </div>
                <div className="flex justify-between items-center border-t pt-3">
                  <span className="text-muted-foreground">Inflation gap</span>
                  <span className="font-mono font-semibold text-destructive">+7.2%</span>
                </div>
              </div>
              <p className="text-sm text-muted-foreground mt-4">
                This 7.2 percentage point gap represents artificial inflation present in many published studies.
                Our subject-level validation ensures no patient appears in both training and test sets.
              </p>
            </div>

            {/* Feature Engineering */}
            <div>
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Activity className="h-5 w-5 text-primary" />
                Feature Engineering
              </h2>
              <p className="text-muted-foreground mb-4">
                We extract 264 features from each EEG recording across multiple domains:
              </p>
              <div className="grid sm:grid-cols-2 gap-4">
                <div className="border rounded-lg p-4">
                  <h4 className="font-medium mb-2">Spectral Power (80 features)</h4>
                  <p className="text-sm text-muted-foreground">
                    Power across delta, theta, alpha, beta, and gamma frequency bands for each channel.
                  </p>
                </div>
                <div className="border rounded-lg p-4">
                  <h4 className="font-medium mb-2">Coherence (30 features)</h4>
                  <p className="text-sm text-muted-foreground">
                    Inter-channel coherence measuring functional connectivity between brain regions.
                  </p>
                </div>
                <div className="border rounded-lg p-4">
                  <h4 className="font-medium mb-2">Complexity (32 features)</h4>
                  <p className="text-sm text-muted-foreground">
                    Sample entropy and fractal dimension capturing nonlinear signal dynamics.
                  </p>
                </div>
                <div className="border rounded-lg p-4">
                  <h4 className="font-medium mb-2">Statistical (96 features)</h4>
                  <p className="text-sm text-muted-foreground">
                    Higher-order moments and temporal statistics for each channel.
                  </p>
                </div>
              </div>
            </div>

            {/* Biological Validation */}
            <div>
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <CheckCircle2 className="h-5 w-5 text-primary" />
                Biological Validation
              </h2>
              <p className="text-muted-foreground mb-4">
                The top predictive features align with established schizophrenia neuroscience:
              </p>
              <ul className="space-y-2">
                <li className="flex items-start gap-2">
                  <CheckCircle2 className="h-4 w-4 text-primary mt-1 flex-shrink-0" />
                  <span className="text-muted-foreground">
                    <strong className="text-foreground">Frontal channel dominance</strong> — Fp1, Fp2, F3 features rank highest,
                    consistent with prefrontal abnormalities in schizophrenia.
                  </span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle2 className="h-4 w-4 text-primary mt-1 flex-shrink-0" />
                  <span className="text-muted-foreground">
                    <strong className="text-foreground">Theta power significance</strong> — Fp1 theta power is the top feature,
                    aligning with documented theta band abnormalities.
                  </span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle2 className="h-4 w-4 text-primary mt-1 flex-shrink-0" />
                  <span className="text-muted-foreground">
                    <strong className="text-foreground">Connectivity markers</strong> — Fp1-Fp2 coherence reflects known
                    interhemispheric connectivity disruptions.
                  </span>
                </li>
              </ul>
            </div>

            {/* Hardware Feasibility */}
            <div>
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Cpu className="h-5 w-5 text-primary" />
                Hardware Accessibility
              </h2>
              <p className="text-muted-foreground mb-4">
                Our research validates that a single frontal channel carries sufficient discriminative power,
                enabling deployment in resource-limited settings.
              </p>
              <div className="bg-muted/50 rounded-lg p-6">
                <div className="grid sm:grid-cols-2 gap-6">
                  <div>
                    <div className="text-sm text-muted-foreground mb-1">Traditional Clinical EEG</div>
                    <div className="text-2xl font-bold">$5,000 - $50,000</div>
                  </div>
                  <div>
                    <div className="text-sm text-muted-foreground mb-1">Our Hardware Prototype</div>
                    <div className="text-2xl font-bold text-primary">~$50</div>
                  </div>
                </div>
                <div className="mt-4 pt-4 border-t text-sm text-muted-foreground">
                  Components: ESP32 microcontroller ($5), BioAmp EXG Pill ($25), dry Ag/AgCl electrodes,
                  single-channel acquisition at 256 Hz.
                </div>
              </div>
            </div>

            {/* Dataset */}
            <div>
              <h2 className="text-xl font-semibold mb-4">Dataset: ASZED-153</h2>
              <p className="text-muted-foreground mb-4">
                Our research uses the publicly available ASZED-153 dataset for full transparency and reproducibility.
              </p>
              <ul className="space-y-2 text-muted-foreground">
                <li>• 153 subjects (77 healthy controls, 76 patients)</li>
                <li>• 16-channel EEG following international 10-20 system</li>
                <li>• 256 Hz sampling rate</li>
                <li>• Multiple paradigms: resting-state, cognitive tasks, MMN, ASSR</li>
                <li>• Available on Zenodo (DOI: 10.5281/zenodo.14178398)</li>
              </ul>
            </div>

            {/* Disclaimer */}
            <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-6">
              <h3 className="font-semibold text-destructive mb-2">Important Disclaimer</h3>
              <p className="text-sm text-muted-foreground">
                MindBloom is a screening tool designed to assist healthcare professionals. It is not a diagnostic
                device and should not replace clinical evaluation. The 93.4% sensitivity makes it suitable for
                initial screening, but all results should be reviewed by qualified medical professionals.
                False positives are expected and resolved through follow-up clinical assessment.
              </p>
            </div>

            {/* CTA */}
            <div className="text-center pt-8">
              <Button asChild size="lg">
                <Link href="/scan">Try EEG Analysis</Link>
              </Button>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t py-8 px-4 sm:px-6 lg:px-8">
        <div className="max-w-3xl mx-auto flex flex-col sm:flex-row justify-between items-center gap-4">
          <div className="flex items-center gap-2">
            <Leaf className="h-5 w-5 text-primary" />
            <span className="text-sm text-muted-foreground">&copy; 2024 MindBloom. All rights reserved.</span>
          </div>
          <nav className="flex gap-6">
            <Link href="/" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
              Home
            </Link>
            <Link href="#" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
              Privacy
            </Link>
          </nav>
        </div>
      </footer>
    </div>
  );
}
