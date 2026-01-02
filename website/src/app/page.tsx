import { Button } from '@/components/ui/button';
import Link from 'next/link';
import { Leaf, Brain, Upload, BarChart3, Shield, FlaskConical, Users, DollarSign } from 'lucide-react';

export default function LandingPage() {
  return (
    <div className="flex flex-col min-h-screen">
      {/* Header */}
      <header className="px-4 lg:px-6 h-16 flex items-center border-b">
        <Link href="/" className="flex items-center gap-2" prefetch={false}>
          <Leaf className="h-6 w-6 text-primary" />
          <span className="font-headline font-semibold text-lg">MindBloom</span>
        </Link>
        <nav className="ml-auto flex gap-4">
          <Link href="/science" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
            Science
          </Link>
          <Link href="/login" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
            Login
          </Link>
        </nav>
      </header>

      <main className="flex-1">
        {/* Hero Section */}
        <section className="py-20 px-4 sm:px-6 lg:px-8">
          <div className="max-w-4xl mx-auto text-center space-y-6">
            <div className="inline-flex items-center gap-2 bg-primary/10 text-primary px-4 py-1.5 rounded-full text-sm font-medium">
              <FlaskConical className="h-4 w-4" />
              Research-Validated Technology
            </div>
            <h1 className="text-4xl font-bold tracking-tight sm:text-5xl md:text-6xl font-headline">
              EEG-Based Schizophrenia Screening
            </h1>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              83.7% accuracy validated on 1,931 recordings. Accessible screening technology with a trauma-informed approach.
            </p>
            <div className="flex flex-col sm:flex-row gap-3 justify-center pt-4">
              <Button asChild size="lg" className="text-base">
                <Link href="/scan">Try EEG Analysis</Link>
              </Button>
              <Button asChild variant="outline" size="lg" className="text-base">
                <Link href="/science">View Research</Link>
              </Button>
            </div>
          </div>
        </section>

        {/* Stats Section */}
        <section className="py-16 px-4 sm:px-6 lg:px-8 bg-muted/50">
          <div className="max-w-5xl mx-auto">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
              <div className="text-center">
                <div className="text-3xl sm:text-4xl font-bold text-primary">83.7%</div>
                <div className="text-sm text-muted-foreground mt-1">Accuracy</div>
              </div>
              <div className="text-center">
                <div className="text-3xl sm:text-4xl font-bold text-primary">93.4%</div>
                <div className="text-sm text-muted-foreground mt-1">Sensitivity</div>
              </div>
              <div className="text-center">
                <div className="text-3xl sm:text-4xl font-bold text-primary">153</div>
                <div className="text-sm text-muted-foreground mt-1">Subjects Validated</div>
              </div>
              <div className="text-center">
                <div className="text-3xl sm:text-4xl font-bold text-primary">1,931</div>
                <div className="text-sm text-muted-foreground mt-1">EEG Recordings</div>
              </div>
            </div>
          </div>
        </section>

        {/* How It Works */}
        <section className="py-20 px-4 sm:px-6 lg:px-8">
          <div className="max-w-5xl mx-auto">
            <h2 className="text-2xl sm:text-3xl font-bold text-center mb-12 font-headline">How It Works</h2>
            <div className="grid md:grid-cols-3 gap-8">
              <div className="text-center space-y-4">
                <div className="w-14 h-14 rounded-full bg-primary/10 flex items-center justify-center mx-auto">
                  <Upload className="h-6 w-6 text-primary" />
                </div>
                <h3 className="font-semibold text-lg">1. Upload EEG</h3>
                <p className="text-muted-foreground text-sm">
                  Upload your EEG recording in EDF format. We support standard clinical formats.
                </p>
              </div>
              <div className="text-center space-y-4">
                <div className="w-14 h-14 rounded-full bg-primary/10 flex items-center justify-center mx-auto">
                  <Brain className="h-6 w-6 text-primary" />
                </div>
                <h3 className="font-semibold text-lg">2. AI Analysis</h3>
                <p className="text-muted-foreground text-sm">
                  Our model extracts 264 features across spectral, coherence, and complexity domains.
                </p>
              </div>
              <div className="text-center space-y-4">
                <div className="w-14 h-14 rounded-full bg-primary/10 flex items-center justify-center mx-auto">
                  <BarChart3 className="h-6 w-6 text-primary" />
                </div>
                <h3 className="font-semibold text-lg">3. Get Results</h3>
                <p className="text-muted-foreground text-sm">
                  Receive screening results with confidence scores and therapeutic recommendations.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Why Trust Us */}
        <section className="py-20 px-4 sm:px-6 lg:px-8 bg-muted/50">
          <div className="max-w-5xl mx-auto">
            <h2 className="text-2xl sm:text-3xl font-bold text-center mb-12 font-headline">Why Trust Our Research</h2>
            <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="bg-background rounded-lg p-6 space-y-3">
                <Shield className="h-8 w-8 text-primary" />
                <h3 className="font-semibold">Honest Metrics</h3>
                <p className="text-sm text-muted-foreground">
                  We report 83.7% accuracy, not inflated 90%+ claims from flawed methodology.
                </p>
              </div>
              <div className="bg-background rounded-lg p-6 space-y-3">
                <Users className="h-8 w-8 text-primary" />
                <h3 className="font-semibold">Public Dataset</h3>
                <p className="text-sm text-muted-foreground">
                  ASZED-153 dataset available on Zenodo for independent verification.
                </p>
              </div>
              <div className="bg-background rounded-lg p-6 space-y-3">
                <FlaskConical className="h-8 w-8 text-primary" />
                <h3 className="font-semibold">Rigorous Validation</h3>
                <p className="text-sm text-muted-foreground">
                  Subject-level cross-validation prevents identity leakage bias.
                </p>
              </div>
              <div className="bg-background rounded-lg p-6 space-y-3">
                <DollarSign className="h-8 w-8 text-primary" />
                <h3 className="font-semibold">Accessible Design</h3>
                <p className="text-sm text-muted-foreground">
                  ~$50 hardware prototype vs $5,000+ traditional clinical systems.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* CTA Section */}
        <section className="py-20 px-4 sm:px-6 lg:px-8">
          <div className="max-w-2xl mx-auto text-center space-y-6">
            <h2 className="text-2xl sm:text-3xl font-bold font-headline">Ready to Get Started?</h2>
            <p className="text-muted-foreground">
              Upload your EEG recording for analysis, or create an account to access therapeutic resources.
            </p>
            <div className="flex flex-col sm:flex-row gap-3 justify-center">
              <Button asChild size="lg">
                <Link href="/scan">Start Analysis</Link>
              </Button>
              <Button asChild variant="outline" size="lg">
                <Link href="/signup">Create Account</Link>
              </Button>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t py-8 px-4 sm:px-6 lg:px-8">
        <div className="max-w-5xl mx-auto flex flex-col sm:flex-row justify-between items-center gap-4">
          <div className="flex items-center gap-2">
            <Leaf className="h-5 w-5 text-primary" />
            <span className="text-sm text-muted-foreground">&copy; 2024 MindBloom. All rights reserved.</span>
          </div>
          <nav className="flex gap-6">
            <Link href="/science" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
              Science
            </Link>
            <Link href="#" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
              Privacy
            </Link>
            <Link href="#" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
              Terms
            </Link>
          </nav>
        </div>
        <div className="max-w-5xl mx-auto mt-4 pt-4 border-t">
          <p className="text-xs text-muted-foreground text-center">
            MindBloom is a screening tool, not a diagnostic device. Results should be reviewed by qualified healthcare professionals.
          </p>
        </div>
      </footer>
    </div>
  );
}
