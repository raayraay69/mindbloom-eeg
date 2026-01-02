'use client';

import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Label } from '@/components/ui/label';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { useToast } from '@/hooks/use-toast';
import type { AnalysisResult } from '@/lib/types';
import { cn, formatProcessingTime } from '@/lib/utils';
import { CheckCircle, Download, Info, RefreshCcw, Smile, Frown, Meh, ChevronDown } from 'lucide-react';
import { useState, useMemo } from 'react';
import { getDisclosure } from '../actions';
import { ChartContainer, ChartTooltip, ChartTooltipContent } from '@/components/ui/chart';
import { Pie, PieChart, Cell } from 'recharts';
import { ValidationDetailsView } from './validation-details';

type Props = {
  result: AnalysisResult;
  onReset: () => void;
};

type EmotionalState = 'calm' | 'anxious' | 'neutral';

export function AnalysisView({ result, onReset }: Props) {
  const [emotionalState, setEmotionalState] = useState<EmotionalState | null>(null);
  const [disclosure, setDisclosure] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [showValidation, setShowValidation] = useState(false);
  const { toast } = useToast();

  const chartData = useMemo(() => [
    { name: 'Confidence', value: result.confidence, fill: 'hsl(var(--primary))' },
    { name: 'Uncertainty', value: 1 - result.confidence, fill: 'hsl(var(--muted))' },
  ], [result.confidence]);

  const handleGetDisclosure = async () => {
    if (!emotionalState) {
      toast({
        title: 'Please select your current feeling',
        variant: 'destructive',
      });
      return;
    }
    setIsLoading(true);
    try {
      const response = await getDisclosure({
        prediction: result.prediction,
        confidence: result.confidence,
        emotionalState: emotionalState,
      });
      setDisclosure(response.disclosureText);
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Could not generate personalized message. Please try again.',
        variant: 'destructive',
      });
    } finally {
      setIsLoading(false);
    }
  };
  
  const handleDownloadReport = () => {
    toast({
        title: "Report Download Started",
        description: "Your PDF report is being generated and will download shortly.",
    });
    // In a real app, this would trigger a PDF generation service.
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
            <CheckCircle className="w-6 h-6 text-green-600"/>
            Analysis Complete
        </CardTitle>
        <CardDescription>
          Your EEG recording has been analyzed. Results are presented below.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="grid md:grid-cols-2 gap-6">
          <Card className="bg-muted/30">
            <CardHeader>
              <CardTitle className="text-lg">Summary</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
               <div className="flex justify-between items-center">
                <span className="text-sm font-medium text-muted-foreground">Prediction</span>
                <span className={cn(
                    "font-semibold px-2 py-1 rounded-full text-sm",
                    result.prediction === 'positive' ? 'bg-orange-100 text-orange-800' : 'bg-green-100 text-green-800'
                )}>
                  {result.prediction === 'positive' ? 'Markers Detected' : 'No Markers Detected'}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium text-muted-foreground">Processing Time</span>
                <span className="font-semibold text-sm">{formatProcessingTime(result.processingTime)}</span>
              </div>
            </CardContent>
          </Card>
          <Card className="bg-muted/30 flex flex-col items-center justify-center">
            <CardHeader className="items-center">
                <CardTitle className="text-lg">Confidence Score</CardTitle>
            </CardHeader>
            <CardContent>
                <div className="w-full h-24 relative">
                    <ChartContainer config={{}} className="w-full h-full">
                        <PieChart accessibilityLayer>
                            <Pie data={chartData} dataKey="value" nameKey="name" innerRadius={35} outerRadius={45} startAngle={90} endAngle={450}>
                                {chartData.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={entry.fill} />
                                ))}
                            </Pie>
                        </PieChart>
                    </ChartContainer>
                    <div className="absolute inset-0 flex items-center justify-center">
                        <span className="text-2xl font-bold text-primary">
                            {Math.round(result.confidence * 100)}%
                        </span>
                    </div>
                </div>
            </CardContent>
          </Card>
        </div>

        {/* Validation Details Section */}
        {result.validation && (
          <Collapsible open={showValidation} onOpenChange={setShowValidation}>
            <Card className="bg-muted/20">
              <CollapsibleTrigger asChild>
                <CardHeader className="cursor-pointer hover:bg-muted/30 transition-colors">
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle className="text-lg">Data Quality & Validation Details</CardTitle>
                      <CardDescription>
                        View channel status, signal quality, and validation information
                      </CardDescription>
                    </div>
                    <ChevronDown className={cn(
                      "w-5 h-5 transition-transform",
                      showValidation && "rotate-180"
                    )} />
                  </div>
                </CardHeader>
              </CollapsibleTrigger>
              <CollapsibleContent>
                <CardContent>
                  <ValidationDetailsView validation={result.validation} />
                </CardContent>
              </CollapsibleContent>
            </Card>
          </Collapsible>
        )}

        {!disclosure ? (
          <Card className="bg-muted/30 border-primary/50 border">
            <CardHeader>
              <CardTitle className="text-lg">Personalized Results</CardTitle>
              <CardDescription>To present your results in the most comfortable way, please tell us how you are feeling right now.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <RadioGroup onValueChange={(value: EmotionalState) => setEmotionalState(value)} className="flex flex-col sm:flex-row gap-4">
                <Label htmlFor="state-calm" className="flex-1 flex items-center gap-3 rounded-md border p-4 hover:bg-background cursor-pointer has-[input:checked]:border-primary has-[input:checked]:bg-primary/10">
                  <RadioGroupItem value="calm" id="state-calm" />
                  <Smile className="w-5 h-5 text-green-600"/>
                  Calm
                </Label>
                <Label htmlFor="state-neutral" className="flex-1 flex items-center gap-3 rounded-md border p-4 hover:bg-background cursor-pointer has-[input:checked]:border-primary has-[input:checked]:bg-primary/10">
                  <RadioGroupItem value="neutral" id="state-neutral" />
                  <Meh className="w-5 h-5 text-yellow-600"/>
                  Neutral
                </Label>
                 <Label htmlFor="state-anxious" className="flex-1 flex items-center gap-3 rounded-md border p-4 hover:bg-background cursor-pointer has-[input:checked]:border-primary has-[input:checked]:bg-primary/10">
                  <RadioGroupItem value="anxious" id="state-anxious" />
                  <Frown className="w-5 h-5 text-red-600"/>
                  Anxious
                </Label>
              </RadioGroup>
              <Button onClick={handleGetDisclosure} disabled={isLoading}>
                {isLoading ? 'Generating...' : 'View My Results'}
              </Button>
            </CardContent>
          </Card>
        ) : (
          <Alert>
            <Info className="h-4 w-4" />
            <AlertTitle>Your Personalized Disclosure</AlertTitle>
            <AlertDescription className="prose prose-sm max-w-none text-foreground">
              <p>{disclosure}</p>
              <p className="mt-2 text-xs text-muted-foreground">This information is not a diagnosis. It is important to discuss these results with a qualified healthcare professional.</p>
            </AlertDescription>
          </Alert>
        )}

        <div className="flex flex-col sm:flex-row gap-2 pt-4 border-t">
          <Button onClick={handleDownloadReport} disabled={!disclosure}>
            <Download className="mr-2 h-4 w-4"/>
            Download Report
          </Button>
          <Button variant="outline" onClick={onReset}>
            <RefreshCcw className="mr-2 h-4 w-4"/>
            Start New Session
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
