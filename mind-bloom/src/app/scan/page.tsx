'use client';

import { useState, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import Link from 'next/link';
import { Leaf, Upload, FileUp, AlertCircle, CheckCircle2, Loader2, Brain } from 'lucide-react';

interface PredictionResult {
  success: boolean;
  prediction: string;
  probability: number;
  risk_level: string;
  confidence: number;
  channels_matched: number;
  recording_length_seconds: number;
  disclaimer: string;
}

export default function ScanPage() {
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && (droppedFile.name.endsWith('.edf') || droppedFile.name.endsWith('.bdf'))) {
      setFile(droppedFile);
      setError(null);
      setResult(null);
    } else {
      setError('Please upload an EDF or BDF file');
    }
  }, []);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      if (selectedFile.name.endsWith('.edf') || selectedFile.name.endsWith('.bdf')) {
        setFile(selectedFile);
        setError(null);
        setResult(null);
      } else {
        setError('Please upload an EDF or BDF file');
      }
    }
  }, []);

  const handleSubmit = async () => {
    if (!file) return;

    setIsLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || '/api';
      const response = await fetch(`${apiUrl}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to analyze EEG');
      }

      const data: PredictionResult = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const getRiskColor = (level: string) => {
    switch (level) {
      case 'Low': return 'text-green-600 bg-green-50 border-green-200';
      case 'Low-Moderate': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'Moderate-High': return 'text-orange-600 bg-orange-50 border-orange-200';
      case 'High': return 'text-red-600 bg-red-50 border-red-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  return (
    <div className="flex flex-col min-h-screen">
      <header className="px-4 lg:px-6 h-14 flex items-center border-b">
        <Link href="/" className="flex items-center justify-center gap-2" prefetch={false}>
          <Leaf className="h-6 w-6 text-primary" />
          <span className="font-semibold">MindBloom</span>
        </Link>
        <nav className="ml-auto flex gap-4">
          <Link href="/" className="text-sm text-muted-foreground hover:text-primary">
            Home
          </Link>
        </nav>
      </header>

      <main className="flex-1 container mx-auto px-4 py-8 max-w-3xl">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-primary/10 mb-4">
            <Brain className="h-8 w-8 text-primary" />
          </div>
          <h1 className="text-3xl font-bold tracking-tight mb-2">EEG Analysis</h1>
          <p className="text-muted-foreground">
            Upload your EEG recording for AI-powered schizophrenia screening analysis
          </p>
        </div>

        {/* Upload Area */}
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`
            border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer
            ${isDragging ? 'border-primary bg-primary/5' : 'border-muted-foreground/25 hover:border-primary/50'}
            ${file ? 'border-green-500 bg-green-50' : ''}
          `}
          onClick={() => document.getElementById('file-input')?.click()}
        >
          <input
            id="file-input"
            type="file"
            accept=".edf,.bdf"
            onChange={handleFileSelect}
            className="hidden"
          />

          {file ? (
            <div className="flex flex-col items-center gap-2">
              <CheckCircle2 className="h-12 w-12 text-green-500" />
              <p className="font-medium">{file.name}</p>
              <p className="text-sm text-muted-foreground">
                {(file.size / 1024 / 1024).toFixed(2)} MB
              </p>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-2">
              <Upload className="h-12 w-12 text-muted-foreground" />
              <p className="font-medium">Drop your EEG file here</p>
              <p className="text-sm text-muted-foreground">
                or click to browse (EDF/BDF format)
              </p>
            </div>
          )}
        </div>

        {/* Error Message */}
        {error && (
          <div className="mt-4 p-4 rounded-lg bg-red-50 border border-red-200 flex items-center gap-2 text-red-700">
            <AlertCircle className="h-5 w-5 flex-shrink-0" />
            <p>{error}</p>
          </div>
        )}

        {/* Analyze Button */}
        <div className="mt-6 flex justify-center">
          <Button
            size="lg"
            onClick={handleSubmit}
            disabled={!file || isLoading}
            className="min-w-[200px]"
          >
            {isLoading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <FileUp className="mr-2 h-4 w-4" />
                Analyze EEG
              </>
            )}
          </Button>
        </div>

        {/* Results */}
        {result && (
          <div className="mt-8 space-y-4">
            <h2 className="text-xl font-semibold text-center">Analysis Results</h2>

            <div className={`p-6 rounded-lg border-2 ${getRiskColor(result.risk_level)}`}>
              <div className="text-center">
                <p className="text-2xl font-bold mb-2">{result.prediction}</p>
                <p className="text-lg">
                  Risk Level: <span className="font-semibold">{result.risk_level}</span>
                </p>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 rounded-lg bg-muted">
                <p className="text-sm text-muted-foreground">Probability Score</p>
                <p className="text-2xl font-bold">{(result.probability * 100).toFixed(1)}%</p>
              </div>
              <div className="p-4 rounded-lg bg-muted">
                <p className="text-sm text-muted-foreground">Confidence</p>
                <p className="text-2xl font-bold">{(result.confidence * 100).toFixed(1)}%</p>
              </div>
              <div className="p-4 rounded-lg bg-muted">
                <p className="text-sm text-muted-foreground">Channels Matched</p>
                <p className="text-2xl font-bold">{result.channels_matched}/16</p>
              </div>
              <div className="p-4 rounded-lg bg-muted">
                <p className="text-sm text-muted-foreground">Recording Length</p>
                <p className="text-2xl font-bold">{result.recording_length_seconds}s</p>
              </div>
            </div>

            <div className="p-4 rounded-lg bg-amber-50 border border-amber-200">
              <div className="flex gap-2">
                <AlertCircle className="h-5 w-5 text-amber-600 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="font-medium text-amber-800">Important Disclaimer</p>
                  <p className="text-sm text-amber-700 mt-1">{result.disclaimer}</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Info Section */}
        <div className="mt-12 p-6 rounded-lg bg-muted/50">
          <h3 className="font-semibold mb-2">About This Tool</h3>
          <p className="text-sm text-muted-foreground mb-4">
            This screening tool uses a machine learning model trained on the ASZED-153 dataset
            to analyze EEG recordings for patterns associated with schizophrenia. The model
            achieved 83.7% accuracy in clinical validation with strict subject-level cross-validation.
          </p>
          <ul className="text-sm text-muted-foreground space-y-1">
            <li>• Supports standard 10-20 EEG montage (16 channels)</li>
            <li>• Accepts EDF and BDF file formats</li>
            <li>• Analyzes spectral power, coherence, and complexity features</li>
            <li>• Results are for screening purposes only</li>
          </ul>
        </div>
      </main>

      <footer className="border-t py-6 px-4 text-center">
        <p className="text-xs text-muted-foreground">
          &copy; 2024 MindBloom. This tool is for research and screening purposes only.
        </p>
      </footer>
    </div>
  );
}
