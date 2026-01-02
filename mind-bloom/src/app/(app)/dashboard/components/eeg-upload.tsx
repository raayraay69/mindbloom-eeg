'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { UploadCloud, File, X, BrainCircuit } from 'lucide-react';
import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { useToast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';
import { performAnalysis } from '../actions';
import type { AnalysisResult } from '@/lib/types';
import { AnalysisView } from './analysis-view';
import { Button } from '@/components/ui/button';

export function EegUpload() {
  const [file, setFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { toast } = useToast();

  const resetState = () => {
    setFile(null);
    setUploadProgress(0);
    setIsUploading(false);
    setIsProcessing(false);
    setAnalysisResult(null);
    setError(null);
  }

  const simulateUpload = () => {
    setIsUploading(true);
    setError(null);
    let progress = 0;
    const interval = setInterval(() => {
      progress += Math.random() * 10;
      if (progress >= 100) {
        setUploadProgress(100);
        clearInterval(interval);
        setIsUploading(false);
        handleProcessing();
      } else {
        setUploadProgress(progress);
      }
    }, 300);
  };
  
  const handleProcessing = async () => {
    setIsProcessing(true);
    try {
      const result = await performAnalysis();
      setAnalysisResult(result);
    } catch (e) {
      setError('An error occurred during analysis. Please try again.');
      toast({
        title: 'Analysis Failed',
        description: 'Could not process the EEG file. Please try again.',
        variant: 'destructive',
      });
    } finally {
      setIsProcessing(false);
    }
  }

  const onDrop = useCallback((acceptedFiles: File[], fileRejections: any[]) => {
    resetState();
    if (fileRejections.length > 0) {
      const message = fileRejections[0].errors[0].message;
      setError(message);
      toast({
        title: 'File Upload Error',
        description: message,
        variant: 'destructive',
      });
      return;
    }

    if (acceptedFiles.length > 0) {
      setFile(acceptedFiles[0]);
      simulateUpload();
    }
  }, [toast]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/octet-stream': ['.edf'] },
    maxSize: 500 * 1024 * 1024, // 500MB
    multiple: false,
  });

  const removeFile = () => {
    resetState();
  };
  
  if (analysisResult) {
    return <AnalysisView result={analysisResult} onReset={resetState} />;
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
            <BrainCircuit className="w-6 h-6"/>
            EEG Analysis
        </CardTitle>
        <CardDescription>
          Upload a 16-channel EEG recording in .EDF format. Max file size: 500MB.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {isUploading || isProcessing || file ? (
          <div className="space-y-4">
            <div className="flex items-center justify-between p-3 rounded-md border bg-muted/50">
              <div className="flex items-center gap-3">
                <File className="w-6 h-6 text-primary" />
                <span className="font-medium text-sm truncate max-w-[200px] sm:max-w-xs md:max-w-md">{file?.name}</span>
              </div>
              <Button variant="ghost" size="icon" className="w-6 h-6" onClick={removeFile} disabled={isUploading || isProcessing}>
                <X className="w-4 h-4" />
              </Button>
            </div>
            {(isUploading || uploadProgress > 0) && !isProcessing && (
              <div>
                <Progress value={uploadProgress} className="h-2 transition-all duration-300 ease-linear" />
                <p className="text-sm text-muted-foreground mt-2">Uploading...</p>
              </div>
            )}
             {isProcessing && (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                 <div className="w-4 h-4 border-2 border-primary/20 border-t-primary rounded-full animate-spin"></div>
                Analyzing recording, please wait. This may take a moment.
              </div>
            )}
          </div>
        ) : (
          <div
            {...getRootProps()}
            className={cn(
              'border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-colors',
              isDragActive ? 'border-primary bg-primary/10' : 'border-border hover:border-primary/50'
            )}
          >
            <input {...getInputProps()} />
            <div className="flex flex-col items-center gap-2 text-muted-foreground">
              <UploadCloud className="w-12 h-12" />
              <p className="font-semibold">
                {isDragActive ? 'Drop the file here' : 'Drag & drop file here, or click to select'}
              </p>
              <p className="text-xs">.EDF format, up to 500MB</p>
            </div>
          </div>
        )}
        {error && <p className="mt-4 text-sm text-destructive">{error}</p>}
      </CardContent>
    </Card>
  );
}
