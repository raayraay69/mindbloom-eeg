'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { AlertCircle, CheckCircle, XCircle, Signal, Activity, Filter } from 'lucide-react';
import type { ValidationDetails, ChannelStatus } from '@/lib/types';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';

interface ValidationDetailsProps {
  validation: ValidationDetails;
}

export function ValidationDetailsView({ validation }: ValidationDetailsProps) {
  const qualityPercentage = Math.round(validation.signal_quality.overall_score * 100);

  // Determine quality color
  const getQualityColor = (score: number) => {
    if (score >= 0.7) return 'text-green-600';
    if (score >= 0.4) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getChannelStatusIcon = (channel: ChannelStatus) => {
    if (!channel.found || channel.is_zero) {
      return <XCircle className="w-4 h-4 text-gray-400" />;
    }
    if (channel.is_noisy) {
      return <AlertCircle className="w-4 h-4 text-yellow-600" />;
    }
    if ((channel.quality_score ?? 0) >= 0.7) {
      return <CheckCircle className="w-4 h-4 text-green-600" />;
    }
    return <AlertCircle className="w-4 h-4 text-yellow-600" />;
  };

  const getChannelStatusColor = (channel: ChannelStatus) => {
    if (!channel.found || channel.is_zero) return 'bg-gray-100 text-gray-600';
    if (channel.is_noisy) return 'bg-yellow-100 text-yellow-800';
    if ((channel.quality_score ?? 0) >= 0.7) return 'bg-green-100 text-green-800';
    return 'bg-yellow-100 text-yellow-800';
  };

  return (
    <div className="space-y-4">
      {/* Validation Errors */}
      {validation.validation_errors.length > 0 && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Validation Errors</AlertTitle>
          <AlertDescription>
            <ul className="list-disc list-inside space-y-1 mt-2">
              {validation.validation_errors.map((error, idx) => (
                <li key={idx}>{error}</li>
              ))}
            </ul>
          </AlertDescription>
        </Alert>
      )}

      {/* Validation Warnings */}
      {validation.validation_warnings.length > 0 && (
        <Alert>
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Validation Warnings</AlertTitle>
          <AlertDescription>
            <ul className="list-disc list-inside space-y-1 mt-2">
              {validation.validation_warnings.map((warning, idx) => (
                <li key={idx} className="text-sm">{warning}</li>
              ))}
            </ul>
          </AlertDescription>
        </Alert>
      )}

      {/* Overall Signal Quality */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            <Signal className="w-5 h-5" />
            Overall Signal Quality
          </CardTitle>
          <CardDescription>
            Quality assessment based on SNR, noise levels, and channel coverage
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <div className="flex justify-between mb-2">
              <span className="text-sm font-medium">Quality Score</span>
              <span className={`text-sm font-bold ${getQualityColor(validation.signal_quality.overall_score)}`}>
                {qualityPercentage}%
              </span>
            </div>
            <Progress value={qualityPercentage} className="h-2" />
          </div>

          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <div className="text-muted-foreground">Channels Found</div>
              <div className="font-semibold">
                {validation.signal_quality.channels_found}/{validation.signal_quality.channels_expected}
              </div>
            </div>
            <div>
              <div className="text-muted-foreground">Zero Channels</div>
              <div className="font-semibold">{validation.signal_quality.zero_channels}</div>
            </div>
            <div>
              <div className="text-muted-foreground">Noisy Channels</div>
              <div className="font-semibold">{validation.signal_quality.noisy_channels}</div>
            </div>
            {validation.signal_quality.average_snr_db !== null && (
              <div>
                <div className="text-muted-foreground">Avg SNR</div>
                <div className="font-semibold">{validation.signal_quality.average_snr_db.toFixed(1)} dB</div>
              </div>
            )}
          </div>

          {/* Preprocessing Status */}
          <div className="pt-2 border-t">
            <div className="flex items-center gap-2 mb-2">
              <Filter className="w-4 h-4" />
              <span className="text-sm font-medium">Preprocessing Filters</span>
            </div>
            <div className="flex flex-wrap gap-2">
              <Badge variant={validation.signal_quality.preprocessing_status.dc_removal ? "default" : "destructive"}>
                DC Removal {validation.signal_quality.preprocessing_status.dc_removal ? '✓' : '✗'}
              </Badge>
              <Badge variant={validation.signal_quality.preprocessing_status.bandpass_filter ? "default" : "destructive"}>
                Bandpass {validation.signal_quality.preprocessing_status.bandpass_filter ? '✓' : '✗'}
              </Badge>
              <Badge variant={validation.signal_quality.preprocessing_status.notch_filter ? "default" : "destructive"}>
                Notch (50Hz) {validation.signal_quality.preprocessing_status.notch_filter ? '✓' : '✗'}
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* File Information */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            <Activity className="w-5 h-5" />
            Recording Information
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div>
              <div className="text-muted-foreground">Format</div>
              <div className="font-semibold">{validation.file_format}</div>
            </div>
            <div>
              <div className="text-muted-foreground">Duration</div>
              <div className="font-semibold">{validation.duration_seconds.toFixed(2)}s</div>
            </div>
            <div>
              <div className="text-muted-foreground">Sampling Rate</div>
              <div className="font-semibold">
                {validation.original_sampling_rate.toFixed(0)} Hz
                {validation.resampled && <span className="text-xs text-muted-foreground ml-1">(resampled)</span>}
              </div>
            </div>
            <div>
              <div className="text-muted-foreground">Total Samples</div>
              <div className="font-semibold">{validation.total_samples.toLocaleString()}</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Channel Status Grid */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Channel Status</CardTitle>
          <CardDescription>
            Individual channel quality and detection status
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {validation.channel_statuses.map((channel) => (
              <div
                key={channel.name}
                className={`flex items-center gap-2 p-2 rounded-md ${getChannelStatusColor(channel)}`}
              >
                {getChannelStatusIcon(channel)}
                <div className="flex-1">
                  <div className="font-semibold text-sm">{channel.name}</div>
                  {channel.quality_score !== null && channel.found && (
                    <div className="text-xs opacity-75">
                      Q: {Math.round(channel.quality_score * 100)}%
                    </div>
                  )}
                  {channel.is_zero && (
                    <div className="text-xs opacity-75">Missing</div>
                  )}
                </div>
              </div>
            ))}
          </div>

          {/* Legend */}
          <div className="mt-4 pt-4 border-t flex flex-wrap gap-3 text-xs">
            <div className="flex items-center gap-1">
              <CheckCircle className="w-3 h-3 text-green-600" />
              <span>Good Quality</span>
            </div>
            <div className="flex items-center gap-1">
              <AlertCircle className="w-3 h-3 text-yellow-600" />
              <span>Low Quality/Noisy</span>
            </div>
            <div className="flex items-center gap-1">
              <XCircle className="w-3 h-3 text-gray-400" />
              <span>Missing/Not Found</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
