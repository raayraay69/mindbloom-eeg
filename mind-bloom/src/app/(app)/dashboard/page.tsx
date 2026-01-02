'use client';
import { useAuth } from '@/lib/auth';
import { EegUpload } from './components/eeg-upload';

export default function DashboardPage() {
  const { user, isGuest } = useAuth();
  
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl md:text-3xl font-bold tracking-tight font-headline">
          Welcome, {isGuest ? 'Guest' : user?.name || 'User'}
        </h1>
        <p className="text-muted-foreground mt-1">
          Begin your session by uploading an EEG recording.
        </p>
      </div>
      <EegUpload />
    </div>
  );
}
