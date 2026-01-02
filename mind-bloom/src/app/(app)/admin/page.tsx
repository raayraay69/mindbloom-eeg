'use client';
import { useAuth } from '@/lib/auth';
import { useRouter } from 'next/navigation';
import { useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { ShieldAlert } from 'lucide-react';
import { Button } from '@/components/ui/button';

export default function AdminPage() {
  const { user, loading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!loading && user?.email !== 'admin@mindbloom.com') {
      router.replace('/dashboard');
    }
  }, [user, loading, router]);

  if (loading || user?.email !== 'admin@mindbloom.com') {
    return (
       <div className="flex flex-col items-center justify-center text-center py-16">
         <div className="w-12 h-12 border-4 border-primary/20 border-t-primary rounded-full animate-spin"></div>
         <p className="mt-4 text-muted-foreground">Verifying access...</p>
       </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl md:text-3xl font-bold tracking-tight font-headline">
          Admin Content Management
        </h1>
        <p className="text-muted-foreground mt-1">
          Manage therapeutic content for the MindBloom app.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Content Management</CardTitle>
          <CardDescription>
            This is a placeholder for the admin panel. In a full application, this area would allow you to add, edit, and remove content from the Therapeutic Content Hub.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
            <Button>Add New Content</Button>
            <div className="border-t pt-4">
                <p className="text-sm text-muted-foreground">Existing content would be listed here for editing or removal.</p>
            </div>
        </CardContent>
      </Card>
       <Card className="border-destructive">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-destructive">
            <ShieldAlert className="w-5 h-5"/>
            Admin Area
          </CardTitle>
          <CardDescription>
            This area is restricted. Any changes made here will affect all users.
          </CardDescription>
        </CardHeader>
      </Card>
    </div>
  );
}
