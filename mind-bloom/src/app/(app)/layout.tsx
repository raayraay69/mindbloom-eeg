import { AppSidebar } from '@/components/app-sidebar';
import { CrisisFooter } from '@/components/crisis-footer';
import { UserNav } from '@/components/user-nav';
import { Button } from '@/components/ui/button';
import { SidebarProvider, SidebarInset, SidebarTrigger } from '@/components/ui/sidebar';
import { Leaf } from 'lucide-react';
import Link from 'next/link';

export default function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset className="flex flex-col">
        <header className="sticky top-0 z-10 flex h-14 items-center gap-4 border-b bg-background/80 px-4 backdrop-blur-sm sm:h-auto sm:border-0 sm:bg-transparent sm:px-6">
          <SidebarTrigger className="md:hidden" />
          <div className="flex items-center gap-2 md:hidden">
            <Leaf className="w-6 h-6 text-primary" />
            <span className="font-bold">MindBloom</span>
          </div>
          <div className="ml-auto flex items-center gap-2">
            <UserNav />
          </div>
        </header>
        <main className="flex-1 overflow-y-auto p-4 sm:p-6">{children}</main>
        <CrisisFooter />
      </SidebarInset>
    </SidebarProvider>
  );
}
