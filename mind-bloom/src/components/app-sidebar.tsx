'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  BrainCircuit,
  HeartHandshake,
  BarChart3,
  BookOpen,
  ShieldCheck,
  Leaf,
} from 'lucide-react';

import {
  Sidebar,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
  SidebarFooter,
} from '@/components/ui/sidebar';
import { cn } from '@/lib/utils';
import { useAuth } from '@/lib/auth';
import { Button } from './ui/button';

const menuItems = [
  {
    href: '/dashboard',
    label: 'Dashboard',
    icon: BrainCircuit,
  },
  {
    href: '/therapy',
    label: 'Therapeutic Hub',
    icon: HeartHandshake,
  },
  {
    href: '/history',
    label: 'Session History',
    icon: BarChart3,
  },
  {
    href: '/resources',
    label: 'Resources',
    icon: BookOpen,
  },
  {
    href: '/admin',
    label: 'Admin',
    icon: ShieldCheck,
  },
];

export function AppSidebar() {
  const pathname = usePathname();
  const { user } = useAuth();

  return (
    <Sidebar collapsible="icon">
      <SidebarHeader>
        <div className="flex items-center gap-2 p-2">
          <Leaf className="w-6 h-6 text-primary" />
          <span className="font-bold text-lg group-data-[collapsible=icon]:hidden">
            MindBloom
          </span>
        </div>
      </SidebarHeader>
      <SidebarMenu className="flex-1">
        {menuItems.map(item => {
          if (item.href === '/admin' && user?.email !== 'admin@mindbloom.com') {
            return null;
          }
          return (
            <SidebarMenuItem key={item.href}>
              <Link href={item.href}>
                <SidebarMenuButton
                  isActive={pathname === item.href}
                  tooltip={{
                    children: item.label,
                  }}
                >
                  <item.icon />
                  <span>{item.label}</span>
                </SidebarMenuButton>
              </Link>
            </SidebarMenuItem>
          );
        })}
      </SidebarMenu>
      <SidebarFooter>
        <div
          className={cn(
            'p-2 text-center text-xs text-muted-foreground',
            'group-data-[collapsible=icon]:hidden'
          )}
        >
          &copy; 2024 MindBloom
        </div>
      </SidebarFooter>
    </Sidebar>
  );
}
