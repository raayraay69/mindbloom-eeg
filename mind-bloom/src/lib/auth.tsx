'use client';

import React, { createContext, useContext, useState, ReactNode, useEffect } from 'react';
import { useRouter, usePathname } from 'next/navigation';

type User = {
  name: string;
  email: string;
};

type AuthContextType = {
  user: User | null;
  isGuest: boolean;
  login: (email: string, password?: string) => void;
  signup: (name: string, email: string, password?: string) => void;
  loginAsGuest: () => void;
  logout: () => void;
  loading: boolean;
};

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isGuest, setIsGuest] = useState<boolean>(false);
  const [loading, setLoading] = useState(true);
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    // Simulate checking for a stored session
    try {
      const storedUser = sessionStorage.getItem('mindbloom_user');
      const storedGuest = sessionStorage.getItem('mindbloom_guest');
      if (storedUser) {
        setUser(JSON.parse(storedUser));
      } else if (storedGuest) {
        setIsGuest(true);
      }
    } catch (error) {
      console.error("Could not parse session storage", error);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    if (!loading) {
      const isAuthenticated = user || isGuest;
      const isAuthPage = pathname === '/login' || pathname === '/signup' || pathname === '/';
      
      if (isAuthenticated && isAuthPage) {
        router.push('/dashboard');
      } else if (!isAuthenticated && !isAuthPage) {
        router.push('/login');
      }
    }
  }, [user, isGuest, loading, pathname, router]);


  const login = (email: string) => {
    // In a real app, you'd validate password.
    // For demo purposes, we'll allow login with just email for the demo account.
    const isDemo = email === 'demo@mindbloom.com' || email === 'admin@mindbloom.com'
    const name = isDemo && email === 'admin@mindbloom.com' ? 'Admin User' : 'Demo User';
    
    const newUser = { name, email };
    setUser(newUser);
    setIsGuest(false);
    sessionStorage.setItem('mindbloom_user', JSON.stringify(newUser));
    sessionStorage.removeItem('mindbloom_guest');
    router.push('/dashboard');
  };

  const signup = (name: string, email: string, password?: string) => {
    // Here you would typically handle the password, e.g., send to a server.
    console.log(`Signing up ${name} with email ${email} and password ${password ? '*****' : 'not provided'}`);
    const newUser = { name, email };
    setUser(newUser);
    setIsGuest(false);
    sessionStorage.setItem('mindbloom_user', JSON.stringify(newUser));
    sessionStorage.removeItem('mindbloom_guest');
    router.push('/dashboard');
  };
  
  const loginAsGuest = () => {
    setUser(null);
    setIsGuest(true);
    sessionStorage.setItem('mindbloom_guest', 'true');
    sessionStorage.removeItem('mindbloom_user');
    router.push('/dashboard');
  };

  const logout = () => {
    setUser(null);
    setIsGuest(false);
    sessionStorage.removeItem('mindbloom_user');
    sessionStorage.removeItem('mindbloom_guest');
    router.push('/login');
  };

  const value = { user, isGuest, login, signup, loginAsGuest, logout, loading };

  if (loading) {
    return <div className="w-screen h-screen flex items-center justify-center bg-background">
        <div className="w-16 h-16 border-4 border-primary/20 border-t-primary rounded-full animate-spin"></div>
    </div>;
  }
  
  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
