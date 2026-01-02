import { Phone, MessageSquare, Globe } from 'lucide-react';
import Link from 'next/link';

export function CrisisFooter() {
  return (
    <footer className="mt-auto border-t border-border/60 bg-background/80 backdrop-blur-sm">
      <div className="container mx-auto px-4 py-4">
        <div className="text-center">
          <h3 className="font-semibold text-base text-destructive">In Crisis? Help is available.</h3>
          <p className="text-sm text-muted-foreground mt-1 mb-3">
            If you or someone you know is in immediate danger, please call 911. For mental health support, you can reach out to the following resources:
          </p>
          <div className="flex flex-wrap justify-center items-center gap-x-6 gap-y-3">
            <div className="flex items-center gap-2">
              <Phone className="w-4 h-4 text-primary" />
              <span className="text-sm font-medium">Suicide & Crisis Lifeline:</span>
              <Link href="tel:988" className="text-sm text-primary hover:underline">988</Link>
            </div>
            <div className="flex items-center gap-2">
              <MessageSquare className="w-4 h-4 text-primary" />
              <span className="text-sm font-medium">Crisis Text Line:</span>
              <span className="text-sm">Text HOME to <Link href="sms:741741" className="text-primary hover:underline">741741</Link></span>
            </div>
             <div className="flex items-center gap-2">
              <Globe className="w-4 h-4 text-primary" />
              <Link href="https://www.thetrevorproject.org/get-help/" target="_blank" rel="noopener noreferrer" className="text-sm text-primary hover:underline">The Trevor Project (for LGBTQ youth)</Link>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
}
