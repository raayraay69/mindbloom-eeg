import { ContentGallery } from './components/content-gallery';

export default function TherapyPage() {
  return (
    <div className="space-y-6">
       <div>
        <h1 className="text-2xl md:text-3xl font-bold tracking-tight font-headline">
          Therapeutic Content Hub
        </h1>
        <p className="text-muted-foreground mt-1">
          Explore a curated library of resources designed for calm and cognitive engagement.
        </p>
      </div>
      <ContentGallery />
    </div>
  );
}
