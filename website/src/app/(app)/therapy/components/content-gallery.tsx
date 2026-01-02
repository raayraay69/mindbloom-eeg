'use client';

import { therapeuticContent } from '@/lib/data';
import type { TherapeuticContent } from '@/lib/types';
import Image from 'next/image';
import { useState, useMemo } from 'react';
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Star } from 'lucide-react';

export function ContentGallery() {
  const [activeCategory, setActiveCategory] = useState('All');
  const [intensityFilter, setIntensityFilter] = useState('All');

  const filteredContent = useMemo(() => {
    return therapeuticContent.filter(item => {
      const categoryMatch = activeCategory === 'All' || item.category === activeCategory;
      const intensityMatch = intensityFilter === 'All' || item.intensity === intensityFilter;
      return categoryMatch && intensityMatch;
    });
  }, [activeCategory, intensityFilter]);

  const categories = ['All', 'Mobile Apps', 'VR Experiences', 'AI-Guided Exercises', 'Ambient Soundscapes'];

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row gap-4 justify-between">
        <Tabs value={activeCategory} onValueChange={setActiveCategory}>
          <TabsList className="grid grid-cols-2 sm:inline-flex sm:grid-cols-none h-auto flex-wrap">
            {categories.map(category => (
              <TabsTrigger key={category} value={category}>{category}</TabsTrigger>
            ))}
          </TabsList>
        </Tabs>
        <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-muted-foreground">Intensity:</span>
            <Select value={intensityFilter} onValueChange={setIntensityFilter}>
                <SelectTrigger className="w-[180px]">
                    <SelectValue placeholder="Filter by intensity" />
                </SelectTrigger>
                <SelectContent>
                    <SelectItem value="All">All</SelectItem>
                    <SelectItem value="Low">Low Stimulation</SelectItem>
                    <SelectItem value="Moderate">Moderate Engagement</SelectItem>
                    <SelectItem value="High">High Engagement</SelectItem>
                </SelectContent>
            </Select>
        </div>
      </div>

      {filteredContent.length > 0 ? (
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {filteredContent.map(item => (
            <Card key={item.id} className="flex flex-col overflow-hidden transition-all hover:shadow-lg">
              <div className="relative aspect-video">
                <Image
                  src={item.thumbnailUrl}
                  alt={item.title}
                  fill
                  sizes="(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 25vw"
                  className="object-cover"
                  data-ai-hint={item.imageHint}
                />
              </div>
              <CardHeader>
                <Badge variant="secondary" className="w-fit mb-2">{item.category}</Badge>
                <CardTitle className="text-lg">{item.title}</CardTitle>
              </CardHeader>
              <CardContent className="flex-1">
                <p className="text-sm text-muted-foreground line-clamp-3">{item.description}</p>
              </CardContent>
              <CardFooter className="flex justify-between items-center text-sm text-muted-foreground">
                <div className="flex items-center gap-1">
                  <Star className="w-4 h-4 text-yellow-500 fill-yellow-400" />
                  <span>{item.rating}</span>
                </div>
                 <Badge variant="outline">{item.intensity}</Badge>
              </CardFooter>
            </Card>
          ))}
        </div>
      ) : (
        <div className="text-center py-16 border border-dashed rounded-lg">
            <p className="text-muted-foreground">No content found matching your filters.</p>
        </div>
      )}
    </div>
  );
}
