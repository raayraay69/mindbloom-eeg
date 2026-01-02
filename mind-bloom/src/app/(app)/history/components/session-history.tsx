'use client';

import { sessionHistory } from '@/lib/data';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  ChartLegend,
  ChartLegendContent
} from '@/components/ui/chart';
import { Area, AreaChart, CartesianGrid, XAxis, YAxis, ResponsiveContainer } from 'recharts';
import { Badge } from '@/components/ui/badge';
import { formatProcessingTime } from '@/lib/utils';
import { format, parseISO } from 'date-fns';

const chartData = sessionHistory.map(session => ({
    date: session.date,
    confidence: session.confidence * 100,
})).sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());


const chartConfig = {
    confidence: {
        label: "Confidence",
        color: "hsl(var(--chart-1))",
    },
};

export function SessionHistory() {

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Confidence Score Over Time</CardTitle>
        </CardHeader>
        <CardContent>
          <ChartContainer config={chartConfig} className="h-64 w-full">
            <AreaChart data={chartData} margin={{ left: -20, right: 20, top: 5, bottom: 5 }}>
              <CartesianGrid vertical={false} strokeDasharray="3 3" />
              <XAxis 
                dataKey="date" 
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                tickFormatter={(value) => format(parseISO(value), 'MMM d')}
              />
              <YAxis 
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                domain={[0, 100]}
                tickFormatter={(value) => `${value}%`}
              />
               <ChartTooltip 
                cursor={false}
                content={<ChartTooltipContent indicator="line" />} 
              />
               <defs>
                    <linearGradient id="fillConfidence" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="var(--color-confidence)" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="var(--color-confidence)" stopOpacity={0.1}/>
                    </linearGradient>
                </defs>
              <Area
                dataKey="confidence"
                type="natural"
                fill="url(#fillConfidence)"
                fillOpacity={0.4}
                stroke="var(--color-confidence)"
                stackId="a"
              />
            </AreaChart>
          </ChartContainer>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Detailed Session Log</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Date</TableHead>
                <TableHead>Prediction</TableHead>
                <TableHead className="text-right">Confidence</TableHead>
                <TableHead className="text-right">Processing Time</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {[...sessionHistory].reverse().map(session => (
                <TableRow key={session.id}>
                  <TableCell className="font-medium">{format(parseISO(session.date), 'MMMM d, yyyy')}</TableCell>
                  <TableCell>
                     <Badge variant={session.prediction === 'positive' ? 'destructive' : 'default'}>
                        {session.prediction === 'positive' ? 'Markers Detected' : 'No Markers Detected'}
                     </Badge>
                  </TableCell>
                  <TableCell className="text-right font-mono">{(session.confidence * 100).toFixed(2)}%</TableCell>
                  <TableCell className="text-right">{formatProcessingTime(session.processingTime)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
