import { Activity, Image as ImageIcon } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/app/components/ui/card';
import type { UltrasoundAnalysis as UltrasoundAnalysisType } from '@/app/types/patient';
import { useEffect, useState } from 'react';

interface UltrasoundAnalysisProps {
  analysis: UltrasoundAnalysisType | null | undefined;
  ultrasoundFile?: File | null;
}

export function UltrasoundAnalysis({ analysis,ultrasoundFile }: UltrasoundAnalysisProps) {
  if (!analysis) {
    return null;
  }

  const [imageUrl, setImageUrl] = useState<string | null>(null);

useEffect(() => {
  if (!ultrasoundFile) {
    setImageUrl(null);
    return;
  }

  const url = URL.createObjectURL(ultrasoundFile);
  setImageUrl(url);

  return () => {
    URL.revokeObjectURL(url);
  };
}, [ultrasoundFile]);

  const byra = analysis.byra;
  const behsof = analysis.behsof;

  return (
    <Card className="border-teal-200 shadow-sm">
      <CardHeader>
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-teal-100 flex items-center justify-center">
            <ImageIcon className="w-5 h-5 text-teal-600" />
          </div>

          <div>
            <CardTitle className="text-lg">
              Ultrasound Analysis
            </CardTitle>

            <p className="text-sm text-gray-500">
              AI-based analysis from two ultrasound models
            </p>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">

        {imageUrl && (
  <div className="rounded-lg border overflow-hidden bg-gray-50">
    <div className="px-4 py-3 border-b bg-white">
      <p className="text-sm font-medium text-gray-900">
        Uploaded Ultrasound
      </p>
      <p className="text-xs text-gray-500 mt-1">
        {ultrasoundFile?.name}
      </p>
    </div>

    <div className="p-4 flex justify-center">
      <img
        src={imageUrl}
        alt="Uploaded ultrasound"
        className="max-h-[400px] w-auto max-w-full object-contain rounded-lg"
      />
    </div>
  </div>
)}

        {/* Byra */}
        <div className="rounded-lg border p-4">
          <div className="flex items-center justify-between mb-3">
            <div>
              <h3 className="font-semibold text-gray-900">
                Byra / IJCARS
              </h3>

              <p className="text-xs text-gray-500">
                Steatosis Assessment
              </p>
            </div>

            <Activity className="w-5 h-5 text-teal-600" />
          </div>

          <div className="grid grid-cols-2 gap-3">

            <div className="bg-gray-50 rounded-lg p-3">
              <p className="text-xs text-gray-500">
                Prediction
              </p>

              <p className="font-semibold text-gray-900 mt-1">
                {byra.prediction}
              </p>
            </div>

            <div className="bg-gray-50 rounded-lg p-3">
              <p className="text-xs text-gray-500">
                Confidence
              </p>

              <p className="font-semibold text-gray-900 mt-1">
                {(byra.confidence * 100).toFixed(1)}%
              </p>
            </div>

            <div className="bg-gray-50 rounded-lg p-3">
              <p className="text-xs text-gray-500">
                Normal
              </p>

              <p className="font-semibold text-gray-900 mt-1">
                {(byra.normal_probability * 100).toFixed(1)}%
              </p>
            </div>

            <div className="bg-gray-50 rounded-lg p-3">
              <p className="text-xs text-gray-500">
                Steatosis
              </p>

              <p className="font-semibold text-gray-900 mt-1">
                {(byra.steatosis_probability * 100).toFixed(1)}%
              </p>
            </div>

          </div>
        </div>

        {/* BEHSOF */}
        <div className="rounded-lg border p-4">
          <div className="flex items-center justify-between mb-3">
            <div>
              <h3 className="font-semibold text-gray-900">
                BEHSOF
              </h3>

              <p className="text-xs text-gray-500">
                NAFLD Assessment
              </p>
            </div>

            <Activity className="w-5 h-5 text-teal-600" />
          </div>

          <div className="grid grid-cols-2 gap-3">

            <div className="bg-gray-50 rounded-lg p-3">
              <p className="text-xs text-gray-500">
                Prediction
              </p>

              <p className="font-semibold text-gray-900 mt-1">
                {behsof.prediction}
              </p>
            </div>

            <div className="bg-gray-50 rounded-lg p-3">
              <p className="text-xs text-gray-500">
                Confidence
              </p>

              <p className="font-semibold text-gray-900 mt-1">
                {(behsof.confidence * 100).toFixed(1)}%
              </p>
            </div>

            <div className="bg-gray-50 rounded-lg p-3">
              <p className="text-xs text-gray-500">
                Non-NAFLD
              </p>

              <p className="font-semibold text-gray-900 mt-1">
                {(behsof.non_nafld_probability * 100).toFixed(1)}%
              </p>
            </div>

            <div className="bg-gray-50 rounded-lg p-3">
              <p className="text-xs text-gray-500">
                NAFLD
              </p>

              <p className="font-semibold text-gray-900 mt-1">
                {(behsof.nafld_probability * 100).toFixed(1)}%
              </p>
            </div>

          </div>
        </div>

        {/* Summary */}
        <div className="rounded-lg bg-teal-50 border border-teal-100 p-4">
          <p className="text-sm font-medium text-teal-900 mb-1">
            Ultrasound Summary
          </p>

          <p className="text-sm text-teal-800">
            {analysis.summary}
          </p>
        </div>

      </CardContent>
    </Card>
  );
}