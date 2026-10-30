'use client'


import jsPDF from 'jspdf'; 
import { CheckSquare, FileDown, Copy, Save, Stethoscope } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/app/components/ui/card';
import { Button } from '@/app/components/ui/button';
import { Badge } from '@/app/components/ui/badge';
import { toast } from 'sonner';
import { PredictionResult, PatientData } from '@/app/types/patient';

interface RecommendationsProps {
  recommendations: string[];
  result: PredictionResult;
  patientData: PatientData;
  ultrasoundFile?: File | null;
}

export function Recommendations({ recommendations, result, patientData,ultrasoundFile }: RecommendationsProps) {
  const handleCopySummary = () => {
    const summary = generateSummaryText();
    navigator.clipboard.writeText(summary);
    toast.success('Summary copied to clipboard!');
  };

  const handleExportPDF = async () => {
  try {
    const doc = new jsPDF();
    const getImageDataUrl = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = () => {
      if (typeof reader.result === 'string') {
        resolve(reader.result);
      } else {
        reject(new Error('Failed to read ultrasound image'));
      }
    };

    reader.onerror = () => {
      reject(new Error('Failed to read ultrasound image'));
    };

    reader.readAsDataURL(file);
  });
};

    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();

    const margin = 15;
    const contentWidth = pageWidth - margin * 2;
    let y = 20;

    // Helper: add text with automatic page handling
    const addText = (
      text: string,
      x: number,
      fontSize = 10,
      options?: {
        bold?: boolean;
        color?: [number, number, number];
        maxWidth?: number;
      }
    ) => {
      if (options?.bold) {
        doc.setFont('helvetica', 'bold');
      } else {
        doc.setFont('helvetica', 'normal');
      }

      doc.setFontSize(fontSize);

      if (options?.color) {
        doc.setTextColor(...options.color);
      } else {
        doc.setTextColor(40, 40, 40);
      }

      const lines = doc.splitTextToSize(
        text,
        options?.maxWidth ?? contentWidth
      );

      const lineHeight = fontSize * 0.45;

      // Page break
      if (y + lines.length * lineHeight > pageHeight - 20) {
        doc.addPage();
        y = 20;
      }

      doc.text(lines, x, y);
      y += lines.length * lineHeight + 3;
    };

    // =========================
    // HEADER
    // =========================

    doc.setFillColor(22, 163, 74);
    doc.rect(0, 0, pageWidth, 35, 'F');

    doc.setTextColor(255, 255, 255);
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(18);

    doc.text(
      'LIVER DISEASE RISK PREDICTION REPORT',
      pageWidth / 2,
      16,
      { align: 'center' }
    );

    doc.setFont('helvetica', 'normal');
    doc.setFontSize(9);

    doc.text(
      'ML-Based Clinical Risk Assessment',
      pageWidth / 2,
      25,
      { align: 'center' }
    );

    y = 48;

    // =========================
    // PATIENT INFORMATION
    // =========================

    addText('PATIENT INFORMATION', margin, 13, {
      bold: true,
      color: [22, 163, 74],
    });

    addText(
      `Patient ID: ${patientData.patientId || 'N/A'}`,
      margin,
      10
    );

    addText(
      `Name: ${patientData.name || 'N/A'}`,
      margin,
      10
    );

    addText(
      `Age: ${patientData.age || 'N/A'}    Gender: ${patientData.gender || 'N/A'}`,
      margin,
      10
    );

    addText(
      `Date: ${result.timestamp.toLocaleString()}`,
      margin,
      10
    );

    y += 5;

      // =========================
  // MULTIMODAL AI RISK SCORE
  // =========================

  if (result.multimodal_score) {
    const multimodal = result.multimodal_score;

    y += 5;

    addText('MULTIMODAL AI RISK SCORE', margin, 13, {
      bold: true,
      color: [22, 163, 74],
    });

    addText(
      'AI-based evidence aggregation from clinical and ultrasound models.',
      margin,
      9
    );

    // =========================
    // OVERALL SCORE
    // =========================

    addText(
      `Multimodal AI Risk Score: ${multimodal.score.toFixed(1)} / 100`,
      margin,
      11,
      {
        bold: true,
      }
    );

    addText(
      `Evidence Agreement: ${multimodal.agreement}`,
      margin + 5,
      9
    );

    y += 5;

    // =========================
    // CLINICAL ML
    // =========================

    addText(
      'Clinical ML',
      margin,
      10,
      {
        bold: true,
      }
    );

    addText(
      `Probability: ${multimodal.clinical.probability.toFixed(1)}%`,
      margin + 5,
      9
    );

    addText(
      `Weight: ${(multimodal.clinical.weight * 100).toFixed(0)}%`,
      margin + 5,
      9
    );

    addText(
      `Contribution: ${multimodal.clinical.contribution.toFixed(2)}`,
      margin + 5,
      9
    );

    addText(
      `Evidence Level: ${multimodal.clinical.evidence_level}`,
      margin + 5,
      9
    );

    y += 3;

    // =========================
    // BYRA
    // =========================

    addText(
      'Byra / IJCARS',
      margin,
      10,
      {
        bold: true,
      }
    );

    addText(
      `Steatosis Probability: ${multimodal.byra.probability.toFixed(1)}%`,
      margin + 5,
      9
    );

    addText(
      `Weight: ${(multimodal.byra.weight * 100).toFixed(0)}%`,
      margin + 5,
      9
    );

    addText(
      `Contribution: ${multimodal.byra.contribution.toFixed(2)}`,
      margin + 5,
      9
    );

    addText(
      `Evidence Level: ${multimodal.byra.evidence_level}`,
      margin + 5,
      9
    );

    y += 3;

    // =========================
    // BEHSOF
    // =========================

    addText(
      'BEHSOF',
      margin,
      10,
      {
        bold: true,
      }
    );

    addText(
      `NAFLD Probability: ${multimodal.behsof.probability.toFixed(1)}%`,
      margin + 5,
      9
    );

    addText(
      `Weight: ${(multimodal.behsof.weight * 100).toFixed(0)}%`,
      margin + 5,
      9
    );

    addText(
      `Contribution: ${multimodal.behsof.contribution.toFixed(2)}`,
      margin + 5,
      9
    );

    addText(
      `Evidence Level: ${multimodal.behsof.evidence_level}`,
      margin + 5,
      9
    );

    y += 3;

    // =========================
    // SCORE INTERPRETATION
    // =========================

    addText(
      'Multimodal Evidence Summary',
      margin,
      10,
      {
        bold: true,
      }
    );

    addText(
      `The multimodal system generated a score of ${multimodal.score.toFixed(1)} / 100 based on weighted outputs from the clinical ML, Byra, and BEHSOF models. Evidence agreement was classified as ${multimodal.agreement}.`,
      margin + 5,
      9,
      {
        maxWidth: contentWidth - 5,
      }
    );

    y += 5;

    // =========================
    // DISCLAIMER
    // =========================

    addText(
      'Note: This score is an AI-based aggregation of model outputs and is not a clinically validated disease probability.',
      margin + 5,
      8,
      {
        maxWidth: contentWidth - 5,
      }
    );

    y += 5;
  }

    // =========================
    // PREDICTION SUMMARY
    // =========================

    addText('PREDICTION SUMMARY', margin, 13, {
      bold: true,
      color: [22, 163, 74],
    });

    const riskStatus = result.hasDiseaseRisk
      ? 'DISEASE RISK DETECTED'
      : 'NO DISEASE DETECTED';

    addText(
      `Risk Status: ${riskStatus}`,
      margin,
      11,
      {
        bold: true,
        color: result.hasDiseaseRisk
          ? [220, 38, 38]
          : [22, 163, 74],
      }
    );

    addText(
      `Risk Probability: ${result.probability}%`,
      margin,
      10
    );

    addText(
      `Model Confidence: ${result.confidence}`,
      margin,
      10
    );

    y += 5;

    // =========================
// ULTRASOUND ANALYSIS
// =========================

if (result.ultrasound_analysis) {
  const ultrasound = result.ultrasound_analysis;

  y += 5;

  addText('ULTRASOUND ANALYSIS', margin, 13, {
    bold: true,
    color: [22, 163, 74],
  });

  addText(
    'AI-based ultrasound assessment using two task-specific models.',
    margin,
    9
  );

  // =========================
  // UPLOADED IMAGE
  // =========================

  if (ultrasoundFile) {
    

    try {
      const imageData = await getImageDataUrl(ultrasoundFile);

      const imageWidth = contentWidth;
      const imageHeight = 80;

      // Make sure image + heading fit on current page
      if (y + imageHeight + 10 > pageHeight - 20) {
        doc.addPage();
        y = 20;
      }

      doc.setDrawColor(220, 220, 220);
      doc.roundedRect(
        margin,
        y,
        imageWidth,
        imageHeight,
        3,
        3,
        'S'
      );

      doc.addImage(
        imageData,
        ultrasoundFile.type === 'image/png'
          ? 'PNG'
          : 'JPEG',
        margin + 3,
        y + 3,
        imageWidth - 6,
        imageHeight - 6,
        undefined,
        'FAST'
      );

      y += imageHeight + 8;
    } catch (imageError) {
      console.error(
        'Failed to add ultrasound image:',
        imageError
      );

      addText(
        'Ultrasound image could not be embedded in the report.',
        margin + 5,
        9,
        {
          color: [180, 30, 30],
        }
      );
    }
  }

  // =========================
  // BYRA / IJCARS
  // =========================

  addText(
    'Byra / IJCARS - Steatosis Assessment',
    margin,
    10,
    { bold: true }
  );

  addText(
    `Prediction: ${ultrasound.byra.prediction}`,
    margin + 5,
    9
  );

  addText(
    `Normal Probability: ${(ultrasound.byra.normal_probability * 100).toFixed(1)}%`,
    margin + 5,
    9
  );

  addText(
    `Steatosis Probability: ${(ultrasound.byra.steatosis_probability * 100).toFixed(1)}%`,
    margin + 5,
    9
  );

  addText(
    `Confidence: ${(ultrasound.byra.confidence * 100).toFixed(1)}%`,
    margin + 5,
    9
  );

  y += 3;

  // =========================
  // BEHSOF
  // =========================

  addText(
    'BEHSOF - NAFLD Assessment',
    margin,
    10,
    { bold: true }
  );

  addText(
    `Prediction: ${ultrasound.behsof.prediction}`,
    margin + 5,
    9
  );

  addText(
    `Non-NAFLD Probability: ${(ultrasound.behsof.non_nafld_probability * 100).toFixed(1)}%`,
    margin + 5,
    9
  );

  addText(
    `NAFLD Probability: ${(ultrasound.behsof.nafld_probability * 100).toFixed(1)}%`,
    margin + 5,
    9
  );

  addText(
    `Confidence: ${(ultrasound.behsof.confidence * 100).toFixed(1)}%`,
    margin + 5,
    9
  );

  y += 3;

  // =========================
  // ULTRASOUND SUMMARY
  // =========================

  addText(
    'Ultrasound Summary',
    margin,
    10,
    { bold: true }
  );

  addText(
    ultrasound.summary,
    margin + 5,
    9,
    {
      maxWidth: contentWidth - 5,
    }
  );

  y += 5;
}


    // =========================
    // CONTRIBUTING FACTORS
    // =========================

    if (result.contributingFactors.length > 0) {
      addText('KEY CONTRIBUTING FACTORS', margin, 13, {
        bold: true,
        color: [22, 163, 74],
      });

      result.contributingFactors.forEach((factor, index) => {
        addText(
          `${index + 1}. ${factor.feature}: ${factor.patientValue} (${factor.status})`,
          margin,
          10,
          { bold: true }
        );

        addText(
          `Normal Range: ${factor.normalRange}`,
          margin + 5,
          9
        );

        addText(
          `Clinical Note: ${factor.clinicalNote}`,
          margin + 5,
          9
        );

        y += 2;
      });
    }

    // =========================
    // CLINICAL INTERPRETATION
    // =========================

    addText('CLINICAL INTERPRETATION', margin, 13, {
      bold: true,
      color: [22, 163, 74],
    });

    addText(
      result.clinicalInterpretation,
      margin,
      10
    );

    y += 5;

    // =========================
    // RECOMMENDATIONS
    // =========================

    addText('RECOMMENDATIONS', margin, 13, {
      bold: true,
      color: [22, 163, 74],
    });

    recommendations.forEach((recommendation, index) => {
      addText(
        `${index + 1}. ${recommendation}`,
        margin,
        10
      );
    });

    // =========================
    // HIGH PRIORITY WARNING
    // =========================

    if (result.probability > 70) {
      y += 5;

      if (y + 25 > pageHeight - 20) {
        doc.addPage();
        y = 20;
      }

      doc.setFillColor(254, 226, 226);
      doc.setDrawColor(239, 68, 68);

      doc.roundedRect(
        margin,
        y,
        contentWidth,
        20,
        3,
        3,
        'FD'
      );

      doc.setTextColor(185, 28, 28);
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(10);

      doc.text(
        'HIGH PRIORITY',
        margin + 5,
        y + 8
      );

      doc.setFont('helvetica', 'normal');
      doc.setFontSize(8);

      doc.text(
        'Urgent clinical review recommended based on high risk probability.',
        margin + 5,
        y + 15
      );

      y += 28;
    }

    // =========================
    // DISCLAIMER
    // =========================

    addText('DISCLAIMER', margin, 13, {
      bold: true,
      color: [22, 163, 74],
    });

    addText(
      'This is an ML-based risk prediction tool. Final diagnosis must be made by a qualified physician.',
      margin,
      9
    );

    addText(
      'Recommendations should be interpreted together with clinical history, physical examination, imaging studies, laboratory findings, comorbidities, medications, and professional medical judgment.',
      margin,
      9
    );

    // =========================
    // FOOTER
    // =========================

    const totalPages = doc.getNumberOfPages();

    for (let page = 1; page <= totalPages; page++) {
      doc.setPage(page);

      doc.setDrawColor(220, 220, 220);
      doc.line(
        margin,
        pageHeight - 12,
        pageWidth - margin,
        pageHeight - 12
      );

      doc.setFont('helvetica', 'normal');
      doc.setFontSize(8);
      doc.setTextColor(120, 120, 120);

      doc.text(
        'Liver Disease Risk Prediction System',
        margin,
        pageHeight - 6
      );

      doc.text(
        `Page ${page} of ${totalPages}`,
        pageWidth - margin,
        pageHeight - 6,
        { align: 'right' }
      );
    }

    // =========================
    // DOWNLOAD
    // =========================

    const patientName = patientData.name
      ? patientData.name.replace(/[^a-zA-Z0-9]/g, '_')
      : 'Patient';

    doc.save(
      `Liver_Risk_Report_${patientName}.pdf`
    );

    toast.success('PDF report exported successfully!');
  } catch (error) {
    console.error('PDF export failed:', error);
    toast.error('Failed to export PDF');
  }
};

  const handleSaveToRecord = () => {
    // In a real application, this would save to a database
    toast.info('Save to patient record functionality would be implemented here');
  };

  const generateSummaryText = () => {
    let text = '=== LIVER DISEASE RISK PREDICTION REPORT ===\n\n';
    text += `Patient ID: ${patientData.patientId || 'N/A'}\n`;
    text += `Name: ${patientData.name || 'N/A'}\n`;
    text += `Age: ${patientData.age} | Gender: ${patientData.gender}\n`;
    text += `Date: ${result.timestamp.toLocaleString()}\n\n`;

    text += '--- PREDICTION SUMMARY ---\n';
    text += `Risk Status: ${result.hasDiseaseRisk ? 'DISEASE RISK DETECTED' : 'NO DISEASE DETECTED'}\n`;
    text += `Risk Probability: ${result.probability}%\n`;
    text += `Model Confidence: ${result.confidence}\n\n`;

    if (result.contributingFactors.length > 0) {
      text += '--- KEY CONTRIBUTING FACTORS ---\n';
      result.contributingFactors.forEach((factor, i) => {
        text += `${i + 1}. ${factor.feature}: ${factor.patientValue} (${factor.status})\n`;
        text += `   ${factor.normalRange}\n`;
        text += `   ${factor.clinicalNote}\n\n`;
      });
    }

    text += '--- CLINICAL INTERPRETATION ---\n';
    text += `${result.clinicalInterpretation}\n\n`;

    text += '--- RECOMMENDATIONS ---\n';
    recommendations.forEach((rec, i) => {
      text += `${i + 1}. ${rec}\n`;
    });

    text += '\n--- DISCLAIMER ---\n';
    text += 'This is an ML-based risk prediction tool. Final diagnosis must be made by a qualified physician.\n';

    return text;
  };

  return (
    <Card className="border-2 border-green-200">
      <CardHeader className="bg-gradient-to-r from-green-50 to-teal-50 border-b">
        <CardTitle className="flex items-center gap-2">
          <Stethoscope className="w-5 h-5 text-green-600" />
          Clinical Recommendations
        </CardTitle>
        <CardDescription>Recommended follow-up actions for physician review</CardDescription>
      </CardHeader>
      <CardContent className="p-6">
        <div className="space-y-6">
          {/* Recommendations Checklist */}
          <div className="space-y-3">
            <div className="flex items-center gap-2 mb-3">
              <CheckSquare className="w-5 h-5 text-green-600" />
              <h4 className="font-semibold text-gray-900">Recommended Follow-ups</h4>
            </div>

            <div className="space-y-2">
              {recommendations.map((recommendation, index) => (
                <div
                  key={index}
                  className="flex items-start gap-3 p-3 bg-white border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
                >
                  <div className="flex-shrink-0 mt-0.5">
                    <div className="w-5 h-5 border-2 border-green-600 rounded flex items-center justify-center">
                      <div className="w-2 h-2 bg-green-600 rounded-full"></div>
                    </div>
                  </div>
                  <p className="text-sm text-gray-700 flex-1">{recommendation}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Urgency Badge */}
          {result.probability > 70 && (
            <div className="bg-red-50 border-l-4 border-red-500 p-4 rounded-r">
              <div className="flex items-center gap-2">
                <Badge variant="destructive" className="bg-red-600">
                  HIGH PRIORITY
                </Badge>
                <p className="text-sm text-red-900 font-semibold">
                  Urgent clinical review recommended based on high risk probability
                </p>
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="pt-4 border-t space-y-3">
            <h4 className="text-sm font-semibold text-gray-700 mb-2">Report Actions</h4>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={handleCopySummary}
                className="w-full"
              >
                <Copy className="w-4 h-4 mr-2" />
                Copy Summary
              </Button>

              <Button
                variant="outline"
                size="sm"
                onClick={handleExportPDF}
                className="w-full"
              >
                <FileDown className="w-4 h-4 mr-2" />
                Export PDF
              </Button>

              <Button
                variant="outline"
                size="sm"
                onClick={handleSaveToRecord}
                className="w-full"
              >
                <Save className="w-4 h-4 mr-2" />
                Save to Record
              </Button>
            </div>
          </div>

          {/* Final Note */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <p className="text-xs text-blue-900">
              <span className="font-semibold">Note for Healthcare Providers:</span> These recommendations
              are generated based on the laboratory findings and prediction results. Please integrate them
              with comprehensive patient assessment, including clinical history, physical examination, imaging
              studies, and your professional judgment. Consider patient-specific factors such as comorbidities,
              medications, and risk factors when determining the appropriate diagnostic and therapeutic approach.
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
