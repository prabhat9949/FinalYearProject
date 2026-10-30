// import { Database, Brain, BarChart3, Shield, FileText, Activity, AlertCircle } from 'lucide-react';
// import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/app/components/ui/card';
// import { Badge } from '@/app/components/ui/badge';

// export function AboutSection() {
//   return (
//     <div className="max-w-5xl mx-auto py-8 px-4 space-y-6">
//       <div className="text-center mb-8">
//         <h2 className="text-3xl font-bold text-gray-900 mb-2">About the Liver Disease Prediction System</h2>
//         <p className="text-gray-600">
//           AI-powered clinical decision support tool for liver disease risk assessment
//         </p>
//       </div>

//       <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
//         {/* Dataset Information */}
//         <Card>
//           <CardHeader className="bg-gradient-to-r from-blue-50 to-cyan-50 border-b">
//             <CardTitle className="flex items-center gap-2">
//               <Database className="w-5 h-5 text-blue-600" />
//               Dataset: ILPD
//             </CardTitle>
//             <CardDescription>Indian Liver Patient Dataset</CardDescription>
//           </CardHeader>
//           <CardContent className="p-6 space-y-3">
//             <p className="text-sm text-gray-700">
//               The system is trained on the <strong>Indian Liver Patient Dataset (ILPD)</strong>, which contains
//               laboratory test results from liver patients and healthy individuals.
//             </p>
//             <div className="bg-blue-50 p-3 rounded-lg">
//               <p className="text-xs text-blue-900 font-semibold mb-1">Dataset Features:</p>
//               <ul className="text-xs text-blue-800 space-y-1">
//                 <li>• Total Bilirubin (TB)</li>
//                 <li>• Direct Bilirubin (DB)</li>
//                 <li>• Alkaline Phosphatase</li>
//                 <li>• SGPT / ALT (Alamine Aminotransferase)</li>
//                 <li>• SGOT / AST (Aspartate Aminotransferase)</li>
//                 <li>• Total Proteins</li>
//                 <li>• Albumin</li>
//                 <li>• Albumin/Globulin Ratio</li>
//               </ul>
//             </div>
//             <p className="text-xs text-gray-600">
//               These features represent standard liver function tests commonly used in clinical practice.
//             </p>
//           </CardContent>
//         </Card>

//         {/* Model Information */}
//         <Card>
//           <CardHeader className="bg-gradient-to-r from-purple-50 to-pink-50 border-b">
//             <CardTitle className="flex items-center gap-2">
//               <Brain className="w-5 h-5 text-purple-600" />
//               Machine Learning Model
//             </CardTitle>
//             <CardDescription>AI Algorithm Details</CardDescription>
//           </CardHeader>
//           <CardContent className="p-6 space-y-3">
//             <p className="text-sm text-gray-700">
//               The prediction system uses advanced <strong>supervised machine learning algorithms</strong>{' '}
//               trained on historical patient data to identify patterns associated with liver disease.
//             </p>
//             <div className="space-y-2">
//               <div className="flex items-center gap-2">
//                 <Badge variant="outline" className="bg-purple-50 text-purple-700 border-purple-300">
//                   Classification
//                 </Badge>
//                 <span className="text-xs text-gray-600">Binary prediction (Disease/No Disease)</span>
//               </div>
//               <div className="flex items-center gap-2">
//                 <Badge variant="outline" className="bg-purple-50 text-purple-700 border-purple-300">
//                   Algorithms
//                 </Badge>
//                 <span className="text-xs text-gray-600">Random Forest / Logistic Regression / XGBoost</span>
//               </div>
//             </div>
//             <div className="bg-purple-50 p-3 rounded-lg">
//               <p className="text-xs text-purple-900">
//                 <strong>Model Training:</strong> The model is trained using cross-validation techniques to
//                 ensure robust performance across diverse patient populations.
//               </p>
//             </div>
//           </CardContent>
//         </Card>

//         {/* Explainability */}
//         <Card>
//           <CardHeader className="bg-gradient-to-r from-amber-50 to-orange-50 border-b">
//             <CardTitle className="flex items-center gap-2">
//               <BarChart3 className="w-5 h-5 text-amber-600" />
//               Model Explainability
//             </CardTitle>
//             <CardDescription>Understanding AI Decisions</CardDescription>
//           </CardHeader>
//           <CardContent className="p-6 space-y-3">
//             <p className="text-sm text-gray-700">
//               The system uses <strong>SHAP (SHapley Additive exPlanations)</strong> and{' '}
//               <strong>LIME (Local Interpretable Model-agnostic Explanations)</strong> techniques to provide
//               transparent insights into predictions.
//             </p>
//             <div className="bg-amber-50 p-3 rounded-lg space-y-2">
//               <p className="text-xs text-amber-900 font-semibold">Key Benefits:</p>
//               <ul className="text-xs text-amber-800 space-y-1">
//                 <li>✓ Identifies which features most influenced the prediction</li>
//                 <li>✓ Quantifies the contribution of each laboratory parameter</li>
//                 <li>✓ Helps clinicians understand the "why" behind predictions</li>
//                 <li>✓ Builds trust in AI-assisted decision making</li>
//               </ul>
//             </div>
//             <p className="text-xs text-gray-600">
//               This transparency is crucial for clinical adoption and helps physicians validate AI recommendations
//               against their domain expertise.
//             </p>
//           </CardContent>
//         </Card>

//         {/* Clinical Usage */}
//         <Card>
//           <CardHeader className="bg-gradient-to-r from-green-50 to-teal-50 border-b">
//             <CardTitle className="flex items-center gap-2">
//               <Activity className="w-5 h-5 text-green-600" />
//               Clinical Usage Guidelines
//             </CardTitle>
//             <CardDescription>Best Practices for Healthcare Providers</CardDescription>
//           </CardHeader>
//           <CardContent className="p-6 space-y-3">
//             <p className="text-sm text-gray-700">
//               This tool is designed as a <strong>clinical decision support system</strong>, not a replacement
//               for medical judgment.
//             </p>
//             <div className="space-y-2">
//               <div className="bg-green-50 p-3 rounded-lg">
//                 <p className="text-xs text-green-900 font-semibold mb-1">✓ Appropriate Uses:</p>
//                 <ul className="text-xs text-green-800 space-y-1">
//                   <li>• Screening and risk stratification</li>
//                   <li>• Supplement to clinical evaluation</li>
//                   <li>• Identifying patients needing further workup</li>
//                   <li>• Educational tool for medical trainees</li>
//                 </ul>
//               </div>
//               <div className="bg-red-50 p-3 rounded-lg">
//                 <p className="text-xs text-red-900 font-semibold mb-1">✗ Not Suitable For:</p>
//                 <ul className="text-xs text-red-800 space-y-1">
//                   <li>• Standalone diagnosis without clinical correlation</li>
//                   <li>• Emergency or critical care decisions</li>
//                   <li>• Replacing comprehensive patient evaluation</li>
//                   <li>• Legal or regulatory documentation</li>
//                 </ul>
//               </div>
//             </div>
//           </CardContent>
//         </Card>
//       </div>

//       {/* Important Information Banner */}
//       <Card className="border-2 border-amber-300 bg-amber-50">
//         <CardHeader>
//           <CardTitle className="flex items-center gap-2 text-amber-900">
//             <AlertCircle className="w-6 h-6 text-amber-600" />
//             Important Information & Limitations
//           </CardTitle>
//         </CardHeader>
//         <CardContent className="space-y-3">
//           <div className="space-y-2 text-sm text-amber-900">
//             <p>
//               <strong>Model Performance:</strong> While the model demonstrates good performance on validation
//               data, real-world clinical scenarios may vary. The model's accuracy depends on:
//             </p>
//             <ul className="list-disc list-inside ml-4 space-y-1 text-xs">
//               <li>Quality and accuracy of input laboratory values</li>
//               <li>Patient population characteristics matching the training dataset</li>
//               <li>Proper interpretation within appropriate clinical context</li>
//             </ul>

//             <p className="pt-2">
//               <strong>Known Limitations:</strong>
//             </p>
//             <ul className="list-disc list-inside ml-4 space-y-1 text-xs">
//               <li>Does not account for patient symptoms, history, or physical examination findings</li>
//               <li>Cannot detect specific liver disease subtypes (viral, alcoholic, autoimmune, etc.)</li>
//               <li>May not generalize well to populations significantly different from ILPD dataset</li>
//               <li>Cannot replace imaging studies or liver biopsy when indicated</li>
//             </ul>

//             <p className="pt-2">
//               <strong>Data Privacy:</strong> This demonstration system does not store or transmit patient data.
//               In production environments, appropriate HIPAA-compliant data handling procedures must be
//               implemented.
//             </p>
//           </div>
//         </CardContent>
//       </Card>

//       {/* Technical Specifications */}
//       <Card>
//         <CardHeader className="bg-gradient-to-r from-gray-50 to-slate-50 border-b">
//           <CardTitle className="flex items-center gap-2">
//             <FileText className="w-5 h-5 text-gray-600" />
//             Technical Specifications
//           </CardTitle>
//         </CardHeader>
//         <CardContent className="p-6">
//           <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm">
//             <div>
//               <h4 className="font-semibold text-gray-900 mb-2">System Architecture</h4>
//               <ul className="space-y-1 text-xs text-gray-700">
//                 <li>• Frontend: React + TypeScript</li>
//                 <li>• Styling: Tailwind CSS v4</li>
//                 <li>• Charts: Recharts library</li>
//                 <li>• Animations: Motion (Framer Motion)</li>
//               </ul>
//             </div>
//             <div>
//               <h4 className="font-semibold text-gray-900 mb-2">ML Integration</h4>
//               <ul className="space-y-1 text-xs text-gray-700">
//                 <li>• Model: Classification (Binary)</li>
//                 <li>• Explainability: SHAP/LIME-based</li>
//                 <li>• Output: Risk probability + factors</li>
//                 <li>• Processing: Real-time inference</li>
//               </ul>
//             </div>
//           </div>
//         </CardContent>
//       </Card>

//       {/* Contact/Support */}
//       <div className="bg-teal-50 border border-teal-200 rounded-lg p-6 text-center">
//         <Shield className="w-8 h-8 text-teal-600 mx-auto mb-2" />
//         <h4 className="font-semibold text-gray-900 mb-1">For Healthcare Professionals</h4>
//         <p className="text-sm text-gray-700">
//           This system is intended for use by licensed medical professionals, laboratory staff, and clinical
//           researchers. For technical support or to report issues, please contact your system administrator.
//         </p>
//       </div>
//     </div>
//   );
// }


import {
  Database,
  Brain,
  BarChart3,
  Shield,
  FileText,
  Activity,
  AlertCircle,
  Image as ImageIcon,
  Layers3,
  Microscope,
} from 'lucide-react';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from '@/app/components/ui/card';
import { Badge } from '@/app/components/ui/badge';

export function AboutSection() {
  return (
    <div className="max-w-5xl mx-auto py-8 px-4 space-y-6">
      {/* Header */}
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-gray-900 mb-2">
          About the Multimodal Liver AI System
        </h2>
        <p className="text-gray-600">
          Explainable AI framework combining clinical biomarkers and ultrasound
          imaging for liver disease risk assessment
        </p>
      </div>

      {/* Dataset Information */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardHeader className="bg-gradient-to-r from-blue-50 to-cyan-50 border-b">
            <CardTitle className="flex items-center gap-2">
              <Database className="w-5 h-5 text-blue-600" />
              Clinical Dataset
            </CardTitle>
            <CardDescription>
              Indian Liver Patient Dataset (ILPD)
            </CardDescription>
          </CardHeader>

          <CardContent className="p-6 space-y-3">
            <p className="text-sm text-gray-700">
              The clinical branch is developed using the{' '}
              <strong>Indian Liver Patient Dataset (ILPD)</strong>, containing
              demographic and laboratory measurements associated with liver
              disease.
            </p>

            <div className="bg-blue-50 p-3 rounded-lg">
              <p className="text-xs text-blue-900 font-semibold mb-1">
                Clinical Features:
              </p>

              <ul className="text-xs text-blue-800 space-y-1">
                <li>• Age and Gender</li>
                <li>• Total Bilirubin</li>
                <li>• Direct Bilirubin</li>
                <li>• Alkaline Phosphatase</li>
                <li>• SGPT / ALT</li>
                <li>• SGOT / AST</li>
                <li>• Total Proteins</li>
                <li>• Albumin</li>
                <li>• Albumin/Globulin Ratio</li>
              </ul>
            </div>

            <p className="text-xs text-gray-600">
              These measurements represent commonly used biochemical indicators
              of liver function and are used as inputs to the clinical
              prediction branch.
            </p>
          </CardContent>
        </Card>

        {/* Ultrasound Dataset */}
        <Card>
          <CardHeader className="bg-gradient-to-r from-purple-50 to-pink-50 border-b">
            <CardTitle className="flex items-center gap-2">
              <ImageIcon className="w-5 h-5 text-purple-600" />
              Ultrasound Analysis
            </CardTitle>

            <CardDescription>
              Deep learning-based liver ultrasound assessment
            </CardDescription>
          </CardHeader>

          <CardContent className="p-6 space-y-3">
            <p className="text-sm text-gray-700">
              The imaging branch analyzes liver ultrasound images using
              <strong> Vision Transformer (ViT-B/16)</strong> models trained
              for complementary liver steatosis and NAFLD assessment tasks.
            </p>

            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Badge
                  variant="outline"
                  className="bg-purple-50 text-purple-700 border-purple-300"
                >
                  Byra
                </Badge>

                <span className="text-xs text-gray-600">
                  Ultrasound-based steatosis assessment
                </span>
              </div>

              <div className="flex items-center gap-2">
                <Badge
                  variant="outline"
                  className="bg-purple-50 text-purple-700 border-purple-300"
                >
                  BEHSOF
                </Badge>

                <span className="text-xs text-gray-600">
                  Ultrasound-based NAFLD assessment
                </span>
              </div>
            </div>

            <div className="bg-purple-50 p-3 rounded-lg">
              <p className="text-xs text-purple-900">
                <strong>Imaging Model:</strong> ViT-B/16 with transfer learning
                and partial fine-tuning is used to extract high-level visual
                representations from liver ultrasound images.
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Multimodal Architecture */}
        <Card>
          <CardHeader className="bg-gradient-to-r from-indigo-50 to-blue-50 border-b">
            <CardTitle className="flex items-center gap-2">
              <Layers3 className="w-5 h-5 text-indigo-600" />
              Multimodal AI Architecture
            </CardTitle>

            <CardDescription>
              Combining complementary clinical and imaging evidence
            </CardDescription>
          </CardHeader>

          <CardContent className="p-6 space-y-3">
            <p className="text-sm text-gray-700">
              The system is designed around a{' '}
              <strong>multimodal liver disease assessment framework</strong>{' '}
              that combines biochemical information with deep visual features
              extracted from ultrasound images.
            </p>

            <div className="bg-indigo-50 p-4 rounded-lg">
              <div className="grid grid-cols-1 gap-3 text-xs">
                <div className="flex items-center gap-2">
                  <Database className="w-4 h-4 text-blue-600" />
                  <span>
                    <strong>Clinical branch:</strong> laboratory and demographic
                    features
                  </span>
                </div>

                <div className="flex items-center gap-2">
                  <ImageIcon className="w-4 h-4 text-purple-600" />
                  <span>
                    <strong>Imaging branch:</strong> ultrasound-based ViT
                    predictions
                  </span>
                </div>

                <div className="flex items-center gap-2">
                  <Layers3 className="w-4 h-4 text-indigo-600" />
                  <span>
                    <strong>Fusion:</strong> complementary evidence aggregation
                  </span>
                </div>

                <div className="flex items-center gap-2">
                  <Activity className="w-4 h-4 text-green-600" />
                  <span>
                    <strong>Output:</strong> liver disease risk assessment
                  </span>
                </div>
              </div>
            </div>

            <p className="text-xs text-gray-600">
              The multimodal framework is intended to investigate whether
              combining different sources of patient information can provide
              more informative risk assessment than individual modalities.
            </p>
          </CardContent>
        </Card>

        {/* Explainability */}
        <Card>
          <CardHeader className="bg-gradient-to-r from-amber-50 to-orange-50 border-b">
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-amber-600" />
              Model Explainability
            </CardTitle>

            <CardDescription>
              Understanding the factors behind AI predictions
            </CardDescription>
          </CardHeader>

          <CardContent className="p-6 space-y-3">
            <p className="text-sm text-gray-700">
              The clinical prediction branch uses{' '}
              <strong>SHAP (SHapley Additive exPlanations)</strong> to identify
              how individual laboratory and demographic features influence a
              prediction.
            </p>

            <div className="bg-amber-50 p-3 rounded-lg space-y-2">
              <p className="text-xs text-amber-900 font-semibold">
                Explainability Features:
              </p>

              <ul className="text-xs text-amber-800 space-y-1">
                <li>✓ Identifies influential clinical features</li>
                <li>✓ Shows direction of feature contribution</li>
                <li>✓ Provides patient-specific explanations</li>
                <li>✓ Displays contributions from individual AI models</li>
                <li>✓ Helps researchers inspect model behaviour</li>
              </ul>
            </div>

            <p className="text-xs text-gray-600">
              Explainability outputs are intended to support model
              interpretation and research analysis rather than replace
              professional clinical judgement.
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Research Objective */}
      <Card className="border-2 border-indigo-200 bg-indigo-50">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-indigo-900">
            <Brain className="w-6 h-6 text-indigo-600" />
            Research Objective
          </CardTitle>
        </CardHeader>

        <CardContent>
          <p className="text-sm text-indigo-900 leading-relaxed">
            The primary research objective is to investigate whether combining
            <strong> clinical biochemical information </strong>
            with
            <strong> ultrasound-based deep learning predictions </strong>
            can provide a more comprehensive framework for liver disease risk
            assessment. The architecture is inspired by recent multimodal AI
            research in steatotic liver disease while emphasizing
            explainability and reproducibility.
          </p>
        </CardContent>
      </Card>

      {/* Clinical Usage */}
      <Card>
        <CardHeader className="bg-gradient-to-r from-green-50 to-teal-50 border-b">
          <CardTitle className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-green-600" />
            Intended Usage
          </CardTitle>

          <CardDescription>
            Research and clinical decision-support context
          </CardDescription>
        </CardHeader>

        <CardContent className="p-6 space-y-2">
          <p className="text-sm text-gray-700">
            This system is designed as an{' '}
            <strong>AI-assisted research and decision-support tool</strong>.
            Its outputs should be interpreted alongside clinical history,
            physical examination, laboratory findings, imaging and professional
            medical judgement.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="bg-green-50 p-3 rounded-lg">
              <p className="text-xs text-green-900 font-semibold mb-1">
                Appropriate Uses
              </p>

              <ul className="text-xs text-green-800 space-y-1">
                <li>• Research and model evaluation</li>
                <li>• Risk stratification research</li>
                <li>• Multimodal AI experimentation</li>
                <li>• Educational demonstrations</li>
                <li>• Decision-support research</li>
              </ul>
            </div>

            <div className="bg-red-50 p-3 rounded-lg">
              <p className="text-xs text-red-900 font-semibold mb-1">
                Not Intended For
              </p>

              <ul className="text-xs text-red-800 space-y-1">
                <li>• Standalone medical diagnosis</li>
                <li>• Emergency treatment decisions</li>
                <li>• Replacing physician evaluation</li>
                <li>• Definitive disease staging without validation</li>
                <li>• Regulatory or legal medical decisions</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Important Limitations */}
      <Card className="border-2 border-amber-300 bg-amber-50">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-amber-900">
            <AlertCircle className="w-6 h-6 text-amber-600" />
            Important Information & Limitations
          </CardTitle>
        </CardHeader>

        <CardContent className="space-y-3">
          <div className="space-y-2 text-sm text-amber-900">
            <p>
              <strong>Model outputs are not clinical diagnoses.</strong> The
              predicted probabilities and evidence scores represent outputs of
              machine learning models and should not be interpreted as
              medically validated disease probabilities.
            </p>

            <p className="pt-2">
              <strong>Dataset limitations:</strong>
            </p>

            <ul className="list-disc list-inside ml-4 space-y-1 text-xs">
              <li>
                Clinical and ultrasound datasets may originate from different
                cohorts.
              </li>
              <li>
                Dataset distributions may differ from real-world patient
                populations.
              </li>
              <li>
                External clinical validation is required before real-world
                deployment.
              </li>
              <li>
                Ultrasound predictions can be affected by acquisition quality,
                operator variability and image characteristics.
              </li>
            </ul>

            <p className="pt-2">
              <strong>Multimodal fusion limitation:</strong>
            </p>

            <p className="text-xs">
              When clinical and ultrasound information comes from different
              datasets or patients, the resulting evidence aggregation should
              not be interpreted as a supervised multimodal prediction for a
              single patient. A properly paired clinical-ultrasound dataset is
              required to train and evaluate a fully supervised multimodal
              model.
            </p>

            <p className="pt-2">
              <strong>Privacy:</strong>
            </p>

            <p className="text-xs">
              Patient information should be handled according to applicable
              institutional privacy, security and data-governance requirements.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Technical Specifications */}
      <Card>
        <CardHeader className="bg-gradient-to-r from-gray-50 to-slate-50 border-b">
          <CardTitle className="flex items-center gap-2">
            <FileText className="w-5 h-5 text-gray-600" />
            Technical Specifications
          </CardTitle>
        </CardHeader>

        <CardContent className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm">
            <div>
              <h4 className="font-semibold text-gray-900 mb-2">
                Application
              </h4>

              <ul className="space-y-1 text-xs text-gray-700">
                <li>• Frontend: React + TypeScript</li>
                <li>• Styling: Tailwind CSS</li>
                <li>• UI Components: shadcn/ui</li>
                <li>• Charts: Recharts</li>
                <li>• Backend: FastAPI</li>
                <li>• Real-time model inference</li>
              </ul>
            </div>

            <div>
              <h4 className="font-semibold text-gray-900 mb-2">
                AI Pipeline
              </h4>

              <ul className="space-y-1 text-xs text-gray-700">
                <li>• Clinical ML ensemble</li>
                <li>• Vision Transformer (ViT-B/16)</li>
                <li>• Ultrasound steatosis analysis</li>
                <li>• Ultrasound NAFLD analysis</li>
                <li>• SHAP-based clinical explanations</li>
                <li>• Multimodal evidence aggregation</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Research Context */}
      <Card>
        <CardHeader className="bg-gradient-to-r from-cyan-50 to-blue-50 border-b">
          <CardTitle className="flex items-center gap-2">
            <Microscope className="w-5 h-5 text-cyan-600" />
            Research Context
          </CardTitle>

          <CardDescription>
            Multimodal AI for steatotic liver disease
          </CardDescription>
        </CardHeader>

        <CardContent className="p-6">
          <p className="text-sm text-gray-700 leading-relaxed">
            This project is inspired by recent multimodal artificial
            intelligence research for steatotic liver disease, including
            approaches that combine medical imaging with clinical and
            laboratory information. The current system investigates a more
            accessible and explainable architecture using liver ultrasound and
            routinely available clinical biomarkers.
          </p>

          <div className="mt-4 flex flex-wrap gap-2">
            <Badge variant="outline">Multimodal AI</Badge>
            <Badge variant="outline">Liver Steatosis</Badge>
            <Badge variant="outline">Ultrasound</Badge>
            <Badge variant="outline">Clinical Biomarkers</Badge>
            <Badge variant="outline">Vision Transformer</Badge>
            <Badge variant="outline">Explainable AI</Badge>
          </div>
        </CardContent>
      </Card>

      {/* Healthcare Professionals */}
      <div className="bg-teal-50 border border-teal-200 rounded-lg p-6 text-center">
        <Shield className="w-8 h-8 text-teal-600 mx-auto mb-2" />

        <h4 className="font-semibold text-gray-900 mb-1">
          For Healthcare Professionals & Researchers
        </h4>

        <p className="text-sm text-gray-700">
          This system is intended primarily for research, education and
          decision-support experimentation. Model outputs should always be
          interpreted by appropriately qualified healthcare professionals and
          should not be used as a standalone diagnostic conclusion.
        </p>
      </div>
    </div>
  );
}