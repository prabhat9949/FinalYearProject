import { Activity, Home, HelpCircle, Upload, ClipboardList, AlertCircle } from 'lucide-react';
import { Button } from '@/app/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/app/components/ui/dialog';
import { Badge } from '@/app/components/ui/badge';

type ViewMode = 'home' | 'predict' | 'bulk' | 'help';

interface DashboardHeaderProps {
  currentView: ViewMode;
  onViewChange: (view: ViewMode) => void;
}

export function DashboardHeader({ currentView, onViewChange }: DashboardHeaderProps) {
  return (
  <header className="sticky top-0 z-50 bg-white border-b border-gray-200 shadow-sm">
    <div className="max-w-7xl mx-auto px-3 sm:px-5 lg:px-8">

      {/* ================= HEADER ================= */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 py-4">

        {/* Logo */}
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-11 h-11 sm:w-12 sm:h-12 rounded-xl bg-gradient-to-br from-teal-500 to-cyan-600 shrink-0">
            <Activity className="w-6 h-6 text-white" />
          </div>

          <div className="min-w-0">
            <h1 className="text-lg sm:text-xl lg:text-2xl font-semibold text-gray-900 leading-tight">
              Liver Disease Prediction System
            </h1>

            <p className="hidden sm:block text-sm text-gray-600">
              ILPD ML-Based Decision Support Dashboard for Clinical Use
            </p>
          </div>
        </div>

        {/* Badge */}
        <div className="self-start lg:self-auto">
          <Badge
            variant="outline"
            className="bg-teal-50 border-teal-300 text-teal-700 px-3 py-1"
          >
            <Activity className="w-3 h-3 mr-1" />
            Clinical System
          </Badge>
        </div>
      </div>

      {/* ================= NAVIGATION ================= */}

      <nav className="border-t border-gray-100 py-3">

        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">

          {/* Navigation Buttons */}

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 w-full lg:w-auto">

            <Button
              variant={currentView === "home" ? "default" : "ghost"}
              size="sm"
              onClick={() => onViewChange("home")}
              className={`w-full ${
                currentView === "home"
                  ? "bg-teal-600 hover:bg-teal-700"
                  : ""
              }`}
            >
              <Home className="w-4 h-4 mr-2" />
              Home
            </Button>

            <Button
              variant={currentView === "predict" ? "default" : "ghost"}
              size="sm"
              onClick={() => onViewChange("predict")}
              className={`w-full ${
                currentView === "predict"
                  ? "bg-teal-600 hover:bg-teal-700"
                  : ""
              }`}
            >
              <ClipboardList className="w-4 h-4 mr-2" />
              Individual  Check
            </Button>

            <Button
              variant={currentView === "bulk" ? "default" : "ghost"}
              size="sm"
              onClick={() => onViewChange("bulk")}
              className={`w-full ${
                currentView === "bulk"
                  ? "bg-teal-600 hover:bg-teal-700"
                  : ""
              }`}
            >
              <Upload className="w-4 h-4 mr-2" />
              Bulk
            </Button>

            <Button
              variant={currentView === "help" ? "default" : "ghost"}
              size="sm"
              onClick={() => onViewChange("help")}
              className={`w-full ${
                currentView === "help"
                  ? "bg-teal-600 hover:bg-teal-700"
                  : ""
              }`}
            >
              <HelpCircle className="w-4 h-4 mr-2" />
              Help
            </Button>

          </div>

          {/* Disclaimer */}

          <Dialog>
            <DialogTrigger asChild>
              <Button
                variant="outline"
                className="w-full lg:w-auto border-amber-300 bg-amber-50 text-amber-800 hover:bg-amber-100"
              >
                <AlertCircle className="w-4 h-4 mr-2" />
                Important Disclaimer
              </Button>
            </DialogTrigger>

            <DialogContent className="max-w-[95vw] sm:max-w-2xl rounded-xl max-h-[90vh] overflow-y-auto">

              <DialogHeader>

                <DialogTitle className="flex items-center gap-2 text-lg sm:text-xl">
                  <AlertCircle className="w-5 h-5 text-amber-600" />
                  Medical Disclaimer
                </DialogTitle>

                <DialogDescription className="pt-4 space-y-4 text-sm sm:text-base">

                  <div className="bg-amber-50 border-l-4 border-amber-500 p-4 rounded">
                    <p className="font-semibold text-amber-900">
                      This tool provides ML-based risk prediction support. It is NOT
                      a medical diagnosis.
                    </p>
                  </div>

                  <div className="space-y-3 text-gray-700">

                    <ul className="list-disc list-inside space-y-2">

                      <li>
                        This system uses machine learning algorithms trained on
                        the Indian Liver Patient Dataset (ILPD).
                      </li>

                      <li>
                        Predictions should only support clinical judgment.
                      </li>

                      <li>
                        Final diagnosis and treatment decisions must be made by
                        qualified healthcare professionals.
                      </li>

                      <li>
                        This tool is not a substitute for medical advice.
                      </li>

                      <li>
                        Always consult qualified physicians.
                      </li>

                      <li>
                        Prediction quality depends on accurate patient data.
                      </li>

                    </ul>

                    <div className="bg-gray-50 rounded-lg p-4">
                      <p className="text-sm">
                        <strong>For Healthcare Professionals:</strong> Validate
                        model predictions with standard clinical protocols.
                      </p>
                    </div>

                  </div>

                </DialogDescription>

              </DialogHeader>

            </DialogContent>

          </Dialog>

        </div>

      </nav>

    </div>
  </header>
);
  
}
